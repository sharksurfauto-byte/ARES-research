"""Temperature scaling for reliability calibration (PRD §4.6).

Fits a temperature parameter T that scales logits to improve calibration.
Minimizes Negative Log-Likelihood (NLL) on a validation set.
"""

from typing import Any

import torch
import torch.nn as nn


class TemperatureScaling(nn.Module):
    """Learnable temperature scaling for probabilistic calibration.

    Based on PRD §4.6: After all training:
    1. Collect reliability scores R(x) and correctness labels on validation set
    2. Fit temperature scaling: optimal T that minimizes NLL
    3. Fit isotonic regression on (R(x), correctness) for non-parametric calibration
    4. Report: ECE before/after calibration, Brier score, reliability diagrams
    """

    def __init__(self, model: nn.Module | None = None, device: torch.device | None = None):
        """Initialize temperature scaling.

        Args:
            model: Optional model to calibrate (should output logits)
            device: Computation device
        """
        super().__init__()
        self.model = model
        self.device = device or torch.device("cpu")
        self.temperature = nn.Parameter(torch.tensor(1.0, device=self.device))

        if self.model is not None:
            # Freeze all model parameters except temperature
            for param in self.model.parameters():
                param.requires_grad = False

        self.temperature.requires_grad = True

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling to logits.

        Args:
            logits: [batch, ...] model logits

        Returns:
            Scaled logits: logits / temperature
        """
        return logits / self.temperature

    def fit(
        self,
        val_logits: torch.Tensor,
        val_labels: torch.Tensor,
        epochs: int = 20,
        lr: float = 0.01,
        use_val_set: bool = True,
    ) -> dict[str, Any]:
        """Fit the temperature parameter to minimize NLL.

        Args:
            val_logits: [N, ...] validation logits
            val_labels: [N] binary correctness labels (0/1)
            epochs: Number of optimization epochs
            lr: Learning rate for temperature
            use_val_set: Whether to use a separate val set (not used here, early stopping)

        Returns:
            Dictionary with fitted temperature and metrics
        """
        # Move to device and detach inputs to avoid backward propagation into model
        val_logits = val_logits.to(self.device).detach()
        val_labels = val_labels.to(self.device).float().detach()

        # Enable grad explicitly even if caller is inside torch.no_grad()
        with torch.enable_grad():
            # Optimizer for temperature only
            optimizer = torch.optim.RMSprop([self.temperature], lr=lr, alpha=0.99)

            best_nll = float("inf")
            best_temperature = self.temperature.item()

            # Training loop
            for epoch in range(epochs):
                optimizer.zero_grad()

                # Apply temperature and compute NLL
                scaled_logits = self.forward(val_logits)
                probs = torch.sigmoid(scaled_logits)  # For binary classification
                nll = -(
                    val_labels * torch.log(probs + 1e-7)
                    + (1 - val_labels) * torch.log(1 - probs + 1e-7)
                ).mean()

                nll.backward()
                optimizer.step()

                # Track best temperature
                if nll.item() < best_nll:
                    best_nll = nll.item()
                    best_temperature = self.temperature.item()

        # Set to best temperature
        self.temperature.data = torch.tensor(best_temperature, device=self.device)

        return {
            "temperature": best_temperature,
            "best_nll": best_nll,
            "final_nll": nll.item(),
        }

    @staticmethod
    def expected_calibration_error(
        probabilities: torch.Tensor,
        confidence: torch.Tensor,
        n_bins: int = 10,
    ) -> float:
        """Compute Expected Calibration Error (ECE).

        Args:
            probabilities: [N] predicted probabilities
            confidence: [N] observed frequencies in each bin
            n_bins: Number of bins for ECE computation

        Returns:
            ECE value (lower is better)
        """
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = torch.tensor(0.0)
        for i in range(n_bins):
            # Calculate the fraction of samples in this bin
            in_bin = (probabilities > bin_lowers[i]) & (probabilities <= bin_uppers[i])
            prop_in_bin = in_bin.float().mean()

            if prop_in_bin.item() > 0:
                # Calculate the average confidence and accuracy in this bin
                avg_confidence_in_bin = probabilities[in_bin].mean()
                accuracy_in_bin = confidence[in_bin].mean()
                # Add to ECE
                ece += prop_in_bin * torch.abs(avg_confidence_in_bin - accuracy_in_bin).item()

        return ece.item()


def fit_temperature_scaling(
    logits: torch.Tensor,
    labels: torch.Tensor,
    epochs: int = 20,
    lr: float = 0.01,
) -> dict[str, Any]:
    """Standalone temperature scaling fit (convenience function).

    Args:
        logits: [N] or [N, 1] logits to calibrate
        labels: [N] binary labels
        epochs: Number of optimization epochs
        lr: Learning rate

    Returns:
        Dictionary with fitted temperature and metrics
    """
    device = logits.device if isinstance(logits, torch.Tensor) else torch.device("cpu")
    scaler = TemperatureScaling(device=device)
    return scaler.fit(logits, labels, epochs=epochs, lr=lr)
