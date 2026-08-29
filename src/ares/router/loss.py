"""Router Loss Functions and Oracle Target Generation (PRD §4.4).

Implements:
1. Switch Transformer auxiliary load-balancing loss (Fedus et al., 2021).
2. Oracle routing target generator for supervised router pretraining.
3. Composite RouterLoss combining classification loss and load-balancing loss.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


def generate_oracle_targets(
    domain_labels: torch.Tensor,
    correctness_labels: torch.Tensor,
    mode: Literal["oracle", "expert_only", "base_only"] = "oracle",
    feasibility_threshold: float = 0.5,
) -> torch.Tensor:
    """Generate oracle routing targets for supervised router pretraining (PRD §4.4 Option A).

    Strategy:
    - If base model is correct / feasible (correctness >= threshold): route to base (class 0).
    - If base model is incorrect / unreliable (correctness < threshold): route to the specialized
      expert for that domain (class domain_id + 1).

    Args:
        domain_labels: [batch] domain indices (0=general, 1=math, 2=code, 3=science, 4=reasoning)
        correctness_labels: [batch] correctness/feasibility scores in [0, 1] or binary {0, 1}
        mode: Target generation mode ('oracle', 'expert_only', 'base_only')
        feasibility_threshold: Threshold above which base model is considered correct

    Returns:
        oracle_targets: [batch] target route index in [0, n_experts] (0 = base, 1..5 = expert_0..4)
    """
    domain_labels = domain_labels.long()
    correctness_labels = correctness_labels.float()

    if mode == "base_only":
        return torch.zeros_like(domain_labels)
    elif mode == "expert_only":
        # Always route to domain expert (1..5)
        return torch.clamp(domain_labels + 1, 1, 5)

    # Oracle mode: base if correct, domain expert if wrong
    is_correct = correctness_labels >= feasibility_threshold
    # Default to base (0)
    targets = torch.zeros_like(domain_labels)
    # For incorrect samples, set target = domain_id + 1 (clamp to valid expert range [1, 5])
    expert_targets = torch.clamp(domain_labels + 1, 1, 5)
    targets = torch.where(is_correct, torch.zeros_like(targets), expert_targets)

    return targets


class SwitchLoadBalancingLoss(nn.Module):
    """Switch Transformer auxiliary load-balancing loss (Fedus et al., 2021).

    Loss formula:
        L_balance = alpha * N * sum_{i=1}^N (f_i * P_i)

    where:
        N = total number of routes (n_experts + 1)
        f_i = fraction of inputs dispatched to route i (discrete argmax, detached)
        P_i = average routing probability assigned to route i (differentiable)
        alpha = loss coefficient multiplier (default: 0.01)

    When routing is perfectly uniform across all routes (f_i = 1/N, P_i = 1/N),
    L_balance = alpha * N * sum(1/N^2) = alpha * 1.0.
    When routing is imbalanced, L_balance increases.
    """

    def __init__(self, n_classes: int = 6, coeff: float = 0.01):
        """Initialize Switch Transformer load balancing loss.

        Args:
            n_classes: Total number of routing classes (e.g. 6 = base + 5 experts)
            coeff: Loss weighting coefficient (alpha)
        """
        super().__init__()
        self.n_classes = n_classes
        self.coeff = coeff

    def forward(self, routing_probs: torch.Tensor) -> torch.Tensor:
        """Compute load balancing auxiliary loss.

        Args:
            routing_probs: Routing probabilities [batch, n_classes] or [batch, seq_len, n_classes]

        Returns:
            Scalar load-balancing loss tensor with gradient tracking to routing_probs.
        """
        if routing_probs.dim() == 3:
            routing_probs = routing_probs.view(-1, routing_probs.size(-1))

        batch_size, n_classes = routing_probs.shape
        if batch_size == 0:
            return torch.tensor(0.0, device=routing_probs.device, requires_grad=True)

        # Discrete assignment (f_i): fraction of tokens routed to expert i
        # Detached so gradients only flow through P_i per Switch Transformer formulation
        with torch.no_grad():
            selected = routing_probs.argmax(dim=-1)
            # Create one-hot dispatch counts
            f = torch.zeros(n_classes, device=routing_probs.device, dtype=torch.float32)
            for i in range(n_classes):
                f[i] = (selected == i).float().mean()

        # Mean routing probability assigned to expert i (P_i)
        p = routing_probs.mean(dim=0)

        # L_balance = n_classes * sum(f_i * P_i)
        loss = n_classes * (f * p).sum()
        return self.coeff * loss


class RouterLoss(nn.Module):
    """Composite loss function for training the Router MLP.

    Combines:
    1. Supervised Cross-Entropy loss against oracle routing targets.
    2. Switch Transformer load-balancing auxiliary loss.
    3. Optional confidence regularization.
    """

    def __init__(
        self,
        n_classes: int = 6,
        lambda_lb: float = 0.01,
        label_smoothing: float = 0.0,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.ce_criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing,
        )
        self.lb_criterion = SwitchLoadBalancingLoss(n_classes=n_classes, coeff=lambda_lb)
        self.lambda_lb = lambda_lb

    def forward(
        self,
        logits: torch.Tensor,
        routing_probs: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute composite router training loss.

        Args:
            logits: Unnormalized logits [batch, n_classes]
            routing_probs: Softmax routing probabilities [batch, n_classes]
            targets: Oracle routing target indices [batch]

        Returns:
            Tuple of (total_loss, dict_of_loss_components)
        """
        ce_loss = self.ce_criterion(logits, targets)
        lb_loss = self.lb_criterion(routing_probs)

        total_loss = ce_loss + lb_loss

        losses = {
            "total_loss": total_loss,
            "ce_loss": ce_loss,
            "lb_loss": lb_loss,
        }
        return total_loss, losses
