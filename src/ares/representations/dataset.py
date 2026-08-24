"""Representation Dataset for loading and managing saved representations (PRD §3.2.2).

Provides a PyTorch Dataset interface over collected backbone representations.
"""

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .collector import RepresentationSample

DOMAIN_MAP = {
    "general": 0,
    "math": 1,
    "code": 2,
    "science": 3,
    "reasoning": 4,
}


class RepresentationDataset(Dataset):
    """PyTorch Dataset for multi-layer backbone representations."""

    def __init__(
        self,
        samples: list[RepresentationSample] | None = None,
        representations: list[torch.Tensor] | None = None,
    ):
        """Initialize dataset.

        Args:
            samples: List of RepresentationSample objects
            representations: List of representation tensors
        """
        self.samples = samples or []
        self.representations = representations or []

    def __len__(self) -> int:
        if self.samples:
            return len(self.samples)
        return len(self.representations)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item: dict[str, Any] = {}
        if idx < len(self.representations):
            item["representation"] = self.representations[idx]

        if idx < len(self.samples):
            sample = self.samples[idx]
            item["sample_id"] = sample.sample_id
            item["domain"] = sample.domain
            item["domain_id"] = DOMAIN_MAP.get(sample.domain, 0)
            item["task"] = sample.task
            item["correctness"] = float(sample.correctness)
            item["confidence"] = sample.confidence
            item["entropy"] = sample.entropy
            item["margin"] = sample.margin
            if "representation" not in item and sample.representation is not None:
                item["representation"] = sample.representation

        return item

    def get_tensors(self) -> dict[str, torch.Tensor]:
        """Convert dataset into stacked PyTorch tensors.

        Returns:
            Dict containing 'representations', 'domain_labels', 'feasibility_labels'
        """
        if len(self) == 0:
            raise ValueError("Cannot extract tensors from empty RepresentationDataset")

        # Extract representations
        if self.representations:
            reps = torch.stack(
                [
                    r.detach().cpu() if isinstance(r, torch.Tensor) else torch.tensor(r)
                    for r in self.representations
                ]
            )
        elif self.samples:
            reps = torch.stack(
                [
                    (
                        s.representation.detach().cpu()
                        if isinstance(s.representation, torch.Tensor)
                        else torch.tensor(s.representation)
                    )
                    for s in self.samples
                ]
            )
        else:
            raise ValueError("No representation data found")

        # Flatten multi-dim representations if stacked as 2D/3D
        if reps.dim() > 2:
            reps = reps.view(reps.size(0), -1)

        # Extract labels
        if self.samples:
            domains = torch.tensor(
                [DOMAIN_MAP.get(s.domain, 0) for s in self.samples], dtype=torch.long
            )
            feasibility = torch.tensor(
                [1.0 if s.correctness else 0.0 for s in self.samples], dtype=torch.float32
            )
        else:
            domains = torch.zeros(len(self), dtype=torch.long)
            feasibility = torch.ones(len(self), dtype=torch.float32)

        return {
            "representations": reps,
            "domain_labels": domains,
            "feasibility_labels": feasibility,
        }

    def save(self, path: str | Path) -> str:
        """Save dataset to .pt file.

        Args:
            path: Output file path

        Returns:
            Path string
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "samples": self.samples,
                "representations": self.representations,
            },
            path,
        )
        return str(path)

    @classmethod
    def load(cls, path: str | Path) -> "RepresentationDataset":
        """Load dataset from .pt file.

        Args:
            path: File path to load

        Returns:
            RepresentationDataset instance
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Representation dataset not found at {path}")
        data = torch.load(path, weights_only=False)
        return cls(
            samples=data.get("samples", []),
            representations=data.get("representations", []),
        )

    def train_test_split(
        self, test_fraction: float = 0.1, seed: int = 42
    ) -> tuple["RepresentationDataset", "RepresentationDataset"]:
        """Split dataset into train and validation datasets.

        Args:
            test_fraction: Fraction for validation set
            seed: Random seed

        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        n = len(self)
        n_val = max(1, int(n * test_fraction))
        n_train = n - n_val

        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n, generator=g).tolist()

        train_indices = perm[:n_train]
        val_indices = perm[n_train:]

        train_samples = [self.samples[i] for i in train_indices] if self.samples else []
        val_samples = [self.samples[i] for i in val_indices] if self.samples else []

        train_reps = (
            [self.representations[i] for i in train_indices] if self.representations else []
        )
        val_reps = [self.representations[i] for i in val_indices] if self.representations else []

        return (
            RepresentationDataset(samples=train_samples, representations=train_reps),
            RepresentationDataset(samples=val_samples, representations=val_reps),
        )
