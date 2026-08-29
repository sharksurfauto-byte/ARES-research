"""Domain-Specific Datasets for ARES Expert Training.

Provides clean dataset loaders for each of the 5 expert domains,
leveraging Hugging Face datasets with graceful offline/synthetic fallbacks.
"""

from __future__ import annotations

import os
import sys
import torch
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch.nn as nn

# Add src to path for ares imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from datasets import Dataset, DatasetDict, load_dataset

# ─── Dataset Mappings ───────────────────────────────────────────────────────

EXPERT_DATASET_MAP: dict[str, dict[str, str]] = {
    "general": {"hf_name": "wikitext", "hf_config": "wikitext-103-raw-v1", "split": "train"},
    "math": {"hf_name": "gsm8k", "hf_config": "main", "split": "train"},
    "code": {"hf_name": "mbpp", "hf_config": "default", "split": "train"},
    "science": {"hf_name": "ai2_arc", "hf_config": "ARC-Challenge", "split": "train"},
    "reasoning": {"hf_name": "custom_reasoning", "hf_config": None, "split": "train"},
}
DOMAIN_CONFIGS = EXPERT_DATASET_MAP

# ─── Fallback Synthetic Generator ──────────────────────────────────────────

class SyntheticDataset:
    """Generates synthetic domain data when HF datasets are unavailable.

    Returns a plain Dataset so callers can use ds["text"], ds["domain"], etc.
    """

    def __init__(self, domain: str, n_samples: int = 1000):
        self.domain = domain
        self.n_samples = n_samples
        # Domain-specific text patterns
        self.patterns: dict[str, list[str]] = {
            "general": [
                "Artificial intelligence is important because",
                "The future of technology will",
                "Climate change affects",
                "Quantum computing relies on",
                "The economy depends on",
            ],
            "math": [
                "John has $x apples. He buys y more. How many does he have now?",
                "If it takes 3 hours to travel 180 km, what is the speed in km/h?",
                "Solve for x: 2x + 5 = 15",
                "The perimeter of a rectangle is 30. If the length is 8, what is the width?",
                "What is the next number in: 2, 5, 10, 17, ...?",
            ],
            "code": [
                "def greet(name): return f'Hello {name}'",
                "def fibonacci(n): a, b = 0, 1\nfor _ in range(n): a, b = b, a + b\nreturn a",
                "import torch\nx = torch.tensor([1, 2, 3])\nprint(x.shape)",
                "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + [pivot] + quicksort(right)",
                "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, email TEXT)",
            ],
            "science": [
                "The chemical formula for water is H2O",
                "Photosynthesis converts light energy into chemical energy",
                "The mitochondria is the powerhouse of the cell",
                "Newton's first law states that an object in motion stays in motion",
                "The periodic table organizes elements by atomic number",
            ],
            "reasoning": [
                "If all cats are mammals and some mammals are dogs, which conclusion is valid?",
                "John is taller than Mary. Mary is taller than Peter. Who is the tallest?",
                "If it rains, the ground gets wet. The ground is wet. Does it necessarily mean it rained?",
                "Complete the sequence: 1, 4, 9, 16, 25, ?",
                "Alice is older than Bob. Bob is older than Carol. Who is the youngest?",
            ],
        }

    def __getitem__(self, idx: int) -> dict[str, Any]:
        import random
        pattern = random.choice(self.patterns[self.domain])
        return {"text": pattern, "domain": self.domain, "idx": idx}

    def __len__(self):
        return self.n_samples


def _synthetic_to_dataset(domain: str, n_samples: int) -> Dataset:
    """Convert SyntheticDataset to a Hugging Face Dataset."""
    sd = SyntheticDataset(domain, n_samples)
    # Build lists from iterating the synthetic dataset
    import random
    random.seed(42)
    texts: list[str] = []
    domains: list[str] = []
    for i in range(n_samples):
        item = sd.__getitem__(i)
        texts.append(item["text"])
        domains.append(item["domain"])
    return Dataset.from_dict({"text": texts, "domain": domains})

# ─── Dataset Loaders ────────────────────────────────────────────────────────

def load_wikitext(n_samples: int = 1000) -> Dataset:
    """Load wikitext-103 for general expert."""
    try:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=f"train[:{n_samples * 4}]")
        texts = [t.strip() for t in ds["text"] if t.strip()][:n_samples]
        if len(texts) < n_samples:
            fallback = _synthetic_to_dataset("general", n_samples - len(texts))
            texts.extend(fallback["text"])
        return Dataset.from_dict({"text": texts, "domain": ["general"] * len(texts)})
    except Exception as e:
        print(f"[Wikitext] HF load failed: {e}. Using synthetic fallback.")
        return _synthetic_to_dataset("general", n_samples)


def load_gsm8k(n_samples: int = 1000) -> Dataset:
    """Load GSM8K for math expert."""
    try:
        ds = load_dataset("gsm8k", "main", split=f"train[:{n_samples * 2}]")
        # Keep question portion
        questions = [q.strip() for q in ds["question"][:n_samples]]
        if len(questions) < n_samples:
            fallback = _synthetic_to_dataset("math", n_samples - len(questions))
            questions.extend(fallback["text"])
        return Dataset.from_dict({"text": questions, "domain": ["math"] * len(questions)})
    except Exception as e:
        print(f"[GSM8K] HF load failed: {e}. Using synthetic fallback.")
        return _synthetic_to_dataset("math", n_samples)


def load_mbpp(n_samples: int = 1000) -> Dataset:
    """Load MBPP for code expert."""
    for cfg_name in ["sanitized", "full", "default"]:
        try:
            ds = load_dataset("mbpp", cfg_name, split=f"train[:{n_samples * 2}]")
            col = "text" if "text" in ds.column_names else ("prompt" if "prompt" in ds.column_names else ds.column_names[0])
            texts = [str(t)[:300] for t in ds[col][:n_samples]]
            if len(texts) < n_samples:
                fallback = _synthetic_to_dataset("code", n_samples - len(texts))
                texts.extend(fallback["text"])
            return Dataset.from_dict({"text": texts, "domain": ["code"] * len(texts)})
        except Exception:
            continue
    print("[MBPP] HF load failed. Using synthetic fallback.")
    return _synthetic_to_dataset("code", n_samples)


def load_ai2_arc(n_samples: int = 1000) -> Dataset:
    """Load AI2 ARC for science expert."""
    try:
        ds = load_dataset("ai2_arc", "ARC-Challenge", split="train[:1%]")
        limit = min(n_samples, len(ds))
        questions = []
        for i in range(limit):
            row = ds[i]
            q = row.get("question", "")
            choices = row.get("choices", {})
            if choices and isinstance(choices, dict):
                text_list = choices.get("text", [])
                label_list = choices.get("label", [])
                if label_list and text_list:
                    choice_text = " ".join([f"{l}: {t}" for l, t in zip(label_list, text_list)])
                    questions.append(f"{q} {choice_text}")
                else:
                    questions.append(str(q))
            else:
                questions.append(str(q))
        return Dataset.from_dict({"text": questions, "domain": ["science"] * len(questions)})
    except Exception as e:
        print(f"[AI2-ARC] HF load failed: {e}. Using synthetic fallback.")
        return _synthetic_to_dataset("science", n_samples)


def load_custom_reasoning(n_samples: int = 1000) -> Dataset:
    """Load custom reasoning dataset."""
    for name in ["commonsense_qa", "piqa", "openbookqa"]:
        try:
            ds = load_dataset(name, split="train[:1%]")
            limit = min(n_samples, len(ds))
            questions = []
            for i in range(limit):
                row = ds[i]
                q = row.get("question", "")
                if not q and "goal" in row:
                    q = row["goal"]
                questions.append(str(q) if q else str(row))
            return Dataset.from_dict({"text": questions, "domain": ["reasoning"] * len(questions)})
        except Exception:
            continue
    print("[Reasoning] No HF dataset found. Using synthetic fallback.")
    return _synthetic_to_dataset("reasoning", n_samples)


# ─── Loader Registry ────────────────────────────────────────────────────────

DATASET_LOADERS: dict[str, callable] = {
    "general": load_wikitext,
    "math": load_gsm8k,
    "code": load_mbpp,
    "science": load_ai2_arc,
    "reasoning": load_custom_reasoning,
}

# ─── Public API ────────────────────────────────────────────────────────.....

def load_domain_dataset(domain: str, n_samples: int = 500) -> Dataset:
    """Load dataset for a given expert domain.

    Args:
        domain: One of "general", "math", "code", "science", "reasoning"
        n_samples: Number of samples to load (takes first N for speed)

    Returns:
        Dataset with fields: text, domain
    """
    loader = DATASET_LOADERS.get(domain)
    if loader is None:
        raise ValueError(f"Unknown domain: {domain}. Valid: {list(DATASET_LOADERS.keys())}")

    try:
        ds = loader(n_samples=n_samples)
        # Ensure it's a proper Dataset object (not SyntheticDataset)
        if not isinstance(ds, Dataset):
            ds = _synthetic_to_dataset(domain, n_samples)
        return ds
    except Exception as e:
        # Fallback to synthetic on any error
        print(f"[{domain}] Error: {e}. Using synthetic fallback.")
        return _synthetic_to_dataset(domain, n_samples)


def get_domain_vocab(domain: str, n_examples: int = 10) -> List[str]:
    """Get a few example texts from a domain for inspection."""
    ds = load_domain_dataset(domain, n_samples=n_examples)
    return list(ds["text"][:n_examples])


# ─── Quick Demo ────────────────────────────────────────────────────────.....

if __name__ == "__main__":
    print("=== ARES Domain Datasets ===\n")
    for domain in ["general", "math", "code", "science", "reasoning"]:
        ds = load_domain_dataset(domain, n_samples=3)
        print(f"▸ {domain:12s}: {len(ds)} samples")
        for i, ex in enumerate(ds):
            print(f"    {i+1}. {ex['text'][:80]}...")
        print()