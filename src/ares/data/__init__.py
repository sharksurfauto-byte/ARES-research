"""ARES Data Module (PRD §4.1).

Provides multi-domain dataset loaders, benchmark evaluation functions,
and representation harvesting tools.
"""

from .domain_datasets import load_domain_dataset, SyntheticDataset, EXPERT_DATASET_MAP
from .benchmark_loader import (
    BenchmarkSample,
    evaluate_prediction,
    extract_math_answer,
    extract_mcq_answer,
    load_gsm8k_samples,
    load_mbpp_samples,
    load_ai2_arc_samples,
    load_wikitext_samples,
    load_reasoning_samples,
    load_all_benchmark_samples,
)

__all__ = [
    "load_domain_dataset",
    "SyntheticDataset",
    "EXPERT_DATASET_MAP",
    "BenchmarkSample",
    "evaluate_prediction",
    "extract_math_answer",
    "extract_mcq_answer",
    "load_gsm8k_samples",
    "load_mbpp_samples",
    "load_ai2_arc_samples",
    "load_wikitext_samples",
    "load_reasoning_samples",
    "load_all_benchmark_samples",
]
