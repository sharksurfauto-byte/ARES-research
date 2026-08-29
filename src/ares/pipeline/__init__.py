"""ARES Pipeline Module (PRD §3, §4.1, §5.2).

Provides the complete end-to-end ARES runtime pipeline, baseline comparisons,
metric calculations, and evaluation reporting.
"""

from .ares_pipeline import (
    ARESPipeline,
    PipelineConfig,
    PipelineResult,
    DEFAULT_EXPERT_NAMES,
)
from .baselines import (
    BaselineComparator,
    BaselineSampleResult,
    DEFAULT_BASELINES,
)
from .metrics import (
    EvaluationReport,
    MetricsCalculator,
    ReliabilityMetrics,
    StrategyMetrics,
)

__all__ = [
    "ARESPipeline",
    "PipelineConfig",
    "PipelineResult",
    "DEFAULT_EXPERT_NAMES",
    "BaselineComparator",
    "BaselineSampleResult",
    "DEFAULT_BASELINES",
    "EvaluationReport",
    "MetricsCalculator",
    "ReliabilityMetrics",
    "StrategyMetrics",
]
