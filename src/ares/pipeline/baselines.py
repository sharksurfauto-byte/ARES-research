"""Baseline Comparison Suite for ARES (PRD §4.1, §5.2).

Implements systematic evaluation across routing strategies:
1. Base Model (no expert invoked)
2. Fixed Expert (always route to designated expert e.g. math/code)
3. Dynamic ARES (learned routing based on representation & dual reliability)
4. Threshold Router (route to expert only when reliability < tau)
5. Random Router (stochastic selection across routes)
6. Oracle Router (perfect domain expert routing using ground truth)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ares.data.benchmark_loader import BenchmarkSample, evaluate_prediction
from ares.pipeline.ares_pipeline import ARESPipeline, PipelineResult


DEFAULT_BASELINES = [
    "BASE",
    "FIXED_EXPERT",
    "DYNAMIC_ARES",
    "THRESHOLD_ROUTER",
    "RANDOM_ROUTER",
    "ORACLE_ROUTER",
]


@dataclass
class BaselineSampleResult:
    """Evaluation result for a single sample across all baseline strategies."""

    sample_id: str
    domain: str
    prompt: str
    target_answer: str
    eval_type: str
    results: Dict[str, PipelineResult]  # strategy -> PipelineResult
    correctness: Dict[str, bool]  # strategy -> is_correct
    latencies_ms: Dict[str, float]  # strategy -> total_ms
    expert_invocations: Dict[str, bool]  # strategy -> bool (True if expert != BASE)


class BaselineComparator:
    """Executes and compares multiple routing strategies on benchmark samples."""

    def __init__(
        self,
        pipeline: ARESPipeline,
        strategies: Optional[List[str]] = None,
        fixed_expert: str = "math",
        threshold: float = 0.5,
    ):
        self.pipeline = pipeline
        self.strategies = strategies or list(DEFAULT_BASELINES)
        self.fixed_expert = fixed_expert
        self.threshold = threshold

    def evaluate_sample(
        self,
        sample: BenchmarkSample,
        max_new_tokens: Optional[int] = None,
    ) -> BaselineSampleResult:
        """Run all configured baseline strategies for a single benchmark sample."""
        results: Dict[str, PipelineResult] = {}
        correctness: Dict[str, bool] = {}
        latencies_ms: Dict[str, float] = {}
        expert_invocations: Dict[str, bool] = {}

        for strategy in self.strategies:
            strat_lower = strategy.lower()

            if strat_lower == "base":
                res = self.pipeline.generate(
                    prompt=sample.prompt,
                    strategy="base",
                    max_new_tokens=max_new_tokens,
                )
            elif "fixed" in strat_lower:
                res = self.pipeline.generate(
                    prompt=sample.prompt,
                    strategy=f"fixed_{self.fixed_expert}",
                    max_new_tokens=max_new_tokens,
                )
            elif "oracle" in strat_lower:
                res = self.pipeline.generate(
                    prompt=sample.prompt,
                    strategy="oracle",
                    oracle_domain=sample.domain,
                    max_new_tokens=max_new_tokens,
                )
            elif "threshold" in strat_lower:
                res = self.pipeline.generate(
                    prompt=sample.prompt,
                    strategy="threshold",
                    max_new_tokens=max_new_tokens,
                )
            elif "random" in strat_lower:
                res = self.pipeline.generate(
                    prompt=sample.prompt,
                    strategy="random",
                    max_new_tokens=max_new_tokens,
                )
            else:
                # Dynamic ARES
                res = self.pipeline.generate(
                    prompt=sample.prompt,
                    strategy="dynamic",
                    max_new_tokens=max_new_tokens,
                )

            is_correct = evaluate_prediction(
                prediction=res.generated_text,
                target=sample.target_answer,
                eval_type=sample.eval_type,
            )

            results[strategy] = res
            correctness[strategy] = is_correct
            latencies_ms[strategy] = res.latency_ms.get("total_ms", 0.0)
            expert_invocations[strategy] = res.route_idx > 0

        return BaselineSampleResult(
            sample_id=sample.sample_id,
            domain=sample.domain,
            prompt=sample.prompt,
            target_answer=sample.target_answer,
            eval_type=sample.eval_type,
            results=results,
            correctness=correctness,
            latencies_ms=latencies_ms,
            expert_invocations=expert_invocations,
        )

    def evaluate_batch(
        self,
        samples: List[BenchmarkSample],
        max_new_tokens: Optional[int] = None,
        verbose: bool = True,
    ) -> List[BaselineSampleResult]:
        """Evaluate a batch of benchmark samples across all baseline strategies."""
        results: List[BaselineSampleResult] = []
        n = len(samples)

        for i, sample in enumerate(samples):
            if verbose and (i % 10 == 0 or i == n - 1):
                print(f"[ARES Baselines] Processing sample {i+1}/{n} (domain: {sample.domain})...")

            sample_res = self.evaluate_sample(sample, max_new_tokens=max_new_tokens)
            results.append(sample_res)

        return results
