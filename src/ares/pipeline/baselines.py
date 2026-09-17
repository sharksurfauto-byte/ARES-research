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
        """Run all configured baseline strategies for a single benchmark sample with SMART GENERATION CACHING.

        Smart Caching Optimization:
        For each sample, base model generation and router inference are run once.
        If the router or baseline selects Route 0 (Base), the already generated Base text
        is reused instead of running a redundant forward pass through the LLM decoder.
        Similarly, if multiple strategies select the same expert, the generated text is cached.
        """
        results: Dict[str, PipelineResult] = {}
        correctness: Dict[str, bool] = {}
        latencies_ms: Dict[str, float] = {}
        expert_invocations: Dict[str, bool] = {}

        # ─── 1. Run Base Generation & Router Inference ───────────────────────
        base_res = self.pipeline.generate(
            prompt=sample.prompt,
            strategy="base",
            max_new_tokens=max_new_tokens,
        )
        route_cache: Dict[str, PipelineResult] = {"BASE": base_res}

        # ─── 2. Evaluate All Configured Strategies ───────────────────────────
        for strategy in self.strategies:
            strat_lower = strategy.lower()

            if strat_lower == "base":
                selected_route = "BASE"
                route_idx = 0
            elif "fixed" in strat_lower:
                target = self.fixed_expert
                if "_" in strat_lower:
                    target = strat_lower.split("_", 1)[1]
                if target in self.pipeline.expert_names:
                    selected_route = target
                    route_idx = self.pipeline.expert_names.index(target) + 1
                else:
                    selected_route = "BASE"
                    route_idx = 0
            elif "oracle" in strat_lower:
                if sample.domain in self.pipeline.expert_names:
                    selected_route = sample.domain
                    route_idx = self.pipeline.expert_names.index(sample.domain) + 1
                else:
                    selected_route = "BASE"
                    route_idx = 0
            elif "threshold" in strat_lower:
                if base_res.global_reliability >= self.threshold:
                    selected_route = "BASE"
                    route_idx = 0
                else:
                    pred_dom = base_res.domain_prediction
                    if pred_dom in self.pipeline.expert_names:
                        selected_route = pred_dom
                        route_idx = self.pipeline.expert_names.index(pred_dom) + 1
                    else:
                        selected_route = "BASE"
                        route_idx = 0
            elif "random" in strat_lower:
                import random
                route_idx = random.randint(0, len(self.pipeline.route_names) - 1)
                selected_route = self.pipeline.route_names[route_idx]
            else:
                # Dynamic ARES (learned router decision)
                if base_res.routing_probs:
                    selected_route = max(base_res.routing_probs, key=base_res.routing_probs.get)
                    route_idx = self.pipeline.route_names.index(selected_route) if selected_route in self.pipeline.route_names else 0
                else:
                    selected_route = "BASE"
                    route_idx = 0

            # ─── 3. Smart Cache Lookup ───────────────────────────────────────
            if selected_route == "BASE" or route_idx == 0:
                # SMART CACHE HIT: Reuse base model generation without redundant forward pass!
                res = PipelineResult(
                    prompt=sample.prompt,
                    generated_text=base_res.generated_text,
                    full_output_text=base_res.full_output_text,
                    selected_route="BASE",
                    route_idx=0,
                    routing_probs=base_res.routing_probs,
                    domain_prediction=base_res.domain_prediction,
                    domain_confidence=base_res.domain_confidence,
                    global_reliability=base_res.global_reliability,
                    feasibility=base_res.feasibility,
                    token_reliability=base_res.token_reliability,
                    failure_risk=base_res.failure_risk,
                    uncertainty_score=base_res.uncertainty_score,
                    latency_ms=dict(base_res.latency_ms),
                    tokens_generated=base_res.tokens_generated,
                    route_confidence=base_res.routing_probs.get("BASE", 0.0),
                )
            elif selected_route in route_cache:
                # SMART CACHE HIT: Reuse existing expert generation
                cached = route_cache[selected_route]
                res = PipelineResult(
                    prompt=sample.prompt,
                    generated_text=cached.generated_text,
                    full_output_text=cached.full_output_text,
                    selected_route=selected_route,
                    route_idx=route_idx,
                    routing_probs=base_res.routing_probs,
                    domain_prediction=base_res.domain_prediction,
                    domain_confidence=base_res.domain_confidence,
                    global_reliability=base_res.global_reliability,
                    feasibility=base_res.feasibility,
                    token_reliability=base_res.token_reliability,
                    failure_risk=base_res.failure_risk,
                    uncertainty_score=base_res.uncertainty_score,
                    latency_ms=dict(cached.latency_ms),
                    tokens_generated=cached.tokens_generated,
                    route_confidence=base_res.routing_probs.get(selected_route, 0.0),
                )
            else:
                # CACHE MISS: Execute generation with selected expert adapter
                res = self.pipeline.generate(
                    prompt=sample.prompt,
                    strategy=f"fixed_{selected_route}",
                    max_new_tokens=max_new_tokens,
                )
                res.selected_route = selected_route
                res.route_idx = route_idx
                res.route_confidence = base_res.routing_probs.get(selected_route, 0.0)
                route_cache[selected_route] = res

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
        checkpoint_callback: Optional[Any] = None,
    ) -> List[BaselineSampleResult]:
        """Evaluate a batch of benchmark samples across all baseline strategies with smart caching."""
        results: List[BaselineSampleResult] = []
        n = len(samples)

        for i, sample in enumerate(samples):
            if verbose and (i % 10 == 0 or i == n - 1):
                print(f"[ARES Baselines] Processing sample {i+1}/{n} (domain: {sample.domain})...")

            sample_res = self.evaluate_sample(sample, max_new_tokens=max_new_tokens)
            results.append(sample_res)

            if checkpoint_callback is not None:
                try:
                    checkpoint_callback(results, i + 1, n)
                except Exception as e:
                    if verbose:
                        print(f"[ARES Baselines] Checkpoint callback warning: {e}")

        return results
