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

import sys
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

        # ─── 1. Run Dynamic Router & Reliability Inference ──────────────────
        # Runs representation extraction, GRM reliability, and learned router
        router_res = self.pipeline.generate(
            prompt=sample.prompt,
            strategy="dynamic_ares",
            max_new_tokens=max_new_tokens,
        )
        route_cache: Dict[str, PipelineResult] = {router_res.selected_route: router_res}

        # ─── 2. Evaluate All Configured Strategies ───────────────────────────
        for strategy in self.strategies:
            strat_lower = strategy.lower()

            if strat_lower == "base":
                selected_route = "BASE"
                route_idx = 0
            elif "fixed" in strat_lower:
                parts = strat_lower.split("_", 1)
                if len(parts) > 1 and parts[1] in self.pipeline.expert_names:
                    target = parts[1]
                else:
                    target = self.fixed_expert
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
                if router_res.global_reliability >= self.threshold:
                    selected_route = "BASE"
                    route_idx = 0
                else:
                    pred_dom = router_res.domain_prediction
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
                if router_res.routing_probs:
                    selected_route = max(router_res.routing_probs, key=router_res.routing_probs.get)
                    route_idx = self.pipeline.route_names.index(selected_route) if selected_route in self.pipeline.route_names else 0
                else:
                    selected_route = router_res.selected_route
                    route_idx = router_res.route_idx

            # ─── 3. Smart Cache Lookup ───────────────────────────────────────
            if selected_route == "BASE" or route_idx == 0:
                if "BASE" not in route_cache:
                    base_res = self.pipeline.generate(
                        prompt=sample.prompt,
                        strategy="base",
                        max_new_tokens=max_new_tokens,
                    )
                    route_cache["BASE"] = base_res
                cached = route_cache["BASE"]
                res = PipelineResult(
                    prompt=sample.prompt,
                    generated_text=cached.generated_text,
                    full_output_text=cached.full_output_text,
                    selected_route="BASE",
                    route_idx=0,
                    routing_probs=router_res.routing_probs,
                    domain_prediction=router_res.domain_prediction,
                    domain_confidence=router_res.domain_confidence,
                    global_reliability=router_res.global_reliability,
                    feasibility=router_res.feasibility,
                    token_reliability=router_res.token_reliability,
                    failure_risk=router_res.failure_risk,
                    uncertainty_score=router_res.uncertainty_score,
                    latency_ms=dict(cached.latency_ms),
                    tokens_generated=cached.tokens_generated,
                    route_confidence=router_res.routing_probs.get("BASE", 0.0),
                )
            elif selected_route in route_cache:
                cached = route_cache[selected_route]
                res = PipelineResult(
                    prompt=sample.prompt,
                    generated_text=cached.generated_text,
                    full_output_text=cached.full_output_text,
                    selected_route=selected_route,
                    route_idx=route_idx,
                    routing_probs=router_res.routing_probs,
                    domain_prediction=router_res.domain_prediction,
                    domain_confidence=router_res.domain_confidence,
                    global_reliability=router_res.global_reliability,
                    feasibility=router_res.feasibility,
                    token_reliability=router_res.token_reliability,
                    failure_risk=router_res.failure_risk,
                    uncertainty_score=router_res.uncertainty_score,
                    latency_ms=dict(cached.latency_ms),
                    tokens_generated=cached.tokens_generated,
                    route_confidence=router_res.routing_probs.get(selected_route, 0.0),
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
                res.route_confidence = router_res.routing_probs.get(selected_route, 0.0)
                route_cache[selected_route] = res

            try:
                is_correct = evaluate_prediction(
                    prediction=res.generated_text,
                    target=sample.target_answer,
                    eval_type=sample.eval_type,
                )
            except Exception:
                is_correct = False

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
                print(f"[ARES Baselines] Processing sample {i+1}/{n} (domain: {sample.domain})...", flush=True)

            sample_res = self.evaluate_sample(sample, max_new_tokens=max_new_tokens)
            results.append(sample_res)

            if checkpoint_callback is not None:
                try:
                    checkpoint_callback(results, i + 1, n)
                except Exception as e:
                    if verbose:
                        print(f"[ARES Baselines] Checkpoint callback warning: {e}", flush=True)

        # Sanity check: Ensure strategies with significantly different invocation rates do not produce byte-identical completions
        if len(results) >= 10:
            for strat_a in self.strategies:
                for strat_b in self.strategies:
                    if strat_a >= strat_b:
                        continue
                    inv_a = sum(1 for r in results if r.expert_invocations.get(strat_a, False)) / len(results)
                    inv_b = sum(1 for r in results if r.expert_invocations.get(strat_b, False)) / len(results)
                    # If invocation rate difference > 25%
                    if abs(inv_a - inv_b) > 0.25:
                        identical_count = sum(
                            1 for r in results
                            if r.results.get(strat_a) and r.results.get(strat_b)
                            and r.results[strat_a].generated_text == r.results[strat_b].generated_text
                        )
                        identical_ratio = identical_count / len(results)
                        if identical_ratio > 0.90:
                            print(
                                f"[ARES SANITY WARNING] Strategy '{strat_a}' (inv={inv_a:.1%}) and '{strat_b}' (inv={inv_b:.1%}) "
                                f"have {identical_ratio:.1%} identical generation outputs! "
                                f"Verify that adapter weights are properly loaded and affecting hidden states.",
                                flush=True,
                            )

        return results
