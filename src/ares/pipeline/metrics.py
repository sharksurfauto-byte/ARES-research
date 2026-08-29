"""ARES Metrics Calculation and Evaluation Reporting (PRD §4.1, §5.2).

Computes:
- Overall & Domain-stratified Accuracy per strategy
- Latency statistics (mean, p50, p95 per strategy)
- Expert Utilization & Compute Savings (vs Always-on expert)
- Routing Distribution (Base % vs Domain Experts %)
- Reliability & Calibration Metrics (ECE, Brier Score, Uncertainty correlation)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ares.calibration.isotonic import compute_brier_score, compute_ece
from ares.pipeline.ares_pipeline import PipelineResult
from ares.pipeline.baselines import BaselineSampleResult


@dataclass
class StrategyMetrics:
    """Aggregated performance metrics for a single routing strategy."""

    strategy_name: str
    accuracy: float
    domain_accuracies: Dict[str, float]
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    expert_invocation_rate: float
    routing_distribution: Dict[str, float]
    total_samples: int


@dataclass
class ReliabilityMetrics:
    """Reliability model calibration and uncertainty metrics."""

    mean_global_reliability: float
    mean_failure_risk: float
    mean_uncertainty: float
    domain_classification_accuracy: float
    ece: Optional[float] = None
    brier_score: Optional[float] = None


@dataclass
class EvaluationReport:
    """Comprehensive evaluation report for ARES pipeline runs."""

    total_samples: int
    domains_evaluated: List[str]
    strategies: List[str]
    strategy_metrics: Dict[str, StrategyMetrics]
    reliability_metrics: Optional[ReliabilityMetrics] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary for JSON serialization."""
        return {
            "total_samples": self.total_samples,
            "domains_evaluated": self.domains_evaluated,
            "strategies": self.strategies,
            "strategy_metrics": {
                k: asdict(v) for k, v in self.strategy_metrics.items()
            },
            "reliability_metrics": asdict(self.reliability_metrics) if self.reliability_metrics else None,
            "metadata": self.metadata,
        }

    def save_json(self, filepath: str | Path):
        """Save report to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def to_markdown(self) -> str:
        """Generate formatted Markdown report tables suitable for research papers."""
        md = []
        md.append("# ARES End-to-End Evaluation Report\n")
        md.append(f"**Total Samples Evaluated**: {self.total_samples} across domains: `{', '.join(self.domains_evaluated)}`\n")

        # ─── Table 1: Baseline Accuracy & Efficiency Comparison ──────────────
        md.append("## 1. Strategy Comparison Summary\n")
        md.append("| Strategy | Overall Acc (%) | Invocation Rate (%) | Mean Latency (ms) | P95 Latency (ms) |")
        md.append("|:---|:---:|:---:|:---:|:---:|")

        for strat in self.strategies:
            m = self.strategy_metrics.get(strat)
            if not m:
                continue
            acc_str = f"{m.accuracy * 100:.2f}%"
            inv_str = f"{m.expert_invocation_rate * 100:.1f}%"
            lat_mean_str = f"{m.mean_latency_ms:.1f}"
            lat_p95_str = f"{m.p95_latency_ms:.1f}"
            md.append(f"| **{strat}** | {acc_str} | {inv_str} | {lat_mean_str} | {lat_p95_str} |")

        # ─── Table 2: Domain-Stratified Accuracy Breakdown ───────────────────
        md.append("\n## 2. Domain-Stratified Accuracy Breakdown\n")
        header = "| Strategy | " + " | ".join([f"{d.capitalize()} (%)" for d in self.domains_evaluated]) + " |"
        sep = "|:---|" + "|:---:".join([""] * (len(self.domains_evaluated) + 1)) + "|"
        md.append(header)
        md.append(sep)

        for strat in self.strategies:
            m = self.strategy_metrics.get(strat)
            if not m:
                continue
            row = [f"**{strat}**"]
            for d in self.domains_evaluated:
                dom_acc = m.domain_accuracies.get(d, 0.0)
                row.append(f"{dom_acc * 100:.2f}%")
            md.append("| " + " | ".join(row) + " |")

        # ─── Table 3: Routing Distribution (Dynamic ARES) ─────────────────────
        ares_metric = self.strategy_metrics.get("DYNAMIC_ARES") or self.strategy_metrics.get("dynamic")
        if ares_metric and ares_metric.routing_distribution:
            md.append("\n## 3. Dynamic ARES Routing Distribution\n")
            md.append("| Route | Usage (%) |")
            md.append("|:---|:---:|")
            for route, pct in ares_metric.routing_distribution.items():
                md.append(f"| `{route}` | {pct * 100:.2f}% |")

        # ─── Section 4: Reliability Diagnostics ──────────────────────────────
        if self.reliability_metrics:
            rm = self.reliability_metrics
            md.append("\n## 4. Reliability & Calibration Diagnostics\n")
            md.append(f"- **Mean Global Reliability $R(x)$**: `{rm.mean_global_reliability:.4f}`")
            md.append(f"- **Mean Local Failure Risk**: `{rm.mean_failure_risk:.4f}`")
            md.append(f"- **Mean Uncertainty Score**: `{rm.mean_uncertainty:.4f}`")
            md.append(f"- **GRM Domain Classification Accuracy**: `{rm.domain_classification_accuracy * 100:.2f}%`")
            if rm.ece is not None:
                md.append(f"- **Expected Calibration Error (ECE)**: `{rm.ece:.4f}`")
            if rm.brier_score is not None:
                md.append(f"- **Brier Score**: `{rm.brier_score:.4f}`")

        return "\n".join(md)

    def print_summary(self):
        """Print summary directly to console."""
        print("\n" + "=" * 70)
        print("  ARES PIPELINE BENCHMARK & BASELINE COMPARISON SUMMARY")
        print("=" * 70)
        print(f"Total Samples: {self.total_samples} | Domains: {', '.join(self.domains_evaluated)}\n")

        print(f"{'Strategy':<20} | {'Acc (%)':<9} | {'Invoked (%)':<12} | {'Mean (ms)':<10} | {'P95 (ms)':<10}")
        print("-" * 70)
        for strat in self.strategies:
            m = self.strategy_metrics.get(strat)
            if not m:
                continue
            print(
                f"{strat:<20} | "
                f"{m.accuracy * 100:>7.2f}% | "
                f"{m.expert_invocation_rate * 100:>10.1f}% | "
                f"{m.mean_latency_ms:>8.1f}ms | "
                f"{m.p95_latency_ms:>8.1f}ms"
            )
        print("=" * 70 + "\n")


class MetricsCalculator:
    """Computes comprehensive benchmark metrics and generates EvaluationReport."""

    @staticmethod
    def calculate_metrics(
        baseline_results: List[BaselineSampleResult],
        strategies: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EvaluationReport:
        """Calculate complete metrics from a list of BaselineSampleResult objects."""
        if not baseline_results:
            raise ValueError("baseline_results cannot be empty.")

        first = baseline_results[0]
        strategies = strategies or list(first.results.keys())
        total_samples = len(baseline_results)
        domains = sorted(list(set(s.domain for s in baseline_results)))

        strategy_metrics: Dict[str, StrategyMetrics] = {}

        # Aggregate per strategy
        for strat in strategies:
            strat_correct = [s.correctness.get(strat, False) for s in baseline_results]
            overall_acc = float(np.mean(strat_correct)) if strat_correct else 0.0

            # Domain breakdown
            domain_accs: Dict[str, float] = {}
            for d in domains:
                d_samples = [s for s in baseline_results if s.domain == d]
                if d_samples:
                    d_corr = [s.correctness.get(strat, False) for s in d_samples]
                    domain_accs[d] = float(np.mean(d_corr))
                else:
                    domain_accs[d] = 0.0

            # Latencies
            latencies = [s.latencies_ms.get(strat, 0.0) for s in baseline_results]
            mean_lat = float(np.mean(latencies)) if latencies else 0.0
            p50_lat = float(np.percentile(latencies, 50)) if latencies else 0.0
            p95_lat = float(np.percentile(latencies, 95)) if latencies else 0.0

            # Invocations
            invocations = [s.expert_invocations.get(strat, False) for s in baseline_results]
            inv_rate = float(np.mean(invocations)) if invocations else 0.0

            # Routing distribution
            routes = [s.results[strat].selected_route for s in baseline_results if strat in s.results]
            route_dist: Dict[str, float] = {}
            if routes:
                for r in set(routes):
                    route_dist[r] = float(routes.count(r) / len(routes))

            strategy_metrics[strat] = StrategyMetrics(
                strategy_name=strat,
                accuracy=overall_acc,
                domain_accuracies=domain_accs,
                mean_latency_ms=mean_lat,
                p50_latency_ms=p50_lat,
                p95_latency_ms=p95_lat,
                expert_invocation_rate=inv_rate,
                routing_distribution=route_dist,
                total_samples=total_samples,
            )

        # Reliability analysis (from DYNAMIC_ARES or first available result)
        dyn_strat = "DYNAMIC_ARES" if "DYNAMIC_ARES" in strategies else strategies[0]
        dyn_results = [s.results[dyn_strat] for s in baseline_results if dyn_strat in s.results]

        reliability_metrics = None
        if dyn_results:
            mean_rel = float(np.mean([r.global_reliability for r in dyn_results]))
            mean_risk = float(np.mean([r.failure_risk for r in dyn_results]))
            mean_unc = float(np.mean([r.uncertainty_score for r in dyn_results]))
            
            # Domain classification accuracy
            domain_matches = [
                r.domain_prediction.lower() == s.domain.lower()
                for r, s in zip(dyn_results, baseline_results)
            ]
            domain_acc = float(np.mean(domain_matches)) if domain_matches else 0.0

            # Calibration metrics (ECE, Brier)
            probs = np.array([r.global_reliability for r in dyn_results])
            targets = np.array([1.0 if s.correctness.get(dyn_strat, False) else 0.0 for s in baseline_results])
            
            ece_val = None
            brier_val = None
            try:
                ece_val = float(compute_ece(probs, targets))
                brier_val = float(compute_brier_score(probs, targets))
            except Exception:
                pass

            reliability_metrics = ReliabilityMetrics(
                mean_global_reliability=mean_rel,
                mean_failure_risk=mean_risk,
                mean_uncertainty=mean_unc,
                domain_classification_accuracy=domain_acc,
                ece=ece_val,
                brier_score=brier_val,
            )

        return EvaluationReport(
            total_samples=total_samples,
            domains_evaluated=domains,
            strategies=strategies,
            strategy_metrics=strategy_metrics,
            reliability_metrics=reliability_metrics,
            metadata=metadata or {},
        )
