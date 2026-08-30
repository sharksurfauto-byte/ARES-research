#!/usr/bin/env python
"""End-to-End ARES Pipeline CLI (PRD §3, §4.1, §5.2).

Links the frozen Backbone -> Representation Extraction -> GRM & LRM Reliability Analysis ->
Router Decision -> Dynamic LoRA Expert Hook Generation -> Text Generation -> Baseline Comparisons -> Metrics & Reports.

Usage Examples:
    # 1. Single prompt interactive run:
    python scripts/run_ares_pipeline.py \
        --prompt "If it takes 3 hours to travel 180 km, what is the speed in km/h?" \
        --model_name "Qwen/Qwen2.5-0.5B" \
        --checkpoints_dir checkpoints

    # 2. Single prompt with all baseline comparisons:
    python scripts/run_ares_pipeline.py \
        --prompt "Write a Python function to compute the Fibonacci sequence." \
        --run_baselines

    # 3. Full benchmark evaluation across 5 domains with report export:
    python scripts/run_ares_pipeline.py \
        --benchmark all \
        --n_samples_per_domain 10 \
        --output_report outputs/benchmark_report.md \
        --output_json outputs/benchmark_report.json
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ares.data.benchmark_loader import (
    BenchmarkSample,
    load_all_benchmark_samples,
    load_gsm8k_samples,
    load_mbpp_samples,
    load_ai2_arc_samples,
    load_wikitext_samples,
    load_reasoning_samples,
)
from ares.pipeline.ares_pipeline import ARESPipeline, PipelineConfig
from ares.pipeline.baselines import BaselineComparator
from ares.pipeline.metrics import MetricsCalculator


def parse_args():
    parser = argparse.ArgumentParser(
        description="ARES End-to-End Dynamic Routing & Reliability Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Input Mode
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--prompt", type=str, help="Input prompt text for single query execution")
    mode_group.add_argument(
        "--benchmark",
        type=str,
        choices=["all", "gsm8k", "mbpp", "ai2_arc", "wikitext", "reasoning"],
        help="Run evaluation on standard benchmark dataset(s)",
    )

    # Model & Checkpoint Configurations
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B", help="Backbone model name or path")
    parser.add_argument(
        "--checkpoints_dir",
        "--expert_dir",
        dest="checkpoints_dir",
        type=str,
        default="checkpoints",
        help="Directory containing module checkpoints",
    )
    parser.add_argument("--grm_checkpoint", type=str, default=None, help="Explicit path to GRM checkpoint (overrides default)")
    parser.add_argument("--lrm_checkpoint", type=str, default=None, help="Explicit path to LRM checkpoint (overrides default)")
    parser.add_argument("--router_checkpoint", type=str, default=None, help="Explicit path to Router checkpoint (overrides default)")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('cuda', 'cpu', or 'auto')")

    # Routing & Baseline Parameters
    parser.add_argument(
        "--strategy",
        type=str,
        default="dynamic",
        choices=["dynamic", "base", "fixed", "threshold", "oracle", "random"],
        help="Routing strategy for single prompt execution",
    )
    parser.add_argument("--fixed_expert", type=str, default="math", help="Target domain expert when using fixed strategy")
    parser.add_argument("--threshold", type=float, default=0.5, help="Reliability threshold for threshold-based routing")
    parser.add_argument("--run_baselines", action="store_true", help="Compare all baseline routing strategies for the prompt/benchmark")

    # Generation Parameters
    parser.add_argument("--max_new_tokens", type=int, default=50, help="Maximum generated tokens")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling probability")
    parser.add_argument("--do_sample", action="store_true", help="Enable sampling during generation")

    # Benchmark & Reporting Parameters
    parser.add_argument(
        "--n_samples_per_domain",
        "--samples_per_domain",
        dest="n_samples_per_domain",
        type=int,
        default=10,
        help="Number of benchmark samples per domain",
    )
    parser.add_argument("--output_report", type=str, default=None, help="Filepath to save generated Markdown report")
    parser.add_argument("--output_json", type=str, default=None, help="Filepath to save JSON metrics report")

    return parser.parse_args()


def run_single_prompt(pipeline: ARESPipeline, args):
    """Execute ARES pipeline for a single prompt."""
    print("\n" + "=" * 70)
    print("  ARES SINGLE PROMPT PIPELINE EXECUTION")
    print("=" * 70)
    print(f"Prompt: '{args.prompt}'\n")

    if args.run_baselines:
        print("[ARES] Running all baseline routing strategies...")
        comparator = BaselineComparator(
            pipeline=pipeline,
            fixed_expert=args.fixed_expert,
            threshold=args.threshold,
        )
        sample = BenchmarkSample(
            sample_id="prompt_0",
            domain="general",
            prompt=args.prompt,
            target_answer="",
            eval_type="general_text",
        )
        res = comparator.evaluate_sample(sample, max_new_tokens=args.max_new_tokens)

        print("\n--- Strategy Outputs ---")
        for strat, r in res.results.items():
            print(f"\n[{strat}] -> Route: {r.selected_route} ({r.route_confidence:.1%}) | Latency: {r.latency_ms.get('total_ms', 0):.1f}ms")
            print(f"Output: {r.generated_text}")
        print("\n" + "=" * 70)
        return

    # Standard single strategy execution
    res = pipeline.generate(
        prompt=args.prompt,
        strategy=args.strategy,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.do_sample,
    )

    print("─── Reliability Diagnostics (GRM & LRM) ──────────────────────────")
    print(f"  • Predicted Domain:    {res.domain_prediction.upper()} (Confidence: {res.domain_confidence:.1%})")
    print(f"  • Global Reliability:  {res.global_reliability:.1%}")
    print(f"  • Feasibility:         {res.feasibility:.1%}")
    print(f"  • Token Reliability:   {res.token_reliability:.1%}")
    print(f"  • Mean Failure Risk:   {res.failure_risk:.1%}")
    print(f"  • Uncertainty Score:   {res.uncertainty_score:.3f}")

    print("\n─── Router Decision ───────────────────────────────────────────────")
    print(f"  • Strategy:            {args.strategy.upper()}")
    print(f"  • Selected Route:      {res.selected_route}")
    print(f"  • Route Confidence:    {res.route_confidence:.1%}")
    print("  • Route Probabilities:")
    for route_name, prob in res.routing_probs.items():
        bar = "█" * int(prob * 20)
        print(f"      - {route_name:<10}: {prob:>6.1%}  {bar}")

    print("\n─── Performance Latencies ─────────────────────────────────────────")
    print(f"  • Backbone Forward:    {res.latency_ms.get('backbone_ms', 0):.1f} ms")
    print(f"  • Dual Reliability:    {res.latency_ms.get('reliability_ms', 0):.1f} ms")
    print(f"  • Router Dispatch:     {res.latency_ms.get('router_ms', 0):.1f} ms")
    print(f"  • Generation Time:     {res.latency_ms.get('generation_ms', 0):.1f} ms ({res.tokens_generated} tokens)")
    print(f"  • Total Latency:       {res.latency_ms.get('total_ms', 0):.1f} ms")

    print("\n─── Generated Text Output ─────────────────────────────────────────")
    print(f"{res.generated_text}\n")
    print("=" * 70)


def run_benchmark_evaluation(pipeline: ARESPipeline, args):
    """Execute multi-domain benchmark evaluation across baselines."""
    print("\n" + "=" * 70)
    print("  ARES MULTI-DOMAIN BENCHMARK EVALUATION")
    print("=" * 70)

    # 1. Load benchmark datasets
    samples: list[BenchmarkSample] = []
    n = args.n_samples_per_domain

    if args.benchmark == "all":
        print(f"[ARES Data] Loading {n} samples across all 5 domains...")
        all_dict = load_all_benchmark_samples(n_samples_per_domain=n)
        for d, s_list in all_dict.items():
            samples.extend(s_list)
    elif args.benchmark == "gsm8k":
        samples = load_gsm8k_samples(n_samples=n)
    elif args.benchmark == "mbpp":
        samples = load_mbpp_samples(n_samples=n)
    elif args.benchmark == "ai2_arc":
        samples = load_ai2_arc_samples(n_samples=n)
    elif args.benchmark == "wikitext":
        samples = load_wikitext_samples(n_samples=n)
    elif args.benchmark == "reasoning":
        samples = load_reasoning_samples(n_samples=n)

    print(f"[ARES Benchmark] Total test samples loaded: {len(samples)}")

    # 2. Run Baseline Comparisons
    comparator = BaselineComparator(
        pipeline=pipeline,
        fixed_expert=args.fixed_expert,
        threshold=args.threshold,
    )
    baseline_results = comparator.evaluate_batch(samples, max_new_tokens=args.max_new_tokens)

    # 3. Calculate Metrics & Generate Report
    report = MetricsCalculator.calculate_metrics(
        baseline_results=baseline_results,
        metadata={
            "model_name": args.model_name,
            "threshold": args.threshold,
            "fixed_expert": args.fixed_expert,
            "max_new_tokens": args.max_new_tokens,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )

    # Print Summary to console
    report.print_summary()

    # 4. Save Outputs
    if args.output_report:
        report_path = Path(args.output_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report.to_markdown())
        print(f"[ARES Report] Markdown report saved to: {report_path}")

    if args.output_json:
        json_path = Path(args.output_json)
        report.save_json(json_path)
        print(f"[ARES Report] JSON report saved to: {json_path}")


def main():
    args = parse_args()

    # Configure Pipeline
    config = PipelineConfig(
        model_name=args.model_name,
        checkpoints_dir=args.checkpoints_dir,
        grm_checkpoint=args.grm_checkpoint,
        lrm_checkpoint=args.lrm_checkpoint,
        router_checkpoint=args.router_checkpoint,
        device=args.device,
        reliability_threshold=args.threshold,
        routing_strategy=args.strategy,
        fixed_expert_name=args.fixed_expert,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.do_sample,
    )

    print(f"[ARES] Initializing ARES Pipeline ({config.model_name})...")
    pipeline = ARESPipeline(config=config)
    print(f"[ARES] Pipeline successfully initialized on device: {pipeline.device}")

    if args.prompt:
        run_single_prompt(pipeline, args)
    elif args.benchmark:
        run_benchmark_evaluation(pipeline, args)


if __name__ == "__main__":
    main()
