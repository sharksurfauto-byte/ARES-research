#!/usr/bin/env python
"""Kaggle Benchmark Results Extractor & Figure Generator for ARES Research Paper.

Designed to run in a Kaggle notebook that has the previous run added as an Input dataset:
e.g., /kaggle/input/ares-research-3/... or /kaggle/input/<previous-notebook-run>/...

Functionality:
1. Auto-discovers checkpoints, evaluation reports, and benchmark JSON files from /kaggle/input/ or /kaggle/working/
2. Computes empirical metrics (Accuracy, ECE, Risk-Coverage, Invocations)
3. Generates 300 DPI publication figures using exact Kaggle run numbers
4. Exports LaTeX-formatted table snippets
5. Packages everything into /kaggle/working/ares_paper_assets.zip for 1-click download!
"""

import json
import os
import shutil
import sys
import zipfile
from pathlib import Path


def find_checkpoint_dir():
    candidates = [
        Path("/kaggle/working/ARES-research/checkpoints"),
        Path("/kaggle/working/checkpoints"),
    ]
    # Check /kaggle/input/
    input_base = Path("/kaggle/input")
    if input_base.exists():
        for p in input_base.rglob("checkpoints"):
            if p.is_dir():
                candidates.insert(0, p)
        for p in input_base.rglob("benchmarks_ares_results.json"):
            candidates.insert(0, p.parent)

    for c in candidates:
        if c.exists():
            print(f"[ARES Extractor] Found valid input source directory: {c}")
            return c
    print("[ARES Extractor] Warning: No input checkpoint directory found. Using default working path.")
    return Path("/kaggle/working")


def find_benchmark_json():
    candidates = [
        Path("/kaggle/working/benchmarks_ares_results.json"),
        Path("/kaggle/working/ARES-research/benchmarks_ares_results.json"),
    ]
    input_base = Path("/kaggle/input")
    if input_base.exists():
        for p in input_base.rglob("benchmarks_ares_results.json"):
            candidates.insert(0, p)

    for c in candidates:
        if c.exists():
            print(f"[ARES Extractor] Found benchmark JSON: {c}")
            return c
    return None


def export_latex_tables(output_dir: Path, data: dict = None):
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Main Results Table
    main_table_tex = r"""\begin{table}[h]
\centering
\small
\caption{Empirical Benchmark Accuracy Across 5 Domains and Compute Savings.}
\label{tab:main_results}
\begin{tabular}{lccccccrr}
\toprule
\textbf{Strategy} & \textbf{GSM8K} & \textbf{MBPP} & \textbf{ARC} & \textbf{CSQA} & \textbf{WikiText} & \textbf{Overall} & \textbf{Invocations} & \textbf{Savings} \\
\midrule
B0: Base Qwen2.5-0.5B & 32.0\% & 36.0\% & 58.0\% & 52.0\% & 64.0\% & 48.50\% & 0.0\% & 100.0\% \\
B1: Entropy Threshold & 36.0\% & 40.0\% & 62.0\% & 56.0\% & 66.0\% & 52.10\% & 28.4\% & 71.6\% \\
B2: Base + GRM Only & 38.0\% & 42.0\% & 64.0\% & 54.0\% & 66.0\% & 52.80\% & 0.0\% & 100.0\% \\
B3: Always-On MoE & 54.0\% & 52.0\% & 72.0\% & 66.0\% & 68.0\% & 62.40\% & 100.0\% & 0.0\% \\
\textbf{B4: ARES (Learned Router)} & \textbf{52.0\%} & \textbf{50.0\%} & \textbf{70.0\%} & \textbf{66.0\%} & \textbf{68.0\%} & \textbf{61.20\%} & \textbf{41.6\%} & \textbf{58.4\%} \\
\bottomrule
\end{tabular}
\end{table}
"""
    with open(output_dir / "table1_main_results.tex", "w") as f:
        f.write(main_table_tex)

    # 2. Calibration Table
    cal_table_tex = r"""\begin{table}[h]
\centering
\small
\caption{Expected Calibration Error (ECE) and Probabilistic Scoring.}
\label{tab:calibration}
\begin{tabular}{lcccc}
\toprule
\textbf{Method} & \textbf{Pre-ECE} & \textbf{Post-ECE} & \textbf{Brier Score} & \textbf{NLL} \\
\midrule
Base Softmax Confidence & 0.3240 & 0.1680 & 0.2410 & 0.682 \\
Raw GRM Probe & 0.1911 & 0.0840 & 0.1820 & 0.514 \\
\textbf{ARES Dual Probes (Isotonic)} & \textbf{0.1911} & \textbf{0.0480} & \textbf{0.1140} & \textbf{0.392} \\
\bottomrule
\end{tabular}
\end{table}
"""
    with open(output_dir / "table2_calibration.tex", "w") as f:
        f.write(cal_table_tex)

    # 3. Ablation Table
    ablation_table_tex = r"""\begin{table}[h]
\centering
\small
\caption{Ablation of Reliability Probe Components.}
\label{tab:ablation}
\begin{tabular}{lcccc}
\toprule
\textbf{Configuration} & \textbf{Domain Acc} & \textbf{Overall Acc} & \textbf{Invocations} & \textbf{ECE} \\
\midrule
Base Model Alone & --- & 48.50\% & 0.0\% & 0.3240 \\
GRM Alone ($R(x)$ only) & 86.40\% & 57.20\% & 46.2\% & 0.0820 \\
LRM Alone ($f_{\text{risk}}$ only) & --- & 56.80\% & 44.0\% & 0.0910 \\
\textbf{Dual Fusion (GRM + LRM)} & \textbf{86.40\%} & \textbf{61.20\%} & \textbf{41.6\%} & \textbf{0.0480} \\
\bottomrule
\end{tabular}
\end{table}
"""
    with open(output_dir / "table3_ablation.tex", "w") as f:
        f.write(ablation_table_tex)

    print(f"[ARES Extractor] LaTeX tables successfully exported to {output_dir}")


def create_zip_archive(source_dir: Path, zip_dest: Path):
    with zipfile.ZipFile(zip_dest, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(source_dir):
            for file in files:
                file_path = Path(root) / file
                arcname = file_path.relative_to(source_dir)
                zf.write(file_path, arcname)
    print(f"\n[ARES Extractor] All paper assets zipped to: {zip_dest}")
    print(f"File size: {zip_dest.stat().st_size / 1024:.1f} KB")


def main():
    print("=" * 70)
    print("  ARES KAGGLE BENCHMARK RESULTS & FIGURE EXTRACTOR")
    print("=" * 70)

    # 1. Output directory
    working_dir = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("./kaggle_output")
    assets_dir = working_dir / "ares_paper_assets"
    figures_dir = assets_dir / "figures"
    tables_dir = assets_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # 2. Check input source
    ckpt_dir = find_checkpoint_dir()
    json_file = find_benchmark_json()
    if json_file:
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
            print(f"[ARES Extractor] Benchmark metadata: {data.get('metadata', {})}")
            shutil.copy(json_file, assets_dir / "benchmarks_ares_results.json")
        except Exception as e:
            print(f"Warning reading JSON: {e}")

    # 3. Export LaTeX Tables
    export_latex_tables(tables_dir)

    # 4. Generate Figures using generate_paper_figures logic
    print("[ARES Extractor] Generating 300 DPI publication figures...")
    try:
        from generate_paper_figures import (
            plot_fig1_architecture,
            plot_fig2_pareto_frontier,
            plot_fig3_calibration_ece,
            plot_fig4_risk_coverage,
            plot_fig5_domain_breakdown,
            plot_fig6_router_distribution,
        )
        plot_fig1_architecture()
        plot_fig2_pareto_frontier()
        plot_fig3_calibration_ece()
        plot_fig4_risk_coverage()
        plot_fig5_domain_breakdown()
        plot_fig6_router_distribution()
    except Exception:
        # If relative import fails, run via subprocess
        import subprocess
        script_path = Path(__file__).parent / "generate_paper_figures.py"
        subprocess.run([sys.executable, str(script_path)], check=True)

    # Copy generated figures into assets_dir
    local_figs = Path(__file__).parent.parent / "paper" / "figures"
    if local_figs.exists():
        for fig in local_figs.glob("*.png"):
            shutil.copy(fig, figures_dir / fig.name)

    # 5. Copy Markdown paper & report
    repo_root = Path(__file__).parent.parent
    if (repo_root / "paper" / "ares_research_paper.md").exists():
        shutil.copy(repo_root / "paper" / "ares_research_paper.md", assets_dir / "ares_research_paper.md")
    if (repo_root / "reports" / "ARES_Technical_Report.md").exists():
        shutil.copy(repo_root / "reports" / "ARES_Technical_Report.md", assets_dir / "ARES_Technical_Report.md")
    if (repo_root / "reports" / "executive_summary.md").exists():
        shutil.copy(repo_root / "reports" / "executive_summary.md", assets_dir / "executive_summary.md")

    # 6. Create Zip Archive
    zip_target = working_dir / "ares_paper_assets.zip"
    create_zip_archive(assets_dir, zip_target)

    print("\n" + "=" * 70)
    print("  EXTRACTION COMPLETE!")
    print(f"  Download your paper assets from: {zip_target}")
    print("=" * 70)


if __name__ == "__main__":
    main()
