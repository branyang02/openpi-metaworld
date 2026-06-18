"""
Exp 4: Analyze denoising step ablation results.

Loads result JSONs from denoising_step_exp/results/ablation/ and produces:
- Summary table (step count x mean success rate)
- Per-task comparison across step counts
- Bar chart saved to denoising_step_exp/results/figures/exp4/
- Identification of tasks where fewer steps cause degradation

Usage:
    uv run denoising_step_exp/exp4_denoising_steps.py
"""

import dataclasses
import json
import logging
import pathlib

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tyro

sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Args:
    results_dir: str = "denoising_step_exp/results/ablation"
    figures_dir: str = "denoising_step_exp/results/figures/exp4"
    # Threshold for flagging degradation (absolute drop from 10-step baseline).
    degradation_threshold: float = 0.1


def main(args: Args) -> None:
    results_dir = pathlib.Path(args.results_dir)
    figures_dir = pathlib.Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load all result files
    step_counts = []
    all_results = {}
    for path in sorted(results_dir.glob("results_*steps.json")):
        with open(path) as f:
            data = json.load(f)
        num_steps = data["num_steps"]
        step_counts.append(num_steps)
        all_results[num_steps] = data

    if not all_results:
        logger.error(f"No result files found in {results_dir}")
        return

    step_counts.sort()
    logger.info(f"Found results for step counts: {step_counts}")

    # === Summary table ===
    print("\n" + "=" * 50)
    print("DENOISING STEP ABLATION SUMMARY")
    print("=" * 50)
    print(f"{'Steps':>6s}  {'Mean SR':>8s}  {'Tasks':>5s}")
    print("-" * 25)
    for ns in step_counts:
        data = all_results[ns]
        print(f"{ns:>6d}  {data['mean_success_rate']:>8.4f}  {data['tasks_evaluated']:>5d}")

    # === Per-task comparison ===
    # Collect all task names from any result file
    all_tasks = set()
    for data in all_results.values():
        all_tasks.update(data["per_task"].keys())
    all_tasks = sorted(all_tasks)

    print(f"\n{'Task':<40s}", end="")
    for ns in step_counts:
        print(f"  {ns:>4d}s", end="")
    print()
    print("-" * (40 + 6 * len(step_counts)))

    for task in all_tasks:
        print(f"{task:<40s}", end="")
        for ns in step_counts:
            data = all_results[ns]
            if task in data["per_task"]:
                sr = data["per_task"][task]["success_rate"]
                print(f"  {sr:>.2f}", end="")
            else:
                print(f"  {'N/A':>5s}", end="")
        print()

    # === Degradation analysis ===
    baseline_steps = max(step_counts)
    if baseline_steps in all_results:
        baseline = all_results[baseline_steps]
        print(f"\n{'=' * 50}")
        print(f"DEGRADATION ANALYSIS (vs {baseline_steps}-step baseline, threshold={args.degradation_threshold})")
        print("=" * 50)

        for ns in step_counts:
            if ns == baseline_steps:
                continue
            data = all_results[ns]
            degraded_tasks = []
            for task in all_tasks:
                if task in baseline["per_task"] and task in data["per_task"]:
                    base_sr = baseline["per_task"][task]["success_rate"]
                    test_sr = data["per_task"][task]["success_rate"]
                    drop = base_sr - test_sr
                    if drop >= args.degradation_threshold:
                        degraded_tasks.append((task, base_sr, test_sr, drop))

            degraded_tasks.sort(key=lambda x: -x[3])
            if degraded_tasks:
                print(f"\nnum_steps={ns}: {len(degraded_tasks)} tasks degraded")
                for task, base_sr, test_sr, drop in degraded_tasks:
                    print(f"  {task:<40s} {base_sr:.2f} -> {test_sr:.2f} (drop={drop:.2f})")
            else:
                print(f"\nnum_steps={ns}: No tasks degraded by >= {args.degradation_threshold}")

    # === Bar chart ===
    mean_srs = [all_results[ns]["mean_success_rate"] for ns in step_counts]

    fig, ax = plt.subplots(figsize=(8, 5))
    palette = sns.color_palette("muted")
    bars = ax.bar([str(ns) for ns in step_counts], mean_srs, color=palette[0], edgecolor="black", linewidth=0.5)

    # Add value labels on bars
    for bar, sr in zip(bars, mean_srs, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005, f"{sr:.3f}", ha="center", va="bottom")

    ax.set_xlabel("Number of Denoising Steps $N$")
    ax.set_ylabel("Mean Success Rate")
    ax.set_title(r"Pi0.5 MetaWorld: Denoising Step Ablation ($N \in \{1,2,3,5,10\}$)")
    ax.set_ylim(0, min(1.0, max(mean_srs) * 1.15))
    ax.grid(axis="y", alpha=0.3)

    chart_path = figures_dir / "denoising_steps_bar.png"
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Bar chart saved to {chart_path}")

    # === Per-task heatmap-style chart ===
    if len(all_tasks) > 1 and len(step_counts) > 1:
        task_sr_matrix = np.zeros((len(all_tasks), len(step_counts)))
        for i, task in enumerate(all_tasks):
            for j, ns in enumerate(step_counts):
                if task in all_results[ns]["per_task"]:
                    task_sr_matrix[i, j] = all_results[ns]["per_task"][task]["success_rate"]

        fig, ax = plt.subplots(figsize=(8, max(6, len(all_tasks) * 0.3)))
        im = ax.imshow(task_sr_matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_xticks(range(len(step_counts)))
        ax.set_xticklabels([str(ns) for ns in step_counts])
        ax.set_yticks(range(len(all_tasks)))
        ax.set_yticklabels(all_tasks, fontsize=7)
        ax.set_xlabel("Number of Denoising Steps")
        ax.set_title("Per-Task Success Rate by Denoising Steps")
        fig.colorbar(im, label="Success Rate")

        heatmap_path = figures_dir / "denoising_steps_heatmap.png"
        fig.savefig(heatmap_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Heatmap saved to {heatmap_path}")

    # Save summary JSON
    summary = {
        "step_counts": step_counts,
        "mean_success_rates": {ns: all_results[ns]["mean_success_rate"] for ns in step_counts},
        "per_task": {
            task: {ns: all_results[ns]["per_task"].get(task, {}).get("success_rate") for ns in step_counts}
            for task in all_tasks
        },
    }
    summary_path = figures_dir / "exp4_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = tyro.cli(Args)
    main(args)
