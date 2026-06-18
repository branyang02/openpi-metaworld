"""Exp 3: Cross-denoising CKA analysis of suffix residual activations.

Measures representation similarity across denoising steps using linear CKA
on suffix residual activations. Produces 2 figures:
  1. CKA matrices (2x2 grid of 10x10 heatmaps, one per layer)
  2. CKA similarity to final step (step 9) vs denoising step, per layer

Usage:
    uv run denoising_step_exp/exp3_denoising_cka.py
"""

from pathlib import Path
import sys

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm

sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_utils import LAYER_INDICES  # noqa: E402
from data_utils import ensure_dir  # noqa: E402
from data_utils import get_step_dirs  # noqa: E402
from data_utils import load_activations  # noqa: E402
from data_utils import load_episode_index  # noqa: E402
from data_utils import sample_episodes  # noqa: E402

OUTPUT_DIR = "denoising_step_exp/results/figures/exp3"
NUM_DENOISING_STEPS = 10
NUM_LAYERS = 4  # len(LAYER_INDICES)
NUM_TOKENS = 32
HIDDEN_DIM = 1024


def linear_cka(x_mat: np.ndarray, y_mat: np.ndarray) -> float:
    """Compute linear CKA between two representation matrices.

    Args:
        x_mat: (n, p) matrix -- n samples, p features.
        y_mat: (n, q) matrix -- n samples, q features.

    Returns:
        CKA similarity in [0, 1].
    """
    x_mat = x_mat - x_mat.mean(0)
    y_mat = y_mat - y_mat.mean(0)
    hsic_xy = np.linalg.norm(x_mat.T @ y_mat, "fro") ** 2
    hsic_xx = np.linalg.norm(x_mat.T @ x_mat, "fro") ** 2
    hsic_yy = np.linalg.norm(y_mat.T @ y_mat, "fro") ** 2
    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-12:
        return 0.0
    return float(hsic_xy / denom)


def load_suffix_residual_step0(episode_dir: str) -> np.ndarray | None:
    """Load suffix_residual from the first inference step of an episode.

    Returns:
        Array of shape (10, 4, 32, 1024) or None on failure.
    """
    step_dirs = get_step_dirs(episode_dir)
    if not step_dirs:
        return None
    try:
        data = load_activations(step_dirs[0], "suffix_residual")
        arr = data.get("all_suffix_residual", next(iter(data.values())))
        if arr.shape != (NUM_DENOISING_STEPS, NUM_LAYERS, NUM_TOKENS, HIDDEN_DIM):
            return None
        return arr
    except Exception:
        return None


def compute_cka_matrices(episodes_data: list[np.ndarray]) -> np.ndarray:
    """Compute CKA matrices across denoising steps for each layer, averaged over episodes.

    Returns:
        cka_matrices: (4, 10, 10)
    """
    n_episodes = len(episodes_data)
    cka_sum = np.zeros((NUM_LAYERS, NUM_DENOISING_STEPS, NUM_DENOISING_STEPS))

    for arr in tqdm(episodes_data, desc="Computing CKA matrices"):
        for layer_idx in range(NUM_LAYERS):
            for i in range(NUM_DENOISING_STEPS):
                for j in range(i, NUM_DENOISING_STEPS):
                    cka_val = linear_cka(arr[i, layer_idx], arr[j, layer_idx])
                    cka_sum[layer_idx, i, j] += cka_val
                    if i != j:
                        cka_sum[layer_idx, j, i] += cka_val

    return cka_sum / n_episodes


def main():
    out = ensure_dir(OUTPUT_DIR)
    print("Loading episode index...")
    index = load_episode_index()
    print(f"Found {len(index)} episodes across {index['task_name'].nunique()} tasks")

    # Sample ~100 episodes
    sampled = sample_episodes(index, n_per_task=3, seed=42)
    print(f"Sampled {len(sampled)} episodes (n_per_task=3)")

    # Load suffix_residual from step_0000
    episodes_data = []
    for _, row in tqdm(sampled.iterrows(), total=len(sampled), desc="Loading activations"):
        arr = load_suffix_residual_step0(row["episode_dir"])
        if arr is not None:
            episodes_data.append(arr)

    n_loaded = len(episodes_data)
    print(f"Successfully loaded {n_loaded} episodes")
    if n_loaded == 0:
        print("ERROR: No episodes loaded. Exiting.")
        return

    # Compute CKA matrices
    print("\nComputing CKA matrices (10x10) for each of 4 layers...")
    cka_matrices = compute_cka_matrices(episodes_data)

    # ---- Figure 1: CKA matrices (2x2 grid) ----
    print("\nPlotting Figure 1: CKA heatmaps...")
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    for idx, ax in enumerate(axes.flat):
        layer = LAYER_INDICES[idx]
        sns.heatmap(
            cka_matrices[idx],
            vmin=0,
            vmax=1,
            cmap="viridis",
            annot=True,
            fmt=".2f",
            annot_kws={"size": 6},
            square=True,
            cbar=idx == 1,  # only show colorbar on top-right
            cbar_kws={"label": "Linear CKA", "shrink": 0.8} if idx == 1 else {},
            xticklabels=range(NUM_DENOISING_STEPS),
            yticklabels=range(NUM_DENOISING_STEPS),
            ax=ax,
        )
        ax.set_title(f"Layer {layer}")
        ax.set_xlabel("Denoising step $j$")
        ax.set_ylabel("Denoising step $i$")
    fig.suptitle(r"Cross-Denoising CKA: $\mathrm{CKA}(h_i^\ell,\, h_j^\ell)$", fontsize=15, y=1.01)
    fig.tight_layout()
    fig.savefig(out / "cka_matrices.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out / 'cka_matrices.png'}")

    # ---- Figure 2: CKA to final step ----
    print("Plotting Figure 2: CKA to final step...")
    palette = sns.color_palette("muted", NUM_LAYERS)
    fig, ax = plt.subplots(figsize=(8, 5))
    for idx in range(NUM_LAYERS):
        layer = LAYER_INDICES[idx]
        cka_to_final = cka_matrices[idx, :, NUM_DENOISING_STEPS - 1]
        ax.plot(range(NUM_DENOISING_STEPS), cka_to_final, "o-", color=palette[idx], label=f"Layer {layer}", linewidth=2)
    ax.set_xlabel("Denoising step $i$")
    ax.set_ylabel(r"$\mathrm{CKA}(h_i^\ell,\, h_9^\ell)$")
    ax.set_title("CKA Similarity to Final Denoising Step")
    ax.set_xticks(range(NUM_DENOISING_STEPS))
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(out / "cka_to_final_step.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {out / 'cka_to_final_step.png'}")

    # ---- Summary Statistics ----
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"\nEpisodes loaded: {n_loaded}")

    for idx in range(NUM_LAYERS):
        layer = LAYER_INDICES[idx]
        off_diag = cka_matrices[idx][np.triu_indices(NUM_DENOISING_STEPS, k=1)]
        cka_final = cka_matrices[idx, :, NUM_DENOISING_STEPS - 1]
        print(f"\nLayer {layer}:")
        print(f"  Off-diagonal CKA: mean={off_diag.mean():.4f}, min={off_diag.min():.4f}")
        print(f"  CKA to step 9: step_0={cka_final[0]:.4f}, step_3={cka_final[3]:.4f}, step_6={cka_final[6]:.4f}")

    print(f"\nFigures saved to: {Path(out).resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
