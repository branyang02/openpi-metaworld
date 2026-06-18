"""Exp 2: adaRMS conditioning analysis.

Analyzes the adaptive RMS conditioning vectors that modulate the action expert
across the 10 denoising steps. Key questions:
  - Is the conditioning deterministic across episodes at each denoising step?
  - How does the conditioning trajectory evolve through denoising?
  - What is the effective dimensionality of the conditioning space?

Produces 4 figures:
  1. Determinism check: max L2 distance per denoising step
  2. PCA trajectory: conditioning in PC1-PC2 space
  3. Norm progression: ||cond_t|| vs denoising step
  4. Effective dimensionality: cumulative variance + scree plot

Usage:
    uv run denoising_step_exp/exp2_adarms.py
"""

from __future__ import annotations

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
from data_utils import ensure_dir  # noqa: E402
from data_utils import get_step_dirs  # noqa: E402
from data_utils import load_activations  # noqa: E402
from data_utils import load_episode_index  # noqa: E402

FIGURE_DIR = "denoising_step_exp/results/figures/exp2"
N_DENOISING_STEPS = 10
COND_DIM = 1024

BLUE = sns.color_palette("muted")[0]


class SimplePCA:
    """Minimal PCA implementation using SVD."""

    def __init__(self, n_components: int | None = None):
        self.n_components = n_components
        self.explained_variance_ratio_: np.ndarray | None = None
        self.components_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None

    def fit(self, data: np.ndarray) -> SimplePCA:
        self.mean_ = data.mean(axis=0)
        centered = data - self.mean_
        _u, svals, vt = np.linalg.svd(centered, full_matrices=False)
        explained_var = (svals**2) / max(data.shape[0] - 1, 1)
        total_var = explained_var.sum()
        self.explained_variance_ratio_ = explained_var / total_var if total_var > 0 else explained_var
        n = self.n_components or len(svals)
        self.components_ = vt[:n]
        self.explained_variance_ratio_ = self.explained_variance_ratio_[:n]
        return self

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        self.fit(data)
        return (data - self.mean_) @ self.components_.T


def load_all_adarms(index) -> np.ndarray:
    """Load adaRMS conditioning from step_0000 for every episode.

    Returns:
        cond: (n_episodes, 10, 1024)
    """
    all_cond = []
    failed = 0
    for _, row in tqdm(index.iterrows(), total=len(index), desc="Loading adaRMS"):
        step_dirs = get_step_dirs(row["episode_dir"])
        if not step_dirs:
            failed += 1
            continue
        try:
            data = load_activations(step_dirs[0], "adarms_cond")
            cond = data["all_adarms_cond"]  # (10, 1024)
            assert cond.shape == (N_DENOISING_STEPS, COND_DIM), f"Unexpected shape {cond.shape}"
            all_cond.append(cond)
        except Exception as e:
            failed += 1
            if failed <= 3:
                print(f"  Warning: failed to load: {e}")
    if failed:
        print(f"  Skipped {failed}/{len(index)} episodes due to errors.")
    return np.stack(all_cond, axis=0)


def main():
    print("=" * 70)
    print("Exp 2: adaRMS Conditioning Analysis")
    print("=" * 70)

    fig_dir = ensure_dir(FIGURE_DIR)
    index = load_episode_index()
    print(f"Total episodes: {len(index)}, Tasks: {index['task_name'].nunique()}")

    # Load conditioning
    print("\nLoading adaRMS conditioning vectors (step_0000)...")
    cond = load_all_adarms(index)
    print(f"Loaded shape: {cond.shape}")

    # ---- Determinism check ----
    print("\nChecking determinism...")
    ref = cond[0]
    max_l2 = np.zeros(N_DENOISING_STEPS)
    for i in range(1, cond.shape[0]):
        dists = np.linalg.norm(cond[i] - ref, axis=1)
        max_l2 = np.maximum(max_l2, dists)
    is_deterministic = bool(np.all(max_l2 < 1e-6))

    for t in range(N_DENOISING_STEPS):
        print(f"  step {t}: max L2 = {max_l2[t]:.6e}")
    status = "DETERMINISTIC" if is_deterministic else "VARIES across episodes"
    print(f"  --> {status}")

    # Figure 1: Determinism bar chart
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(
        x=list(range(N_DENOISING_STEPS)), y=max_l2.tolist(), color=BLUE, edgecolor="black", linewidth=0.5, ax=ax
    )
    ax.set_xlabel("Denoising step $t$")
    ax.set_ylabel(r"$\max_i \| c_t^{(i)} - c_t^{(0)} \|_2$")
    ax.set_title(f"adaRMS Conditioning Determinism Check ({status})")
    fig.tight_layout()
    fig.savefig(fig_dir / "determinism_check.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {fig_dir / 'determinism_check.png'}")

    # ---- Figure 2: PCA trajectory ----
    print("\nComputing PCA trajectory...")
    if is_deterministic:
        vectors = cond[0]  # (10, 1024)
        pca = SimplePCA(n_components=2)
        proj = pca.fit_transform(vectors)

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(proj[:, 0], proj[:, 1], "o-", color=BLUE, markersize=8, linewidth=2)
        for t in range(N_DENOISING_STEPS):
            ax.annotate(f"$t={t}$", (proj[t, 0], proj[t, 1]), textcoords="offset points", xytext=(7, 5), fontsize=9)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
        ax.set_title("adaRMS Conditioning PCA Trajectory (Deterministic)")
    else:
        all_vectors = cond.reshape(-1, COND_DIM)
        pca = SimplePCA(n_components=2)
        all_proj = pca.fit_transform(all_vectors).reshape(cond.shape[0], N_DENOISING_STEPS, 2)

        fig, ax = plt.subplots(figsize=(7, 6))
        n_show = min(20, cond.shape[0])
        palette = sns.color_palette("husl", n_show)
        for i in range(n_show):
            ax.plot(all_proj[i, :, 0], all_proj[i, :, 1], "o-", color=palette[i], markersize=3, alpha=0.5)
        mean_proj = all_proj.mean(axis=0)
        ax.plot(mean_proj[:, 0], mean_proj[:, 1], "s-", color="black", markersize=6, linewidth=2, label="Mean")
        for t in range(N_DENOISING_STEPS):
            ax.annotate(
                f"$t={t}$", (mean_proj[t, 0], mean_proj[t, 1]), textcoords="offset points", xytext=(7, 5), fontsize=9
            )
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
        ax.set_title(f"adaRMS Conditioning PCA Trajectory ({n_show} episodes)")
        ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(fig_dir / "pca_trajectory.png", dpi=150)
    plt.close(fig)
    print(f"  PC1: {pca.explained_variance_ratio_[0]:.1%}, PC2: {pca.explained_variance_ratio_[1]:.1%}")
    print(f"  Saved {fig_dir / 'pca_trajectory.png'}")

    # ---- Figure 3: Norm progression ----
    print("\nComputing norm progression...")
    norms = np.linalg.norm(cond, axis=2)  # (N, 10)

    fig, ax = plt.subplots(figsize=(8, 4))
    t_steps = np.arange(N_DENOISING_STEPS)
    if is_deterministic:
        ax.plot(t_steps, norms[0], "o-", color=BLUE, markersize=8, linewidth=2)
        ax.set_title(r"adaRMS Conditioning $\|c_t\|_2$ (Deterministic)")
    else:
        mean_n, std_n = norms.mean(0), norms.std(0)
        ax.plot(t_steps, mean_n, "o-", color=BLUE, markersize=8, linewidth=2, label="Mean")
        ax.fill_between(t_steps, mean_n - std_n, mean_n + std_n, alpha=0.2, color=BLUE, label=r"$\pm 1\sigma$")
        ax.set_title(r"adaRMS Conditioning $\|c_t\|_2$")
        ax.legend(fontsize=9)
    ax.set_xlabel("Denoising step $t$")
    ax.set_ylabel(r"$\|c_t\|_2$")
    ax.set_xticks(t_steps)
    fig.tight_layout()
    fig.savefig(fig_dir / "norm_progression.png", dpi=150)
    plt.close(fig)
    print(f"  Norm range: [{norms.min():.4f}, {norms.max():.4f}]")
    print(f"  Saved {fig_dir / 'norm_progression.png'}")

    # ---- Figure 4: Effective dimensionality ----
    print("\nComputing effective dimensionality...")
    vectors = cond[0] if is_deterministic else cond.reshape(-1, COND_DIM)
    n_comp = min(vectors.shape[0], vectors.shape[1])
    pca_full = SimplePCA(n_components=n_comp)
    pca_full.fit(vectors)

    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_for_95 = int(np.searchsorted(cumvar, 0.95) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(np.arange(1, len(cumvar) + 1), cumvar, "o-", color=BLUE, markersize=4)
    ax1.axhline(0.95, color=sns.color_palette("muted")[3], linestyle="--", alpha=0.8, label="95%")
    ax1.axvline(n_for_95, color=sns.color_palette("muted")[3], linestyle=":", alpha=0.8, label=f"$k={n_for_95}$")
    ax1.set_xlabel("Number of components $k$")
    ax1.set_ylabel("Cumulative explained variance")
    ax1.set_title("Effective Dimensionality of adaRMS Conditioning")
    ax1.legend(fontsize=9)

    n_show = min(30, len(pca_full.explained_variance_ratio_))
    sns.barplot(
        x=list(range(1, n_show + 1)),
        y=pca_full.explained_variance_ratio_[:n_show].tolist(),
        color=BLUE,
        edgecolor="black",
        linewidth=0.5,
        ax=ax2,
    )
    ax2.set_xlabel("Component")
    ax2.set_ylabel("Explained variance ratio")
    ax2.set_title(f"Scree Plot (first {n_show} components)")
    # Thin out x-tick labels to avoid crowding
    ax2.set_xticks([0, 4, 9, 14, 19, 24, 29][:n_show])
    ax2.set_xticklabels([1, 5, 10, 15, 20, 25, 30][:n_show])

    fig.tight_layout()
    fig.savefig(fig_dir / "effective_dimensionality.png", dpi=150)
    plt.close(fig)
    print(f"  Components for 95% variance: {n_for_95}")
    print(f"  Saved {fig_dir / 'effective_dimensionality.png'}")

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Episodes loaded:        {cond.shape[0]}")
    print(f"  Deterministic:          {is_deterministic}")
    print(f"  Norm range:             [{norms.min():.4f}, {norms.max():.4f}]")
    print(f"  Components for 95% var: {n_for_95}")
    print(f"\nFigures saved to: {fig_dir}/")


if __name__ == "__main__":
    main()
