"""Exp 1: Denoising trajectory geometry analysis.

Analyzes the geometry of flow-matching denoising trajectories across episodes,
comparing successful vs failed episodes. Produces 4 figures:
  1. Velocity norm vs denoising step (success vs failure)
  2. Consecutive velocity cosine similarity vs denoising step
  3. Straightness metric distribution (success vs failure histogram)
  4. Variance decay across denoising steps

Usage:
    uv run denoising_step_exp/exp1_denoising.py
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
from data_utils import ensure_dir  # noqa: E402
from data_utils import get_step_dirs  # noqa: E402
from data_utils import load_activations  # noqa: E402
from data_utils import load_episode_index  # noqa: E402

OUTPUT_DIR = "denoising_step_exp/results/figures/exp1"
NUM_DENOISING_STEPS = 10
ACTION_HORIZON = 32
ACTION_DIM = 32
FLAT_DIM = ACTION_HORIZON * ACTION_DIM  # 1024

BLUE = sns.color_palette("muted")[0]
RED = sns.color_palette("muted")[3]


def sample_step_dirs(step_dirs: list[str], n: int = 3) -> list[str]:
    """Sample n inference steps: first, middle, last."""
    if len(step_dirs) <= 2:
        return step_dirs
    mid = len(step_dirs) // 2
    return [step_dirs[i] for i in [0, mid, len(step_dirs) - 1]][:n]


def compute_episode_stats(episode_dir: str) -> dict | None:
    """Load denoising data for sampled steps and compute per-episode statistics."""
    step_dirs = get_step_dirs(episode_dir)
    if not step_dirs:
        return None

    sampled = sample_step_dirs(step_dirs, n=3)
    n_sampled = len(sampled)

    velocity_norms = np.zeros((n_sampled, NUM_DENOISING_STEPS))
    cosine_sims = np.zeros((n_sampled, NUM_DENOISING_STEPS - 1))
    straightness = np.zeros(n_sampled)
    variance_per_denoise = np.zeros((n_sampled, NUM_DENOISING_STEPS))

    for i, sd in enumerate(sampled):
        try:
            data = load_activations(sd, "denoising")
        except Exception:
            return None

        all_x_t = data["all_x_t"]  # (10, 32, 32)
        all_v_t = data["all_v_t"]  # (10, 32, 32)

        v_flat = all_v_t.reshape(NUM_DENOISING_STEPS, FLAT_DIM)
        x_flat = all_x_t.reshape(NUM_DENOISING_STEPS, FLAT_DIM)

        # Velocity norms
        velocity_norms[i] = np.linalg.norm(v_flat, axis=1)

        # Cosine similarity between consecutive velocity vectors
        for t in range(NUM_DENOISING_STEPS - 1):
            v_a, v_b = v_flat[t], v_flat[t + 1]
            denom = np.linalg.norm(v_a) * np.linalg.norm(v_b)
            cosine_sims[i, t] = np.dot(v_a, v_b) / denom if denom > 1e-10 else 0.0

        # Straightness: chord / arc
        chord = np.linalg.norm(x_flat[0] - x_flat[-1])
        arc = sum(np.linalg.norm(x_flat[t] - x_flat[t + 1]) for t in range(NUM_DENOISING_STEPS - 1))
        straightness[i] = chord / arc if arc > 1e-10 else 1.0

        # Variance across action dims at each denoising step
        variance_per_denoise[i] = np.var(x_flat, axis=1)

    return {
        "velocity_norms": velocity_norms,
        "cosine_sims": cosine_sims,
        "straightness": straightness,
        "variance_per_denoise": variance_per_denoise,
    }


def main():
    out = ensure_dir(OUTPUT_DIR)
    print("Loading episode index...")
    index = load_episode_index()
    print(f"Found {len(index)} episodes across {index['task_name'].nunique()} tasks")
    print(f"  Success: {index['episode_success'].sum()}, Failure: {(~index['episode_success']).sum()}")

    # Collect stats split by success/failure
    vn_s, vn_f = [], []
    cs_s, cs_f = [], []
    st_s, st_f = [], []
    var_s, var_f = [], []

    for _, row in tqdm(index.iterrows(), total=len(index), desc="Processing episodes"):
        stats = compute_episode_stats(row["episode_dir"])
        if stats is None:
            continue
        if row["episode_success"]:
            vn_s.append(stats["velocity_norms"])
            cs_s.append(stats["cosine_sims"])
            st_s.extend(stats["straightness"].tolist())
            var_s.append(stats["variance_per_denoise"])
        else:
            vn_f.append(stats["velocity_norms"])
            cs_f.append(stats["cosine_sims"])
            st_f.extend(stats["straightness"].tolist())
            var_f.append(stats["variance_per_denoise"])

    vn_success = np.concatenate(vn_s) if vn_s else np.empty((0, NUM_DENOISING_STEPS))
    vn_failure = np.concatenate(vn_f) if vn_f else np.empty((0, NUM_DENOISING_STEPS))
    cs_success = np.concatenate(cs_s) if cs_s else np.empty((0, NUM_DENOISING_STEPS - 1))
    cs_failure = np.concatenate(cs_f) if cs_f else np.empty((0, NUM_DENOISING_STEPS - 1))
    var_success = np.concatenate(var_s) if var_s else np.empty((0, NUM_DENOISING_STEPS))
    var_failure = np.concatenate(var_f) if var_f else np.empty((0, NUM_DENOISING_STEPS))

    steps = np.arange(NUM_DENOISING_STEPS)
    steps_mid = np.arange(NUM_DENOISING_STEPS - 1)

    # ---- Figure 1: Velocity norms ----
    print("\nPlotting Figure 1: Velocity norms...")
    fig, ax = plt.subplots(figsize=(8, 5))
    if vn_success.shape[0] > 0:
        m, s = vn_success.mean(0), vn_success.std(0)
        ax.plot(steps, m, "o-", color=BLUE, label=f"Success ($n={vn_success.shape[0]}$)")
        ax.fill_between(steps, m - s, m + s, color=BLUE, alpha=0.2)
    if vn_failure.shape[0] > 0:
        m, s = vn_failure.mean(0), vn_failure.std(0)
        ax.plot(steps, m, "s-", color=RED, label=f"Failure ($n={vn_failure.shape[0]}$)")
        ax.fill_between(steps, m - s, m + s, color=RED, alpha=0.2)
    ax.set_xlabel("Denoising step $t$")
    ax.set_ylabel(r"$\|v_t\|_2$")
    ax.set_title("Velocity Norm Across Denoising Steps")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "velocity_norms.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {out / 'velocity_norms.png'}")

    # ---- Figure 2: Cosine similarity ----
    print("Plotting Figure 2: Cosine similarity...")
    fig, ax = plt.subplots(figsize=(8, 5))
    if cs_success.shape[0] > 0:
        m, s = cs_success.mean(0), cs_success.std(0)
        ax.plot(steps_mid, m, "o-", color=BLUE, label=f"Success ($n={cs_success.shape[0]}$)")
        ax.fill_between(steps_mid, m - s, m + s, color=BLUE, alpha=0.2)
    if cs_failure.shape[0] > 0:
        m, s = cs_failure.mean(0), cs_failure.std(0)
        ax.plot(steps_mid, m, "s-", color=RED, label=f"Failure ($n={cs_failure.shape[0]}$)")
        ax.fill_between(steps_mid, m - s, m + s, color=RED, alpha=0.2)
    ax.set_xlabel("Denoising step $t$")
    ax.set_ylabel(r"$\cos(v_t,\, v_{t+1})$")
    ax.set_title("Consecutive Velocity Cosine Similarity")
    ax.legend()
    ax.set_ylim(-0.1, 1.05)
    fig.tight_layout()
    fig.savefig(out / "cosine_similarity.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {out / 'cosine_similarity.png'}")

    # ---- Figure 3: Straightness histogram ----
    print("Plotting Figure 3: Straightness distribution...")
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1.05, 40)
    if st_s:
        ax.hist(st_s, bins=bins, alpha=0.6, color=BLUE, label=f"Success ($n={len(st_s)}$)", density=True)
    if st_f:
        ax.hist(st_f, bins=bins, alpha=0.6, color=RED, label=f"Failure ($n={len(st_f)}$)", density=True)
    ax.set_xlabel(r"Straightness $\;\|x_0 - x_T\|\; / \;\sum_t \|x_t - x_{t+1}\|$")
    ax.set_ylabel("Density")
    ax.set_title("Denoising Trajectory Straightness")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "straightness_hist.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {out / 'straightness_hist.png'}")

    # ---- Figure 4: Variance decay ----
    print("Plotting Figure 4: Variance decay...")
    fig, ax = plt.subplots(figsize=(8, 5))
    if var_success.shape[0] > 0:
        m, s = var_success.mean(0), var_success.std(0)
        ax.plot(steps, m, "o-", color=BLUE, label=f"Success ($n={var_success.shape[0]}$)")
        ax.fill_between(steps, m - s, m + s, color=BLUE, alpha=0.2)
    if var_failure.shape[0] > 0:
        m, s = var_failure.mean(0), var_failure.std(0)
        ax.plot(steps, m, "o-", color=RED, label=f"Failure ($n={var_failure.shape[0]}$)")
        ax.fill_between(steps, m - s, m + s, color=RED, alpha=0.2)
    ax.set_xlabel("Denoising step $t$")
    ax.set_ylabel(r"$\mathrm{Var}(x_t)$ across action dims")
    ax.set_title("Action Variance Decay During Denoising")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "variance_decay.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {out / 'variance_decay.png'}")

    # ---- Summary Statistics ----
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    if vn_success.shape[0] > 0:
        print("\nVelocity norms (success):")
        print(f"  First step: {vn_success[:, 0].mean():.4f} +/- {vn_success[:, 0].std():.4f}")
        print(f"  Last step:  {vn_success[:, -1].mean():.4f} +/- {vn_success[:, -1].std():.4f}")
    if vn_failure.shape[0] > 0:
        print("Velocity norms (failure):")
        print(f"  First step: {vn_failure[:, 0].mean():.4f} +/- {vn_failure[:, 0].std():.4f}")
        print(f"  Last step:  {vn_failure[:, -1].mean():.4f} +/- {vn_failure[:, -1].std():.4f}")

    if cs_success.shape[0] > 0:
        print(f"\nCosine similarity (success) - mean: {cs_success.mean():.4f}")
    if cs_failure.shape[0] > 0:
        print(f"Cosine similarity (failure) - mean: {cs_failure.mean():.4f}")

    if st_s:
        arr = np.array(st_s)
        print(f"\nStraightness (success): mean={arr.mean():.4f}, median={np.median(arr):.4f}, std={arr.std():.4f}")
    if st_f:
        arr = np.array(st_f)
        print(f"Straightness (failure): mean={arr.mean():.4f}, median={np.median(arr):.4f}, std={arr.std():.4f}")

    if var_success.shape[0] > 0:
        print(f"\nVariance (success) - step 0: {var_success[:, 0].mean():.6f}, step 9: {var_success[:, -1].mean():.6f}")
    if var_failure.shape[0] > 0:
        print(f"Variance (failure) - step 0: {var_failure[:, 0].mean():.6f}, step 9: {var_failure[:, -1].mean():.6f}")

    print(f"\nFigures saved to: {out.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
