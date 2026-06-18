"""Shared data loading utilities for activation analysis experiments."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_BASE_DIR = "ml45-activations-15/5000"

# Task categories for semantic grouping
TASK_CATEGORIES = {
    "press": [
        "button-press-topdown-v3",
        "button-press-topdown-wall-v3",
        "button-press-v3",
        "button-press-wall-v3",
        "handle-press-side-v3",
        "handle-press-v3",
    ],
    "push": ["coffee-push-v3", "push-back-v3", "push-v3", "push-wall-v3", "stick-push-v3"],
    "pull": ["coffee-pull-v3", "handle-pull-side-v3", "handle-pull-v3", "lever-pull-v3", "stick-pull-v3"],
    "pick/place": ["pick-out-of-hole-v3", "pick-place-v3", "pick-place-wall-v3", "shelf-place-v3"],
    "slide": [
        "plate-slide-back-side-v3",
        "plate-slide-back-v3",
        "plate-slide-side-v3",
        "plate-slide-v3",
    ],
    "open/close": [
        "door-close-v3",
        "door-open-v3",
        "drawer-close-v3",
        "drawer-open-v3",
        "faucet-close-v3",
        "faucet-open-v3",
        "window-close-v3",
        "window-open-v3",
    ],
    "reach": ["reach-v3", "reach-wall-v3"],
    "other": [
        "assembly-v3",
        "basketball-v3",
        "coffee-button-v3",
        "dial-turn-v3",
        "disassemble-v3",
        "hammer-v3",
        "peg-insert-side-v3",
        "peg-unplug-side-v3",
        "soccer-v3",
        "sweep-into-v3",
        "sweep-v3",
    ],
}

# Reverse mapping: task -> category
TASK_TO_CATEGORY = {}
for cat, tasks in TASK_CATEGORIES.items():
    for t in tasks:
        TASK_TO_CATEGORY[t] = cat

# Layer indices collected (Action Expert layers 0, 5, 11, 17)
LAYER_INDICES = [0, 5, 11, 17]


def load_episode_index(base_dir: str = DEFAULT_BASE_DIR) -> pd.DataFrame:
    """Load all episode-level metadata into a DataFrame.

    Returns DataFrame with columns: task_name, episode_id, env_id, episode_success,
    total_reward, steps_to_success, total_env_steps, total_inference_steps, prompt,
    episode_dir (absolute path), category.
    """
    rows = []
    base = Path(base_dir)
    for task_dir in sorted(base.iterdir()):
        if not task_dir.is_dir():
            continue
        for ep_dir in sorted(task_dir.iterdir()):
            if not ep_dir.is_dir():
                continue
            meta_path = ep_dir / "metadata.json"
            if not meta_path.exists():
                continue
            with open(meta_path) as f:
                meta = json.load(f)
            meta["episode_dir"] = str(ep_dir)
            meta["category"] = TASK_TO_CATEGORY.get(meta["task_name"], "other")
            rows.append(meta)
    return pd.DataFrame(rows)


def load_step_metadata(episode_dir: str) -> list[dict]:
    """Load all step-level metadata for an episode, sorted by step number."""
    ep = Path(episode_dir)
    steps = []
    for step_dir in sorted(ep.iterdir()):
        if not step_dir.is_dir() or not step_dir.name.startswith("step_"):
            continue
        meta_path = step_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            meta["step_dir"] = str(step_dir)
            steps.append(meta)
    return sorted(steps, key=lambda x: x["step"])


def load_activations(step_dir: str, file_type: str) -> dict[str, np.ndarray]:
    """Load a specific .npz file from a step directory.

    Args:
        step_dir: Path to step_XXXX directory.
        file_type: One of 'denoising', 'adarms_cond', 'suffix_residual', 'suffix_mlp_hidden'.

    Returns:
        Dict of array name -> numpy array.
    """
    path = Path(step_dir) / f"{file_type}.npz"
    data = np.load(path)
    return dict(data)


def load_rewards(episode_dir: str) -> dict[str, np.ndarray]:
    """Load reward trajectory for an episode."""
    path = Path(episode_dir) / "rewards.npz"
    data = np.load(path)
    return dict(data)


def get_step_dirs(episode_dir: str) -> list[str]:
    """Get sorted list of step directories for an episode."""
    ep = Path(episode_dir)
    return [str(d) for d in sorted(ep.iterdir()) if d.is_dir() and d.name.startswith("step_")]


def sample_episodes(
    index: pd.DataFrame,
    tasks: list[str] | None = None,
    success: bool | None = None,
    n_per_task: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Filter and sample episodes from the index.

    Args:
        index: Episode index DataFrame from load_episode_index.
        tasks: If set, filter to these tasks only.
        success: If set, filter to success=True or success=False.
        n_per_task: If set, sample at most this many episodes per task.
        seed: Random seed for sampling.
    """
    filtered = index.copy()
    if tasks is not None:
        filtered = filtered[filtered["task_name"].isin(tasks)]
    if success is not None:
        filtered = filtered[filtered["episode_success"] == success]
    if n_per_task is not None:
        filtered = (
            filtered.groupby("task_name")
            .apply(
                lambda x: x.sample(min(len(x), n_per_task), random_state=seed),
                include_groups=False,
            )
            .reset_index(level=0, drop=False)
        )
    return filtered.reset_index(drop=True)


def get_partial_success_tasks(index: pd.DataFrame, min_rate: float = 0.05, max_rate: float = 0.95) -> list[str]:
    """Get tasks with success rate between min_rate and max_rate (exclusive)."""
    rates = index.groupby("task_name")["episode_success"].mean()
    return sorted(rates[(rates > min_rate) & (rates < max_rate)].index.tolist())


def get_task_success_rates(index: pd.DataFrame) -> dict[str, float]:
    """Return per-task success rate as a dict."""
    return index.groupby("task_name")["episode_success"].mean().to_dict()


def ensure_dir(path: str) -> Path:
    """Create directory if it doesn't exist, return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
