"""MetaWorld adapter: in-process rollout + eval against a loaded openpi policy.

Uses the same env/inference loop as ``examples/metaworld/main.py`` so behavior stays
in sync with the canonical eval scripts. ML45 train/test splits come from
``metaworld.ML45()``.

Both rollout and eval run in-process in the root venv: MetaWorld's env library is
light enough to coexist with the training JAX stack and a loaded PyTorch policy in
the same Python process, so we don't need subprocess orchestration here.
"""

from __future__ import annotations

import collections
from collections.abc import Sequence
import logging
import os
import sys

import gymnasium as gym
import metaworld
import numpy as np
from tqdm import tqdm

from experiments.filtered_bc.envs.adapter import EpisodeRollout
from experiments.filtered_bc.envs.adapter import EvalResult
from experiments.filtered_bc.envs.adapter import InferenceSample
from experiments.filtered_bc.envs.adapter import RolloutConfig

# ``examples/metaworld/main.py`` is not a package; add its directory to sys.path so
# we can import the shared env wrapper + task-prompt table.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_METAWORLD_DIR = os.path.join(_REPO_ROOT, "examples", "metaworld")
if _METAWORLD_DIR not in sys.path:
    sys.path.insert(0, _METAWORLD_DIR)

from main import CAMERA_IDS  # noqa: E402, F401
from main import TASK_TO_PROMPT  # noqa: E402
from main import MultiCameraWrapper  # noqa: E402

# ML45 train/test splits via metaworld's registry (source of truth post-merge).
_ML45 = metaworld.ML45()
ML45_TRAIN: list[str] = list(_ML45.train_classes.keys())
ML45_TEST: list[str] = list(_ML45.test_classes.keys())


logger = logging.getLogger(__name__)


def _make_env(task_name: str, seed: int, cfg: RolloutConfig) -> gym.Env:
    return MultiCameraWrapper(
        gym.make(
            "Meta-World/MT1",
            env_name=task_name,
            seed=seed,
            width=cfg.width,
            height=cfg.height,
        ),
        ["corner", "corner4", "gripperPOV"],
    )


def _rollout_impl(
    policy,
    task_name: str,
    num_episodes: int,
    cfg: RolloutConfig,
    *,
    record_samples: bool,
    seed_offset: int = 0,
    desc_prefix: str = "rollout",
) -> list[EpisodeRollout]:
    """Shared loop for rollout (record_samples=True) and eval (record_samples=False).

    seed_offset lets eval use a disjoint seed range from rollout so the held-out
    eval episodes aren't in the training data.
    """
    if task_name not in TASK_TO_PROMPT:
        raise ValueError(f"Unknown MetaWorld task: {task_name}")
    prompt = TASK_TO_PROMPT[task_name]
    num_envs = num_episodes
    base_seed = cfg.seed + seed_offset

    env_fns = [lambda i=i: _make_env(task_name, base_seed + i, cfg) for i in range(num_envs)]
    env = gym.vector.AsyncVectorEnv(env_fns, context="spawn")

    per_env_samples: list[list[InferenceSample]] = [[] for _ in range(num_envs)]

    try:
        obs, info = env.reset(seed=base_seed)
        camera_views = info["cameras"]
        success = np.zeros(num_envs, dtype=bool)
        cumulative_reward = np.zeros(num_envs, dtype=np.float64)
        steps_to_success = np.full(num_envs, -1, dtype=int)
        action_plan: collections.deque = collections.deque()

        max_steps = cfg.max_steps if cfg.max_steps is not None else 300
        pbar = tqdm(range(max_steps), desc=f"{desc_prefix} {task_name}")
        for step in pbar:
            if not action_plan:
                obs_dict = {
                    "observation/image": camera_views["corner4"],
                    "observation/wrist_image": camera_views["gripperPOV"],
                    "observation/state": obs.astype(np.float32)[..., :4],
                    "prompt": [prompt] * num_envs,
                }
                result = policy.infer(obs_dict)
                # Training targets must be the *raw* policy outputs, not the clipped ones —
                # otherwise filtered-BC self-distillation has a corrective gradient pulling
                # the policy away from its own saturated behaviour, independent of whether
                # LoRA init is zero. The env clips internally at step time anyway, so we
                # only clip the copy we dispatch to env.step.
                raw_action_chunk = np.asarray(result["actions"], dtype=np.float32)
                exec_action_chunk = np.clip(raw_action_chunk, -1.0, 1.0)

                if record_samples:
                    for env_id in range(num_envs):
                        if success[env_id]:
                            continue
                        per_env_samples[env_id].append(
                            InferenceSample(
                                image=np.asarray(camera_views["corner4"][env_id], dtype=np.uint8).copy(),
                                wrist_image=np.asarray(camera_views["gripperPOV"][env_id], dtype=np.uint8).copy(),
                                state=obs_dict["observation/state"][env_id].copy(),
                                prompt=prompt,
                                action_chunk=raw_action_chunk[env_id].copy(),
                            )
                        )

                for t in range(cfg.replan_steps):
                    action_plan.append(exec_action_chunk[:, t, :])

            action = action_plan.popleft()
            obs, reward, terminated, truncated, info = env.step(action)
            camera_views = info["cameras"]
            cumulative_reward += reward

            step_success = np.asarray(info.get("success", np.zeros(num_envs)), dtype=bool)
            for env_id in range(num_envs):
                if step_success[env_id] and steps_to_success[env_id] == -1:
                    steps_to_success[env_id] = step
            success |= step_success
            if success.all():
                break

            pbar.set_postfix(
                reward=f"{cumulative_reward.mean():.1f}",
                success=f"{success.mean():.0%}",
            )

        total_env_steps = step + 1

        return [
            EpisodeRollout(
                task_name=task_name,
                env_id=i,
                success=bool(success[i]),
                total_reward=float(cumulative_reward[i]),
                steps_to_success=int(steps_to_success[i]),
                total_env_steps=total_env_steps,
                samples=per_env_samples[i],
            )
            for i in range(num_envs)
        ]
    finally:
        env.close()


class MetaWorldAdapter:
    """In-process MetaWorld adapter. Expects a live openpi Policy object."""

    name = "metaworld"
    training_config = "pi05_metaworld_low_mem_finetune"

    @property
    def train_tasks(self) -> Sequence[str]:
        return ML45_TRAIN

    @property
    def test_tasks(self) -> Sequence[str]:
        return ML45_TEST

    def rollout(
        self,
        policy_or_ckpt,
        task_name: str,
        num_episodes: int,
        cfg: RolloutConfig | None = None,
    ) -> list[EpisodeRollout]:
        if cfg is None:
            cfg = RolloutConfig()
        return _rollout_impl(
            policy_or_ckpt,
            task_name,
            num_episodes,
            cfg,
            record_samples=True,
            seed_offset=0,
            desc_prefix="rollout",
        )

    def eval(
        self,
        policy_or_ckpt,
        task_name: str,
        num_episodes: int,
        cfg: RolloutConfig | None = None,
    ) -> EvalResult:
        if cfg is None:
            cfg = RolloutConfig()
        # Offset seed by 10_000 so eval episodes are disjoint from rollout.
        rollouts = _rollout_impl(
            policy_or_ckpt,
            task_name,
            num_episodes,
            cfg,
            record_samples=False,
            seed_offset=10_000,
            desc_prefix="eval",
        )
        n_succ = sum(1 for r in rollouts if r.success)
        rewards = [r.total_reward for r in rollouts]
        succ_steps = [r.steps_to_success for r in rollouts if r.steps_to_success >= 0]
        return EvalResult(
            task_name=task_name,
            num_episodes=len(rollouts),
            num_success=n_succ,
            success_rate=n_succ / len(rollouts) if rollouts else 0.0,
            mean_reward=float(np.mean(rewards)) if rewards else 0.0,
            mean_steps_to_success=float(np.mean(succ_steps)) if succ_steps else float("nan"),
        )


__all__ = [
    "ML45_TEST",
    "ML45_TRAIN",
    "TASK_TO_PROMPT",
    "MetaWorldAdapter",
]
