"""Env-neutral types and Protocol for the filtered-BC baseline.

Each env (MetaWorld, LIBERO, RoboCasa) provides a concrete adapter implementing
:class:`EnvAdapter`. The orchestrator in ``run_filtered_bc.py`` calls
``adapter.rollout(...)`` and ``adapter.eval(...)`` without caring whether the
policy runs in-process (MetaWorld) or in a subprocess websocket server
(LIBERO / RoboCasa).

Shared sample shape across envs is ``{image, wrist_image, state, prompt,
action_chunk}``; per-env dims (state: 4/8/16, action: 4/7/12, action_horizon:
32/10/50) live inside the numpy arrays, not in the schema.
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
from typing import Any, Protocol

import numpy as np


@dataclasses.dataclass
class InferenceSample:
    """A single (obs, action_chunk) pair produced by one inference call during rollout."""

    image: np.ndarray  # (H, W, 3) uint8, primary camera
    wrist_image: np.ndarray  # (H, W, 3) uint8, wrist camera
    state: np.ndarray  # (state_dim,) float32 — env-specific dim
    prompt: str
    action_chunk: np.ndarray  # (action_horizon, action_dim) float32 — env-specific dims


@dataclasses.dataclass
class EpisodeRollout:
    task_name: str
    env_id: int
    success: bool
    total_reward: float
    steps_to_success: int  # -1 if never succeeded
    total_env_steps: int
    samples: list[InferenceSample]


@dataclasses.dataclass
class EvalResult:
    task_name: str
    num_episodes: int
    num_success: int
    success_rate: float
    mean_reward: float
    mean_steps_to_success: float  # NaN if none succeeded


@dataclasses.dataclass
class RolloutConfig:
    width: int = 224
    height: int = 224
    # None means "use the adapter's env-specific default": 300 for MetaWorld, the
    # suite-specific value from ``SUITE_MAX_STEPS`` for LIBERO, and
    # ``1.5 * task_horizon`` for RoboCasa. Pass an integer to override.
    max_steps: int | None = None
    replan_steps: int = 10
    seed: int = 69_420
    # Extra env-specific knobs the adapter may consume (e.g. task-suite name).
    extra: dict[str, Any] = dataclasses.field(default_factory=dict)


class EnvAdapter(Protocol):
    """Contract every env-specific adapter must satisfy.

    ``policy_or_ckpt`` is either a live ``openpi.policies.policy.Policy`` (for
    in-process envs like MetaWorld) or a checkpoint directory path (for
    server-client envs like LIBERO / RoboCasa that spawn a policy server).
    Concrete adapters document which they expect.
    """

    name: str  # e.g. "metaworld", "libero", "robocasa"
    training_config: str  # TrainConfig name to load via openpi.training.config.get_config()

    @property
    def train_tasks(self) -> Sequence[str]: ...

    @property
    def test_tasks(self) -> Sequence[str]: ...

    def rollout(
        self,
        policy_or_ckpt: Any,
        task_name: str,
        num_episodes: int,
        cfg: RolloutConfig | None = None,
    ) -> list[EpisodeRollout]: ...

    def eval(
        self,
        policy_or_ckpt: Any,
        task_name: str,
        num_episodes: int,
        cfg: RolloutConfig | None = None,
    ) -> EvalResult: ...


def filter_successful(episodes: list[EpisodeRollout]) -> list[InferenceSample]:
    """Flatten rollouts, keep only samples from successful episodes."""
    return [s for ep in episodes if ep.success for s in ep.samples]


def get_adapter(name: str) -> EnvAdapter:
    """Lazy-import and return the adapter for a given env name.

    Lazy imports so e.g. a LIBERO-only run doesn't pay the MetaWorld import cost.
    """
    name = name.lower()
    if name == "metaworld":
        from experiments.filtered_bc.envs.metaworld import MetaWorldAdapter

        return MetaWorldAdapter()
    if name == "libero":
        from experiments.filtered_bc.envs.libero import LiberoAdapter

        return LiberoAdapter()
    if name == "robocasa":
        from experiments.filtered_bc.envs.robocasa import RoboCasaAdapter

        return RoboCasaAdapter()
    raise ValueError(f"Unknown env: {name!r}. Expected one of: metaworld, libero, robocasa.")


__all__ = [
    "EnvAdapter",
    "EpisodeRollout",
    "EvalResult",
    "InferenceSample",
    "RolloutConfig",
    "filter_successful",
    "get_adapter",
]
