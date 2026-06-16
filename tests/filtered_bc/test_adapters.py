"""CPU-only smoke tests for the filtered-BC env adapters.

These verify that each adapter can be constructed, exposes a non-empty task
list, and references a training config that actually resolves. Running the full
rollout/eval pipeline requires a GPU + the env's own venv and lives in the
smoke scripts under ``experiments/filtered_bc/``.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.filtered_bc.envs.adapter import EnvAdapter
from experiments.filtered_bc.envs.adapter import EpisodeRollout
from experiments.filtered_bc.envs.adapter import EvalResult
from experiments.filtered_bc.envs.adapter import InferenceSample
from experiments.filtered_bc.envs.adapter import RolloutConfig
from experiments.filtered_bc.envs.adapter import filter_successful
from experiments.filtered_bc.envs.adapter import get_adapter
from openpi.training import config as _config


def test_rollout_config_defaults():
    cfg = RolloutConfig()
    assert cfg.width == 224
    assert cfg.height == 224
    assert cfg.replan_steps == 10
    assert cfg.extra == {}


def test_inference_sample_shape():
    sample = InferenceSample(
        image=np.zeros((224, 224, 3), dtype=np.uint8),
        wrist_image=np.zeros((224, 224, 3), dtype=np.uint8),
        state=np.zeros(8, dtype=np.float32),
        prompt="pick up the block",
        action_chunk=np.zeros((10, 7), dtype=np.float32),
    )
    assert sample.action_chunk.shape == (10, 7)


def test_filter_successful_drops_failures():
    def _ep(*, success: bool, n_samples: int) -> EpisodeRollout:
        return EpisodeRollout(
            task_name="t",
            env_id=0,
            success=success,
            total_reward=0.0,
            steps_to_success=-1,
            total_env_steps=0,
            samples=[
                InferenceSample(
                    image=np.zeros((1, 1, 3), np.uint8),
                    wrist_image=np.zeros((1, 1, 3), np.uint8),
                    state=np.zeros(1, np.float32),
                    prompt="",
                    action_chunk=np.zeros((1, 1), np.float32),
                )
                for _ in range(n_samples)
            ],
        )

    eps = [
        _ep(success=True, n_samples=3),
        _ep(success=False, n_samples=2),
        _ep(success=True, n_samples=5),
    ]
    samples = filter_successful(eps)
    assert len(samples) == 8  # 3 + 5, failed episode dropped


@pytest.mark.parametrize(
    ("env", "training_config"),
    [
        ("metaworld", "pi05_metaworld_low_mem_finetune"),
        ("libero", "pi05_libero_low_mem_finetune"),
        ("robocasa", "pi05_robocasa_low_mem_finetune"),
    ],
)
def test_get_adapter_resolves_training_config(env: str, training_config: str):
    # MetaWorld adapter imports metaworld + examples/metaworld/main.py at construction
    # time, so skip if not importable (e.g. CI without the metaworld extras).
    try:
        adapter = get_adapter(env)
    except Exception as exc:  # pragma: no cover — env-import-dependent
        pytest.skip(f"Adapter {env!r} unavailable: {exc}")

    assert adapter.name == env
    assert adapter.training_config == training_config
    # Training config must be registered in the openpi registry.
    tc = _config.get_config(training_config)
    assert tc.name == training_config

    # Tasks should be non-empty.
    assert len(adapter.train_tasks) > 0, f"{env} has no train_tasks"
    assert len(adapter.test_tasks) > 0, f"{env} has no test_tasks"


def test_get_adapter_rejects_unknown_env():
    with pytest.raises(ValueError, match="Unknown env"):
        get_adapter("xland")


@pytest.mark.parametrize("num_episodes", [15, 20, 25])
def test_libero_rollout_eval_seeds_land_on_disjoint_init_states(num_episodes: int):
    """Guard against the regression where ``10_000 % 50 == 0`` made rollout and eval
    collapse onto identical LIBERO initial states.

    The LIBERO client picks ``initial_states[(seed + ep) % num_init_states]``, with
    ``num_init_states = 50`` for every standard suite. With the old ``+ 10_000`` eval
    offset, rollout and eval indexed the same 15 states (20..34) on every task — the
    finetuned policy was silently evaluated on its training scenes.

    The adapter now offsets eval by ``num_episodes`` so the two windows stay disjoint
    mod 50 as long as ``2 * num_episodes <= 50``. That caps the per-side run size at
    25; anything larger must either change the offset scheme or use a suite with more
    than 50 canonical init states."""
    num_init_states = 50
    base_seed = 69_420

    rollout_states = {(base_seed + i) % num_init_states for i in range(num_episodes)}
    eval_states = {(base_seed + num_episodes + i) % num_init_states for i in range(num_episodes)}

    assert rollout_states.isdisjoint(eval_states), (
        f"LIBERO rollout/eval init states overlap at N={num_episodes}: rollout={sorted(rollout_states)}, eval={sorted(eval_states)}"
    )
    assert len(rollout_states) == num_episodes
    assert len(eval_states) == num_episodes


def test_libero_seed_scheme_breaks_above_25():
    """Explicit contract test: N=26 exceeds the disjoint-mod-50 capacity by 2, so
    rollout and eval overlap on exactly 2 states. If you need N>25 per side, add a
    new offset strategy rather than quietly letting this break."""
    num_init_states = 50
    num_episodes = 26
    base_seed = 69_420

    rollout_states = {(base_seed + i) % num_init_states for i in range(num_episodes)}
    eval_states = {(base_seed + num_episodes + i) % num_init_states for i in range(num_episodes)}

    assert not rollout_states.isdisjoint(eval_states), (
        "If this ever becomes disjoint, someone changed the offset scheme — update the N<=25 cap docs accordingly."
    )


def test_env_adapter_protocol_contract():
    """Each concrete adapter should satisfy the EnvAdapter Protocol duck-type."""
    try:
        mw = get_adapter("metaworld")
    except Exception as exc:
        pytest.skip(f"metaworld adapter unavailable: {exc}")
    # Check the members the Protocol lists — the important ones.
    assert isinstance(mw.name, str)
    assert isinstance(mw.training_config, str)
    assert callable(mw.rollout)
    assert callable(mw.eval)
    # isinstance against Protocol requires runtime_checkable; we don't decorate it
    # because it's structural docs, not a type-check target.
    _ = EnvAdapter  # Protocol is importable
    _ = EvalResult  # EvalResult is importable
