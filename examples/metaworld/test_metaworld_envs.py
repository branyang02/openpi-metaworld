"""Tests for Metaworld environment creation and basic functionality.

Run with:
    MUJOCO_GL=egl uv run pytest examples/metaworld/test_metaworld_envs.py -v

Note: MT50 creates 50 parallel environments and may be slow.
"""

from main import make_env
import numpy as np
import pytest

WORKING_BENCHMARKS = ["MT10", "MT50", "ML10", "ML45"]
UNIMPLEMENTED_BENCHMARKS = ["MT1", "ML1"]

SEED = 42
CAMERA_NAMES = ["gripperPOV", "corner", "corner2"]


@pytest.mark.parametrize("benchmark_name", WORKING_BENCHMARKS)
def test_benchmark_creates_and_runs(benchmark_name):
    """Test that each benchmark can be created, reset, and stepped through."""
    env = make_env(benchmark_name, env_name=None, seed=SEED, camera_names=CAMERA_NAMES)
    try:
        assert env.num_envs > 0, f"{benchmark_name}: expected num_envs > 0"

        # Reset and validate observations
        obs, info = env.reset(seed=SEED)
        assert obs.shape[0] == env.num_envs, (
            f"{benchmark_name}: obs batch dim {obs.shape[0]} != num_envs {env.num_envs}"
        )
        assert "cameras" in info, f"{benchmark_name}: 'cameras' missing from reset info"
        assert len(info["cameras"]) == env.num_envs, (
            f"{benchmark_name}: cameras list length {len(info['cameras'])} != num_envs {env.num_envs}"
        )

        # Validate camera images from reset
        for i, cam_dict in enumerate(info["cameras"]):
            for cam_name in CAMERA_NAMES:
                assert cam_name in cam_dict, f"{benchmark_name} env[{i}]: camera '{cam_name}' missing"
                img = cam_dict[cam_name]
                assert img.ndim == 3, f"{benchmark_name} env[{i}] '{cam_name}': expected 3D array, got {img.ndim}D"
                assert img.shape[2] == 3, (
                    f"{benchmark_name} env[{i}] '{cam_name}': expected 3 channels, got {img.shape[2]}"
                )
                assert img.dtype == np.uint8, f"{benchmark_name} env[{i}] '{cam_name}': expected uint8, got {img.dtype}"

        # Take one step with a random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        assert obs.shape[0] == env.num_envs, (
            f"{benchmark_name}: step obs batch dim {obs.shape[0]} != num_envs {env.num_envs}"
        )
        assert reward.shape == (env.num_envs,), f"{benchmark_name}: reward shape {reward.shape} != ({env.num_envs},)"
        assert terminated.shape == (env.num_envs,), (
            f"{benchmark_name}: terminated shape {terminated.shape} != ({env.num_envs},)"
        )
        assert truncated.shape == (env.num_envs,), (
            f"{benchmark_name}: truncated shape {truncated.shape} != ({env.num_envs},)"
        )
        assert "cameras" in info, f"{benchmark_name}: 'cameras' missing from step info"
        assert len(info["cameras"]) == env.num_envs
    finally:
        env.close()


@pytest.mark.parametrize("benchmark_name", UNIMPLEMENTED_BENCHMARKS)
def test_unimplemented_benchmarks_raise(benchmark_name):
    """Test that MT1 and ML1 raise NotImplementedError as expected."""
    with pytest.raises(NotImplementedError):
        make_env(benchmark_name, env_name=None, seed=SEED, camera_names=CAMERA_NAMES)
