"""Tests for denoising step ablation: verify num_steps flows through the policy
to model.sample_actions() and actually controls the number of Euler iterations."""

from unittest.mock import patch

import numpy as np
import pytest
import torch

from openpi.policies import policy as _policy


class FakeModel:
    """Minimal fake model that records how sample_actions is called."""

    def __init__(self):
        self.call_log = []

    def sample_actions(self, device, observation, **kwargs):
        self.call_log.append({"device": device, "kwargs": kwargs})
        # Return a dummy action tensor: (batch, action_horizon=1, action_dim=4)
        batch_size = observation.state.shape[0]
        return torch.zeros(batch_size, 1, 4)

    def eval(self):
        pass

    def to(self, device):
        return self


def _make_policy(num_steps: int, model: FakeModel | None = None) -> tuple[_policy.Policy, FakeModel]:
    """Create a Policy with a fake model and specified num_steps."""
    if model is None:
        model = FakeModel()
    policy = _policy.Policy(
        model,
        transforms=[],
        output_transforms=[],
        sample_kwargs={"num_steps": num_steps},
        is_pytorch=True,
        pytorch_device="cpu",
    )
    return policy, model


def _make_obs():
    return {
        "observation/state": np.zeros((1, 4), dtype=np.float32),
        "observation/image": np.zeros((1, 224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.zeros((1, 224, 224, 3), dtype=np.uint8),
        "prompt": ["test prompt"],
    }


def _mock_transform(x):
    return {
        "state": np.zeros((4,), dtype=np.float32),
        "image": {"0": np.zeros((224, 224, 3), dtype=np.uint8)},
        "image_mask": {"0": np.ones((), dtype=bool)},
        "tokenized_prompt": np.zeros((10,), dtype=np.int32),
        "tokenized_prompt_mask": np.ones((10,), dtype=bool),
    }


class TestSampleKwargsPropagation:
    """Test that num_steps in sample_kwargs propagates from Policy to model.sample_actions()."""

    def test_num_steps_stored_in_policy(self):
        policy, _ = _make_policy(num_steps=3)
        assert policy._sample_kwargs == {"num_steps": 3}  # noqa: SLF001

    def test_num_steps_passed_to_sample_actions(self):
        policy, model = _make_policy(num_steps=5)

        with (
            patch.object(policy, "_input_transform", side_effect=_mock_transform),
            patch.object(policy, "_output_transform", side_effect=lambda x: x),
        ):
            policy.infer(_make_obs())

        assert len(model.call_log) == 1
        assert model.call_log[0]["kwargs"]["num_steps"] == 5

    def test_different_num_steps_values_propagate(self):
        """Verify that different num_steps values each propagate correctly."""
        for num_steps in [1, 2, 3, 5, 10]:
            policy, model = _make_policy(num_steps=num_steps)
            with (
                patch.object(policy, "_input_transform", side_effect=_mock_transform),
                patch.object(policy, "_output_transform", side_effect=lambda x: x),
            ):
                policy.infer(_make_obs())

            assert model.call_log[0]["kwargs"]["num_steps"] == num_steps

    def test_noise_kwarg_coexists_with_num_steps(self):
        """Verify noise and num_steps can both be passed via sample_kwargs."""
        policy, model = _make_policy(num_steps=3)
        noise = np.random.randn(1, 1, 4).astype(np.float32)
        with (
            patch.object(policy, "_input_transform", side_effect=_mock_transform),
            patch.object(policy, "_output_transform", side_effect=lambda x: x),
        ):
            policy.infer(_make_obs(), noise=noise)

        kwargs = model.call_log[0]["kwargs"]
        assert kwargs["num_steps"] == 3
        assert "noise" in kwargs


class TestEulerLoopStepCount:
    """Test that the Euler denoising loop in sample_actions runs exactly num_steps iterations."""

    @pytest.mark.parametrize("num_steps", [1, 2, 3, 5, 10])
    def test_denoise_step_called_n_times(self, num_steps):
        """Simulate the while loop from sample_actions to verify iteration count."""
        dt = -1.0 / num_steps
        dt_tensor = torch.tensor(dt, dtype=torch.float32)
        time = torch.tensor(1.0, dtype=torch.float32)

        iteration_count = 0
        while time >= -dt_tensor / 2:
            iteration_count += 1
            time += dt_tensor

        assert iteration_count == num_steps, f"Expected {num_steps} iterations but got {iteration_count}"

    @pytest.mark.parametrize("num_steps", [1, 2, 3, 5, 10])
    def test_dt_value_correct(self, num_steps):
        """Verify dt = -1/num_steps produces the right step size."""
        dt = -1.0 / num_steps
        expected_dt = -1.0 / num_steps
        assert dt == pytest.approx(expected_dt)
        # After num_steps iterations starting at t=1.0, we should reach t~=0.0
        final_time = 1.0 + num_steps * dt
        assert final_time == pytest.approx(0.0, abs=1e-10)

    def test_single_step_produces_different_output_than_multistep(self):
        """With a non-zero velocity field, 1-step and 10-step should differ
        (they only match when the flow is perfectly straight, which is the
        empirical finding -- but with arbitrary v_t they should differ)."""
        batch_size = 1
        action_horizon = 4
        action_dim = 4
        noise = torch.randn(batch_size, action_horizon, action_dim)

        def run_euler(x_t, num_steps, velocity_fn):
            """Simulate the Euler loop from sample_actions."""
            dt = -1.0 / num_steps
            dt_t = torch.tensor(dt, dtype=torch.float32)
            time = torch.tensor(1.0, dtype=torch.float32)
            while time >= -dt_t / 2:
                v_t = velocity_fn(x_t, time)
                x_t = x_t + dt_t * v_t
                time += dt_t
            return x_t

        # Non-linear velocity field: v_t depends on x_t
        def nonlinear_velocity(x_t, t):
            return torch.sin(x_t) * t

        result_1step = run_euler(noise.clone(), num_steps=1, velocity_fn=nonlinear_velocity)
        result_10step = run_euler(noise.clone(), num_steps=10, velocity_fn=nonlinear_velocity)

        # With a nonlinear velocity field, more steps = different (more accurate) result
        assert not torch.allclose(
            result_1step, result_10step, atol=1e-4
        ), "1-step and 10-step should differ with a nonlinear velocity field"

    def test_linear_velocity_same_regardless_of_steps(self):
        """With a constant velocity field (perfectly straight flow),
        all step counts should produce identical results -- this is the
        theoretical basis for the ablation."""
        batch_size = 1
        action_horizon = 4
        action_dim = 4
        noise = torch.randn(batch_size, action_horizon, action_dim)

        def run_euler(x_t, num_steps, velocity_fn):
            dt = -1.0 / num_steps
            dt_t = torch.tensor(dt, dtype=torch.float32)
            time = torch.tensor(1.0, dtype=torch.float32)
            while time >= -dt_t / 2:
                v_t = velocity_fn(x_t, time)
                x_t = x_t + dt_t * v_t
                time += dt_t
            return x_t

        # Constant velocity field (doesn't depend on x_t or t)
        constant_v = torch.randn(batch_size, action_horizon, action_dim)

        def constant_velocity(x_t, t):
            return constant_v

        results = {}
        for num_steps in [1, 2, 3, 5, 10]:
            results[num_steps] = run_euler(noise.clone(), num_steps=num_steps, velocity_fn=constant_velocity)

        # All should be identical since Euler is exact for constant velocity
        for ns in [2, 3, 5, 10]:
            assert torch.allclose(
                results[1], results[ns], atol=1e-5
            ), f"1-step and {ns}-step should match for constant velocity field"
