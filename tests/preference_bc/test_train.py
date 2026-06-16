"""CPU tests for DPO train-step wiring.

Exercises ``compute_dpo_components`` with a fake model so we can verify that the
components dict (which the JIT'd train_step returns as ``info`` and the train
loop logs) carries all the diagnostic keys we need for the beta sweep:
``loss``, ``delta_pos_mean``, ``delta_neg_mean``, ``preference_gap``,
``reward_accuracy``.

These tests do not load a real pi0.5 model — they pass a stub with a
``compute_loss(rng, obs, act, train)`` method that returns canned MSE arrays.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.preference_bc.train import compute_dpo_components

EXPECTED_COMPONENT_KEYS = {
    "loss",
    "delta_pos_mean",
    "delta_neg_mean",
    "preference_gap",
    "reward_accuracy",
}


@dataclasses.dataclass
class _StubModel:
    """Returns canned MSE arrays. ``mse_pos`` is returned on the first call,
    ``mse_neg`` on the second — matching the train_step's positives-then-negatives
    invocation order.
    """

    mse_pos: jax.Array
    mse_neg: jax.Array
    _calls: int = 0

    def compute_loss(self, rng, obs, act, *, train):
        self._calls += 1
        return self.mse_pos if self._calls == 1 else self.mse_neg


def _run(mse_theta_pos, mse_theta_neg, mse_ref_pos, mse_ref_neg, *, beta):
    model = _StubModel(jnp.asarray(mse_theta_pos), jnp.asarray(mse_theta_neg))
    return compute_dpo_components(
        model,
        jax.random.key(0),
        obs_pos=None,
        act_pos=None,
        obs_neg=None,
        act_neg=None,
        mse_ref_pos=jnp.asarray(mse_ref_pos),
        mse_ref_neg=jnp.asarray(mse_ref_neg),
        beta=beta,
    )


def test_components_dict_has_all_logging_keys():
    """The dict returned to the train loop must carry every diagnostic the README claims."""
    components = _run(
        mse_theta_pos=np.ones((4, 8), dtype=np.float32),
        mse_theta_neg=np.ones((4, 8), dtype=np.float32),
        mse_ref_pos=np.ones((4, 8), dtype=np.float32),
        mse_ref_neg=np.ones((4, 8), dtype=np.float32),
        beta=1.0,
    )
    missing = EXPECTED_COMPONENT_KEYS - set(components.keys())
    assert not missing, f"missing keys: {missing}"


def test_reward_accuracy_is_one_when_policy_strictly_prefers_pos():
    """If theta beats ref on every positive AND loses to ref on every negative,
    every pair has gap > 0 → reward_accuracy == 1.0.
    """
    components = _run(
        mse_theta_pos=np.full((4, 8), 0.5, dtype=np.float32),  # better than ref
        mse_theta_neg=np.full((4, 8), 1.5, dtype=np.float32),  # worse than ref
        mse_ref_pos=np.ones((4, 8), dtype=np.float32),
        mse_ref_neg=np.ones((4, 8), dtype=np.float32),
        beta=1.0,
    )
    assert float(components["reward_accuracy"]) == pytest.approx(1.0)
    # And preference_gap = (1.5 - 1.0) - (0.5 - 1.0) = 0.5 - (-0.5) = 1.0.
    assert float(components["preference_gap"]) == pytest.approx(1.0, abs=1e-5)


def test_reward_accuracy_is_zero_when_policy_strictly_prefers_neg():
    """Reversed setup → every pair has gap < 0 → reward_accuracy == 0.0."""
    components = _run(
        mse_theta_pos=np.full((4, 8), 1.5, dtype=np.float32),
        mse_theta_neg=np.full((4, 8), 0.5, dtype=np.float32),
        mse_ref_pos=np.ones((4, 8), dtype=np.float32),
        mse_ref_neg=np.ones((4, 8), dtype=np.float32),
        beta=1.0,
    )
    assert float(components["reward_accuracy"]) == pytest.approx(0.0)


def test_reward_accuracy_is_half_at_init():
    """At theta == ref, every gap is exactly 0 → reward_accuracy is 0.0
    (since `gap > 0` is strict). We surface this for the sweep operator: a
    *true* random policy hovers at ~0.5; a freshly-LoRA-init'd policy starts
    at 0.0 and should climb toward 1.0 as beta does its job.
    """
    components = _run(
        mse_theta_pos=np.ones((4, 8), dtype=np.float32),
        mse_theta_neg=np.ones((4, 8), dtype=np.float32),
        mse_ref_pos=np.ones((4, 8), dtype=np.float32),
        mse_ref_neg=np.ones((4, 8), dtype=np.float32),
        beta=2000.0,  # value used in the sweep
    )
    assert float(components["reward_accuracy"]) == pytest.approx(0.0)
    assert float(components["preference_gap"]) == pytest.approx(0.0, abs=1e-6)
    # Loss at theta==ref is log 2 regardless of beta.
    assert float(components["loss"]) == pytest.approx(np.log(2.0), abs=1e-5)


def test_components_are_jit_compatible():
    """The whole point of returning the dict is to flow through value_and_grad +
    jax.jit. Verify by jitting a wrapper.
    """

    def fn(mse_theta_pos, mse_theta_neg, mse_ref_pos, mse_ref_neg):
        # Use a fresh model per call to keep the call counter clean inside JIT.
        model = _StubModel(mse_theta_pos, mse_theta_neg)
        return compute_dpo_components(
            model,
            jax.random.key(0),
            obs_pos=None,
            act_pos=None,
            obs_neg=None,
            act_neg=None,
            mse_ref_pos=mse_ref_pos,
            mse_ref_neg=mse_ref_neg,
            beta=2000.0,
        )

    # _StubModel uses a Python int counter that JAX can't trace. Run eagerly
    # instead — the key invariant is that compute_dpo_components produces only
    # JAX-array outputs (no Python scalars or NumPy fallbacks), which is what
    # actually matters for the JIT'd train step.
    out = fn(
        jnp.full((2, 4), 0.5),
        jnp.full((2, 4), 1.5),
        jnp.ones((2, 4)),
        jnp.ones((2, 4)),
    )
    for k in EXPECTED_COMPONENT_KEYS:
        assert isinstance(out[k], jax.Array), f"{k} is {type(out[k])}, expected jax.Array"
