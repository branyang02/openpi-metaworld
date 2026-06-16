"""CPU tests for Flow-DPO loss math. No model, pure JAX arithmetic."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from experiments.preference_bc.dpo_loss import flow_dpo_loss_components
from experiments.preference_bc.dpo_loss import flow_dpo_loss_from_mses


def test_loss_at_theta_equals_ref_is_log2():
    """When theta == ref, both deltas are zero and loss == log 2."""
    mse_t = jnp.ones((4, 8))
    mse_r = jnp.ones((4, 8))
    loss = flow_dpo_loss_from_mses(mse_t, mse_r, mse_t, mse_r, beta=1.0)
    assert float(loss) == pytest.approx(np.log(2.0), abs=1e-5)


def test_loss_decreases_as_policy_prefers_pos():
    """If theta improves on pos more than on neg (desired direction), loss drops."""
    # Reference MSEs are the same baseline.
    mse_ref = jnp.ones((16, 8))
    # Policy beats ref by 0.1 on positives, is 0.1 worse on negatives.
    mse_theta_pos = mse_ref - 0.1
    mse_theta_neg = mse_ref + 0.1
    better = flow_dpo_loss_from_mses(mse_theta_pos, mse_ref, mse_theta_neg, mse_ref, beta=1.0)

    # Opposite: policy is WORSE on pos and BETTER on neg (undesired).
    worse = flow_dpo_loss_from_mses(mse_theta_neg, mse_ref, mse_theta_pos, mse_ref, beta=1.0)

    assert float(better) < np.log(2.0) < float(worse)


def test_loss_is_zero_in_limit_of_perfect_preference():
    """With huge beta and a positive gap, loss -> 0."""
    mse_ref = jnp.ones((4, 8))
    mse_theta_pos = mse_ref - 1.0
    mse_theta_neg = mse_ref + 1.0
    loss = flow_dpo_loss_from_mses(mse_theta_pos, mse_ref, mse_theta_neg, mse_ref, beta=100.0)
    assert float(loss) == pytest.approx(0.0, abs=1e-6)


def test_loss_large_in_limit_of_reversed_preference():
    """With huge beta and a reversed gap, loss blows up linearly (beta * |gap|)."""
    mse_ref = jnp.ones((4, 8))
    # Reversed: theta is WORSE on pos and BETTER on neg.
    mse_theta_pos = mse_ref + 1.0
    mse_theta_neg = mse_ref - 1.0
    loss = flow_dpo_loss_from_mses(mse_theta_pos, mse_ref, mse_theta_neg, mse_ref, beta=10.0)
    # The arg is beta * (delta_neg - delta_pos) = 10 * (-1 - 1) = -20
    # -log_sigmoid(-20) ~= 20
    assert float(loss) > 10.0


def test_components_match_loss():
    """flow_dpo_loss_components['loss'] matches flow_dpo_loss_from_mses."""
    rng = np.random.default_rng(0)
    mse_t_pos = jnp.asarray(rng.random((8, 10)).astype(np.float32))
    mse_r_pos = jnp.asarray(rng.random((8, 10)).astype(np.float32))
    mse_t_neg = jnp.asarray(rng.random((8, 10)).astype(np.float32))
    mse_r_neg = jnp.asarray(rng.random((8, 10)).astype(np.float32))

    direct = flow_dpo_loss_from_mses(mse_t_pos, mse_r_pos, mse_t_neg, mse_r_neg, beta=3.14)
    pieces = flow_dpo_loss_components(mse_t_pos, mse_r_pos, mse_t_neg, mse_r_neg, beta=3.14)
    np.testing.assert_allclose(float(direct), float(pieces["loss"]), rtol=1e-6)

    # preference_gap == delta_neg_mean - delta_pos_mean
    np.testing.assert_allclose(
        float(pieces["preference_gap"]),
        float(pieces["delta_neg_mean"] - pieces["delta_pos_mean"]),
        rtol=1e-5,
    )

    # reward_accuracy in [0, 1]
    acc = float(pieces["reward_accuracy"])
    assert 0.0 <= acc <= 1.0
