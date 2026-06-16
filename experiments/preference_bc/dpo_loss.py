"""Flow-DPO loss for pi0.5.

Diffusion-DPO (Wallace et al. 2023) adapted to flow matching. Each training sample
is a (pos, neg) pair drawn under the same noise schedule; the loss pushes the LoRA
policy to improve on positives more than it improves on negatives, both measured
as the flow-matching MSE shift relative to a frozen reference policy.

Loss (all four MSEs are the pi0 ``compute_loss`` return — ``mean((v - u_t)**2, axis=-1)``,
shape ``[B, H]``; we reduce over H inside this module):

    Delta_pos = MSE_theta(pos) - MSE_ref(pos)     # <0 when theta better than ref on positives
    Delta_neg = MSE_theta(neg) - MSE_ref(neg)     # >0 when theta worse than ref on negatives (desired)
    L = -E[ log sigma( -beta * (Delta_pos - Delta_neg) ) ]
      = -E[ log sigma( beta * (Delta_neg - Delta_pos) ) ]

Higher beta pushes the policy more aggressively toward positives vs negatives.
At ``theta == ref`` both deltas are zero, L = log 2 = 0.693. Gradient signal lives
entirely in how much theta can beat/lose to ref on each sample, which is what we want.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def flow_dpo_loss_from_mses(
    mse_theta_pos: jax.Array,
    mse_ref_pos: jax.Array,
    mse_theta_neg: jax.Array,
    mse_ref_neg: jax.Array,
    *,
    beta: float = 2000.0,
) -> jax.Array:
    """Flow-DPO loss given the 4 per-sample flow-matching MSEs (shape ``[B, H]``).

    Returns a scalar loss averaged over the pair batch.
    """
    mse_theta_pos = mse_theta_pos.mean(axis=-1)  # (B,)
    mse_ref_pos = mse_ref_pos.mean(axis=-1)
    mse_theta_neg = mse_theta_neg.mean(axis=-1)
    mse_ref_neg = mse_ref_neg.mean(axis=-1)

    delta_pos = mse_theta_pos - mse_ref_pos  # want negative
    delta_neg = mse_theta_neg - mse_ref_neg  # want positive

    # L = -log sigma(beta * (delta_neg - delta_pos))
    arg = beta * (delta_neg - delta_pos)
    # jax.nn.log_sigmoid is numerically stable for large |arg|.
    return -jax.nn.log_sigmoid(arg).mean()


def flow_dpo_loss_components(
    mse_theta_pos: jax.Array,
    mse_ref_pos: jax.Array,
    mse_theta_neg: jax.Array,
    mse_ref_neg: jax.Array,
    *,
    beta: float = 2000.0,
) -> dict:
    """Same as :func:`flow_dpo_loss_from_mses` but returns intermediate pieces for logging.

    Useful fields to track during training:

        delta_pos_mean : scalar, avg (MSE_theta - MSE_ref) on positives
        delta_neg_mean : scalar, avg (MSE_theta - MSE_ref) on negatives
        preference_gap : scalar, avg (delta_neg - delta_pos); positive = policy improving in desired direction
        reward_accuracy : scalar in [0,1], fraction of pairs where policy prefers pos over neg relative to ref
    """
    mse_theta_pos_s = mse_theta_pos.mean(axis=-1)
    mse_ref_pos_s = mse_ref_pos.mean(axis=-1)
    mse_theta_neg_s = mse_theta_neg.mean(axis=-1)
    mse_ref_neg_s = mse_ref_neg.mean(axis=-1)
    delta_pos = mse_theta_pos_s - mse_ref_pos_s
    delta_neg = mse_theta_neg_s - mse_ref_neg_s
    gap = delta_neg - delta_pos
    loss = -jax.nn.log_sigmoid(beta * gap).mean()
    return {
        "loss": loss,
        "delta_pos_mean": delta_pos.mean(),
        "delta_neg_mean": delta_neg.mean(),
        "preference_gap": gap.mean(),
        "reward_accuracy": (gap > 0).astype(jnp.float32).mean(),
    }


__all__ = ["flow_dpo_loss_components", "flow_dpo_loss_from_mses"]
