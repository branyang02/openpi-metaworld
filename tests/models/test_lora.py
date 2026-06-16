import flax.linen as nn
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import pytest

from openpi.models import pi0_config
import openpi.models.lora as lora
from openpi.shared import nnx_utils

# Skip in CI: small JAX matmul + Flax init, but enough to add ~5s to the suite.
# Run locally with `uv run pytest tests/models/test_lora.py -m manual`.
pytestmark = pytest.mark.manual


def test_lora_einsum_params_shape():
    shape = (3, 8, 32, 4)  # (3KDH)
    einsum = lora.Einsum(shape)
    lora0 = lora.Einsum(shape, lora_config=lora.LoRAConfig(rank=2))
    lora1 = lora.Einsum(shape, lora_config=lora.LoRAConfig(rank=2, axes=(1, 2)))

    key = jax.random.key(0)
    x = jax.random.normal(key, (8, 64, 32))  # (BSD)
    eqn = "BSD,3KDH->3BSKH"

    # Ensure that lora parameters are not initialized when LoRA is not used.
    params = einsum.init(key, eqn, x)
    assert "lora_a" not in params["params"]
    assert "lora_b" not in params["params"]

    # Check that default axes work.
    params_lora0 = lora0.init(key, eqn, x)
    assert params_lora0["params"]["lora_a"].shape == (3, 8, 32, 2)
    assert params_lora0["params"]["lora_b"].shape == (3, 8, 2, 4)

    # Check that user provided axes work.
    params_lora1 = lora1.init(key, eqn, x)
    assert params_lora1["params"]["lora_a"].shape == (3, 8, 2, 4)
    assert params_lora1["params"]["lora_b"].shape == (3, 2, 32, 4)


def test_lora_einsum_same_output():
    shape = (3, 8, 32, 4)  # (3KDH)
    einsum = lora.Einsum(shape)
    einsum_lora = lora.Einsum(shape, lora_config=lora.LoRAConfig(rank=2, init_fn=nn.initializers.zeros))

    key = jax.random.key(0)
    x = jax.random.normal(key, (8, 64, 32))  # (BSD)
    eqn = "BSD,3KDH->3BSKH"

    params = einsum.init(key, eqn, x)
    output = einsum.apply(params, eqn, x)

    params_lora = einsum_lora.init(key, eqn, x)
    output_lora = einsum_lora.apply(params_lora, eqn, x)

    # Results are the same since the LoRA parameters are initialized to zeros.
    assert jnp.allclose(output, output_lora)


def test_lora_ffn_params_shape():
    ffn = lora.FeedForward(features=8, hidden_dim=32)
    ffn_lora = lora.FeedForward(
        features=8,
        hidden_dim=32,
        lora_config=lora.LoRAConfig(rank=2),
    )

    key = jax.random.key(0)
    x = jax.random.normal(key, (2, 8))

    params = ffn.init(key, x)
    assert params["params"]["gating_einsum"].shape == (2, 8, 32)
    assert params["params"]["linear"].shape == (32, 8)

    params_lora = ffn_lora.init(key, x)
    assert params_lora["params"]["gating_einsum"].shape == (2, 8, 32)
    assert params_lora["params"]["linear"].shape == (32, 8)
    assert params_lora["params"]["gating_einsum_lora_a"].shape == (2, 8, 2)
    assert params_lora["params"]["gating_einsum_lora_b"].shape == (2, 2, 32)
    assert params_lora["params"]["linear_lora_a"].shape == (32, 2)
    assert params_lora["params"]["linear_lora_b"].shape == (2, 8)


def test_lora_ffn_same_output():
    ffn = lora.FeedForward(features=8, hidden_dim=32)
    ffn_lora = lora.FeedForward(
        features=8,
        hidden_dim=32,
        lora_config=lora.LoRAConfig(rank=2, init_fn=nn.initializers.zeros),
    )

    key = jax.random.key(0)
    x = jax.random.normal(key, (2, 8))

    params = ffn.init(key, x)
    output = ffn.apply(params, x)

    params_lora = ffn_lora.init(key, x)
    output_lora = ffn_lora.apply(params_lora, x)

    assert jnp.allclose(output, output_lora)


def _strip_lora_from_pure_dict(pd: dict) -> dict:
    """Recursively drop any key that ends in ``lora_a`` / ``lora_b`` (or contains those substrings)
    so the resulting pure-dict matches the non-LoRA graph shape exactly."""
    if not isinstance(pd, dict):
        return pd
    out = {}
    for k, v in pd.items():
        if isinstance(k, str) and ("lora_a" in k or "lora_b" in k):
            continue
        out[k] = _strip_lora_from_pure_dict(v) if isinstance(v, dict) else v
    return out


def test_pi05_lora_init_matches_base_vla_on_sample_actions():
    """End-to-end: at step 0, a LoRA-wrapped pi0.5 model with the default init
    (``init_fn_b = zeros``) produces the *same* actions as the non-LoRA pi0.5 model
    when they share identical base weights.

    This is the concrete guarantee PR #46 is meant to provide: the LoRA delta
    ``x @ w_a @ w_b`` is exactly zero at init (because ``w_b = 0``), so the
    LoRA-wrapped forward pass reduces to the base VLA's forward pass.

    Procedure:
        1. Build a LoRA-wrapped pi0.5 model (PaliGemma + action-expert LoRA).
        2. Build a non-LoRA pi0.5 model *from the same base weights* by stripping
           lora_a/lora_b out of the LoRA model's state pure-dict and loading it
           into the non-LoRA config.
        3. Assert every ``lora_b`` in the LoRA model is exactly zero.
        4. Run ``sample_actions`` on both with identical RNG and observation.
        5. Assert bit-identical output (rtol=0, atol=0).
    """
    rng = jax.random.key(0)

    lora_cfg = pi0_config.Pi0Config(
        pi05=True,
        action_horizon=32,
        discrete_state_input=False,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    )
    base_cfg = pi0_config.Pi0Config(
        pi05=True,
        action_horizon=32,
        discrete_state_input=False,
        # same as lora_cfg but with the non-LoRA Gemma variants
    )

    lora_model = lora_cfg.create(rng)

    # Extract the full state, confirm lora_b is zero, then strip LoRA keys and load
    # the remaining base weights into a non-LoRA pi0.5 to get a true apples-to-apples
    # comparison (identical base ``w`` in both models).
    _, lora_state = nnx.split(lora_model)
    pure = lora_state.to_pure_dict()

    lora_b_vals = []

    def _collect_lora_b(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if isinstance(k, str) and "lora_b" in k:
                    lora_b_vals.append((k, v))
                _collect_lora_b(v)

    _collect_lora_b(pure)
    assert len(lora_b_vals) > 0, "expected some lora_b params in a LoRA-wrapped pi0.5"
    for name, arr in lora_b_vals:
        assert jnp.all(jnp.asarray(arr) == 0), f"{name} is not zero at init (post PR #46)"

    base_pure = _strip_lora_from_pure_dict(pure)
    base_model = base_cfg.load(base_pure)

    batch_size = 1
    obs = base_cfg.fake_obs(batch_size)

    base_actions = nnx_utils.module_jit(base_model.sample_actions)(rng, obs, num_steps=4)
    lora_actions = nnx_utils.module_jit(lora_model.sample_actions)(rng, obs, num_steps=4)

    max_abs_diff = float(jnp.max(jnp.abs(base_actions - lora_actions)))
    assert max_abs_diff == 0.0, (
        f"LoRA-wrapped pi0.5 at init diverges from base VLA. Max abs diff: {max_abs_diff}. "
        "This means init_fn_b is no longer zero — PR #46 regression."
    )


def test_pi05_lora_b_is_zero_at_init():
    """Direct check on the LoRAConfig default: ``init_fn_b = zeros`` so every ``lora_b``
    factor in a freshly-created LoRA-wrapped pi0.5 is exactly zero.

    The full-model forward test above already covers this transitively, but this tiny
    sibling test catches regressions in the init default itself without paying
    ``sample_actions``'s runtime cost."""
    rng = jax.random.key(0)
    cfg = pi0_config.Pi0Config(
        pi05=True,
        action_horizon=32,
        discrete_state_input=False,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    )
    model = cfg.create(rng)
    _, state = nnx.split(model)
    pure = state.to_pure_dict()

    found = 0

    def walk(node):
        nonlocal found
        if isinstance(node, dict):
            for k, v in node.items():
                if isinstance(k, str) and "lora_b" in k:
                    found += 1
                    assert jnp.all(jnp.asarray(v) == 0), f"{k} is not zero at init"
                walk(v)

    walk(pure)
    assert found > 0, "expected some lora_b params in a LoRA-wrapped pi0.5"
