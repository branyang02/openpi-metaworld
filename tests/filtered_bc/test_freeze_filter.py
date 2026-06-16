"""Tests for the LoRA-config freeze filters.

Verifies that the `pi05_*_low_mem_finetune` configs freeze:
  1. PaliGemma LM dense weights (excluding LoRA adapters)
  2. Action expert LM dense weights (excluding LoRA adapters)
  3. SigLip vision tower (img branch) — added 2026-04-29 to fix the
     filtered-BC catastrophic regression (coffee-push-v3: 83% → 0%)

And leaves trainable:
  - LoRA adapters (lora_a, lora_b) on both LM variants
  - Action / time projection heads (action_in_proj, action_out_proj,
    time_mlp_in, time_mlp_out)

Test (a) is a pure unit test on synthetic paths — fast, CI-friendly.
Test (b) is integration: applies the filter to the base ckpt's actual
params and asserts every img-* path is frozen.
"""

from __future__ import annotations

import pathlib

import flax.traverse_util as traverse_util
import numpy as np
import pytest

from openpi.training import config as _config

_BASE_CKPT = pathlib.Path("/home/kim34/projects_brandon/openpi-metaworld/checkpoints/openpi-metaworld-5000")
_REQUIRES_BASE_CKPT = pytest.mark.skipif(not _BASE_CKPT.exists(), reason=f"Base ckpt not found at {_BASE_CKPT}")


def _is_frozen(freeze_filter, path_tuple: tuple) -> bool:
    """Apply the freeze_filter to a path and return True if it's frozen.

    Filters in flax.nnx are callables: filter(path: PathParts, x: Any) -> bool.
    Our freeze_filter only inspects the path (via PathRegex), so we can pass
    any value for x.
    """
    return bool(freeze_filter(path_tuple, None))


# =============================================================================
# Test A — Synthetic-path unit test (fast, CI-friendly)
# =============================================================================


@pytest.mark.parametrize(
    ("config_name", "paths_frozen", "paths_trainable"),
    [
        (
            "pi05_metaworld_low_mem_finetune",
            # Frozen
            [
                # PaliGemma LM dense weights (no LoRA suffix)
                ("PaliGemma", "llm", "layers", "attn", "q_einsum", "w"),
                ("PaliGemma", "llm", "layers", "attn", "kv_einsum", "w"),
                ("PaliGemma", "llm", "layers", "attn", "attn_vec_einsum", "w"),
                ("PaliGemma", "llm", "layers", "mlp", "gating_einsum"),
                ("PaliGemma", "llm", "layers", "mlp", "linear"),
                ("PaliGemma", "llm", "layers", "pre_attention_norm", "scale"),
                ("PaliGemma", "llm", "final_norm", "scale"),
                # Action expert LM dense weights
                ("PaliGemma", "llm_1", "layers", "attn", "q_einsum", "w"),
                ("PaliGemma", "llm_1", "layers", "mlp", "gating_einsum"),
                # Vision tower (NEW — added by this change)
                ("PaliGemma", "img", "embedding", "kernel"),
                ("PaliGemma", "img", "Transformer", "encoderblock", "LayerNorm_0", "scale"),
                ("PaliGemma", "img", "Transformer", "encoderblock", "MlpBlock_0", "Dense_0", "kernel"),
                ("PaliGemma", "img", "Transformer", "encoderblock", "MlpBlock_0", "Dense_0", "bias"),
                ("PaliGemma", "img", "head", "kernel"),
                ("PaliGemma", "img", "pos_embedding"),
            ],
            # Trainable
            [
                # LoRA adapters on PaliGemma LM
                ("PaliGemma", "llm", "layers", "attn", "q_einsum", "lora_a"),
                ("PaliGemma", "llm", "layers", "attn", "q_einsum", "lora_b"),
                ("PaliGemma", "llm", "layers", "mlp", "gating_einsum_lora_a"),
                ("PaliGemma", "llm", "layers", "mlp", "gating_einsum_lora_b"),
                # LoRA adapters on action expert
                ("PaliGemma", "llm_1", "layers", "attn", "q_einsum", "lora_a"),
                ("PaliGemma", "llm_1", "layers", "mlp", "linear_lora_b"),
                # Action / time projection heads (top-level, no PaliGemma prefix)
                ("action_in_proj", "kernel"),
                ("action_in_proj", "bias"),
                ("action_out_proj", "kernel"),
                ("action_out_proj", "bias"),
                ("time_mlp_in", "kernel"),
                ("time_mlp_in", "bias"),
                ("time_mlp_out", "kernel"),
                ("time_mlp_out", "bias"),
            ],
        ),
        (
            "pi05_libero_low_mem_finetune",
            # Frozen — same expectation as MetaWorld
            [
                ("PaliGemma", "llm", "layers", "attn", "q_einsum", "w"),
                ("PaliGemma", "img", "embedding", "kernel"),
                ("PaliGemma", "img", "Transformer", "encoderblock", "LayerNorm_0", "scale"),
            ],
            [
                ("PaliGemma", "llm", "layers", "attn", "q_einsum", "lora_a"),
                ("action_in_proj", "kernel"),
            ],
        ),
        (
            "pi05_robocasa_low_mem_finetune",
            [
                ("PaliGemma", "llm", "layers", "attn", "q_einsum", "w"),
                ("PaliGemma", "img", "embedding", "kernel"),
                ("PaliGemma", "img", "head", "bias"),
            ],
            [
                ("PaliGemma", "llm", "layers", "attn", "q_einsum", "lora_a"),
                ("action_in_proj", "kernel"),
                ("time_mlp_out", "bias"),
            ],
        ),
    ],
)
def test_freeze_filter_classifies_synthetic_paths(config_name, paths_frozen, paths_trainable):
    """For each LoRA config, verify the freeze_filter freezes the right families."""
    cfg = _config.get_config(config_name)
    freeze_filter = cfg.freeze_filter
    assert freeze_filter is not None, f"{config_name} has no freeze_filter"

    misclassified = [
        {"path": "/".join(path), "expected": "FROZEN", "got": "trainable"}
        for path in paths_frozen
        if not _is_frozen(freeze_filter, path)
    ] + [
        {"path": "/".join(path), "expected": "trainable", "got": "FROZEN"}
        for path in paths_trainable
        if _is_frozen(freeze_filter, path)
    ]

    if misclassified:
        msg = f"\n{config_name}: {len(misclassified)} path(s) misclassified:\n"
        for m in misclassified:
            msg += f"  {m['path']}: expected {m['expected']}, got {m['got']}\n"
        pytest.fail(msg)


def test_vision_tower_freeze_is_specific_to_img_branch():
    """Sanity: the new 'img' regex doesn't accidentally freeze unrelated paths.

    We're using `.*img.*` which is permissive — verify it doesn't freeze, e.g.,
    a hypothetical 'trimming' or 'IMGBuf' param. Pi0.5 doesn't have any such
    params, so this is just a guard for future param naming changes.
    """
    cfg = _config.get_config("pi05_metaworld_low_mem_finetune")
    freeze_filter = cfg.freeze_filter

    # These paths must NOT be frozen by the new img regex.
    safe_paths = [
        ("action_in_proj", "kernel"),
        ("time_mlp_in", "kernel"),
        ("action_out_proj", "bias"),
    ]
    for path in safe_paths:
        assert not _is_frozen(freeze_filter, path), (
            f"Path {'/'.join(path)} is being frozen but shouldn't be — the .*img.* regex is over-matching."
        )


# =============================================================================
# Test B — Integration: apply filter to actual base-ckpt params
# =============================================================================


@pytest.mark.manual
@_REQUIRES_BASE_CKPT
def test_freeze_filter_on_real_base_ckpt_params():
    """Load the actual base ckpt params, apply the freeze filter, and verify:
      - Every path under PaliGemma/img/* is FROZEN
      - Every path under PaliGemma/llm/* (and llm_1/*) is FROZEN (no LoRA in base)
      - Every path under action_in_proj/, time_mlp_in/, etc. is TRAINABLE

    Since the base ckpt has no LoRA leaves, this verifies the FROZEN side
    cleanly. LoRA classification is covered in the synthetic-path test above.
    """
    from openpi.models import model as _model

    cfg = _config.get_config("pi05_metaworld_low_mem_finetune")
    freeze_filter = cfg.freeze_filter

    print(f"\nLoading base ckpt JAX params from {_BASE_CKPT}/params ...")
    params = _model.restore_params(str(_BASE_CKPT / "params"), restore_type=np.ndarray, dtype="float32")
    flat = dict(traverse_util.flatten_dict(params))
    print(f"Total params: {len(flat)}")

    img_paths = [p for p in flat if "img" in p]
    llm_paths = [p for p in flat if "llm" in p and "llm_1" not in p]
    expert_paths = [p for p in flat if "llm_1" in p]
    proj_paths = [p for p in flat if p[0] in ("action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out")]
    print(f"  img params:     {len(img_paths)}")
    print(f"  llm params:     {len(llm_paths)}")
    print(f"  expert params:  {len(expert_paths)}")
    print(f"  proj params:    {len(proj_paths)}")

    img_frozen = sum(1 for p in img_paths if _is_frozen(freeze_filter, p))
    llm_frozen = sum(1 for p in llm_paths if _is_frozen(freeze_filter, p))
    expert_frozen = sum(1 for p in expert_paths if _is_frozen(freeze_filter, p))
    proj_frozen = sum(1 for p in proj_paths if _is_frozen(freeze_filter, p))
    print(f"\n  img    frozen: {img_frozen}/{len(img_paths)}  (expect ALL frozen)")
    print(f"  llm    frozen: {llm_frozen}/{len(llm_paths)}  (expect ALL frozen — no lora in base)")
    print(f"  llm_1  frozen: {expert_frozen}/{len(expert_paths)}  (expect ALL frozen — no lora in base)")
    print(f"  proj   frozen: {proj_frozen}/{len(proj_paths)}  (expect 0 — projections are trainable)")

    if img_paths and img_frozen != len(img_paths):
        not_frozen = [p for p in img_paths if not _is_frozen(freeze_filter, p)]
        pytest.fail(
            f"{len(not_frozen)}/{len(img_paths)} img-branch params are NOT frozen. "
            f"Examples: {[' / '.join(p) for p in not_frozen[:5]]}"
        )
    if llm_paths and llm_frozen != len(llm_paths):
        not_frozen = [p for p in llm_paths if not _is_frozen(freeze_filter, p)]
        pytest.fail(
            f"{len(not_frozen)}/{len(llm_paths)} llm-branch params are NOT frozen. "
            f"Examples: {[' / '.join(p) for p in not_frozen[:5]]}"
        )
    if expert_paths and expert_frozen != len(expert_paths):
        not_frozen = [p for p in expert_paths if not _is_frozen(freeze_filter, p)]
        pytest.fail(
            f"{len(not_frozen)}/{len(expert_paths)} llm_1-branch params NOT frozen. "
            f"Examples: {[' / '.join(p) for p in not_frozen[:5]]}"
        )
    if proj_paths and proj_frozen != 0:
        is_frozen = [p for p in proj_paths if _is_frozen(freeze_filter, p)]
        pytest.fail(
            f"{len(is_frozen)} projection-head params are FROZEN but should be trainable: "
            f"{[' / '.join(p) for p in is_frozen]}"
        )

    print("\n✅ All families classified correctly.")
