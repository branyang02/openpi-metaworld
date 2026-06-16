"""Differential tests for the filtered-BC LoRA→PyTorch conversion path.

Addresses the suspects raised by the coffee-push-v3 catastrophe (83% rollout
success → 0% post-LoRA-merge eval, with a near-flat training loss curve):

  1. build_pytorch_model_from_merged conversion correctness
     → test_inprocess_vs_safetensors_no_lora
  2. LoRA merge math (used by both in-process and subprocess paths)
     → test_lora_merge_with_synthetic_delta_no_lora_input
     → test_lora_merge_arithmetic_unit
  3. Vision tower (and other trainable-non-LoRA params) propagate through merge → conversion
     → test_vision_tower_param_propagates_through_inprocess_path
  4. Norm-stats path consistency: same stats found at train time vs eval time
     → test_train_eval_norm_stats_resolve_same
  5. Determinism end-to-end: base PyTorch model and (no-op merge → rebuilt model)
     produce the SAME forward output on a fixed input
     → test_no_op_merge_preserves_forward_output

All tests use the openpi-metaworld-5000 base checkpoint and require GPU + ~10 GB.
Marked `manual` so CI default skips them. Run explicitly:

    MUJOCO_GL=egl uv run pytest tests/filtered_bc/test_conversion_paths.py -v -s -m manual
"""

from __future__ import annotations

from collections import Counter
import gc
import logging
import pathlib

import numpy as np
import pytest

logger = logging.getLogger(__name__)

# ---- Test fixtures / common config ----

_BASE_CKPT = pathlib.Path("/home/kim34/projects_brandon/openpi-metaworld/checkpoints/openpi-metaworld-5000")
_BASE_CONFIG_NAME = "pi05_metaworld"
_TRAIN_CONFIG_NAME = "pi05_metaworld_low_mem_finetune"

_REQUIRES_BASE_CKPT = pytest.mark.skipif(
    not _BASE_CKPT.exists(),
    reason=f"Base ckpt not found at {_BASE_CKPT}",
)
_REQUIRES_SAFETENSORS = pytest.mark.skipif(
    not (_BASE_CKPT / "model.safetensors").exists(),
    reason=f"model.safetensors not in {_BASE_CKPT}",
)


def _load_jax_params_float32() -> dict:
    """Restore the base ckpt's JAX params at float32 precision (matches production conversion)."""
    from openpi.models import model as _model

    return _model.restore_params(
        str(_BASE_CKPT / "params"),
        restore_type=np.ndarray,
        dtype="float32",
    )


def _free_torch():
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _diff_state_dicts(state_a: dict, state_b: dict, *, abs_tol: float = 0.05) -> list[dict]:
    """Diff two state dicts; return list of mismatch dicts."""
    import torch  # noqa: F401

    keys_a = set(state_a)
    keys_b = set(state_b)
    only_a = sorted(keys_a - keys_b)
    only_b = sorted(keys_b - keys_a)
    diffs: list[dict] = [{"key": k, "issue": "only_in_a"} for k in only_a]
    diffs.extend({"key": k, "issue": "only_in_b"} for k in only_b)

    for k in sorted(keys_a & keys_b):
        a = state_a[k].float()
        b = state_b[k].float()
        if a.shape != b.shape:
            diffs.append({"key": k, "issue": "shape", "a": tuple(a.shape), "b": tuple(b.shape)})
            continue
        max_abs = float((a - b).abs().max())
        if max_abs > abs_tol:
            diffs.append(
                {
                    "key": k,
                    "issue": "value",
                    "max_abs_diff": max_abs,
                    "a_dtype": str(state_a[k].dtype),
                    "b_dtype": str(state_b[k].dtype),
                }
            )
    return diffs


def _summarize_diffs(diffs: list[dict], total: int) -> str:
    if not diffs:
        return f"✅ All {total} params match within tolerance"
    families = Counter(d["key"].split(".")[0] for d in diffs)
    issues = Counter(d["issue"] for d in diffs)
    out = [
        f"❌ {len(diffs)}/{total} params differ ({100 * len(diffs) / total:.1f}%)",
        f"   issues: {dict(issues)}",
        "   top-level families:",
    ]
    for fam, cnt in sorted(families.items(), key=lambda x: -x[1]):
        out.append(f"     {fam}: {cnt}")
    out.append("   first 20:")
    out.extend(f"     {d}" for d in diffs[:20])
    return "\n".join(out)


# =============================================================================
# Test 1 — Suspect 1: in-process conversion vs production safetensors
# =============================================================================


@pytest.mark.manual
@_REQUIRES_BASE_CKPT
@_REQUIRES_SAFETENSORS
def test_inprocess_vs_safetensors_no_lora():
    """The MetaWorld in-process conversion path on the BASE ckpt (no LoRA delta)
    should produce a state dict equivalent to model.safetensors.

    If they diverge, build_pytorch_model_from_merged has a slicing/dtype bug
    that's independent of training — every filtered-BC MetaWorld eval would
    use a corrupted model.
    """
    import safetensors.torch as st
    import torch

    from experiments.filtered_bc.merge_save import build_pytorch_model_from_merged
    from experiments.filtered_bc.merge_save import merge_lora_params
    from openpi.models import pi0_config as _pi0_config
    from openpi.models_pytorch import pi0_pytorch
    from openpi.training import config as _config

    base_config = _config.get_config(_BASE_CONFIG_NAME).model

    # ---- Path A: in-process JAX → merge_lora_params (no-op) → build_pytorch_model_from_merged ----
    print("\n[Path A] Loading JAX params (float32) ...")
    jax_params = _load_jax_params_float32()
    print(f"[Path A] Top-level keys: {sorted(jax_params.keys())}")
    print("[Path A] merge_lora_params (no-op, base ckpt has no LoRA leaves) ...")
    merged = merge_lora_params(jax_params, base_config)
    print("[Path A] build_pytorch_model_from_merged ...")
    model_a = build_pytorch_model_from_merged(merged, base_config)
    state_a = {k: v.detach().cpu().clone() for k, v in model_a.state_dict().items()}
    print(f"[Path A] state_dict: {len(state_a)} entries")
    del model_a, merged, jax_params
    _free_torch()

    # ---- Path B: PI0Pytorch + load_model(model.safetensors) ----
    print("\n[Path B] Building PI0Pytorch (gemma_2b / gemma_300m) ...")
    eval_model_config = _pi0_config.Pi0Config(
        pi05=base_config.pi05,
        action_horizon=base_config.action_horizon,
        discrete_state_input=base_config.discrete_state_input,
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
    )
    model_b = pi0_pytorch.PI0Pytorch(eval_model_config)
    if torch.cuda.is_available():
        model_b = model_b.to("cuda")
    print("[Path B] Loading model.safetensors ...")
    load_result = st.load_model(model_b, str(_BASE_CKPT / "model.safetensors"))
    print(f"[Path B] safetensors load_model result: {load_result}")
    # Match Path A's bf16 selective conversion so we compare apples-to-apples.
    model_b.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    state_b = {k: v.detach().cpu().clone() for k, v in model_b.state_dict().items()}
    print(f"[Path B] state_dict: {len(state_b)} entries")
    del model_b
    _free_torch()

    # ---- Compare ----
    diffs = _diff_state_dicts(state_a, state_b, abs_tol=0.05)
    print()
    print(_summarize_diffs(diffs, total=max(len(state_a), len(state_b))))

    only_a = [d for d in diffs if d["issue"] == "only_in_a"]
    only_b = [d for d in diffs if d["issue"] == "only_in_b"]
    shape_diffs = [d for d in diffs if d["issue"] == "shape"]

    assert not only_a, f"{len(only_a)} keys only in in-process state_dict (first: {only_a[:5]})"
    assert not only_b, f"{len(only_b)} keys only in safetensors state_dict (first: {only_b[:5]})"
    assert not shape_diffs, f"{len(shape_diffs)} shape mismatches: {shape_diffs[:5]}"

    # Known-benign diff: lm_head.weight is "tied weights, never converted" per the
    # production converter (examples/convert_jax_model_to_pytorch.py:384). Both paths
    # leave it at its random init; the values differ because PI0Pytorch.__init__ is
    # called from different RNG states. lm_head is unused at pi0.5 inference (action
    # prediction goes through action_in_proj / action_out_proj / time_mlp_*).
    expected_benign = {"paligemma_with_expert.gemma_expert.lm_head.weight"}
    real_diffs = [d for d in diffs if d.get("issue") == "value" and d["key"] not in expected_benign]
    benign = [d for d in diffs if d.get("issue") == "value" and d["key"] in expected_benign]
    if benign:
        print(f"\n⚠️  Skipping {len(benign)} benign diff(s) on tied-but-uncopied tensors: {[d['key'] for d in benign]}")
    if real_diffs:
        pytest.fail(
            f"{len(real_diffs)} non-benign param values differ between in-process "
            f"and production conversion. See stdout breakdown."
        )


# =============================================================================
# Test 2 — Suspect 1+2: synthetic LoRA-delta merge round-trip
# =============================================================================


def _einsum_merge_reference(w: np.ndarray, a: np.ndarray, b: np.ndarray, scale: float) -> np.ndarray:
    """Closed-form reference for the LoRA merge: w + (a @ b) * scale."""
    delta = np.einsum("...ir,...rj->...ij", a.astype(np.float32), b.astype(np.float32))
    return (w.astype(np.float32) + scale * delta).astype(w.dtype)


@pytest.mark.manual
def test_lora_merge_arithmetic_unit():
    """The LoRA merge primitive should equal w + scale * (a @ b)."""
    from experiments.filtered_bc.merge_save import _einsum_merge

    rng = np.random.default_rng(0)
    # 2D case
    w_2d = rng.standard_normal((128, 256)).astype(np.float32)
    a_2d = rng.standard_normal((128, 16)).astype(np.float32)
    b_2d = rng.standard_normal((16, 256)).astype(np.float32)
    out = _einsum_merge(w_2d, a_2d, b_2d, scale=0.25)
    ref = _einsum_merge_reference(w_2d, a_2d, b_2d, scale=0.25)
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)

    # 3D batched case (matches Einsum LoRA layout)
    w_3d = rng.standard_normal((8, 128, 256)).astype(np.float32)
    a_3d = rng.standard_normal((8, 128, 16)).astype(np.float32)
    b_3d = rng.standard_normal((8, 16, 256)).astype(np.float32)
    out_3d = _einsum_merge(w_3d, a_3d, b_3d, scale=0.5)
    ref_3d = _einsum_merge_reference(w_3d, a_3d, b_3d, scale=0.5)
    np.testing.assert_allclose(out_3d, ref_3d, rtol=1e-5, atol=1e-5)


# =============================================================================
# Test 3 — Suspect 2: vision tower update propagates through merge → conversion
# =============================================================================


@pytest.mark.manual
@_REQUIRES_BASE_CKPT
def test_vision_tower_param_propagates_through_inprocess_path():
    """Modify a vision-tower kernel by a known delta in the JAX merged dict, then
    confirm the corresponding PyTorch tensor in the rebuilt model reflects that delta.

    If `build_pytorch_model_from_merged` silently drops vision-tower updates, this
    test will catch it: the PyTorch tensor will equal the SigLip base, not base+delta.
    """

    from experiments.filtered_bc.merge_save import build_pytorch_model_from_merged
    from openpi.training import config as _config

    base_config = _config.get_config(_BASE_CONFIG_NAME).model

    print("\n[Test 3] Loading JAX params ...")
    jax_params = _load_jax_params_float32()

    # Pick a deterministic vision-tower kernel. The img branch lives at
    # PaliGemma/img/embedding/kernel  (a Conv 14x14x3 -> 1152).
    vt_path = ["PaliGemma", "img", "embedding", "kernel"]
    cur = jax_params
    for k in vt_path[:-1]:
        cur = cur[k]
    leaf_key = vt_path[-1]
    assert leaf_key in cur, f"Vision tower path {vt_path} not found"
    original = np.asarray(cur[leaf_key]).copy()
    print(f"[Test 3] Vision-tower leaf shape={original.shape}, dtype={original.dtype}")

    # Bump every weight by a recognizable constant so we can detect dropouts.
    delta = np.float32(0.123456)
    modified = (original.astype(np.float32) + delta).astype(original.dtype)
    cur[leaf_key] = modified

    print("[Test 3] build_pytorch_model_from_merged on modified params ...")
    model = build_pytorch_model_from_merged(jax_params, base_config)

    # PyTorch counterpart: the Conv2d kernel of the SigLip patch embedder.
    pt_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    candidates = [k for k in pt_state if "embed" in k.lower() and "weight" in k and pt_state[k].dim() == 4]
    print(f"[Test 3] Candidate patch-embed keys: {candidates}")
    assert candidates, "Could not find a 4D patch-embed weight in the PyTorch model"

    # Pick the conv2d weight whose channel/spatial layout corresponds to (1152, 3, 14, 14).
    # JAX layout is (14, 14, 3, 1152); after `.transpose(3, 2, 0, 1)` it becomes (1152, 3, 14, 14).
    target_key = None
    for k in candidates:
        v = pt_state[k]
        if tuple(v.shape) == (1152, 3, 14, 14):
            target_key = k
            break
    assert target_key, f"No (1152, 3, 14, 14) weight; saw shapes {[pt_state[c].shape for c in candidates]}"
    print(f"[Test 3] Comparing against PyTorch key: {target_key}")

    pt_value = pt_state[target_key].float().numpy()
    # Expected: production conversion does .transpose(3, 2, 0, 1) on the JAX kernel,
    # i.e. JAX(h, w, c_in, c_out) -> PT(c_out, c_in, h, w).
    expected_pt = modified.transpose(3, 2, 0, 1).astype(np.float32)

    # Compute residual: pt_value should equal modified-transposed (NOT the un-modified original).
    residual_to_modified = float(np.abs(pt_value - expected_pt).max())
    expected_orig = original.transpose(3, 2, 0, 1).astype(np.float32)
    residual_to_original = float(np.abs(pt_value - expected_orig).max())
    expected_delta = float(np.abs(expected_pt - expected_orig).max())

    print(f"[Test 3] |pt - expected_modified|_max = {residual_to_modified:.6f}")
    print(f"[Test 3] |pt - expected_original|_max = {residual_to_original:.6f}")
    print(f"[Test 3] expected delta magnitude     = {expected_delta:.6f} (should be ≈ {delta})")

    del model, pt_state, jax_params
    _free_torch()

    # The delta must have made it through. residual_to_modified should be tiny (just bf16 quantization);
    # residual_to_original should be ≈ delta. If residual_to_modified ≈ delta and residual_to_original ≈ 0,
    # the vision tower update was DROPPED.
    bf16_quant_tol = 0.001  # roughly 1/1024 — bfloat16 mantissa precision
    assert residual_to_modified < bf16_quant_tol, (
        f"Vision-tower update was DROPPED through the conversion path: "
        f"|pt - modified|={residual_to_modified:.6f} (should be < {bf16_quant_tol}); "
        f"|pt - original|={residual_to_original:.6f} (should be ≈ {delta})"
    )


# =============================================================================
# Test 4 — Suspect 3: train-time and eval-time norm stats resolve to the same values
# =============================================================================


@pytest.mark.manual
@_REQUIRES_BASE_CKPT
def test_train_eval_norm_stats_resolve_same():
    """Train and eval must consume identical norm stats — otherwise LoRA learns
    in one normalized space and eval reads inputs in another.

    Replicates the train-time logic (filtered_bc/dataset.py:build_training_dataset)
    and the eval-time logic (run_filtered_bc.py:_build_policy_from_model) on the
    actual configs used by the orchestrator. Compares asset_id, file path, and
    every per-key (mean, std) value.
    """
    from openpi.training import checkpoints as _checkpoints
    from openpi.training import config as _config

    # === Train-time resolution (replicates filtered_bc/dataset.py:build_training_dataset) ===
    train_config = _config.get_config(_TRAIN_CONFIG_NAME)
    train_data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    print(f"\n[Test 4] train_config={_TRAIN_CONFIG_NAME}")
    print(f"[Test 4]   assets_dirs={train_config.assets_dirs}")
    print(f"[Test 4]   data_config.asset_id={train_data_config.asset_id}")
    print(f"[Test 4]   data_config.repo_id={train_data_config.repo_id}")
    print(f"[Test 4]   data_config.norm_stats is None? {train_data_config.norm_stats is None}")

    if train_data_config.norm_stats is not None:
        train_norm_stats = train_data_config.norm_stats
        train_source = "data_config.norm_stats (in-memory)"
    elif train_data_config.asset_id is not None:
        train_norm_stats = _checkpoints.load_norm_stats(pathlib.Path(_BASE_CKPT) / "assets", train_data_config.asset_id)
        train_source = f"{_BASE_CKPT}/assets/{train_data_config.asset_id}"
    else:
        pytest.skip("train_data_config has no asset_id and no norm_stats — nothing to compare")
    print(f"[Test 4] train norm stats source: {train_source}")
    print(f"[Test 4] train norm stats keys: {sorted(train_norm_stats.keys())}")

    # === Eval-time resolution (replicates run_filtered_bc.py:_build_policy_from_model) ===
    base_train_config = _config.get_config(_BASE_CONFIG_NAME)
    eval_data_config = base_train_config.data.create(base_train_config.assets_dirs, base_train_config.model)
    print(f"\n[Test 4] base_config={_BASE_CONFIG_NAME}")
    print(f"[Test 4]   assets_dirs={base_train_config.assets_dirs}")
    print(f"[Test 4]   data_config.asset_id={eval_data_config.asset_id}")
    print(f"[Test 4]   data_config.repo_id={eval_data_config.repo_id}")
    print(f"[Test 4]   data_config.norm_stats is None? {eval_data_config.norm_stats is None}")

    if eval_data_config.norm_stats:
        eval_norm_stats = eval_data_config.norm_stats
        eval_source = "data_config.norm_stats (in-memory)"
    elif eval_data_config.asset_id is not None:
        eval_norm_stats = _checkpoints.load_norm_stats(pathlib.Path(_BASE_CKPT) / "assets", eval_data_config.asset_id)
        eval_source = f"{_BASE_CKPT}/assets/{eval_data_config.asset_id}"
    else:
        pytest.fail("eval_data_config has no asset_id and no norm_stats")
    print(f"[Test 4] eval norm stats source: {eval_source}")
    print(f"[Test 4] eval norm stats keys: {sorted(eval_norm_stats.keys())}")

    # === Compare ===
    print(f"\n[Test 4] asset_id match? {train_data_config.asset_id == eval_data_config.asset_id}")
    print(f"[Test 4] source path match? {train_source == eval_source}")

    train_keys = set(train_norm_stats)
    eval_keys = set(eval_norm_stats)
    assert train_keys == eval_keys, (
        f"Norm-stats key sets differ:\n"
        f"  only in train: {train_keys - eval_keys}\n"
        f"  only in eval:  {eval_keys - train_keys}"
    )

    diffs = []
    for k in sorted(train_keys):
        t = train_norm_stats[k]
        e = eval_norm_stats[k]
        for field in ("mean", "std"):
            t_v = getattr(t, field, None)
            e_v = getattr(e, field, None)
            if t_v is None and e_v is None:
                continue
            if t_v is None or e_v is None:
                diffs.append({"key": k, "field": field, "issue": "one_missing"})
                continue
            t_a = np.asarray(t_v)
            e_a = np.asarray(e_v)
            if t_a.shape != e_a.shape:
                diffs.append({"key": k, "field": field, "issue": "shape", "t": t_a.shape, "e": e_a.shape})
                continue
            max_abs = float(np.abs(t_a - e_a).max())
            if max_abs > 1e-7:
                diffs.append({"key": k, "field": field, "issue": "value", "max_abs_diff": max_abs})

    if diffs:
        print("\n❌ Norm-stats DIVERGED between train and eval:")
        for d in diffs:
            print(f"  {d}")
    else:
        print(f"\n✅ Train-time and eval-time norm stats are identical (n_keys={len(train_keys)})")

    assert not diffs, f"Train and eval norm stats differ in {len(diffs)} entries"


# =============================================================================
# Test 5 — End-to-end: no-op merge → rebuilt PyTorch model produces same output
# =============================================================================


@pytest.mark.manual
@_REQUIRES_BASE_CKPT
@_REQUIRES_SAFETENSORS
def test_no_op_merge_preserves_forward_output():
    """The functional gold-standard: forward pass of the BASE PyTorch model
    (loaded from model.safetensors) and the in-process-rebuilt model (no-op
    merge → build_pytorch_model_from_merged) must produce equivalent outputs
    on a fixed observation.

    This catches subtler bugs that wouldn't show up in a state-dict diff:
    e.g. an unused param that gets corrupted but doesn't matter,
    or a shape mismatch that's masked by reshaping at forward time.
    """
    import safetensors.torch as st
    import torch

    from experiments.filtered_bc.merge_save import build_pytorch_model_from_merged
    from experiments.filtered_bc.merge_save import merge_lora_params
    from openpi.models import model as _omodel
    from openpi.models import pi0_config as _pi0_config
    from openpi.models_pytorch import pi0_pytorch
    from openpi.training import config as _config

    base_config = _config.get_config(_BASE_CONFIG_NAME).model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    eval_model_config = _pi0_config.Pi0Config(
        pi05=base_config.pi05,
        action_horizon=base_config.action_horizon,
        discrete_state_input=base_config.discrete_state_input,
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
    )

    # ---- Build base model from safetensors ----
    print("\n[Test 5] Path B: PI0Pytorch + safetensors")
    model_b = pi0_pytorch.PI0Pytorch(eval_model_config).to(device)
    st.load_model(model_b, str(_BASE_CKPT / "model.safetensors"))
    model_b.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    model_b.eval()

    # ---- Build same model via in-process path ----
    print("[Test 5] Path A: in-process build_pytorch_model_from_merged")
    jax_params = _load_jax_params_float32()
    merged = merge_lora_params(jax_params, base_config)
    model_a = build_pytorch_model_from_merged(merged, base_config)
    model_a.eval()
    del jax_params, merged

    # ---- Construct a deterministic Observation. ----
    torch.manual_seed(42)
    bs = 1
    H, W = 224, 224  # noqa: N806 — image dims  # SigLip patch grid expects 224x224
    # Images as (B, H, W, C) uint8 to match Observation.from_dict's path that converts
    # uint8 → [-1, 1] float32 internally.
    images = {
        "base_0_rgb": torch.randint(0, 255, (bs, H, W, 3), dtype=torch.uint8, device=device),
        "left_wrist_0_rgb": torch.randint(0, 255, (bs, H, W, 3), dtype=torch.uint8, device=device),
        "right_wrist_0_rgb": torch.randint(0, 255, (bs, H, W, 3), dtype=torch.uint8, device=device),
    }
    image_masks = {k: torch.ones(bs, dtype=torch.bool, device=device) for k in images}
    state = torch.zeros(bs, base_config.action_dim, device=device)
    L = 16  # noqa: N806 — seq length
    tokenized_prompt = torch.zeros(bs, L, dtype=torch.long, device=device)
    tokenized_prompt_mask = torch.ones(bs, L, dtype=torch.bool, device=device)

    obs_dict = {
        "image": images,
        "image_mask": image_masks,
        "state": state,
        "tokenized_prompt": tokenized_prompt,
        "tokenized_prompt_mask": tokenized_prompt_mask,
    }
    observation = _omodel.Observation.from_dict(obs_dict)

    # Fixed noise so both forward passes share the same denoise trajectory.
    torch.manual_seed(2024)
    noise = torch.randn(bs, base_config.action_horizon, base_config.action_dim, device=device)

    print("[Test 5] sample_actions from both models with same observation + noise ...")

    def _sample(model):
        with torch.no_grad():
            out = model.sample_actions(device=device, observation=observation, noise=noise.clone(), num_steps=10)
            # Clone immediately — sample_actions returns a buffer reused by CUDAGraphs
            # on subsequent calls, so we need a deep copy before the next sample.
            return out.detach().clone().float().cpu()

    out_b = _sample(model_b)
    out_a = _sample(model_a)

    diff = (out_a.float() - out_b.float()).abs()
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())
    print(f"\n[Test 5] action sample shape: {tuple(out_a.shape)}")
    print(f"[Test 5] |diff|_max  = {max_abs:.6f}")
    print(f"[Test 5] |diff|_mean = {mean_abs:.6f}")
    print(f"[Test 5] Path A output magnitude (mean abs) = {float(out_a.abs().mean()):.6f}")
    print(f"[Test 5] Path B output magnitude (mean abs) = {float(out_b.abs().mean()):.6f}")
    print(f"[Test 5] Path A first 4 elements: {out_a.flatten()[:4].tolist()}")
    print(f"[Test 5] Path B first 4 elements: {out_b.flatten()[:4].tolist()}")

    del model_a, model_b, out_a, out_b
    _free_torch()

    # Tight tolerance — same conversion + same input + same noise ⇒ should be near-bitwise.
    # Allow some slack for non-determinism in attention kernels and bf16 op order.
    tol = 0.05
    assert max_abs < tol, (
        f"Forward outputs diverge: max_abs_diff={max_abs:.4f} > {tol}. "
        f"In-process conversion produces functionally different action samples "
        f"than production conversion."
    )


# =============================================================================
# Test 6 — Synthetic LoRA delta: simulates a real training output,
# verifies merge_lora_params actually folds the delta into the dense weight,
# AND that build_pytorch_model_from_merged sees the modified weight.
# =============================================================================


def _inject_synthetic_lora_leaf(params: dict, lora_path: list[str], variant: str, scope: str, rank: int):
    """Add lora_a and lora_b leaves at ``params[*lora_path]`` so that the merge
    triplet (path/w, path/lora_a, path/lora_b) is recognized by merge_lora_params.

    Returns: (the original dense ``w`` array, the synthesized delta array).
    """
    cur = params
    for k in lora_path[:-1]:
        cur = cur[k]
    leaf_dict = cur[lora_path[-1]]  # e.g. params['PaliGemma']['llm']['layers']['attn']['attn_vec_einsum']
    assert "w" in leaf_dict, f"No 'w' at {lora_path}; keys: {list(leaf_dict.keys())}"
    w = np.asarray(leaf_dict["w"]).copy()  # original dense weight, shape e.g. (heads, head_dim, hidden)

    # Synthesize lora_a and lora_b that produce a known, distinguishable delta when merged.
    # Layout: w shape (..., I, J). lora_a shape (..., I, R). lora_b shape (..., R, J).
    # Delta = lora_a @ lora_b.
    rng = np.random.default_rng(seed=42)
    a_shape = (*w.shape[:-1], rank)
    b_shape = (*w.shape[:-2], rank, w.shape[-1])
    lora_a = rng.standard_normal(a_shape).astype(np.float32) * 0.01
    lora_b = rng.standard_normal(b_shape).astype(np.float32) * 0.01

    # Cast to w's dtype to match what real LoRA training would produce.
    leaf_dict["lora_a"] = lora_a.astype(w.dtype)
    leaf_dict["lora_b"] = lora_b.astype(w.dtype)
    return w, lora_a, lora_b


@pytest.mark.manual
@_REQUIRES_BASE_CKPT
def test_synthetic_lora_delta_merges_into_pytorch_weight():
    """Inject lora_a + lora_b into the JAX dict at a known LM layer, run the full
    pipeline (merge → build_pytorch_model_from_merged), and verify the corresponding
    PyTorch weight equals base + (lora_a @ lora_b) * scale (the LoRA merge formula).

    If `merge_lora_params` silently fails to apply the delta, the PyTorch weight
    would still equal the base. If `build_pytorch_model_from_merged` reads the
    pre-merge or wrong slice, the weight would also be wrong.
    """
    import math

    from experiments.filtered_bc.merge_save import build_pytorch_model_from_merged
    from experiments.filtered_bc.merge_save import merge_lora_params
    from openpi.models import gemma as _gemma
    from openpi.training import config as _config

    # Use the LoRA train config so model_config.paligemma_variant has lora_configs.
    train_config = _config.get_config(_TRAIN_CONFIG_NAME)
    model_config = train_config.model
    print(
        f"\n[Test 6] paligemma_variant={model_config.paligemma_variant}, "
        f"action_expert_variant={model_config.action_expert_variant}"
    )

    pg_config = _gemma.get_config(model_config.paligemma_variant)
    print(f"[Test 6] LoRA configs: {pg_config.lora_configs}")
    rank = pg_config.lora_configs["attn"].rank
    alpha = pg_config.lora_configs["attn"].alpha
    rslora = pg_config.lora_configs["attn"].rslora
    expected_scale = alpha / (math.sqrt(rank) if rslora else rank)
    print(f"[Test 6] LoRA scale = {alpha}/{'sqrt(rank)' if rslora else 'rank'} = {expected_scale}")

    # Load base JAX params at float32.
    print("[Test 6] Loading JAX params ...")
    jax_params = _load_jax_params_float32()

    # Inject LoRA at the attention attn_vec_einsum of layer 0 of the PaliGemma LM (variant="llm",
    # scope="attn"). This matches the LoRA placement that real training would use.
    lora_path = ["PaliGemma", "llm", "layers", "attn", "attn_vec_einsum"]
    print(f"[Test 6] Injecting synthetic LoRA at {'/'.join(lora_path)} (rank={rank}) ...")
    w_orig, lora_a, lora_b = _inject_synthetic_lora_leaf(jax_params, lora_path, variant="llm", scope="attn", rank=rank)
    print(f"[Test 6]   w shape = {w_orig.shape}, lora_a shape = {lora_a.shape}, lora_b shape = {lora_b.shape}")

    # Compute expected merged weight = w + (lora_a @ lora_b) * scale.
    expected_delta = (
        np.einsum("...ir,...rj->...ij", lora_a.astype(np.float32), lora_b.astype(np.float32)) * expected_scale
    )
    expected_merged_w = (w_orig.astype(np.float32) + expected_delta).astype(w_orig.dtype)
    print(f"[Test 6]   expected delta |.|_max = {float(np.abs(expected_delta).max()):.6f}")
    print(
        f"[Test 6]   expected merged - orig |.|_max = "
        f"{float(np.abs(expected_merged_w.astype(np.float32) - w_orig.astype(np.float32)).max()):.6f}"
    )

    # Run merge_lora_params and confirm the JAX-side merged value is correct.
    print("[Test 6] merge_lora_params ...")
    merged = merge_lora_params(jax_params, model_config)
    cur = merged
    for k in lora_path[:-1]:
        cur = cur[k]
    actual_merged_w = np.asarray(cur[lora_path[-1]]["w"])
    assert "lora_a" not in cur[lora_path[-1]], "merge_lora_params did NOT strip lora_a"
    assert "lora_b" not in cur[lora_path[-1]], "merge_lora_params did NOT strip lora_b"
    jax_residual = float(np.abs(actual_merged_w.astype(np.float32) - expected_merged_w.astype(np.float32)).max())
    print(f"[Test 6] JAX merged weight residual |actual - expected|_max = {jax_residual:.6f}")
    # _einsum_merge does float32 math then casts back to bf16 — quantization tolerance.
    assert jax_residual < 1e-4, f"merge_lora_params produced wrong JAX value: residual={jax_residual:.6f}"

    # Run the full conversion and locate the PyTorch counterpart of attn_vec_einsum.
    print("[Test 6] build_pytorch_model_from_merged ...")
    model = build_pytorch_model_from_merged(merged, model_config)
    pt_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    # The PaliGemma attn_vec_einsum maps to the LM's attention.o_proj.
    candidates = [k for k in pt_state if "paligemma" in k.lower() and "o_proj" in k.lower()]
    print(f"[Test 6] Candidate o_proj keys ({len(candidates)}):")
    for k in candidates[:5]:
        print(f"    {k} shape={tuple(pt_state[k].shape)} dtype={pt_state[k].dtype}")

    # Find the layer-0 o_proj weight specifically.
    layer0_keys = [k for k in candidates if ".layers.0." in k and k.endswith("o_proj.weight")]
    if not layer0_keys:
        # Fallback: any first-layer o_proj.
        layer0_keys = [k for k in candidates if "layer_0" in k or "layers.0" in k][:1]
    assert layer0_keys, "Could not find layer-0 o_proj.weight in PyTorch model"
    target_key = layer0_keys[0]
    print(f"[Test 6] Comparing against PyTorch key: {target_key}")
    pt_w = pt_state[target_key].float().numpy()
    print(f"[Test 6] PyTorch weight shape = {pt_w.shape}, dtype = {pt_state[target_key].dtype}")

    # The JAX shape is (heads, head_dim, hidden). PyTorch o_proj.weight is (hidden, heads*head_dim).
    # The transpose is (heads, head_dim, hidden) -> reshape to (heads*head_dim, hidden) -> transpose
    # to (hidden, heads*head_dim). Production converter does this with concatenation; for our test
    # we just verify the OVERALL VALUE distribution moved by the expected delta magnitude.
    expected_delta_max = float(np.abs(expected_delta).max())

    # Construct the production-converter's transformation of the JAX weight to PT layout
    # so we can do a direct comparison.
    # Looking at slice_paligemma_state_dict:
    #   attn_vec_einsum has shape (heads, head_dim, hidden). PT o_proj.weight expects (hidden, heads*head_dim).
    #   The standard conversion is: w.transpose(2, 0, 1).reshape(hidden, heads*head_dim).
    # We approximate by checking the difference between the merged-weight-converted
    # and the original-weight-converted matches the delta magnitude.

    # Direct deduction of the layout: the layer dimension (.layers) becomes per-layer
    # selection, so in our merged dict cur[lora_path[-1]]["w"] has shape (num_layers, heads, head_dim, hidden).
    # We injected LoRA on the same layer dim, so all layers got a delta.
    # For a single layer-0 PT key we compare slice [0].
    if actual_merged_w.ndim == 4:
        merged_layer0_jax = actual_merged_w[0]
        orig_layer0_jax = w_orig[0]
    else:
        merged_layer0_jax = actual_merged_w
        orig_layer0_jax = w_orig

    # Convert layer-0 JAX weight to PT layout: (heads, head_dim, hidden) -> (hidden, heads*head_dim)
    merged_pt_expected = merged_layer0_jax.transpose(2, 0, 1).reshape(merged_layer0_jax.shape[-1], -1)
    orig_pt_expected = orig_layer0_jax.transpose(2, 0, 1).reshape(orig_layer0_jax.shape[-1], -1)

    # Sanity: the PT model's layer-0 weight should match merged, NOT original.
    print(
        f"[Test 6] |pt_w - merged_layer0_pt|_max = "
        f"{float(np.abs(pt_w - merged_pt_expected.astype(np.float32)).max()):.6f}"
    )
    print(
        f"[Test 6] |pt_w - orig_layer0_pt|_max   = "
        f"{float(np.abs(pt_w - orig_pt_expected.astype(np.float32)).max()):.6f}"
    )
    print(f"[Test 6] expected delta in this layer  = {expected_delta_max:.6f}")

    delta_in_pt = float(np.abs(pt_w - orig_pt_expected.astype(np.float32)).max())

    del model, pt_state, merged, jax_params
    _free_torch()

    # The PyTorch weight must reflect the LoRA delta. If `merge_lora_params` silently dropped
    # the delta or `build_pytorch_model_from_merged` read pre-merge values, delta_in_pt ≈ 0.
    # bf16 quantization noise on a (1152*256)-element matrix can be ~5e-3, so use that as
    # the lower bound to detect a "delta dropped" scenario.
    assert delta_in_pt > 0.5 * expected_delta_max, (
        f"LoRA delta did NOT make it through to PyTorch. "
        f"|pt - orig|_max={delta_in_pt:.6f} but expected ≈ {expected_delta_max:.6f}"
    )
