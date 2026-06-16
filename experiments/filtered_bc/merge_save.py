"""LoRA merge + in-process PyTorch model build.

After LoRA training, we need to evaluate the updated policy. Because the SLURM cgroup
caps CPU RAM at 16 GB and any on-disk checkpoint save (orbax or safetensors) requires
~2x the model size in memory for staging, we instead:

1. Merge LoRA adapters into base weights (JAX-layout numpy dict).
2. Free the JAX train state.
3. Remap JAX → PyTorch layout via the same slice functions the upstream converter uses.
4. Instantiate :class:`PI0Pytorch` and load the remapped state_dict.
5. Evaluate in-process, then drop the model.

No persistent checkpoint is written (that's Option B from the discussion); a future run
with more RAM should switch back to :func:`save_merged_checkpoint` once written.
"""

from __future__ import annotations

import logging
import math
import pathlib
from typing import Any

import flax.traverse_util as traverse_util
import numpy as np

from openpi.models import gemma as _gemma
from openpi.models import pi0_config

logger = logging.getLogger(__name__)


# ---- LoRA merging ------------------------------------------------------------------


def _lora_scaling(lora_config) -> float:
    """Scaling factor used by both :class:`lora.Einsum` and :class:`lora.FeedForward`."""
    if lora_config.rslora:
        return lora_config.alpha / math.sqrt(lora_config.rank)
    return lora_config.alpha / lora_config.rank


def _einsum_merge(w, lora_a, lora_b, scale: float):
    """Merge an Einsum LoRA pair: w_merged = w + (lora_a @ lora_b) * scale.

    Streams the outer axis when w has a leading batch dim so the per-chunk float32 buffer
    stays under ~512 MB. Numpy's einsum doesn't support bfloat16, so we compute in float32
    and cast the result back to w's dtype.
    """
    w_np = np.asarray(w)
    a_np = np.asarray(lora_a)
    b_np = np.asarray(lora_b)
    out_dtype = w_np.dtype

    out = np.empty_like(w_np)
    if w_np.ndim <= 2:
        delta_f32 = np.einsum("...ir,...rj->...ij", a_np, b_np).astype(np.float32, copy=False)
        if scale != 1.0:
            delta_f32 *= np.float32(scale)
        np.add(w_np.astype(np.float32), delta_f32, out=delta_f32)
        out[...] = delta_f32.astype(out_dtype)
        del delta_f32
    else:
        bytes_per_chunk = 512 * 1024 * 1024
        per_unit = int(np.prod(w_np.shape[1:])) * 4
        chunk = max(1, bytes_per_chunk // max(per_unit, 1))
        for i in range(0, w_np.shape[0], chunk):
            s = slice(i, min(i + chunk, w_np.shape[0]))
            delta_f32 = np.einsum("...ir,...rj->...ij", a_np[s], b_np[s]).astype(np.float32, copy=False)
            if scale != 1.0:
                delta_f32 *= np.float32(scale)
            np.add(w_np[s].astype(np.float32), delta_f32, out=delta_f32)
            out[s] = delta_f32.astype(out_dtype)
            del delta_f32
    return out


def _resolve_lora_configs(model_config) -> dict[str, dict[str, Any]]:
    """Build the {variant_name: {scope: lora_cfg}} map used by the merge loop.

    pi0/pi0.5 have a separate action expert under ``llm_1``; pi0-FAST has no
    action expert (autoregressive decode through PaliGemma's LM), so it only
    contributes the ``llm`` variant.
    """
    out: dict[str, dict[str, Any]] = {}
    paligemma = _gemma.get_config(model_config.paligemma_variant)
    if paligemma.lora_configs:
        out["llm"] = dict(paligemma.lora_configs)
    expert_variant = getattr(model_config, "action_expert_variant", None)
    if expert_variant is not None:
        expert = _gemma.get_config(expert_variant)
        if expert.lora_configs:
            out["llm_1"] = dict(expert.lora_configs)
    return out


_EINSUM_BASE = "w"
_EINSUM_A = "lora_a"
_EINSUM_B = "lora_b"
_FFN_BASES = ("gating_einsum", "linear")
_FFN_SUFFIX_A = "_lora_a"
_FFN_SUFFIX_B = "_lora_b"


def _which_variant(path_tuple: tuple[str, ...]) -> str | None:
    if "llm_1" in path_tuple:
        return "llm_1"
    if "llm" in path_tuple:
        return "llm"
    return None


def _scope_for_einsum(path_tuple: tuple[str, ...]) -> str:
    joined = "/".join(path_tuple)
    if "/attn/" in joined or joined.endswith("/attn"):
        return "attn"
    if "/mlp/" in joined or "/ffn/" in joined:
        return "ffn"
    return "attn"


def merge_lora_params(params: dict, model_config) -> dict:
    """Return a new param tree with LoRA folded into base weights and LoRA leaves dropped.

    Accepts either :class:`pi0_config.Pi0Config` (pi0/pi0.5) or
    :class:`pi0_fast.Pi0FASTConfig` — the merge logic only reads ``paligemma_variant``
    and (optionally) ``action_expert_variant`` via ``_resolve_lora_configs``.

    Mutates the flattened view in-place, dropping ``lora_a``/``lora_b`` and the
    superseded ``w`` immediately after each merge so Python can reclaim memory.
    """
    lora_configs = _resolve_lora_configs(model_config)
    if not lora_configs:
        logger.info("Model config has no LoRA variants; merge is a no-op.")
        return params

    flat = dict(traverse_util.flatten_dict(params))

    triplets: list[tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], float]] = []
    for path in list(flat.keys()):
        last = path[-1]
        parent = path[:-1]

        if last == _EINSUM_A:
            path_b = (*parent, _EINSUM_B)
            path_w = (*parent, _EINSUM_BASE)
            if path_b not in flat or path_w not in flat:
                continue
            variant = _which_variant(parent)
            scope = _scope_for_einsum(parent)
            if variant is None or variant not in lora_configs or scope not in lora_configs[variant]:
                logger.warning(f"Unexpected LoRA leaves at {parent!r}; dropping only.")
                flat.pop(path, None)
                flat.pop(path_b, None)
                continue
            cfg = lora_configs[variant][scope]
            triplets.append((path_w, path, path_b, _lora_scaling(cfg)))
            continue

        for base in _FFN_BASES:
            if last == base + _FFN_SUFFIX_A:
                path_b = (*parent, base + _FFN_SUFFIX_B)
                path_w = (*parent, base)
                if path_b not in flat or path_w not in flat:
                    break
                variant = _which_variant(parent)
                scope = _scope_for_einsum(parent)
                if variant is None or variant not in lora_configs or scope not in lora_configs[variant]:
                    logger.warning(f"Unexpected FFN LoRA leaves at {parent!r}; dropping only.")
                    flat.pop(path, None)
                    flat.pop(path_b, None)
                    break
                cfg = lora_configs[variant][scope]
                triplets.append((path_w, path, path_b, _lora_scaling(cfg)))
                break

    n_merged = 0
    for path_w, path_a, path_b, scale in triplets:
        w = flat.pop(path_w)
        a = flat.pop(path_a)
        b = flat.pop(path_b)
        flat[path_w] = _einsum_merge(w, a, b, scale)
        del w, a, b
        n_merged += 1
        if n_merged % 20 == 0:
            logger.info(f"merge_lora_params: merged {n_merged}/{len(triplets)} triplets")
            import gc

            gc.collect()

    logger.info(f"merge_lora_params: merged {n_merged} triplets")
    return traverse_util.unflatten_dict(flat)


# ---- In-process PyTorch model build ------------------------------------------------


def build_pytorch_model_from_merged(merged: dict, model_config: pi0_config.Pi0Config):
    """Remap JAX-layout merged params to PyTorch layout and load into PI0Pytorch.

    Reuses slice_paligemma_state_dict / slice_gemma_state_dict from
    examples/convert_jax_model_to_pytorch.py. Monkey-patches torch.from_numpy to accept
    ml_dtypes.bfloat16 via a uint16-view reinterpret.

    Returns an eval-ready PI0Pytorch model on GPU (if available).
    """
    import gc
    import importlib.util

    from flax.nnx import traversals
    import jax
    import ml_dtypes as _ml_dtypes
    import torch

    from openpi.models_pytorch import pi0_pytorch

    # Ensure every leaf is numpy (JAX arrays pulled to host).
    def _walk_to_numpy_inplace(d: dict) -> None:
        for k, v in list(d.items()):
            if isinstance(v, dict):
                _walk_to_numpy_inplace(v)
                continue
            if not isinstance(v, np.ndarray):
                d[k] = np.asarray(jax.device_get(v))

    _walk_to_numpy_inplace(merged)
    gc.collect()

    # Resolve the converter module (ships with the repo).
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    convert_path = repo_root / "examples" / "convert_jax_model_to_pytorch.py"
    spec = importlib.util.spec_from_file_location("openpi_convert_script", convert_path)
    convert_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(convert_mod)

    # PaliGemma config matching what the upstream converter uses.
    class _PaliGemmaCfg:
        def __init__(self):
            self.vision_config = type(
                "obj",
                (object,),
                {
                    "hidden_size": 1152,
                    "num_hidden_layers": 27,
                    "num_attention_heads": 16,
                    "intermediate_size": 4304,
                    "patch_size": 14,
                    "projection_dim": 2048,
                },
            )()
            self.text_config = type(
                "obj",
                (object,),
                {
                    "hidden_size": 2048,
                    "num_hidden_layers": 18,
                    "num_attention_heads": 8,
                    "head_dim": 256,
                    "intermediate_size": 16384,
                },
            )()

    paligemma_config = _PaliGemmaCfg()
    action_expert_config = _gemma.get_config("gemma_300m")

    # Monkey-patch torch.from_numpy to accept bfloat16 numpy arrays.
    _original_from_numpy = torch.from_numpy

    def _bf16_tolerant_from_numpy(arr):
        arr = np.asarray(arr)
        if arr.dtype == _ml_dtypes.bfloat16:
            buf = np.ascontiguousarray(arr).view(np.uint16)
            return _original_from_numpy(buf).view(torch.bfloat16)
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        return _original_from_numpy(arr)

    convert_mod.torch.from_numpy = _bf16_tolerant_from_numpy

    # Flatten PaliGemma branch and drop the nested ref so pop() inside the slice
    # function actually frees memory as it goes.
    flat_paligemma = traversals.flatten_mapping(merged["PaliGemma"], sep="/")
    projection_src = {k: v for k, v in merged.items() if k != "PaliGemma"}
    merged.clear()
    gc.collect()

    # Projection layers (action_in_proj, etc).
    if model_config.pi05:
        proj_keys = ["action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"]
    else:
        proj_keys = [
            "state_proj",
            "action_in_proj",
            "action_out_proj",
            "action_time_mlp_in",
            "action_time_mlp_out",
        ]
    projection_params: dict[str, torch.Tensor] = {}
    for key in proj_keys:
        kernel = projection_src[key]["kernel"]
        bias = projection_src[key]["bias"]
        if isinstance(kernel, dict):
            kernel = kernel["value"]
            bias = bias["value"]
        projection_params[f"{key}.weight"] = _bf16_tolerant_from_numpy(np.asarray(kernel).T)
        projection_params[f"{key}.bias"] = _bf16_tolerant_from_numpy(np.asarray(bias))

    try:
        paligemma_params, expert_params = convert_mod.slice_paligemma_state_dict(flat_paligemma, paligemma_config)
        gemma_params = convert_mod.slice_gemma_state_dict(
            expert_params,
            action_expert_config,
            num_expert=1,
            checkpoint_dir="<in-memory>",  # only used for logging
            pi05=model_config.pi05,
        )
    finally:
        torch.from_numpy = _original_from_numpy

    # Build the eval model and move to GPU BEFORE constructing the full state_dict.
    # PI0Pytorch's __init__ allocates ~7 GB of random weights on CPU; we push those to
    # GPU immediately so CPU RAM isn't holding both them and the merged dict.
    eval_model_config = pi0_config.Pi0Config(
        pi05=model_config.pi05,
        action_horizon=model_config.action_horizon,
        discrete_state_input=model_config.discrete_state_input,
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
    )
    model = pi0_pytorch.PI0Pytorch(eval_model_config)
    if torch.cuda.is_available():
        model = model.to("cuda")
        gc.collect()

    # Load state dict tensor-by-tensor, pushing each to GPU and freeing the CPU copy,
    # so peak CPU RAM during load stays at one tensor's worth instead of the full
    # state dict.
    device = next(model.parameters()).device
    named_params = dict(model.named_parameters())
    named_buffers = dict(model.named_buffers())
    missing_count = 0
    total = 0
    all_sources: list[dict] = [paligemma_params, gemma_params, projection_params]
    for src in all_sources:
        for k, v in list(src.items()):
            total += 1
            tensor = v if isinstance(v, torch.Tensor) else _bf16_tolerant_from_numpy(v)
            if tensor.device.type != device.type:
                tensor = tensor.to(device, non_blocking=False)
            target = named_params.get(k)
            if target is None:
                target = named_buffers.get(k)
            if target is None:
                missing_count += 1
                del tensor
                src.pop(k, None)
                continue
            with torch.no_grad():
                target.copy_(tensor.to(target.dtype))
            del tensor
            src.pop(k, None)  # free CPU reference for this entry
    logger.info(
        f"build_pytorch_model_from_merged: loaded {total - missing_count}/{total} params (missing {missing_count})"
    )

    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    model.eval()

    del paligemma_params, gemma_params, projection_params, projection_src, flat_paligemma
    gc.collect()

    return model


# ---- Disk serialization (for server-client envs that restart the policy server) ---


def save_merged_jax_checkpoint(
    merged: dict,
    out_dir: str | pathlib.Path,
    base_ckpt: str | pathlib.Path | None = None,
) -> pathlib.Path:
    """Write a merged param tree to disk in the structure ``serve_policy.py`` expects.

    The LIBERO / RoboCasa filtered-BC flow cannot keep the eval policy in-process
    (their env libs live in separate venvs behind a websocket client), so we need
    to launch a fresh server after merging. This writer produces a directory
    layout identical to a normal openpi training checkpoint:

        <out_dir>/
            params/                     (orbax PyTree; loaded by restore_params)
            assets/<asset_id>/...       (norm stats; copied/symlinked from base_ckpt)
            _CHECKPOINT_METADATA        (orbax-written; needed by ensure_pytorch_checkpoint)

    ``ensure_pytorch_checkpoint`` will then add ``model.safetensors`` +
    ``.pytorch_conversion_hash`` on first server launch with ``--pytorch``.
    """
    import shutil

    import orbax.checkpoint as ocp

    out_dir = pathlib.Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Wrap as {"params": <tree>} to match what restore_params expects.
    save_tree = {"params": merged}
    params_dir = out_dir / "params"
    if params_dir.exists():
        shutil.rmtree(params_dir)

    with ocp.PyTreeCheckpointer() as ckptr:
        ckptr.save(params_dir, save_tree)

    # Write _CHECKPOINT_METADATA so ensure_pytorch_checkpoint can hash it.
    meta_path = out_dir / "_CHECKPOINT_METADATA"
    if not meta_path.exists():
        meta_path.write_text('{"item_handlers": {"params": "pi0_lora_merge"}, "custom_metadata": {}}\n')

    # Copy norm-stats assets from the base ckpt. The training data config's asset_id
    # points into these; without them the served policy can't normalize observations.
    if base_ckpt is not None:
        base_assets = pathlib.Path(base_ckpt) / "assets"
        if base_assets.exists():
            dst_assets = out_dir / "assets"
            if dst_assets.exists():
                shutil.rmtree(dst_assets)
            shutil.copytree(base_assets, dst_assets, symlinks=False)
            logger.info(f"Copied assets from {base_assets} → {dst_assets}")
        else:
            logger.warning(f"No assets/ found at {base_assets}; norm stats may be missing.")

    logger.info(f"save_merged_jax_checkpoint: wrote merged ckpt to {out_dir}")
    return out_dir


__all__ = [
    "build_pytorch_model_from_merged",
    "merge_lora_params",
    "save_merged_jax_checkpoint",
]
