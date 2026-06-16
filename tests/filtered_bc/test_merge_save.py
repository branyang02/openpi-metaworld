"""Round-trip tests for merge_save.save_merged_jax_checkpoint.

Exercise just the save path — it's the bridge between the in-process LoRA merge
and the out-of-process policy server used by LIBERO/RoboCasa. No GPU needed.
"""

from __future__ import annotations

import pathlib
import tempfile

import numpy as np

from experiments.filtered_bc.merge_save import save_merged_jax_checkpoint
from openpi.models.model import restore_params


def test_save_round_trip_plain_dict():
    """restore_params should reproduce bit-identical arrays from a saved dict."""
    merged = {
        "action_in_proj": {
            "kernel": np.random.default_rng(0).random((4, 8)).astype(np.float32),
            "bias": np.zeros(8, dtype=np.float32),
        },
        "time_mlp_in": {
            "kernel": np.random.default_rng(1).random((8, 8)).astype(np.float32),
        },
    }

    with tempfile.TemporaryDirectory() as td:
        out = save_merged_jax_checkpoint(merged, pathlib.Path(td) / "ckpt")

        assert (out / "params").is_dir()
        assert (out / "params" / "_METADATA").is_file()
        assert (out / "_CHECKPOINT_METADATA").is_file()

        restored = restore_params(out / "params", restore_type=np.ndarray)

    np.testing.assert_array_equal(merged["action_in_proj"]["kernel"], restored["action_in_proj"]["kernel"])
    np.testing.assert_array_equal(merged["action_in_proj"]["bias"], restored["action_in_proj"]["bias"])
    np.testing.assert_array_equal(merged["time_mlp_in"]["kernel"], restored["time_mlp_in"]["kernel"])


def test_save_copies_assets_from_base_ckpt():
    """When base_ckpt is provided, assets/ should be copied (e.g. for norm_stats)."""
    merged = {"p": {"w": np.zeros((1, 1), np.float32)}}

    with tempfile.TemporaryDirectory() as td:
        base_ckpt = pathlib.Path(td) / "base"
        (base_ckpt / "assets" / "some_id").mkdir(parents=True)
        (base_ckpt / "assets" / "some_id" / "norm_stats.json").write_text('{"mean": [0, 0, 0]}')

        out = save_merged_jax_checkpoint(merged, pathlib.Path(td) / "merged", base_ckpt=base_ckpt)

        assert (out / "assets" / "some_id" / "norm_stats.json").is_file()
        assert (out / "assets" / "some_id" / "norm_stats.json").read_text() == '{"mean": [0, 0, 0]}'


def test_save_ckpt_metadata_hashable_by_ensure_pytorch():
    """ensure_pytorch_checkpoint's hash function must succeed on our output."""
    from openpi.models_pytorch.convert import _compute_checkpoint_hash

    merged = {"p": {"w": np.zeros((2, 2), np.float32)}}
    with tempfile.TemporaryDirectory() as td:
        out = save_merged_jax_checkpoint(merged, pathlib.Path(td) / "merged")
        h = _compute_checkpoint_hash(out, "pi05_libero_low_mem_finetune")

        assert isinstance(h, str)
        assert len(h) == 64  # sha256 hex digest

        # Stable across invocations of the same input.
        h2 = _compute_checkpoint_hash(out, "pi05_libero_low_mem_finetune")
        assert h == h2
