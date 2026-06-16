"""In-memory torch Dataset over successful rollout samples.

Bypasses on-disk LeRobot datasets entirely. Emits dicts in the already-repacked
format that matches what LeRobotMetaworldDataConfig produces AFTER its
``repack_transforms`` stage, so we only need to apply data_transforms +
Normalize + model_transforms on top.
"""

from __future__ import annotations

from collections.abc import Sequence
import logging
import pathlib

from experiments.filtered_bc.envs.adapter import InferenceSample
from openpi import transforms as _transforms
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

logger = logging.getLogger(__name__)


class InferenceSampleDataset:
    """Raw dataset over InferenceSample objects. Emits LeRobot-repacked-style dicts."""

    def __init__(self, samples: Sequence[InferenceSample]):
        self._samples = list(samples)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        s = self._samples[idx]
        return {
            "observation/image": s.image,
            "observation/wrist_image": s.wrist_image,
            "observation/state": s.state,
            "actions": s.action_chunk,
            "prompt": s.prompt,
        }


def build_training_dataset(
    samples: Sequence[InferenceSample],
    train_config: _config.TrainConfig,
    *,
    skip_norm_stats: bool = False,
    base_ckpt_fallback: str | None = None,
) -> _data_loader.Dataset:
    """Return a fully-transformed Dataset ready for training.

    Applies the same transform stack as `data_loader.transform_dataset`:
        data_transforms.inputs -> Normalize(norm_stats) -> model_transforms.inputs

    The ``repack_transforms`` step is skipped because our rollout emits dicts in
    the post-repack internal-key format directly (``observation/image`` etc.).

    If ``base_ckpt_fallback`` is provided, norm_stats are loaded from
    ``<base_ckpt_fallback>/assets/<asset_id>/`` when the train_config's own assets
    dir doesn't contain them. This matches how :func:`_build_policy_from_model`
    resolves norm_stats for the merged-eval policy and lets the LoRA-finetune
    configs reuse the base model's stats without a manual symlink.
    """
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)

    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None and base_ckpt_fallback and data_config.asset_id:
            try:
                norm_stats = _checkpoints.load_norm_stats(
                    pathlib.Path(base_ckpt_fallback) / "assets", data_config.asset_id
                )
                logger.info(f"[dataset] loaded norm_stats from {base_ckpt_fallback}/assets/{data_config.asset_id}")
            except Exception as exc:
                logger.warning(f"[dataset] base_ckpt norm_stats fallback failed: {exc}")
                norm_stats = None
        elif data_config.norm_stats is not None:
            norm_stats = data_config.norm_stats
        if not norm_stats:
            raise ValueError(
                "Normalization stats not found for "
                f"{train_config.name}. Run `scripts/compute_norm_stats.py --config-name=<name>` "
                "OR copy norm stats from the base-model assets dir (recommended for this baseline)."
            )

    transforms = [
        *data_config.data_transforms.inputs,
        _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.model_transforms.inputs,
    ]

    raw = InferenceSampleDataset(samples)
    return _data_loader.TransformedDataset(raw, transforms)


__all__ = ["InferenceSampleDataset", "build_training_dataset"]
