"""In-memory pair dataset for Flow-DPO fine-tuning.

Generalizes :mod:`experiments.filtered_bc.dataset` from an unpaired ``Sequence[InferenceSample]``
to a **paired** view over a (positive, negative) cross-product. Each sample emitted by
:class:`PreferencePairDataset` is a dict with two nested halves keyed by ``pos/*`` and
``neg/*`` (same schema as the filtered-BC dataset in each half). The Flow-DPO train step
unpacks the two halves, passes them through the same LoRA model under a shared noise
schedule, and differences the two flow-matching MSEs against the frozen reference model.

The transform stack (data_transforms -> Normalize -> model_transforms) is applied
identically to both halves via two inner :class:`TransformedDataset` instances so the
(pos, neg) pair stays in lockstep.
"""

from __future__ import annotations

from collections.abc import Sequence
import logging

from experiments.filtered_bc.dataset import InferenceSampleDataset
from experiments.filtered_bc.envs.adapter import InferenceSample
from openpi import transforms as _transforms
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

logger = logging.getLogger(__name__)


def _build_norm_stats(
    train_config: _config.TrainConfig,
    *,
    skip_norm_stats: bool,
    base_ckpt_fallback: str | None,
) -> tuple[dict, _config.DataConfig]:
    """Resolve norm_stats for the training data config, with base-ckpt fallback."""
    import pathlib

    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    norm_stats: dict = {}
    if data_config.repo_id == "fake" or skip_norm_stats:
        return norm_stats, data_config

    if data_config.norm_stats is None and base_ckpt_fallback and data_config.asset_id:
        try:
            norm_stats = _checkpoints.load_norm_stats(pathlib.Path(base_ckpt_fallback) / "assets", data_config.asset_id)
            logger.info(f"[dataset] loaded norm_stats from {base_ckpt_fallback}/assets/{data_config.asset_id}")
        except Exception as exc:
            logger.warning(f"[dataset] base_ckpt norm_stats fallback failed: {exc}")
            norm_stats = {}
    elif data_config.norm_stats is not None:
        norm_stats = data_config.norm_stats

    if not norm_stats:
        raise ValueError(
            f"Normalization stats not found for {train_config.name}. "
            "Run `scripts/compute_norm_stats.py --config-name=<name>` OR copy norm stats "
            "from the base-model assets dir (recommended for this baseline)."
        )
    return norm_stats, data_config


class PreferencePairDataset:
    """Pair-wise view: emits (pos, neg) pairs via cartesian indexing.

    With ``N_pos`` positives and ``N_neg`` negatives, the dataset has
    ``N_pos * N_neg`` entries. Each entry is a dict with ``pos/*`` and ``neg/*``
    keys carrying the already-transformed sample dicts.
    """

    def __init__(self, pos_ds, neg_ds):
        self._pos = pos_ds
        self._neg = neg_ds
        self._n_pos = len(pos_ds)
        self._n_neg = len(neg_ds)
        if self._n_pos == 0 or self._n_neg == 0:
            raise ValueError(
                f"PreferencePairDataset requires at least 1 pos and 1 neg; got "
                f"n_pos={self._n_pos}, n_neg={self._n_neg}."
            )

    def __len__(self) -> int:
        return self._n_pos * self._n_neg

    def __getitem__(self, idx: int) -> dict:
        i_pos = idx % self._n_pos
        i_neg = idx // self._n_pos
        pos = self._pos[i_pos]
        neg = self._neg[i_neg]
        return {
            **{f"pos/{k}": v for k, v in pos.items()},
            **{f"neg/{k}": v for k, v in neg.items()},
        }


def build_preference_dataset(
    pos_samples: Sequence[InferenceSample],
    neg_samples: Sequence[InferenceSample],
    train_config: _config.TrainConfig,
    *,
    skip_norm_stats: bool = False,
    base_ckpt_fallback: str | None = None,
) -> _data_loader.Dataset:
    """Build a paired dataset ready for Flow-DPO training.

    Both positives and negatives pass through the same transform stack
    (data_transforms -> Normalize -> model_transforms). The resulting Dataset's
    items are dicts with ``pos/*`` and ``neg/*`` sub-keys.
    """
    if len(pos_samples) == 0 or len(neg_samples) == 0:
        raise ValueError(
            f"build_preference_dataset needs >=1 pos AND >=1 neg sample "
            f"(got {len(pos_samples)} pos, {len(neg_samples)} neg). "
            "Tasks with all-success or all-failure rollouts must be skipped upstream."
        )

    norm_stats, data_config = _build_norm_stats(
        train_config, skip_norm_stats=skip_norm_stats, base_ckpt_fallback=base_ckpt_fallback
    )

    transforms = [
        *data_config.data_transforms.inputs,
        _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.model_transforms.inputs,
    ]

    pos_ds = _data_loader.TransformedDataset(InferenceSampleDataset(pos_samples), transforms)
    neg_ds = _data_loader.TransformedDataset(InferenceSampleDataset(neg_samples), transforms)

    return PreferencePairDataset(pos_ds, neg_ds)


__all__ = ["PreferencePairDataset", "build_preference_dataset"]
