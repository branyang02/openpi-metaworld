"""CPU-only tests for PreferencePairDataset.

Focus on pair schema + error handling. No JAX, no GPU, no training.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.filtered_bc.envs.adapter import InferenceSample
from experiments.preference_bc.dataset import PreferencePairDataset
from experiments.preference_bc.dataset import build_preference_dataset


def _fake_sample(state_dim: int = 4, action_horizon: int = 8, action_dim: int = 4) -> InferenceSample:
    return InferenceSample(
        image=np.zeros((224, 224, 3), dtype=np.uint8),
        wrist_image=np.zeros((224, 224, 3), dtype=np.uint8),
        state=np.zeros(state_dim, dtype=np.float32),
        prompt="do the thing",
        action_chunk=np.zeros((action_horizon, action_dim), dtype=np.float32),
    )


def test_pair_cartesian_indexing():
    """N_pos * N_neg pair count; indexing tiles pos fast and neg slow."""

    class _DS:
        def __init__(self, n, prefix):
            self.n = n
            self.prefix = prefix

        def __len__(self):
            return self.n

        def __getitem__(self, i):
            return {"tag": f"{self.prefix}{i}"}

    pos = _DS(3, "P")
    neg = _DS(4, "N")
    ds = PreferencePairDataset(pos, neg)
    assert len(ds) == 12

    # idx=0 -> (P0, N0); idx=1 -> (P1, N0); idx=3 -> (P0, N1)
    assert ds[0] == {"pos/tag": "P0", "neg/tag": "N0"}
    assert ds[1] == {"pos/tag": "P1", "neg/tag": "N0"}
    assert ds[3] == {"pos/tag": "P0", "neg/tag": "N1"}
    assert ds[11] == {"pos/tag": "P2", "neg/tag": "N3"}


def test_pair_requires_both_classes():
    class _DS:
        def __init__(self, n):
            self.n = n

        def __len__(self):
            return self.n

        def __getitem__(self, i):
            return {"k": i}

    with pytest.raises(ValueError, match="at least 1 pos and 1 neg"):
        PreferencePairDataset(_DS(0), _DS(2))
    with pytest.raises(ValueError, match="at least 1 pos and 1 neg"):
        PreferencePairDataset(_DS(3), _DS(0))


def test_build_preference_dataset_rejects_empty_halves():
    from openpi.training.config import get_config

    cfg = get_config("pi05_metaworld_low_mem_finetune")
    pos = [_fake_sample()]
    neg: list = []
    with pytest.raises(ValueError, match="needs >=1 pos AND >=1 neg"):
        build_preference_dataset(pos, neg, cfg, skip_norm_stats=True)


def test_build_preference_dataset_emits_pos_neg_keys():
    """End-to-end: small metaworld config, 2 pos / 3 neg, pos/neg halves in output."""
    from openpi.training.config import get_config

    cfg = get_config("pi05_metaworld_low_mem_finetune")
    pos = [_fake_sample() for _ in range(2)]
    neg = [_fake_sample() for _ in range(3)]
    ds = build_preference_dataset(pos, neg, cfg, skip_norm_stats=True)

    assert len(ds) == 6
    item = ds[0]
    pos_keys = [k for k in item if k.startswith("pos/")]
    neg_keys = [k for k in item if k.startswith("neg/")]
    assert len(pos_keys) > 0, f"no pos/* keys in {list(item)}"
    assert len(neg_keys) > 0, f"no neg/* keys in {list(item)}"
    # pos and neg halves carry the same set of sub-keys (post-transforms).
    pos_subkeys = {k[4:] for k in pos_keys}
    neg_subkeys = {k[4:] for k in neg_keys}
    assert pos_subkeys == neg_subkeys
