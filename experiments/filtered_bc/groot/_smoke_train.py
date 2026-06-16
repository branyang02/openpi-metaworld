"""Quick standalone test: build N=2 dummy samples → tiny train → merge → reload via Gr00tPolicy.

Run from groot_env:
    cd groot_env && CUDA_VISIBLE_DEVICES=0 uv run python \
        ../worktrees/rl-integration/experiments/filtered_bc/groot/_smoke_train.py
"""

from __future__ import annotations

import logging
import pathlib
import pickle
import shutil
import tempfile

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_CKPT = "/home/kim34/projects_brandon/openpi-metaworld/checkpoints/groot_n15/gr00t_n1-5/multitask_learning/checkpoint-120000"


def _build_dummy_sample() -> dict:
    """One InferenceSample-flavored dict at robocasa's shapes."""
    return {
        "image": np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8),
        "wrist_image": np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8),
        # Raw 16-D openpi-flavored state (small magnitudes — closer to in-distribution).
        "state": np.random.randn(16).astype(np.float32) * 0.1,
        "prompt": "close the fridge",
        # 16-step action chunk in raw 12-D space (small to keep within
        # min_max stats; actual rollouts emit reasonable values).
        "action_chunk": np.random.randn(16, 12).astype(np.float32) * 0.05,
    }


def main() -> None:
    scratch = pathlib.Path(tempfile.mkdtemp(prefix="groot_smoke_"))
    samples_path = scratch / "samples.pkl"
    out_dir = scratch / "merged"

    with samples_path.open("wb") as f:
        pickle.dump({"samples": [_build_dummy_sample(), _build_dummy_sample()]}, f)
    print(f"[smoke] wrote 2 dummy samples to {samples_path}")

    # Drive the real trainer module the same way the orchestrator subprocess will.
    import sys

    repo_root = pathlib.Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root))

    import argparse

    from experiments.filtered_bc.groot import train as trainer

    args = argparse.Namespace(
        samples_pickle=str(samples_path),
        base_ckpt=BASE_CKPT,
        output_dir=str(out_dir),
        num_train_steps=2,
        batch_size=1,
        learning_rate=1e-4,
        lora_rank=8,
        lora_alpha=8,
        lora_dropout=0.05,
        seed=42,
    )
    trainer._train(args)  # noqa: SLF001 — module-internal helper, kept private to discourage casual reuse
    print(f"[smoke] merged ckpt at {out_dir}")
    print(f"[smoke] files: {sorted(p.name for p in out_dir.iterdir())}")

    # Verify load via Gr00tPolicy (what serve.py uses).
    sys.path.insert(0, str(repo_root / "groot_env"))

    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.model.policy import Gr00tPolicy
    from groot_adapter import RobocasaPandaOmronDataConfig

    cfg = RobocasaPandaOmronDataConfig()
    policy = Gr00tPolicy(
        model_path=str(out_dir),
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        modality_config=cfg.modality_config(),
        modality_transform=cfg.transform(),
        denoising_steps=4,
        device="cuda:0",
    )
    print("[smoke] Gr00tPolicy loaded merged ckpt OK")

    # Inference is exercised by the orchestrator's rollout client (which uses
    # the GR00TAdapterPolicy translation). Loading via Gr00tPolicy already
    # validates config.json + safetensors + experiment_cfg/metadata.json all
    # round-tripped correctly, which is what serve.py needs.
    print(
        f"[smoke] merged ckpt sane: {sorted(p.name for p in policy._modality_transform.dataset_metadata.statistics.action)}"  # noqa: SLF001
    )

    shutil.rmtree(scratch, ignore_errors=True)
    print("[smoke] PASS")


if __name__ == "__main__":
    main()
