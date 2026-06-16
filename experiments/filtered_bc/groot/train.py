"""GR00T N1.5 LoRA filtered-BC trainer (runs in groot_env).

Standalone subprocess invoked by the orchestrator. Reads a pickle of
:class:`InferenceSample` rollouts (filtered to successes by the orchestrator),
runs LoRA fine-tuning on the underlying ``GR00T_N1_5`` model via PEFT, merges
the LoRA adapters back into the base weights with ``merge_and_unload``, and
saves a directory loadable by ``groot_env/serve.py`` (i.e., ``Gr00tPolicy``).

The orchestrator stays in the root venv; this script must run in groot_env
(``cd groot_env && uv run python …``) because GR00T's deps (``torch==2.5.1``,
``transformers``, ``peft``, ``gr00t``) live there.

Pickle wire format (one ``dict`` written by orchestrator → one read by trainer):

    {
        "samples": [
            {"image": (H, W, 3) uint8,
             "wrist_image": (H, W, 3) uint8,
             "state": (16,) float32,           # raw, openpi-flavored
             "prompt": str,
             "action_chunk": (T, 12) float32}, # raw, denormalized
            ...
        ],
        "base_ckpt": str,
    }

Inputs are openpi-flavored (16-D state, single agentview); we synthesize the
missing ``observation/image2`` (right view) by duplicating ``image`` — same
fallback ``GR00TAdapterPolicy`` uses at inference time. State is split into
the five GR00T keys following the order the openpi robocasa client uses:
``ee_pos(3) + ee_rot(4) + base_pos(3) + base_rot(4) + gripper_qpos(2)``.
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import pickle
import shutil
import sys

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


# Action-chunk concat order from groot_adapter.ROBOCASA_ACTION_KEYS:
# ee_pos(3) + ee_rot_aa(3) + gripper(1) + base_motion(4) + control_mode(1) = 12
_ACTION_SLICES: tuple[tuple[str, slice], ...] = (
    ("action.end_effector_position", slice(0, 3)),
    ("action.end_effector_rotation", slice(3, 6)),
    ("action.gripper_close", slice(6, 7)),
    ("action.base_motion", slice(7, 11)),
    ("action.control_mode", slice(11, 12)),
)
# State concat order from openpi robocasa client `build_state` (verified by
# groot_adapter.build_robocasa_state_dict): differs from the metadata.json key
# enumeration — base comes BEFORE gripper in the flat vector.
_STATE_SLICES: tuple[tuple[str, slice], ...] = (
    ("state.end_effector_position_relative", slice(0, 3)),
    ("state.end_effector_rotation_relative", slice(3, 7)),
    ("state.base_position", slice(7, 10)),
    ("state.base_rotation", slice(10, 14)),
    ("state.gripper_qpos", slice(14, 16)),
)
_VIDEO_KEYS = (
    "video.robot0_agentview_left",
    "video.robot0_agentview_right",
    "video.robot0_eye_in_hand",
)
_LANGUAGE_KEY = "annotation.human.action.task_description"

_ACTION_HORIZON = 16  # matches checkpoint-120000/config.json: action_head_cfg.action_horizon


def _resize_to_256(img: np.ndarray) -> np.ndarray:
    """Match GR00T N1.5 robocasa's declared input resolution (see groot_adapter._resize_to_256)."""
    import cv2

    if img.shape[0] == 256 and img.shape[1] == 256:
        return img
    return cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)


def _sample_to_groot_raw(sample: dict, action_horizon: int = _ACTION_HORIZON) -> dict:
    """Translate one openpi-flavored InferenceSample dict to GR00T's flat raw format.

    Pads/truncates the action chunk to exactly ``action_horizon`` so every batch
    element shares the action shape the DiT head expects.
    """
    state = np.asarray(sample["state"], dtype=np.float32)
    if state.shape[-1] != 16:
        raise ValueError(f"Expected 16-D robocasa state, got shape {state.shape}.")

    action = np.asarray(sample["action_chunk"], dtype=np.float32)
    if action.ndim != 2 or action.shape[-1] != 12:
        raise ValueError(f"Expected (T, 12) robocasa action chunk, got shape {action.shape}.")
    if action.shape[0] < action_horizon:
        pad = np.tile(action[-1:], (action_horizon - action.shape[0], 1))
        action = np.concatenate([action, pad], axis=0)
    elif action.shape[0] > action_horizon:
        action = action[:action_horizon]

    img = _resize_to_256(np.asarray(sample["image"], dtype=np.uint8))
    wrist = _resize_to_256(np.asarray(sample["wrist_image"], dtype=np.uint8))

    out: dict = {
        # Right view: openpi robocasa client emits observation/image2 separately,
        # but at training time we don't have it captured in the InferenceSample,
        # so we duplicate the left view (same fallback GR00TAdapterPolicy uses at
        # inference). Mild stereo signal degradation; shape contract preserved.
        "video.robot0_agentview_left": img[None],
        "video.robot0_agentview_right": img[None],
        "video.robot0_eye_in_hand": wrist[None],
    }
    for key, sl in _STATE_SLICES:
        out[key] = state[sl][None]  # (1, D)
    for key, sl in _ACTION_SLICES:
        out[key] = action[:, sl]  # (T, D)
    out[_LANGUAGE_KEY] = [str(sample["prompt"])]
    return out


class FilteredBCInMemoryDataset(Dataset):
    """In-memory PyTorch dataset over filtered InferenceSamples.

    Each ``__getitem__`` builds a raw GR00T-format dict and runs it through
    the same ``ComposedModalityTransform`` that ``LeRobotSingleDataset`` uses,
    yielding a dict ready for ``GR00T_N1_5.forward()``.
    """

    def __init__(self, samples: list[dict], modality_transform):
        self._samples = list(samples)
        self._transform = modality_transform

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        return self._transform(_sample_to_groot_raw(self._samples[idx]))

    def set_epoch(self, epoch: int) -> None:  # pragma: no cover — DistributedSampler hook
        pass


def _build_modality_transform_and_collator():
    """Return (modality_transform, data_collator) by reusing groot_env's adapter.

    Reuses ``RobocasaPandaOmronDataConfig`` from ``groot_adapter`` so we get
    the exact same normalization stats / video aug pipeline GR00T was trained
    with. The data collator is GR00T's default (stacks dicts of tensors).
    """
    # groot_env/groot_adapter.py is the canonical source. groot_env is on
    # sys.path because train.py is invoked via `cd groot_env && uv run python`.
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    groot_env_dir = repo_root / "groot_env"
    if str(groot_env_dir) not in sys.path:
        sys.path.insert(0, str(groot_env_dir))

    from gr00t.model.transforms import DefaultDataCollator
    from groot_adapter import RobocasaPandaOmronDataConfig

    cfg = RobocasaPandaOmronDataConfig()
    transform = cfg.transform()
    return transform, DefaultDataCollator()


def _train(args: argparse.Namespace) -> None:
    """Load samples, LoRA-finetune GR00T N1.5, merge, save serve-loadable ckpt."""
    from gr00t.experiment.trainer import DualBrainTrainer
    from gr00t.model.gr00t_n1 import GR00T_N1_5
    from gr00t.utils.peft import get_lora_model
    from transformers import TrainingArguments

    base_ckpt = pathlib.Path(args.base_ckpt).resolve()
    output_dir = pathlib.Path(args.output_dir).resolve()
    samples_path = pathlib.Path(args.samples_pickle).resolve()

    with samples_path.open("rb") as f:
        payload = pickle.load(f)
    samples: list[dict] = payload["samples"]
    if not samples:
        raise RuntimeError(f"No samples in {samples_path}; orchestrator should skip.")
    logger.info("Loaded %d filtered samples from %s", len(samples), samples_path)

    # Modality transform reuses normalization stats from the base ckpt's
    # metadata.json + the GR00TTransform wrapping the model expects.
    transform, collator = _build_modality_transform_and_collator()

    # Robocasa's multitask checkpoint uses the "new_embodiment" tag (verified in
    # checkpoints/groot_n15/.../experiment_cfg/metadata.json top-level key).
    # The DatasetMetadata wires the embodiment_tag through itself, mirroring
    # Gr00tPolicy._load_metadata exactly — no separate set_embodiment_tag call
    # exists on ComposedModalityTransform.
    transform.set_metadata(_load_embodiment_metadata(base_ckpt))

    dataset = FilteredBCInMemoryDataset(samples, transform)

    logger.info("Loading base GR00T N1.5 from %s", base_ckpt)
    # Match the pattern in third_party/Isaac-GR00T/scripts/gr00t_finetune.py
    # exactly: load at default dtype, then flip compute_dtype to bf16 (the
    # action head's flow-matching beta sampler does Dirichlet draws which only
    # support fp32, so the model itself stays fp32; bf16 happens via Trainer
    # autocast at the right boundaries).
    model = GR00T_N1_5.from_pretrained(
        str(base_ckpt),
        tune_llm=False,
        tune_visual=False,
        tune_projector=True,
        tune_diffusion_model=True,
    )
    model.compute_dtype = "bfloat16"
    model.config.compute_dtype = "bfloat16"
    model = get_lora_model(
        model,
        rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        action_head_only=True,
    )
    model.train()
    if torch.cuda.is_available():
        model = model.cuda()

    # We don't need optimizer/scheduler/checkpointing emitted by HF Trainer's
    # default save — we'll merge + save ourselves at the end. So save_steps is
    # set huge so HF Trainer never auto-saves intermediate.
    training_args = TrainingArguments(
        output_dir=str(output_dir / "_hf_train_state"),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        max_steps=args.num_train_steps,
        learning_rate=args.learning_rate,
        weight_decay=1e-5,
        warmup_steps=min(20, max(1, args.num_train_steps // 10)),
        bf16=True,
        tf32=True,
        logging_steps=max(1, args.num_train_steps // 10),
        save_strategy="no",
        report_to=[],  # no wandb/tensorboard for filtered-BC self-distill loop
        seed=args.seed,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        label_names=[],  # GR00T computes loss itself; no Trainer-built labels
        gradient_checkpointing=False,
    )

    trainer = DualBrainTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        compute_dtype=torch.bfloat16,
    )

    logger.info(
        "Training: %d steps, batch_size=%d on %d samples (eff. epochs ≈ %.1f)",
        args.num_train_steps,
        args.batch_size,
        len(dataset),
        args.num_train_steps * args.batch_size / max(1, len(dataset)),
    )
    trainer.train()

    logger.info("Merging LoRA adapters and saving to %s", output_dir)
    # peft's PeftModel exposes merge_and_unload(); returns the underlying
    # GR00T_N1_5 with merged weights.
    merged = model.merge_and_unload()
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(output_dir))

    # serve.py needs experiment_cfg/metadata.json next to the model files;
    # save_pretrained doesn't copy it. Pull from the base ckpt verbatim.
    base_meta_dir = base_ckpt / "experiment_cfg"
    out_meta_dir = output_dir / "experiment_cfg"
    if base_meta_dir.exists():
        if out_meta_dir.exists():
            shutil.rmtree(out_meta_dir)
        shutil.copytree(base_meta_dir, out_meta_dir)
        logger.info("Copied experiment_cfg from %s", base_meta_dir)
    else:
        raise FileNotFoundError(f"No experiment_cfg/ found at {base_meta_dir}")

    # Drop the HF Trainer's intermediate state dir (we never used save_strategy
    # other than "no", but Trainer still mkdirs it).
    hf_state_dir = output_dir / "_hf_train_state"
    if hf_state_dir.exists():
        shutil.rmtree(hf_state_dir, ignore_errors=True)

    logger.info("Done. Merged checkpoint at %s", output_dir)


def _load_embodiment_metadata(base_ckpt: pathlib.Path, embodiment_tag: str = "new_embodiment"):
    """Parse ``experiment_cfg/metadata.json`` for the given embodiment.

    Mirrors ``gr00t.model.policy.Gr00tPolicy._load_metadata`` — ``DatasetMetadata``
    is what ``ComposedModalityTransform.set_metadata`` consumes (drives min/max
    normalization, target rotation, etc.).
    """
    import json

    from gr00t.data.dataset import DatasetMetadata

    meta_path = base_ckpt / "experiment_cfg" / "metadata.json"
    with meta_path.open() as f:
        all_meta = json.load(f)
    if embodiment_tag not in all_meta:
        raise ValueError(f"No metadata for embodiment {embodiment_tag!r} in {meta_path}; available: {sorted(all_meta)}")
    return DatasetMetadata.model_validate(all_meta[embodiment_tag])


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples-pickle", required=True, help="Path to pickle of filtered samples.")
    p.add_argument("--base-ckpt", required=True, help="Path to base GR00T N1.5 checkpoint dir.")
    p.add_argument("--output-dir", required=True, help="Where to write merged ckpt.")
    p.add_argument("--num-train-steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    _train(_parse_args())


if __name__ == "__main__":
    main()
