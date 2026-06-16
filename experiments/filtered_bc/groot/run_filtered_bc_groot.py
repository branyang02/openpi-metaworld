"""Filtered-BC orchestrator for GR00T N1.5 on RoboCasa.

Per task in the configured subset:

1. Roll out N episodes against the base GR00T checkpoint via
   :class:`GrootRoboCasaAdapter` (launches ``groot_env/serve.py`` + the
   robocasa client subprocess).
2. Filter to successful episodes; pickle the kept samples to scratch.
3. Spawn the GR00T LoRA trainer
   (`experiments/filtered_bc/groot/train.py`, run from ``groot_env/``):
   loads base ckpt → wraps with PEFT LoRA → trains on filtered samples →
   merges → saves a serve.py-loadable checkpoint dir.
4. Eval the merged checkpoint via the same adapter (fresh server +
   ``--eval-only`` client).
5. Free everything; move on.

Results land incrementally in ``--results-json``: same schema as
``run_filtered_bc.py`` so existing analysis scripts (paired-t / pooled-z) work
unchanged.

This stays in the root venv. The trainer + servers run in ``groot_env/``; the
robocasa client runs in ``examples/robocasa_env/``. All cross-venv handoff is
via files (pickles + checkpoint dirs).
"""

from __future__ import annotations

import contextlib
import dataclasses
import gc
import json
import logging
import os
import pathlib
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

import tyro

from experiments.filtered_bc.envs.adapter import InferenceSample
from experiments.filtered_bc.envs.adapter import RolloutConfig
from experiments.filtered_bc.envs.adapter import filter_successful
from experiments.filtered_bc.groot.envs.robocasa_groot import GrootRoboCasaAdapter

logger = logging.getLogger(__name__)

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
# groot_env (with flash-attn installed) lives at project root; see comment in
# experiments/filtered_bc/groot/envs/robocasa_groot.py.
_GROOT_ENV_DIR = pathlib.Path(os.environ.get(
    "OPENPI_GROOT_ENV_DIR",
    "/home/kim34/projects_brandon/openpi-metaworld/groot_env",
))
_TRAINER_SCRIPT = _REPO_ROOT / "experiments" / "filtered_bc" / "groot" / "train.py"

_DEFAULT_BASE_CKPT = (
    "/home/kim34/projects_brandon/openpi-metaworld/checkpoints/groot_n15"
    "/gr00t_n1-5/multitask_learning/checkpoint-120000"
)


@dataclasses.dataclass
class Args:
    base_ckpt: str = _DEFAULT_BASE_CKPT
    """Absolute path to the base GR00T N1.5 checkpoint dir."""

    tasks: list[str] = dataclasses.field(default_factory=list)
    """Specific robocasa env_names. Empty -> the 7-task subset (matches
    examples/robocasa_env/eval_all.py:SUBSET)."""
    split: str = "pretrain"
    """robocasa init-state split."""

    num_rollouts: int = 30
    """Episodes to roll out for collecting filtered-BC training data."""
    max_steps: int | None = None
    """Per-episode max steps. None → 1.5 * task_horizon (client default)."""
    replan_steps: int = 5
    seed: int = 69_420

    num_train_steps: int = 200
    batch_size: int = 8
    learning_rate: float = 1e-4
    lora_rank: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05

    eval_num_episodes: int = 30
    """Held-out eval episodes (seed offset by 10_000 for disjointness)."""

    scratch_dir: str = ""
    """Directory for per-task scratch (server logs, sample pickles, merged
    ckpts). Empty → mktemp."""
    keep_scratch: bool = False
    results_json: str = "experiments/filtered_bc/results_robocasa_groot.json"


def _resolve_tasks(args: Args, adapter: GrootRoboCasaAdapter) -> list[str]:
    return list(args.tasks) if args.tasks else list(adapter.train_tasks)


def _pickle_samples(samples: list[InferenceSample], path: pathlib.Path) -> None:
    """Serialize InferenceSamples in the format the trainer expects."""
    payload = {
        "samples": [
            {
                "image": s.image,
                "wrist_image": s.wrist_image,
                "state": s.state,
                "prompt": s.prompt,
                "action_chunk": s.action_chunk,
            }
            for s in samples
        ],
    }
    with path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def _run_trainer(
    samples_pickle: pathlib.Path,
    base_ckpt: str,
    output_dir: pathlib.Path,
    args: Args,
    log_path: pathlib.Path,
) -> None:
    """Spawn the GR00T LoRA trainer subprocess in groot_env. Trainer cwd is
    groot_env/, so all path args must be absolute."""
    abs_samples = pathlib.Path(samples_pickle).resolve()
    abs_output = pathlib.Path(output_dir).resolve()
    abs_base_ckpt = pathlib.Path(base_ckpt).resolve()
    abs_log = pathlib.Path(log_path).resolve()
    abs_log.parent.mkdir(parents=True, exist_ok=True)
    abs_output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "uv",
        "run",
        "python",
        "-u",
        str(_TRAINER_SCRIPT),
        f"--samples-pickle={abs_samples}",
        f"--base-ckpt={abs_base_ckpt}",
        f"--output-dir={abs_output}",
        f"--num-train-steps={args.num_train_steps}",
        f"--batch-size={args.batch_size}",
        f"--learning-rate={args.learning_rate}",
        f"--lora-rank={args.lora_rank}",
        f"--lora-alpha={args.lora_alpha}",
        f"--lora-dropout={args.lora_dropout}",
        f"--seed={args.seed}",
    ]
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env["PYTHONUNBUFFERED"] = "1"
    logger.info(f"Spawning trainer → log: {abs_log}; cmd (from {_GROOT_ENV_DIR}): {' '.join(cmd)}")
    with abs_log.open("w") as f:
        result = subprocess.run(cmd, cwd=_GROOT_ENV_DIR, env=env, stdout=f, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"GR00T trainer failed (code={result.returncode}). See log: {abs_log}")


def _write_results(path: pathlib.Path, results: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w") as f:
        json.dump(results, f, indent=2)
    tmp.replace(path)


def _run_one_task(
    task_name: str,
    args: Args,
    adapter: GrootRoboCasaAdapter,
    scratch_root: pathlib.Path,
) -> dict:
    record: dict = {"task": task_name, "status": "started"}
    rollout_cfg = RolloutConfig(
        max_steps=args.max_steps,
        replan_steps=args.replan_steps,
        seed=args.seed,
        extra={"split": args.split},
    )
    task_scratch = scratch_root / task_name.replace(":", "_").replace("/", "_")
    task_scratch.mkdir(parents=True, exist_ok=True)

    # 1. Rollout against the base ckpt.
    t0 = time.time()
    episodes = adapter.rollout(args.base_ckpt, task_name, args.num_rollouts, rollout_cfg, task_scratch)
    record["t_rollout_sec"] = round(time.time() - t0, 2)
    n_success = sum(1 for ep in episodes if ep.success)
    record.update(
        {
            "num_rollouts": len(episodes),
            "num_success_rollout": n_success,
            "rollout_success_rate": n_success / len(episodes) if episodes else 0.0,
        }
    )
    logger.info(
        f"[{task_name}] rollout: {n_success}/{len(episodes)} successful "
        f"({100 * n_success / max(1, len(episodes)):.0f}%), {record['t_rollout_sec']:.1f}s"
    )

    samples = filter_successful(episodes)
    record["num_train_samples"] = len(samples)
    if not samples:
        logger.warning(f"[{task_name}] 0 successful rollouts — skipping training + eval.")
        record["status"] = "skipped_no_successes"
        return record

    # 2. Pickle filtered samples → 3. Spawn trainer → produces merged ckpt.
    samples_pickle = task_scratch / "filtered_samples.pkl"
    merged_ckpt = task_scratch / "merged"
    trainer_log = task_scratch / "trainer.log"
    _pickle_samples(samples, samples_pickle)

    t0 = time.time()
    _run_trainer(samples_pickle, args.base_ckpt, merged_ckpt, args, trainer_log)
    record["t_train_sec"] = round(time.time() - t0, 2)
    logger.info(f"[{task_name}] train+merge: {record['t_train_sec']:.1f}s → {merged_ckpt}")

    # 4. Eval merged ckpt against held-out episodes.
    t0 = time.time()
    eval_result = adapter.eval(merged_ckpt, task_name, args.eval_num_episodes, rollout_cfg, task_scratch)
    record.update(
        {
            "eval_num_episodes": eval_result.num_episodes,
            "eval_num_success": eval_result.num_success,
            "eval_success_rate": eval_result.success_rate,
            "eval_mean_reward": eval_result.mean_reward,
            "eval_mean_steps_to_success": eval_result.mean_steps_to_success,
            "t_eval_sec": round(time.time() - t0, 2),
        }
    )
    logger.info(
        f"[{task_name}] eval: {eval_result.num_success}/{eval_result.num_episodes} "
        f"({100 * eval_result.success_rate:.0f}%), {record['t_eval_sec']:.1f}s"
    )

    # 5. Free per-task scratch unless asked to keep.
    if not args.keep_scratch:
        with contextlib.suppress(Exception):
            shutil.rmtree(task_scratch)

    record["status"] = "ok"
    return record


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
    if not _TRAINER_SCRIPT.exists():
        sys.exit(f"Missing trainer script: {_TRAINER_SCRIPT}")
    if not _GROOT_ENV_DIR.exists():
        sys.exit(f"Missing groot_env dir: {_GROOT_ENV_DIR}")

    adapter = GrootRoboCasaAdapter(tasks=args.tasks if args.tasks else None, split=args.split)

    if args.scratch_dir:
        scratch_root = pathlib.Path(args.scratch_dir)
        scratch_root.mkdir(parents=True, exist_ok=True)
    else:
        scratch_root = pathlib.Path(tempfile.mkdtemp(prefix="filtered_bc_groot_robocasa_"))
    logger.info(f"Scratch root: {scratch_root}")

    tasks = _resolve_tasks(args, adapter)
    logger.info(f"[robocasa_groot] running filtered-BC on {len(tasks)} tasks: {tasks}")

    results_path = pathlib.Path(args.results_json)
    results: dict = {
        "args": {
            "base_ckpt": args.base_ckpt,
            "num_rollouts": args.num_rollouts,
            "num_train_steps": args.num_train_steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "eval_num_episodes": args.eval_num_episodes,
            "split": args.split,
            "seed": args.seed,
        },
        "tasks": {},
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _write_results(results_path, results)

    for task in tasks:
        try:
            record = _run_one_task(task, args, adapter, scratch_root)
        except Exception as exc:
            logger.error(f"[{task}] FAILED: {exc}")
            record = {
                "task": task,
                "status": "failed",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        results["tasks"][task] = record
        results["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        _write_results(results_path, results)
        gc.collect()

    results["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _write_results(results_path, results)

    if not args.keep_scratch:
        with contextlib.suppress(Exception):
            shutil.rmtree(scratch_root)

    logger.info("=" * 60)
    logger.info("SWEEP COMPLETE. Per-task summary:")
    for task, rec in results["tasks"].items():
        status = rec.get("status", "?")
        if status == "ok":
            logger.info(
                f"  {task}: eval {rec['eval_num_success']}/{rec['eval_num_episodes']} "
                f"= {100 * rec['eval_success_rate']:.0f}%, "
                f"train_samples={rec['num_train_samples']}"
            )
        else:
            logger.info(f"  {task}: {status}")
    logger.info(f"Full results: {results_path}")


if __name__ == "__main__":
    tyro.cli(main)
