"""Flow-DPO baseline orchestrator across MetaWorld, LIBERO, and RoboCasa.

Per task in the configured split:

1. Roll out N episodes with the base pi0.5 policy.
2. Partition into successful (``positives``) and failed (``negatives``) samples.
   NO filter that drops failures — both classes are training signal for DPO.
3. If the rollout was all-success or all-failure, skip the task (no pairs available).
4. Flow-DPO fine-tune using a cartesian (pos x neg) pair pool.
5. Merge LoRA into base weights.
6. Evaluate the merged policy on held-out episodes.
7. Record per-task metrics; move on.

Dispatch on ``--args.env``:

- MetaWorld: in-process (live Policy, AsyncVectorEnv rollouts + eval)
- LIBERO / RoboCasa: subprocess (policy server + rollout client in env venv)

Reuses the :class:`EnvAdapter` protocol + three concrete adapters from
``experiments/filtered_bc/envs/`` — those pieces are env-neutral and work
identically for both filtered-BC and preference-BC.
"""

from __future__ import annotations

import contextlib
import dataclasses
import gc
import json
import logging
import pathlib
import shutil
import tempfile
import time
import traceback
from typing import Literal

import tyro

from experiments.filtered_bc.envs.adapter import RolloutConfig
from experiments.filtered_bc.envs.adapter import get_adapter
from experiments.filtered_bc.merge_save import build_pytorch_model_from_merged
from experiments.filtered_bc.merge_save import merge_lora_params
from experiments.filtered_bc.merge_save import save_merged_jax_checkpoint
from experiments.preference_bc.train import train_dpo
from openpi import transforms as _transforms
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

logger = logging.getLogger(__name__)


_DEFAULT_BASE_CKPT = {
    "metaworld": "/home/kim34/projects_brandon/openpi-metaworld/checkpoints/openpi-metaworld-5000",
    "libero": "/home/kim34/projects_brandon/openpi-metaworld/checkpoints/openpi-libero-2000",
    "robocasa": "/home/kim34/projects_brandon/openpi-metaworld/checkpoints/pi05_pretrain_human300/multitask_learning/75000",
}
_DEFAULT_BASE_CONFIG = {
    "metaworld": "pi05_metaworld",
    "libero": "pi05_libero",
    "robocasa": "pi05_robocasa",
}
_DEFAULT_TRAIN_CONFIG = {
    "metaworld": "pi05_metaworld_low_mem_finetune",
    "libero": "pi05_libero_low_mem_finetune",
    "robocasa": "pi05_robocasa_low_mem_finetune",
}


@dataclasses.dataclass
class Args:
    env: Literal["metaworld", "libero", "robocasa"] = "metaworld"

    base_ckpt: str = ""
    base_config: str = ""
    train_config: str = ""

    tasks: list[str] = dataclasses.field(default_factory=list)
    split: Literal["train", "test"] = "train"
    libero_suite: str = "libero_spatial"
    robocasa_task_set: str = "subset"
    robocasa_split: str = "pretrain"

    # Rollout budget (must match steering).
    num_rollouts: int = 15
    max_steps: int | None = None
    replan_steps: int = 10
    seed: int = 69_420

    # Training.
    num_train_steps: int | None = None
    batch_size: int | None = None
    skip_norm_stats: bool = False

    # Flow-DPO hyperparameter.
    beta: float = 2000.0

    # Eval.
    eval_num_episodes: int = 20

    # Scratch / output.
    scratch_dir: str = ""
    keep_scratch: bool = False
    results_json: str = "experiments/preference_bc/results.json"


def _fill_defaults(args: Args) -> Args:
    updates = {}
    if not args.base_ckpt:
        updates["base_ckpt"] = _DEFAULT_BASE_CKPT[args.env]
    if not args.base_config:
        updates["base_config"] = _DEFAULT_BASE_CONFIG[args.env]
    if not args.train_config:
        updates["train_config"] = _DEFAULT_TRAIN_CONFIG[args.env]
    return dataclasses.replace(args, **updates) if updates else args


def _make_adapter(args: Args):
    if args.env == "metaworld":
        return get_adapter("metaworld")
    if args.env == "libero":
        from experiments.filtered_bc.envs.libero import LiberoAdapter

        return LiberoAdapter(task_suite_name=args.libero_suite)
    if args.env == "robocasa":
        from experiments.filtered_bc.envs.robocasa import RoboCasaAdapter

        return RoboCasaAdapter(task_set=args.robocasa_task_set, split=args.robocasa_split)
    raise ValueError(f"Unknown env: {args.env}")


def _resolve_tasks(args: Args, adapter) -> list[str]:
    if args.tasks:
        return list(args.tasks)
    if args.split == "train":
        return list(adapter.train_tasks)
    return list(adapter.test_tasks)


def _override_config(base: _config.TrainConfig, args: Args) -> _config.TrainConfig:
    overrides = {}
    if args.num_train_steps is not None:
        overrides["num_train_steps"] = args.num_train_steps
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if overrides:
        return dataclasses.replace(base, **overrides)
    return base


def _write_results(path: pathlib.Path, results: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w") as f:
        json.dump(results, f, indent=2)
    tmp.replace(path)


def _partition_pos_neg(episodes: list) -> tuple[list, list]:
    """Split InferenceSamples from episodes into (positives, negatives) by episode success."""
    pos: list = []
    neg: list = []
    for ep in episodes:
        target = pos if ep.success else neg
        target.extend(ep.samples)
    return pos, neg


def _build_policy_from_model(model, policy_train_config: _config.TrainConfig, base_ckpt: str) -> _policy.Policy:
    import torch

    from openpi.training import checkpoints as _checkpoints

    data_config = policy_train_config.data.create(policy_train_config.assets_dirs, policy_train_config.model)
    norm_stats = data_config.norm_stats
    if not norm_stats and data_config.asset_id is not None:
        try:
            norm_stats = _checkpoints.load_norm_stats(pathlib.Path(base_ckpt) / "assets", data_config.asset_id)
            logger.info(f"[policy] loaded norm_stats from {base_ckpt}/assets/{data_config.asset_id}")
        except Exception as exc:
            logger.warning(f"[policy] couldn't load norm_stats from base ckpt: {exc}")
            norm_stats = {}

    return _policy.Policy(
        model=model,
        model_type=policy_train_config.model.model_type,
        transforms=[
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            _transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
        ],
        is_pytorch=True,
        pytorch_device="cuda" if torch.cuda.is_available() else "cpu",
    )


# ---- Per-task flows --------------------------------------------------------------


def _run_one_task_inprocess(
    base_policy,
    task_name: str,
    args: Args,
    train_config: _config.TrainConfig,
    adapter,
) -> dict:
    import torch

    record: dict = {"task": task_name, "status": "started"}
    rollout_cfg = RolloutConfig(max_steps=args.max_steps, replan_steps=args.replan_steps, seed=args.seed)

    t0 = time.time()
    episodes = adapter.rollout(base_policy, task_name, num_episodes=args.num_rollouts, cfg=rollout_cfg)
    t_rollout = time.time() - t0
    num_success = sum(ep.success for ep in episodes)
    record.update(
        {
            "num_rollouts": len(episodes),
            "num_success_rollout": num_success,
            "rollout_success_rate": num_success / len(episodes) if episodes else 0.0,
            "t_rollout_sec": round(t_rollout, 2),
        }
    )
    logger.info(
        f"[{task_name}] rollout: {num_success}/{len(episodes)} "
        f"({100 * num_success / max(len(episodes), 1):.0f}%), {t_rollout:.1f}s"
    )

    pos, neg = _partition_pos_neg(episodes)
    record["num_pos_samples"] = len(pos)
    record["num_neg_samples"] = len(neg)
    if len(pos) == 0 or len(neg) == 0:
        why = "no successes" if len(pos) == 0 else "no failures"
        logger.warning(f"[{task_name}] skipping DPO: {why} in the rollout pool.")
        record["status"] = f"skipped_{why.replace(' ', '_')}"
        return record

    t0 = time.time()
    train_state = train_dpo(
        train_config,
        pos,
        neg,
        base_ckpt=args.base_ckpt,
        num_train_steps=args.num_train_steps,
        beta=args.beta,
        skip_norm_stats=args.skip_norm_stats,
    )
    record["t_train_sec"] = round(time.time() - t0, 2)

    t0 = time.time()
    merged = merge_lora_params(train_state.params.to_pure_dict(), train_config.model)
    record["t_merge_sec"] = round(time.time() - t0, 2)

    del train_state
    with contextlib.suppress(Exception):
        import jax

        jax.clear_caches()
    try:
        base_policy._model = base_policy._model.to("cpu")  # noqa: SLF001
        if hasattr(base_policy, "_pytorch_device"):
            base_policy._pytorch_device = "cpu"  # noqa: SLF001
        torch.cuda.empty_cache()
    except Exception as exc:
        logger.warning(f"[{task_name}] couldn't move base policy to CPU: {exc}")
    gc.collect()

    t0 = time.time()
    pytorch_model = build_pytorch_model_from_merged(merged, train_config.model)
    del merged
    gc.collect()
    record["t_build_pt_sec"] = round(time.time() - t0, 2)

    policy_train_config = _config.get_config(args.base_config)
    policy = _build_policy_from_model(pytorch_model, policy_train_config, args.base_ckpt)
    t0 = time.time()
    eval_result = adapter.eval(policy, task_name, num_episodes=args.eval_num_episodes, cfg=rollout_cfg)
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

    del policy, pytorch_model
    with contextlib.suppress(Exception):
        torch.cuda.empty_cache()
    gc.collect()
    try:
        base_policy._model = base_policy._model.to("cuda")  # noqa: SLF001
        if hasattr(base_policy, "_pytorch_device"):
            base_policy._pytorch_device = "cuda"  # noqa: SLF001
    except Exception as exc:
        logger.warning(f"[{task_name}] couldn't restore base policy to GPU: {exc}")

    record["status"] = "ok"
    return record


def _run_one_task_subprocess(
    task_name: str,
    args: Args,
    train_config: _config.TrainConfig,
    adapter,
    scratch_root: pathlib.Path,
) -> dict:
    record: dict = {"task": task_name, "status": "started"}
    rollout_cfg = RolloutConfig(
        max_steps=args.max_steps,
        replan_steps=args.replan_steps,
        seed=args.seed,
        extra={"config_name": args.base_config},
    )

    t0 = time.time()
    episodes = adapter.rollout(args.base_ckpt, task_name, num_episodes=args.num_rollouts, cfg=rollout_cfg)
    t_rollout = time.time() - t0
    num_success = sum(ep.success for ep in episodes)
    record.update(
        {
            "num_rollouts": len(episodes),
            "num_success_rollout": num_success,
            "rollout_success_rate": num_success / len(episodes) if episodes else 0.0,
            "t_rollout_sec": round(t_rollout, 2),
        }
    )
    logger.info(
        f"[{task_name}] rollout: {num_success}/{len(episodes)} "
        f"({100 * num_success / max(len(episodes), 1):.0f}%), {t_rollout:.1f}s"
    )

    pos, neg = _partition_pos_neg(episodes)
    record["num_pos_samples"] = len(pos)
    record["num_neg_samples"] = len(neg)
    if len(pos) == 0 or len(neg) == 0:
        why = "no successes" if len(pos) == 0 else "no failures"
        logger.warning(f"[{task_name}] skipping DPO: {why} in the rollout pool.")
        record["status"] = f"skipped_{why.replace(' ', '_')}"
        return record

    t0 = time.time()
    train_state = train_dpo(
        train_config,
        pos,
        neg,
        base_ckpt=args.base_ckpt,
        num_train_steps=args.num_train_steps,
        beta=args.beta,
        skip_norm_stats=args.skip_norm_stats,
    )
    record["t_train_sec"] = round(time.time() - t0, 2)

    t0 = time.time()
    merged = merge_lora_params(train_state.params.to_pure_dict(), train_config.model)
    record["t_merge_sec"] = round(time.time() - t0, 2)

    del train_state
    with contextlib.suppress(Exception):
        import jax

        jax.clear_caches()
    gc.collect()

    t0 = time.time()
    task_scratch = scratch_root / task_name.replace(":", "_").replace("/", "_")
    if task_scratch.exists():
        shutil.rmtree(task_scratch)
    save_merged_jax_checkpoint(merged, task_scratch, base_ckpt=args.base_ckpt)
    del merged
    gc.collect()
    record["t_save_ckpt_sec"] = round(time.time() - t0, 2)

    t0 = time.time()
    eval_cfg = RolloutConfig(
        max_steps=args.max_steps,
        replan_steps=args.replan_steps,
        seed=args.seed,
        extra={"config_name": args.base_config},
    )
    eval_result = adapter.eval(str(task_scratch), task_name, num_episodes=args.eval_num_episodes, cfg=eval_cfg)
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

    if not args.keep_scratch:
        with contextlib.suppress(Exception):
            shutil.rmtree(task_scratch)

    record["status"] = "ok"
    return record


# ---- Main -----------------------------------------------------------------------


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)

    args = _fill_defaults(args)
    import torch

    torch.compile = lambda fn=None, *a, **kw: (fn if fn is not None else (lambda f: f))

    train_config = _override_config(_config.get_config(args.train_config), args)
    adapter = _make_adapter(args)

    in_process = args.env == "metaworld"
    base_policy = None
    if in_process:
        from openpi.models_pytorch.convert import ensure_pytorch_checkpoint

        base_train_config = _config.get_config(args.base_config)
        ensure_pytorch_checkpoint(args.base_ckpt, args.base_config)
        base_policy = _policy_config.create_trained_policy(base_train_config, args.base_ckpt)
        logger.info(f"Loaded base policy from {args.base_ckpt}")

    scratch_root: pathlib.Path | None = None
    if not in_process:
        if args.scratch_dir:
            scratch_root = pathlib.Path(args.scratch_dir)
            scratch_root.mkdir(parents=True, exist_ok=True)
        else:
            scratch_root = pathlib.Path(tempfile.mkdtemp(prefix=f"preference_bc_{args.env}_"))
        logger.info(f"Merged-ckpt scratch root: {scratch_root}")

    tasks = _resolve_tasks(args, adapter)
    logger.info(
        f"[{args.env}] Running Flow-DPO preference-BC on {len(tasks)} task(s): "
        f"{tasks[:5]}{'...' if len(tasks) > 5 else ''}"
    )

    results_json_path = pathlib.Path(args.results_json)
    results: dict = {
        "args": {
            "env": args.env,
            "base_ckpt": args.base_ckpt,
            "base_config": args.base_config,
            "train_config": args.train_config,
            "num_rollouts": args.num_rollouts,
            "num_train_steps": args.num_train_steps or train_config.num_train_steps,
            "batch_size": train_config.batch_size,
            "beta": args.beta,
            "eval_num_episodes": args.eval_num_episodes,
            "split": args.split,
            "seed": args.seed,
        },
        "tasks": {},
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _write_results(results_json_path, results)

    for task in tasks:
        try:
            if in_process:
                record = _run_one_task_inprocess(base_policy, task, args, train_config, adapter)
            else:
                record = _run_one_task_subprocess(task, args, train_config, adapter, scratch_root)
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
        _write_results(results_json_path, results)
        gc.collect()

    results["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _write_results(results_json_path, results)

    if (not in_process) and (not args.keep_scratch) and scratch_root is not None:
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
                f"pos={rec['num_pos_samples']}, neg={rec['num_neg_samples']}"
            )
        else:
            logger.info(f"  {task}: {status}")
    logger.info(f"Full results: {results_json_path}")


if __name__ == "__main__":
    tyro.cli(main)
