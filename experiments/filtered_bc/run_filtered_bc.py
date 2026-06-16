"""Filtered-BC baseline across MetaWorld, LIBERO, and RoboCasa.

Per task in the configured split:

1. Roll out N episodes with the base pi0.5 policy.
2. Filter to successful episodes.
3. LoRA fine-tune on the (obs, action_chunk) pairs (JAX).
4. Merge LoRA into base weights (numpy dict).
5. Evaluate the merged policy for ``--eval_num_episodes`` rollouts.
6. Free everything, move to the next task.

Two execution modes, dispatched by ``--args.env``:

- **MetaWorld** — in-process: the base Policy is loaded in this Python process,
  rollout + eval happen via :class:`MetaWorldAdapter`'s ``AsyncVectorEnv`` loop,
  the merged model is rebuilt into :class:`PI0Pytorch` on GPU and re-wrapped in
  a ``Policy`` for eval. No disk I/O beyond the incremental ``results.json``.

- **LIBERO / RoboCasa** — subprocess: their env libraries live in separate
  venvs, so the base policy runs in a ``scripts/serve_policy.py`` subprocess
  (root venv), and the rollout client runs in the respective ``examples/{env}``
  venv, talking over WebSocket. After merging, a new ckpt is written to a
  scratch dir and a second server subprocess serves it for eval.

Results are written incrementally to ``--args.results_json`` after every task,
so a mid-sweep crash still leaves partial data on disk.
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
from experiments.filtered_bc.envs.adapter import filter_successful
from experiments.filtered_bc.envs.adapter import get_adapter
from experiments.filtered_bc.merge_save import build_pytorch_model_from_merged
from experiments.filtered_bc.merge_save import merge_lora_params
from experiments.filtered_bc.merge_save import save_merged_jax_checkpoint
from experiments.filtered_bc.train import train_lora
from openpi import transforms as _transforms
from openpi.models import pi0_fast as _pi0_fast
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


def _is_pi0_fast(model_config) -> bool:
    """True for the pi0-FAST model family. pi0/pi0.5 return False."""
    return isinstance(model_config, _pi0_fast.Pi0FASTConfig)


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
    # --- Env to run against ---
    env: Literal["metaworld", "libero", "robocasa"] = "metaworld"

    # --- Input policy to adapt ---
    # If empty, filled from _DEFAULT_BASE_CKPT[env].
    base_ckpt: str = ""
    base_config: str = ""
    train_config: str = ""

    # --- Task selection ---
    tasks: list[str] = dataclasses.field(default_factory=list)
    split: Literal["train", "test"] = "train"
    # LIBERO only: which task suite to draw tasks from.
    libero_suite: str = "libero_spatial"
    # RoboCasa only: which task set + split.
    robocasa_task_set: str = "subset"
    robocasa_split: str = "pretrain"

    # --- Rollout budget (must match steering; default 15) ---
    num_rollouts: int = 15
    # None = env-specific default: 300 for MetaWorld, suite-specific for LIBERO,
    # ``1.5 * task_horizon`` for RoboCasa.
    max_steps: int | None = None
    replan_steps: int = 10
    seed: int = 69_420

    # --- Training ---
    num_train_steps: int | None = None
    batch_size: int | None = None
    skip_norm_stats: bool = False

    # --- Eval ---
    eval_num_episodes: int = 20

    # --- Scratch / output ---
    # Directory for server-based envs to stage merged ckpts. If empty, uses a
    # new mktemp per run.
    scratch_dir: str = ""
    keep_scratch: bool = False
    results_json: str = "experiments/filtered_bc/results.json"


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


def _write_results(results_json_path: pathlib.Path, results: dict) -> None:
    results_json_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = results_json_path.with_suffix(".json.tmp")
    with tmp.open("w") as f:
        json.dump(results, f, indent=2)
    tmp.replace(results_json_path)


def _build_policy_from_model(
    model,
    policy_train_config: _config.TrainConfig,
    base_ckpt: str,
) -> _policy.Policy:
    """Wrap a PI0Pytorch model in a Policy with the right transform stack."""
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


# ---- In-process run (MetaWorld) --------------------------------------------------


def _run_one_task_inprocess(
    base_policy,
    task_name: str,
    args: Args,
    train_config: _config.TrainConfig,
    adapter,
    scratch_root: pathlib.Path | None = None,
) -> dict:
    """MetaWorld: base policy is a live Policy, rollout + eval are in-process.

    Two eval-policy-build paths depending on model family:
      * pi0 / pi0.5: convert merged JAX params → PyTorch model in-process via
        :func:`build_pytorch_model_from_merged`, no disk I/O.
      * pi0-FAST: save merged JAX ckpt to ``scratch_root`` then reload via
        :func:`create_trained_policy(use_pytorch=False)`. The PyTorch converter only
        knows about :class:`Pi0Config`, so for pi0-FAST we keep everything in JAX.
    """
    import torch

    is_fast = _is_pi0_fast(train_config.model)
    record: dict = {"task": task_name, "status": "started"}
    rollout_cfg = RolloutConfig(
        max_steps=args.max_steps,
        replan_steps=args.replan_steps,
        seed=args.seed,
    )

    t0 = time.time()
    episodes = adapter.rollout(base_policy, task_name, num_episodes=args.num_rollouts, cfg=rollout_cfg)
    t_rollout = time.time() - t0
    num_success = sum(ep.success for ep in episodes)
    record.update(
        {
            "num_rollouts": len(episodes),
            "num_success_rollout": num_success,
            "rollout_success_rate": num_success / len(episodes),
            "t_rollout_sec": round(t_rollout, 2),
        }
    )
    logger.info(
        f"[{task_name}] rollout: {num_success}/{len(episodes)} successful "
        f"({100 * num_success / len(episodes):.0f}%), {t_rollout:.1f}s"
    )

    samples = filter_successful(episodes)
    record["num_train_samples"] = len(samples)
    if not samples:
        logger.warning(f"[{task_name}] 0 successful rollouts; skipping training + eval.")
        record["status"] = "skipped_no_successes"
        return record

    t0 = time.time()
    train_state = train_lora(
        train_config,
        samples,
        base_ckpt=args.base_ckpt,
        num_train_steps=args.num_train_steps,
        skip_norm_stats=args.skip_norm_stats,
    )
    record["t_train_sec"] = round(time.time() - t0, 2)

    t0 = time.time()
    merged = merge_lora_params(train_state.params.to_pure_dict(), train_config.model)
    record["t_merge_sec"] = round(time.time() - t0, 2)

    del train_state
    try:
        import jax

        jax.clear_caches()
    except Exception:
        pass

    # PyTorch base only: shuffle to CPU during eval to free GPU for the eval model.
    # (pi0-FAST keeps the base on GPU since both eval and base are JAX.)
    if not is_fast:
        try:
            base_policy._model = base_policy._model.to("cpu")  # noqa: SLF001
            if hasattr(base_policy, "_pytorch_device"):
                base_policy._pytorch_device = "cpu"  # noqa: SLF001
            torch.cuda.empty_cache()
        except Exception as exc:
            logger.warning(f"[{task_name}] couldn't move base policy to CPU: {exc}")
    gc.collect()

    policy_train_config = _config.get_config(args.base_config)
    task_scratch: pathlib.Path | None = None
    if is_fast:
        if scratch_root is None:
            raise RuntimeError("scratch_root is required for pi0-FAST in-process eval")
        t0 = time.time()
        task_scratch = scratch_root / task_name.replace(":", "_").replace("/", "_")
        if task_scratch.exists():
            shutil.rmtree(task_scratch)
        save_merged_jax_checkpoint(merged, task_scratch, base_ckpt=args.base_ckpt)
        del merged
        gc.collect()
        record["t_save_ckpt_sec"] = round(time.time() - t0, 2)

        t0 = time.time()
        policy = _policy_config.create_trained_policy(policy_train_config, str(task_scratch), use_pytorch=False)
        record["t_build_eval_sec"] = round(time.time() - t0, 2)
        pytorch_model = None
    else:
        t0 = time.time()
        pytorch_model = build_pytorch_model_from_merged(merged, train_config.model)
        del merged
        gc.collect()
        record["t_build_pt_sec"] = round(time.time() - t0, 2)
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

    del policy
    if pytorch_model is not None:
        del pytorch_model
    with contextlib.suppress(Exception):
        torch.cuda.empty_cache()
    if is_fast:
        try:
            import jax

            jax.clear_caches()
        except Exception:
            pass
    gc.collect()

    if is_fast and task_scratch is not None and not args.keep_scratch:
        with contextlib.suppress(Exception):
            shutil.rmtree(task_scratch)

    if not is_fast:
        try:
            base_policy._model = base_policy._model.to("cuda")  # noqa: SLF001
            if hasattr(base_policy, "_pytorch_device"):
                base_policy._pytorch_device = "cuda"  # noqa: SLF001
        except Exception as exc:
            logger.warning(f"[{task_name}] couldn't restore base policy to GPU: {exc}")

    record["status"] = "ok"
    return record


# ---- Subprocess run (LIBERO / RoboCasa) ------------------------------------------


def _run_one_task_subprocess(
    task_name: str,
    args: Args,
    train_config: _config.TrainConfig,
    adapter,
    scratch_root: pathlib.Path,
) -> dict:
    """LIBERO/RoboCasa: base policy lives in a server subprocess; eval ditto.

    Rollout spawns a server at args.base_ckpt + a client in the env's venv.
    After LoRA merge, we write a fresh JAX ckpt to scratch and launch a second
    server for eval.
    """
    record: dict = {"task": task_name, "status": "started"}
    rollout_cfg = RolloutConfig(
        max_steps=args.max_steps,
        replan_steps=args.replan_steps,
        seed=args.seed,
        extra={"config_name": args.base_config},
    )

    # 1. Rollout with base ckpt.
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
        f"[{task_name}] rollout: {num_success}/{len(episodes)} successful "
        f"({100 * num_success / len(episodes):.0f}%), {t_rollout:.1f}s"
    )

    samples = filter_successful(episodes)
    record["num_train_samples"] = len(samples)
    if not samples:
        logger.warning(f"[{task_name}] 0 successful rollouts; skipping training + eval.")
        record["status"] = "skipped_no_successes"
        return record

    # 2. LoRA train.
    t0 = time.time()
    train_state = train_lora(
        train_config,
        samples,
        base_ckpt=args.base_ckpt,
        num_train_steps=args.num_train_steps,
        skip_norm_stats=args.skip_norm_stats,
    )
    record["t_train_sec"] = round(time.time() - t0, 2)

    # 3. Merge LoRA → numpy dict.
    t0 = time.time()
    merged = merge_lora_params(train_state.params.to_pure_dict(), train_config.model)
    record["t_merge_sec"] = round(time.time() - t0, 2)

    del train_state
    try:
        import jax

        jax.clear_caches()
    except Exception:
        pass
    gc.collect()

    # 4. Write merged ckpt to scratch so the next server can load it.
    t0 = time.time()
    task_scratch = scratch_root / task_name.replace(":", "_").replace("/", "_")
    if task_scratch.exists():
        shutil.rmtree(task_scratch)
    save_merged_jax_checkpoint(merged, task_scratch, base_ckpt=args.base_ckpt)
    del merged
    gc.collect()
    record["t_save_ckpt_sec"] = round(time.time() - t0, 2)

    # 5. Eval the merged ckpt via a fresh server.
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

    # 6. Clean up scratch ckpt unless asked to keep it.
    if not args.keep_scratch:
        with contextlib.suppress(Exception):
            shutil.rmtree(task_scratch)

    record["status"] = "ok"
    return record


# ---- Main -------------------------------------------------------------------------


def main(args: Args) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        force=True,
    )

    args = _fill_defaults(args)
    import torch

    # torch.compile's max-autotune costs ~5 min of first-call autotuning; disable so
    # the per-task overhead stays predictable.
    torch.compile = lambda fn=None, *a, **kw: (fn if fn is not None else (lambda f: f))

    train_config = _override_config(_config.get_config(args.train_config), args)
    adapter = _make_adapter(args)

    is_fast = _is_pi0_fast(train_config.model)

    # In-process envs (just MetaWorld for now) preload the base policy; subprocess
    # envs don't — the base server is launched per-task.
    in_process = args.env == "metaworld"
    base_policy = None
    if in_process:
        base_train_config = _config.get_config(args.base_config)
        if is_fast:
            # pi0-FAST has no PyTorch converter (the existing one is pi0-only), so
            # load directly as JAX. eval is also in-process JAX, see _run_one_task_inprocess.
            base_policy = _policy_config.create_trained_policy(base_train_config, args.base_ckpt, use_pytorch=False)
            logger.info(f"Loaded base pi0-FAST policy (JAX) from {args.base_ckpt}")
        else:
            from openpi.models_pytorch.convert import ensure_pytorch_checkpoint

            ensure_pytorch_checkpoint(args.base_ckpt, args.base_config)
            base_policy = _policy_config.create_trained_policy(base_train_config, args.base_ckpt)
            logger.info(f"Loaded base policy (PyTorch) from {args.base_ckpt}")

    # Scratch root for merged ckpts. Subprocess envs always need it. In-process envs
    # only need it for pi0-FAST (PyTorch flow keeps everything in memory).
    scratch_root: pathlib.Path | None = None
    needs_scratch = (not in_process) or is_fast
    if needs_scratch:
        if args.scratch_dir:
            scratch_root = pathlib.Path(args.scratch_dir)
            scratch_root.mkdir(parents=True, exist_ok=True)
        else:
            scratch_root = pathlib.Path(tempfile.mkdtemp(prefix=f"filtered_bc_{args.env}_"))
        logger.info(f"Merged-ckpt scratch root: {scratch_root}")

    tasks = _resolve_tasks(args, adapter)
    logger.info(
        f"[{args.env}] Running filtered-BC on {len(tasks)} task(s): {tasks[:5]}{'...' if len(tasks) > 5 else ''}"
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
                record = _run_one_task_inprocess(
                    base_policy, task, args, train_config, adapter, scratch_root=scratch_root
                )
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

    if needs_scratch and (not args.keep_scratch) and scratch_root is not None:
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
    logger.info(f"Full results: {results_json_path}")


if __name__ == "__main__":
    tyro.cli(main)
