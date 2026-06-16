"""RoboCasa adapter: server-client rollout + eval against a pi05 policy server.

Mirrors :mod:`experiments.filtered_bc.envs.libero` but spawns the client in
``examples/robocasa_env/`` instead and enumerates task names from
``robocasa.utils.dataset_registry.TASK_SET_REGISTRY``.

``policy_or_ckpt`` must be a checkpoint directory path; the policy itself lives
in the server subprocess.
"""

from __future__ import annotations

from collections.abc import Sequence
import importlib
import logging
import os
import pathlib
import pickle
import socket
import subprocess
import sys
import tempfile
import time

import numpy as np

from experiments.filtered_bc.envs.adapter import EpisodeRollout
from experiments.filtered_bc.envs.adapter import EvalResult
from experiments.filtered_bc.envs.adapter import InferenceSample
from experiments.filtered_bc.envs.adapter import RolloutConfig

logger = logging.getLogger(__name__)

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_ROBOCASA_ENV_DIR = _REPO_ROOT / "examples" / "robocasa_env"
_ROBOCASA_THIRDPARTY = _REPO_ROOT / "third_party" / "robocasa"

# Fallback task list if TASK_SET_REGISTRY can't be imported from the root venv
# (the robocasa package isn't installed there — only in examples/robocasa_env/'s venv).
# This matches the "subset" curated list in examples/robocasa_env/eval_all.py.
_SUBSET_TASKS = [
    "CloseFridge",
    "CoffeeSetupMug",
    "OpenDrawer",
    "OpenStandMixerHead",
    "PickPlaceCounterToCabinet",
    "PickPlaceCounterToStove",
    "TurnOnElectricKettle",
]

_SERVER_STARTUP_TIMEOUT_S = 600  # pi0.5 takes 1-2 min to cold-load params on GPU
_SERVER_SHUTDOWN_TIMEOUT_S = 30


def _load_task_set(task_set: str) -> list[str]:
    """Read a TASK_SET_REGISTRY entry by importing directly from the third_party dir."""
    if task_set == "subset":
        return list(_SUBSET_TASKS)
    registry_path = _ROBOCASA_THIRDPARTY / "robocasa" / "utils" / "dataset_registry.py"
    sys.path.insert(0, str(_ROBOCASA_THIRDPARTY))
    try:
        mod = importlib.util.spec_from_file_location("robocasa_dataset_registry", registry_path)
        module = importlib.util.module_from_spec(mod)
        mod.loader.exec_module(module)
        if task_set not in module.TASK_SET_REGISTRY:
            raise ValueError(
                f"Unknown RoboCasa task_set {task_set!r}. Available: {sorted(module.TASK_SET_REGISTRY.keys())}"
            )
        return list(module.TASK_SET_REGISTRY[task_set])
    finally:
        sys.path.pop(0)


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_server(port: int, timeout_s: float, proc: subprocess.Popen) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"Policy server exited early (code={proc.returncode}).")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1.0):
                logger.info(f"Policy server ready on port {port}")
                return
        except OSError:
            time.sleep(2.0)
    raise TimeoutError(f"Server didn't accept connections on port {port} within {timeout_s}s.")


def _launch_server(
    ckpt_dir: str | pathlib.Path,
    config_name: str,
    port: int,
    log_path: pathlib.Path,
) -> subprocess.Popen:
    cmd = [
        "uv",
        "run",
        "python",
        "-u",
        "experiments/filtered_bc/_serve_policy_nocompile.py",
        "--pytorch",
        f"--port={port}",
        "policy:checkpoint",
        f"--policy.config={config_name}",
        f"--policy.dir={ckpt_dir}",
    ]
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    env.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
    env["PYTHONUNBUFFERED"] = "1"
    logger.info(f"Launching policy server → log: {log_path}; cmd: {' '.join(cmd)}")
    f = log_path.open("w")
    return subprocess.Popen(cmd, cwd=_REPO_ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)


def _shutdown_server(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=_SERVER_SHUTDOWN_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        logger.warning("Server didn't SIGTERM in time; SIGKILLing.")
        proc.kill()
        proc.wait()


def _run_client(
    env_name: str,
    split: str,
    num_episodes: int,
    port: int,
    samples_out: pathlib.Path,
    max_steps: int | None,
    replan_steps: int,
    seed: int,
    *,
    eval_only: bool,
    log_path: pathlib.Path,
) -> None:
    cmd = [
        "uv",
        "run",
        "python",
        "-u",
        "filtered_bc_client.py",
        "--host=127.0.0.1",
        f"--port={port}",
        f"--env-name={env_name}",
        f"--split={split}",
        f"--num-episodes={num_episodes}",
        f"--replan-steps={replan_steps}",
        f"--seed={seed}",
        f"--samples-out={samples_out}",
    ]
    if max_steps is not None:
        cmd.append(f"--max-steps={max_steps}")
    cmd.append("--eval-only" if eval_only else "--no-eval-only")
    env = os.environ.copy()
    env.setdefault("MUJOCO_GL", "egl")
    env["PYTHONUNBUFFERED"] = "1"
    logger.info(f"Launching RoboCasa client → log: {log_path}; cmd (from {_ROBOCASA_ENV_DIR}): {' '.join(cmd)}")
    with log_path.open("w") as f:
        result = subprocess.run(cmd, cwd=_ROBOCASA_ENV_DIR, env=env, stdout=f, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"RoboCasa client failed with code {result.returncode}. See log: {log_path}")


def _hydrate_rollouts(payload: dict, task_name: str) -> list[EpisodeRollout]:
    rollouts: list[EpisodeRollout] = []
    for ep in payload["episodes"]:
        samples = [
            InferenceSample(
                image=np.asarray(s["image"], dtype=np.uint8),
                wrist_image=np.asarray(s["wrist_image"], dtype=np.uint8),
                state=np.asarray(s["state"], dtype=np.float32),
                prompt=str(s["prompt"]),
                action_chunk=np.asarray(s["action_chunk"], dtype=np.float32),
            )
            for s in ep["samples"]
        ]
        rollouts.append(
            EpisodeRollout(
                task_name=task_name,
                env_id=int(ep["env_id"]),
                success=bool(ep["success"]),
                total_reward=float(ep["total_reward"]),
                steps_to_success=int(ep["steps_to_success"]),
                total_env_steps=int(ep["total_env_steps"]),
                samples=samples,
            )
        )
    return rollouts


class RoboCasaAdapter:
    """Server-client RoboCasa adapter. Expects a checkpoint directory path."""

    name = "robocasa"
    training_config = "pi05_robocasa_low_mem_finetune"
    base_config = "pi05_robocasa"

    def __init__(self, task_set: str = "subset", split: str = "pretrain"):
        self.task_set = task_set
        self.split = split
        self._tasks = _load_task_set(task_set)

    @property
    def train_tasks(self) -> Sequence[str]:
        return list(self._tasks)

    @property
    def test_tasks(self) -> Sequence[str]:
        return list(self._tasks)

    def _run_one(
        self,
        ckpt_dir: str | pathlib.Path,
        task_name: str,
        num_episodes: int,
        cfg: RolloutConfig,
        *,
        eval_only: bool,
    ) -> list[EpisodeRollout]:
        seed = cfg.seed + (10_000 if eval_only else 0)
        config_name = cfg.extra.get("config_name", self.base_config)
        split = cfg.extra.get("split", self.split)
        # Honor ``RolloutConfig.max_steps`` (typed field) before falling back to the
        # ``extra`` override, before finally leaving ``None`` so the client uses
        # ``1.5 * task_horizon`` as the default.
        max_steps = cfg.max_steps if cfg.max_steps is not None else cfg.extra.get("max_steps")

        port = _pick_free_port()
        scratch = pathlib.Path(tempfile.mkdtemp(prefix="filtered_bc_robocasa_"))
        server_log = scratch / "server.log"
        client_log = scratch / "client.log"
        samples_path = scratch / "samples.pkl"

        server = _launch_server(ckpt_dir, config_name, port, server_log)
        try:
            _wait_for_server(port, _SERVER_STARTUP_TIMEOUT_S, server)
            _run_client(
                env_name=task_name,
                split=split,
                num_episodes=num_episodes,
                port=port,
                samples_out=samples_path,
                max_steps=max_steps,
                replan_steps=cfg.replan_steps,
                seed=seed,
                eval_only=eval_only,
                log_path=client_log,
            )
        finally:
            _shutdown_server(server)

        with samples_path.open("rb") as f:
            payload = pickle.load(f)
        return _hydrate_rollouts(payload, task_name)

    def rollout(
        self,
        policy_or_ckpt,
        task_name: str,
        num_episodes: int,
        cfg: RolloutConfig | None = None,
    ) -> list[EpisodeRollout]:
        if cfg is None:
            cfg = RolloutConfig()
        return self._run_one(policy_or_ckpt, task_name, num_episodes, cfg, eval_only=False)

    def eval(
        self,
        policy_or_ckpt,
        task_name: str,
        num_episodes: int,
        cfg: RolloutConfig | None = None,
    ) -> EvalResult:
        if cfg is None:
            cfg = RolloutConfig()
        rollouts = self._run_one(policy_or_ckpt, task_name, num_episodes, cfg, eval_only=True)
        n_succ = sum(1 for r in rollouts if r.success)
        rewards = [r.total_reward for r in rollouts]
        succ_steps = [r.steps_to_success for r in rollouts if r.steps_to_success >= 0]
        return EvalResult(
            task_name=task_name,
            num_episodes=len(rollouts),
            num_success=n_succ,
            success_rate=n_succ / len(rollouts) if rollouts else 0.0,
            mean_reward=float(np.mean(rewards)) if rewards else 0.0,
            mean_steps_to_success=float(np.mean(succ_steps)) if succ_steps else float("nan"),
        )


__all__ = ["RoboCasaAdapter"]
