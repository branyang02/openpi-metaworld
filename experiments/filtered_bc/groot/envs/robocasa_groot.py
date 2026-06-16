"""RoboCasa adapter for filtered-BC against a GR00T N1.5 policy server.

Mirrors :mod:`experiments.filtered_bc.envs.robocasa` but launches
``groot_env/serve.py`` (separate Python 3.10 venv, ``torch==2.5.1`` pinned)
instead of ``scripts/serve_policy.py --pytorch``. The robocasa client itself is
unchanged — it speaks the same WebSocket protocol both backends share.

We re-implement the full launch / shutdown / client loop here (rather than
extending the pi05 adapter) so the GR00T flow stays self-contained: orchestrator
state, scratch dirs, log files, and exception paths all live next to each other,
mirroring the structure the orchestrator (`run_filtered_bc_groot.py`) expects.
"""

from __future__ import annotations

from collections.abc import Sequence
import logging
import os
import pathlib
import pickle
import socket
import subprocess
import time

import numpy as np

from experiments.filtered_bc.envs.adapter import EpisodeRollout
from experiments.filtered_bc.envs.adapter import EvalResult
from experiments.filtered_bc.envs.adapter import InferenceSample
from experiments.filtered_bc.envs.adapter import RolloutConfig

logger = logging.getLogger(__name__)

# Worktree root (where the orchestrator + scripts live).
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
# groot_env's venv (with flash-attn installed) lives at the **project** root —
# worktrees share git state but each has its own venvs, and only the project-root
# groot_env has been fully set up (uv sync + flash-attn). All worktrees point
# here; the README and groot-env rule both place groot_env at the project root.
# robocasa_env's venv is set up per-worktree; use the worktree-local copy.
_GROOT_ENV_DIR = pathlib.Path(os.environ.get(
    "OPENPI_GROOT_ENV_DIR",
    "/home/kim34/projects_brandon/openpi-metaworld/groot_env",
))
_ROBOCASA_ENV_DIR = _REPO_ROOT / "examples" / "robocasa_env"
_FILTERED_BC_CLIENT = _ROBOCASA_ENV_DIR / "filtered_bc_client.py"

# GR00T N1.5 cold-loads its 7B-param backbone (~10s with warm cache, ~1 min
# cold). Match the libero adapter's generous default to absorb cluster jitter.
_SERVER_STARTUP_TIMEOUT_S = 600
_SERVER_SHUTDOWN_TIMEOUT_S = 30


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_server(port: int, timeout_s: float, proc: subprocess.Popen) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"GR00T server exited early (code={proc.returncode}) — see its log.")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1.0):
                logger.info(f"GR00T server ready on port {port}")
                return
        except OSError:
            time.sleep(2.0)
    raise TimeoutError(f"GR00T server didn't accept connections on port {port} within {timeout_s}s.")


def _launch_server(
    ckpt_dir: str | pathlib.Path,
    port: int,
    log_path: pathlib.Path,
    *,
    embodiment: str = "robocasa",
    denoising_steps: int = 4,
) -> subprocess.Popen:
    """Start groot_env/serve.py pointing at ``ckpt_dir``. Caller owns shutdown.

    serve.py interprets ``--model-path`` as relative to its own cwd
    (``groot_env/``), so we always pass an absolute path.
    """
    cmd = [
        "uv",
        "run",
        "python",
        "-u",
        "serve.py",
        f"--model-path={pathlib.Path(ckpt_dir).resolve()}",
        f"--embodiment={embodiment}",
        f"--port={port}",
        f"--denoising-steps={denoising_steps}",
        "--device=cuda:0",
    ]
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env["PYTHONUNBUFFERED"] = "1"
    logger.info(f"Launching GR00T server → log: {log_path}; cmd (from {_GROOT_ENV_DIR}): {' '.join(cmd)}")
    f = log_path.open("w")
    return subprocess.Popen(cmd, cwd=_GROOT_ENV_DIR, env=env, stdout=f, stderr=subprocess.STDOUT)


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
    # Always pass absolute paths to the client — its cwd is examples/robocasa_env/.
    abs_samples_out = pathlib.Path(samples_out).resolve()
    abs_log_path = pathlib.Path(log_path).resolve()
    abs_samples_out.parent.mkdir(parents=True, exist_ok=True)
    abs_log_path.parent.mkdir(parents=True, exist_ok=True)
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
        f"--samples-out={abs_samples_out}",
    ]
    if max_steps is not None:
        cmd.append(f"--max-steps={max_steps}")
    cmd.append("--eval-only" if eval_only else "--no-eval-only")
    env = os.environ.copy()
    env.setdefault("MUJOCO_GL", "egl")
    env["PYTHONUNBUFFERED"] = "1"
    logger.info(f"Launching RoboCasa client → log: {abs_log_path}; cmd (from {_ROBOCASA_ENV_DIR}): {' '.join(cmd)}")
    with abs_log_path.open("w") as f:
        result = subprocess.run(cmd, cwd=_ROBOCASA_ENV_DIR, env=env, stdout=f, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"RoboCasa client failed with code {result.returncode}. See log: {abs_log_path}")


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


# Default 7-task subset matches groot_env/README.md and
# examples/robocasa_env/eval_all.py:SUBSET — same tasks the GR00T base eval ran on.
_DEFAULT_SUBSET = [
    "CloseFridge",
    "CoffeeSetupMug",
    "OpenDrawer",
    "OpenStandMixerHead",
    "PickPlaceCounterToCabinet",
    "PickPlaceCounterToStove",
    "TurnOnElectricKettle",
]


class GrootRoboCasaAdapter:
    """Server-client RoboCasa adapter targeting a ``groot_env/serve.py`` process.

    Methods accept the same shape as :class:`RoboCasaAdapter` (env_name strings;
    ``policy_or_ckpt`` is the absolute path to a checkpoint dir loadable by
    serve.py — base or filtered-BC-merged) so the orchestrator can use the same
    rollout/eval contract.
    """

    name = "robocasa_groot"

    def __init__(
        self,
        tasks: Sequence[str] | None = None,
        split: str = "pretrain",
        denoising_steps: int = 4,
    ):
        self._tasks = list(tasks) if tasks else list(_DEFAULT_SUBSET)
        self.split = split
        self.denoising_steps = denoising_steps

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
        scratch: pathlib.Path,
        *,
        eval_only: bool,
    ) -> list[EpisodeRollout]:
        seed = cfg.seed + (10_000 if eval_only else 0)
        split = cfg.extra.get("split", self.split)
        max_steps = cfg.max_steps if cfg.max_steps is not None else cfg.extra.get("max_steps")

        port = _pick_free_port()
        scratch.mkdir(parents=True, exist_ok=True)
        suffix = "eval" if eval_only else "rollout"
        server_log = scratch / f"server-{suffix}.log"
        client_log = scratch / f"client-{suffix}.log"
        samples_path = scratch / f"samples-{suffix}.pkl"

        server = _launch_server(ckpt_dir, port, server_log, denoising_steps=self.denoising_steps)
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

        # _run_client resolves samples_out to absolute under the hood, so the
        # original (possibly relative) Path may not exist; round-trip through
        # resolve() to read the file the client actually wrote.
        with samples_path.resolve().open("rb") as f:
            payload = pickle.load(f)
        return _hydrate_rollouts(payload, task_name)

    def rollout(
        self,
        ckpt_dir: str | pathlib.Path,
        task_name: str,
        num_episodes: int,
        cfg: RolloutConfig,
        scratch: pathlib.Path,
    ) -> list[EpisodeRollout]:
        return self._run_one(ckpt_dir, task_name, num_episodes, cfg, scratch, eval_only=False)

    def eval(
        self,
        ckpt_dir: str | pathlib.Path,
        task_name: str,
        num_episodes: int,
        cfg: RolloutConfig,
        scratch: pathlib.Path,
    ) -> EvalResult:
        rollouts = self._run_one(ckpt_dir, task_name, num_episodes, cfg, scratch, eval_only=True)
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


__all__ = ["GrootRoboCasaAdapter"]
