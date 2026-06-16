"""LIBERO adapter: server-client rollout + eval against a pi05 policy server.

LIBERO's env library lives in its own Python 3.8 venv at ``examples/libero_env/``
(incompatible with the root venv's JAX stack), so the filtered-BC flow can't run
everything in-process the way MetaWorld does. Instead:

1. Spawn a policy server in the root venv pointing at a checkpoint dir.
2. Spawn a rollout client in the libero_env venv that talks to the server.
3. Client records (obs, action_chunk) pairs per replan and pickles them.
4. Adapter re-hydrates the pickle into :class:`InferenceSample` / :class:`EpisodeRollout`.
5. Kill the server.

``policy_or_ckpt`` for this adapter is always a **checkpoint directory path**
(``str`` or ``pathlib.Path``) — never a live Policy — because the policy has to
live in a separate process (the server).
"""

from __future__ import annotations

from collections.abc import Sequence
import logging
import os
import pathlib
import pickle
import socket
import subprocess
import tempfile
import time

import numpy as np

from experiments.filtered_bc.envs.adapter import EpisodeRollout
from experiments.filtered_bc.envs.adapter import EvalResult
from experiments.filtered_bc.envs.adapter import InferenceSample
from experiments.filtered_bc.envs.adapter import RolloutConfig

logger = logging.getLogger(__name__)

# Repo root: .../worktrees/rl-integration (experiments/filtered_bc/envs/libero.py)
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_LIBERO_ENV_DIR = _REPO_ROOT / "examples" / "libero_env"
_FILTERED_BC_CLIENT = _LIBERO_ENV_DIR / "filtered_bc_client.py"

# Suites we enumerate by default; adapter users can override via RolloutConfig.extra.
_DEFAULT_SUITE = "libero_spatial"

# How many tasks each suite has (matches what benchmark_dict[suite].n_tasks reports).
_SUITE_SIZES = {
    "libero_spatial": 10,
    "libero_object": 10,
    "libero_goal": 10,
    "libero_10": 10,
    "libero_90": 90,
}

_SERVER_STARTUP_TIMEOUT_S = 600  # pi0.5 takes 1-2 min to cold-load params on GPU
_SERVER_SHUTDOWN_TIMEOUT_S = 30


def _config_is_pi0_fast(config_name: str) -> bool:
    """True if the named TrainConfig builds a Pi0FASTConfig model."""
    from openpi.models import pi0_fast as _pi0_fast
    from openpi.training import config as _config

    return isinstance(_config.get_config(config_name).model, _pi0_fast.Pi0FASTConfig)


def _pick_free_port() -> int:
    """Bind to port 0 and read back what the OS handed us, then release."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_server(port: int, timeout_s: float, proc: subprocess.Popen) -> None:
    """Poll localhost:port until the server accepts TCP connections, or timeout."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"Policy server exited early (code={proc.returncode}) — inspect its stderr.")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1.0):
                logger.info(f"Policy server ready on port {port}")
                return
        except OSError:
            time.sleep(2.0)
    raise TimeoutError(f"Policy server didn't accept connections on port {port} within {timeout_s}s")


def _launch_server(
    ckpt_dir: str | pathlib.Path,
    config_name: str,
    port: int,
    log_path: pathlib.Path,
    *,
    use_pytorch: bool = True,
) -> subprocess.Popen:
    """Start a policy server subprocess in the root venv. Caller owns shutdown.

    ``use_pytorch=False`` is required for pi0-FAST (no PyTorch port for the
    autoregressive decode); pi0/pi0.5 use the PyTorch path so the
    ``torch.compile`` no-op shim still applies.
    """
    cmd = [
        "uv",
        "run",
        "python",
        "-u",
        "experiments/filtered_bc/_serve_policy_nocompile.py",
        f"--port={port}",
        "policy:checkpoint",
        f"--policy.config={config_name}",
        f"--policy.dir={ckpt_dir}",
    ]
    if use_pytorch:
        cmd.insert(5, "--pytorch")
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
    task_suite_name: str,
    task_id: int,
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
    """Invoke the LIBERO rollout client in its own venv. Blocks until done.

    ``max_steps=None`` means "use the client's per-suite default from SUITE_MAX_STEPS".
    """
    cmd = [
        "uv",
        "run",
        "python",
        "-u",
        "filtered_bc_client.py",
        "--host=127.0.0.1",
        f"--port={port}",
        f"--task-suite-name={task_suite_name}",
        f"--task-id={task_id}",
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
    logger.info(f"Launching LIBERO client → log: {log_path}; cmd (from {_LIBERO_ENV_DIR}): {' '.join(cmd)}")
    with log_path.open("w") as f:
        result = subprocess.run(cmd, cwd=_LIBERO_ENV_DIR, env=env, stdout=f, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"LIBERO client failed with code {result.returncode}. See log: {log_path}")


def _parse_task(task: str) -> tuple[str, int]:
    if ":" not in task:
        raise ValueError(f"LIBERO task name must be '<suite>:<id>', got {task!r}. Example: 'libero_spatial:0'.")
    suite, id_str = task.split(":", 1)
    try:
        tid = int(id_str)
    except ValueError as exc:
        raise ValueError(f"Invalid task_id {id_str!r} in {task!r}") from exc
    return suite, tid


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


class LiberoAdapter:
    """Server-client LIBERO adapter. Expects a checkpoint directory path."""

    name = "libero"
    training_config = "pi05_libero_low_mem_finetune"
    base_config = "pi05_libero"

    def __init__(self, task_suite_name: str = _DEFAULT_SUITE):
        self.task_suite_name = task_suite_name

    @property
    def train_tasks(self) -> Sequence[str]:
        n = _SUITE_SIZES.get(self.task_suite_name)
        if n is None:
            raise ValueError(f"Unknown LIBERO suite: {self.task_suite_name}")
        return [f"{self.task_suite_name}:{i}" for i in range(n)]

    @property
    def test_tasks(self) -> Sequence[str]:
        # LIBERO has no canonical train/test split per suite — reuse train_tasks.
        return self.train_tasks

    def _run_one(
        self,
        ckpt_dir: str | pathlib.Path,
        task_name: str,
        num_episodes: int,
        cfg: RolloutConfig,
        *,
        eval_only: bool,
    ) -> list[EpisodeRollout]:
        suite, tid = _parse_task(task_name)
        # Eval offsets the seed so held-out rollouts are disjoint from training.
        # LIBERO-specific: the client picks initial states via ``(seed + ep) % num_init_states``
        # (typically num_init_states=50), so the offset must be coprime with (or at least
        # not a multiple of) num_init_states. 10_000 % 50 == 0 would alias rollout and
        # eval to identical init states — use ``num_episodes`` instead, which keeps the
        # rollout window [seed, seed+num_episodes) disjoint from the eval window
        # [seed+num_episodes, seed+2*num_episodes) modulo num_init_states as long as
        # 2 * num_episodes <= num_init_states (50 for all standard LIBERO suites).
        seed = cfg.seed + (num_episodes if eval_only else 0)
        config_name = cfg.extra.get("config_name", self.base_config)

        port = _pick_free_port()
        scratch = pathlib.Path(tempfile.mkdtemp(prefix="filtered_bc_libero_"))
        server_log = scratch / "server.log"
        client_log = scratch / "client.log"
        samples_path = scratch / "samples.pkl"

        # pi0-FAST has no PyTorch port for the autoregressive decode, so its server
        # must serve the JAX checkpoint directly (no --pytorch flag).
        use_pytorch = not _config_is_pi0_fast(config_name)
        server = _launch_server(ckpt_dir, config_name, port, server_log, use_pytorch=use_pytorch)
        try:
            _wait_for_server(port, _SERVER_STARTUP_TIMEOUT_S, server)
            _run_client(
                task_suite_name=suite,
                task_id=tid,
                num_episodes=num_episodes,
                port=port,
                samples_out=samples_path,
                max_steps=cfg.max_steps,
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


__all__ = ["LiberoAdapter"]
