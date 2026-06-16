"""Filtered-BC LIBERO rollout/eval client.

Thin fork of ``main.py``'s episode loop that talks to a running policy server and
records (obs, action_chunk) pairs at each replan boundary. Writes a single pickle
file at ``--samples-out`` so the filtered-BC orchestrator (which runs in a
different venv) can re-hydrate rollouts without importing LIBERO.

This file MUST stay Python 3.8 compatible (``from typing import`` instead of PEP
585, no ``match``, no ``X | None`` union syntax) because it runs in the isolated
``examples/libero_env/`` venv.

Usage (invoked by LiberoAdapter; not typically run by hand):

    MUJOCO_GL=egl uv run python filtered_bc_client.py \\
        --host localhost --port 8123 \\
        --task_suite_name libero_spatial --task_id 0 \\
        --num_episodes 3 --samples_out /tmp/rollouts.pkl
"""

from __future__ import annotations

import collections
import dataclasses
import logging
import math
import os
import pathlib
import pickle
from typing import Any, Deque, Dict, List, Optional

import numpy as np
import tyro
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
from tqdm import tqdm

logger = logging.getLogger(__name__)

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
CAMERA_KEYS = {
    "agentview": "agentview_image",
    "eye_in_hand": "robot0_eye_in_hand_image",
}
SUITE_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000

    task_suite_name: str = "libero_spatial"
    task_id: int = 0

    num_episodes: int = 3
    num_steps_wait: int = 10
    max_steps: Optional[int] = None
    replan_steps: int = 5
    resize_size: int = 224
    seed: int = 69_420

    samples_out: str = "/tmp/filtered_bc_libero_samples.pkl"
    eval_only: bool = False


# --- Helpers lifted verbatim from main.py -------------------------------------------


def _get_task_suite(task_suite_name: str):
    benchmark_dict = benchmark.get_benchmark_dict()
    if task_suite_name not in benchmark_dict:
        raise ValueError(
            "Unknown task_suite_name {!r}. Available: {}".format(
                task_suite_name, sorted(benchmark_dict.keys())
            )
        )
    return benchmark_dict[task_suite_name]()


def _make_env(task, resolution: int, seed: int) -> OffScreenRenderEnv:
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    env = OffScreenRenderEnv(
        bddl_file_name=str(task_bddl_file),
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env


def _rotate(image: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(image[::-1, ::-1])


def _quat_to_axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.array(quat, dtype=np.float32, copy=True)
    quat[3] = float(np.clip(quat[3], -1.0, 1.0))
    denominator = float(np.sqrt(1.0 - quat[3] * quat[3]))
    if math.isclose(denominator, 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * 2.0 * math.acos(float(quat[3])) / denominator).astype(np.float32)


def _build_state(obs: Dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate(
        [
            obs["robot0_eef_pos"],
            _quat_to_axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        ],
        axis=0,
    ).astype(np.float32)


def _prepare_policy_inputs(
    obs: Dict[str, np.ndarray], resize_size: int
) -> Dict[str, np.ndarray]:
    base_image = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(
            _rotate(obs[CAMERA_KEYS["agentview"]]), resize_size, resize_size
        )
    )
    wrist_image = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(
            _rotate(obs[CAMERA_KEYS["eye_in_hand"]]), resize_size, resize_size
        )
    )
    return {
        "observation/image": base_image,
        "observation/wrist_image": wrist_image,
        "observation/state": _build_state(obs),
    }


def _get_max_steps(task_suite_name: str, max_steps: Optional[int]) -> int:
    if max_steps is not None:
        return max_steps
    if task_suite_name not in SUITE_MAX_STEPS:
        raise ValueError(
            "No default max_steps registered for task suite {!r}".format(
                task_suite_name
            )
        )
    return SUITE_MAX_STEPS[task_suite_name]


# --- The rollout loop ---------------------------------------------------------------


def run_task(args: Args) -> Dict[str, Any]:
    """Run num_episodes rollouts, return a dict with episodes + their (obs, action_chunk) samples."""
    policy = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    logger.info("Server metadata: %s", policy.get_server_metadata())

    task_suite = _get_task_suite(args.task_suite_name)
    if args.task_id < 0 or args.task_id >= task_suite.n_tasks:
        raise ValueError(
            "task_id must be in [0, {}), got {}".format(
                task_suite.n_tasks, args.task_id
            )
        )
    task = task_suite.get_task(args.task_id)
    task_name = getattr(task, "name", "task_{:02d}".format(args.task_id))
    task_description = str(task.language)
    initial_states = task_suite.get_task_init_states(args.task_id)
    if args.num_episodes > len(initial_states):
        raise ValueError(
            "Requested {} episodes but only {} initial states available".format(
                args.num_episodes, len(initial_states)
            )
        )

    max_steps = _get_max_steps(args.task_suite_name, args.max_steps)
    env = _make_env(task, LIBERO_ENV_RESOLUTION, args.seed)

    # --seed also acts as an offset into LIBERO's canonical initial-state list so that
    # different seeds evaluate on disjoint start conditions. Mirrors the fix in
    # examples/libero_env/main.py from PR #48. Without this, rollout and eval runs
    # (which pass different --seed) would silently land on the same initial states,
    # because LIBERO's init-state selection would only depend on the loop index.
    num_init_states = len(initial_states)

    episodes: List[Dict[str, Any]] = []
    try:
        for episode in range(args.num_episodes):
            state_idx = (args.seed + episode) % num_init_states
            env.reset()
            obs = env.set_init_state(initial_states[state_idx])
            action_plan: Deque[np.ndarray] = collections.deque()
            success = False
            total_reward = 0.0
            steps_to_success = -1
            samples: List[Dict[str, Any]] = []

            pbar = tqdm(
                range(args.num_steps_wait + max_steps),
                desc="[{}] episode {}/{}".format(
                    task_name, episode + 1, args.num_episodes
                ),
                leave=False,
            )
            step = 0
            rollout_step = 0
            for step in pbar:
                if step < args.num_steps_wait:
                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                    continue
                rollout_step = step - args.num_steps_wait

                if not action_plan:
                    element = _prepare_policy_inputs(obs, args.resize_size)
                    element["prompt"] = task_description

                    action_chunk = np.asarray(
                        policy.infer(element)["actions"], dtype=np.float32
                    )
                    if action_chunk.ndim != 2:
                        raise ValueError(
                            "Server must return (H, D) chunk, got shape {}".format(
                                action_chunk.shape
                            )
                        )
                    if action_chunk.shape[0] < args.replan_steps:
                        raise ValueError(
                            "Server must return >= {} actions, got {}".format(
                                args.replan_steps, action_chunk.shape[0]
                            )
                        )

                    if (not args.eval_only) and (not success):
                        samples.append(
                            {
                                "image": np.asarray(
                                    element["observation/image"], dtype=np.uint8
                                ).copy(),
                                "wrist_image": np.asarray(
                                    element["observation/wrist_image"], dtype=np.uint8
                                ).copy(),
                                "state": np.asarray(
                                    element["observation/state"], dtype=np.float32
                                ).copy(),
                                "prompt": task_description,
                                "action_chunk": action_chunk.copy(),
                            }
                        )
                    action_plan.extend(action_chunk[: args.replan_steps])

                action = action_plan.popleft()
                obs, reward, done, info = env.step(action.tolist())
                total_reward += float(reward)
                if bool(done) and not success:
                    steps_to_success = rollout_step
                success = bool(done) or success
                pbar.set_postfix(success=str(success))
                if success:
                    break

            episodes.append(
                {
                    "env_id": episode,
                    "success": bool(success),
                    "total_reward": float(total_reward),
                    "steps_to_success": int(steps_to_success),
                    "total_env_steps": int(rollout_step + 1),
                    "samples": samples,
                }
            )
            logger.info(
                "[%s] episode %d/%d: success=%s, %d samples",
                task_name,
                episode + 1,
                args.num_episodes,
                success,
                len(samples),
            )
    finally:
        env.close()

    return {
        "task_name": task_name,
        "task_suite_name": args.task_suite_name,
        "task_id": args.task_id,
        "task_description": task_description,
        "episodes": episodes,
    }


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO)
    result = run_task(args)

    os.makedirs(
        os.path.dirname(os.path.abspath(args.samples_out)) or ".", exist_ok=True
    )
    with open(args.samples_out, "wb") as f:
        pickle.dump(result, f, protocol=4)

    n_succ = sum(1 for ep in result["episodes"] if ep["success"])
    logger.info(
        "[%s/%d] %d/%d successes. Wrote %s",
        args.task_suite_name,
        args.task_id,
        n_succ,
        len(result["episodes"]),
        args.samples_out,
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
