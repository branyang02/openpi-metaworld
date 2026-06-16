"""Filtered-BC RoboCasa rollout/eval client.

Runs in the isolated ``examples/robocasa_env/`` venv and talks to a policy server
in the root venv over WebSocket. Records ``(obs, action_chunk)`` pairs per replan,
pickles them to ``--samples-out`` so the filtered-BC orchestrator can re-hydrate
them without importing RoboCasa.

Usage (invoked by RoboCasaAdapter; not typically run by hand):

    MUJOCO_GL=egl uv run python filtered_bc_client.py \\
        --host localhost --port 8123 --env_name CloseFridge \\
        --num-episodes 3 --samples-out /tmp/rollouts.pkl
"""

from __future__ import annotations

import collections
import dataclasses
import logging
import os
import pickle

import gymnasium as gym
import numpy as np
import robocasa  # noqa: F401
import tyro
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
from robocasa.utils.dataset_registry_utils import get_task_horizon
from robocasa.utils.env_utils import convert_action
from tqdm import tqdm

logger = logging.getLogger(__name__)

CAMERA_KEYS = {
    "agentview_left": "video.robot0_agentview_left",
    "agentview_right": "video.robot0_agentview_right",
    "eye_in_hand": "video.robot0_eye_in_hand",
}


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000

    env_name: str = "CloseFridge"
    split: str = "pretrain"

    num_episodes: int = 3
    max_steps: int | None = None
    replan_steps: int = 5
    resize_size: int = 224
    seed: int = 7

    samples_out: str = "/tmp/filtered_bc_robocasa_samples.pkl"
    eval_only: bool = False


def _make_env(env_name: str, split: str, seed: int) -> gym.Env:
    return gym.make(f"robocasa/{env_name}", split=split, seed=seed)


def _build_state(obs: dict) -> np.ndarray:
    return np.concatenate(
        [
            obs["state.end_effector_position_relative"],
            obs["state.end_effector_rotation_relative"],
            obs["state.base_position"],
            obs["state.base_rotation"],
            obs["state.gripper_qpos"],
        ],
        axis=0,
    ).astype(np.float32)


def _make_policy_obs(obs: dict, resize_size: int) -> dict:
    img = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(
            obs[CAMERA_KEYS["agentview_left"]], resize_size, resize_size
        )
    )
    img2 = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(
            obs[CAMERA_KEYS["agentview_right"]], resize_size, resize_size
        )
    )
    wrist_img = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(
            obs[CAMERA_KEYS["eye_in_hand"]], resize_size, resize_size
        )
    )
    return {
        "observation/image": img,
        "observation/image2": img2,
        "observation/wrist_image": wrist_img,
        "observation/state": _build_state(obs),
    }


def run_task(args: Args) -> dict:
    policy = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    logger.info(f"Server metadata: {policy.get_server_metadata()}")

    env = _make_env(env_name=args.env_name, split=args.split, seed=args.seed)
    task_horizon = get_task_horizon(args.env_name)
    max_steps = (
        args.max_steps if args.max_steps is not None else int(task_horizon * 1.5)
    )

    episodes: list[dict] = []
    try:
        for episode in range(args.num_episodes):
            obs, info = env.reset()
            task_lang = str(obs["annotation.human.task_description"])
            action_plan: collections.deque = collections.deque()
            success = False
            total_reward = 0.0
            steps_to_success = -1
            samples: list[dict] = []
            total_steps = 0

            pbar = tqdm(
                range(max_steps),
                desc=f"[{args.env_name}] ep {episode + 1}/{args.num_episodes}",
                leave=False,
            )
            for step in pbar:
                total_steps = step + 1
                if not action_plan:
                    element = _make_policy_obs(obs, args.resize_size)
                    element["prompt"] = task_lang

                    result = policy.infer(element)
                    action_chunk = np.asarray(result["actions"], dtype=np.float32)
                    if action_chunk.ndim != 2:
                        raise ValueError(
                            f"Server must return (H, D) chunk, got shape {action_chunk.shape}"
                        )
                    if action_chunk.shape[0] < args.replan_steps:
                        raise ValueError(
                            f"Server must return >= {args.replan_steps} actions, got {action_chunk.shape[0]}"
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
                                "prompt": task_lang,
                                "action_chunk": action_chunk.copy(),
                            }
                        )
                    for t in range(args.replan_steps):
                        action_plan.append(action_chunk[t])

                action = convert_action(action_plan.popleft())
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                step_success = bool(info.get("success", False))
                if step_success and not success:
                    steps_to_success = step
                success = step_success or success
                pbar.set_postfix(success=str(success))
                if success:
                    break

            episodes.append(
                {
                    "env_id": episode,
                    "success": bool(success),
                    "total_reward": float(total_reward),
                    "steps_to_success": int(steps_to_success),
                    "total_env_steps": int(total_steps),
                    "samples": samples,
                }
            )
            logger.info(
                f"[{args.env_name}] ep {episode + 1}/{args.num_episodes}: success={success}, samples={len(samples)}"
            )
    finally:
        env.close()

    return {
        "task_name": args.env_name,
        "env_name": args.env_name,
        "split": args.split,
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
        f"[{args.env_name}] {n_succ}/{len(result['episodes'])} successes. Wrote {args.samples_out}"
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
