"""
Evaluate pi0.5 MetaWorld checkpoint with different numbers of flow matching denoising steps.

Tests whether the default 10-step Euler denoising is necessary or if fewer steps suffice.
Motivated by the "Much Ado About Noising" paper (ICLR 2026) and empirical observations
that denoising trajectories are nearly perfectly straight.

Uses PyTorch inference loaded in-process (no WebSocket server).

Usage:
    export CUDA_VISIBLE_DEVICES=1
    MUJOCO_GL=egl uv run denoising_step_exp/eval_denoising_steps.py \
        --policy.config=pi05_metaworld \
        --policy.dir=checkpoints/pi05_metaworld/pi05_metaworld_test/5000/ \
        --num_steps 1 --split train
"""

import collections
import dataclasses
import json
import logging
import pathlib
from typing import Literal

import gymnasium as gym
import metaworld  # noqa: F401
import numpy as np
from tqdm import tqdm
import tyro

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

logger = logging.getLogger(__name__)

CAMERA_IDS = {
    "topview": 0,
    "corner": 1,
    "corner2": 2,
    "corner3": 3,
    "corner4": 4,
    "behindGripper": 5,
    "gripperPOV": 6,
}

TASK_TO_PROMPT = {
    "assembly-v3": "pick up the nut and place it onto the peg",
    "disassemble-v3": "pick up the nut and remove it from the peg",
    "basketball-v3": "dunk the basketball into the hoop",
    "soccer-v3": "kick the soccer ball into the goal",
    "bin-picking-v3": "pick up the object and place it into the bin",
    "box-close-v3": "grasp the cover and close the box",
    "button-press-v3": "press the button",
    "button-press-topdown-v3": "press the button from the top",
    "button-press-topdown-wall-v3": "press the button on the wall from the top",
    "button-press-wall-v3": "press the button on the wall",
    "coffee-button-v3": "push the button on the coffee machine",
    "coffee-pull-v3": "pull the mug away from the coffee machine",
    "coffee-push-v3": "push the mug under the coffee machine",
    "dial-turn-v3": "rotate the dial",
    "lever-pull-v3": "pull the lever down",
    "door-close-v3": "close the door",
    "door-lock-v3": "lock the door by rotating the lock",
    "door-open-v3": "open the door",
    "door-unlock-v3": "unlock the door by rotating the lock",
    "drawer-close-v3": "push the drawer closed",
    "drawer-open-v3": "pull the drawer open",
    "faucet-close-v3": "rotate the faucet handle to close it",
    "faucet-open-v3": "rotate the faucet handle to open it",
    "hammer-v3": "hammer the nail into the board",
    "hand-insert-v3": "insert the gripper into the hole",
    "handle-press-v3": "press the handle down",
    "handle-press-side-v3": "press the handle down sideways",
    "handle-pull-v3": "pull the handle up",
    "handle-pull-side-v3": "pull the handle sideways",
    "peg-insert-side-v3": "insert the peg into the hole sideways",
    "peg-unplug-side-v3": "unplug the peg from the hole sideways",
    "pick-out-of-hole-v3": "pick the object out of the hole",
    "pick-place-v3": "pick up the object and place it at the goal",
    "pick-place-wall-v3": "pick up the object and place it at the goal behind the wall",
    "plate-slide-v3": "slide the plate to the goal",
    "plate-slide-back-v3": "slide the plate backwards to the goal",
    "plate-slide-back-side-v3": "slide the plate backwards and sideways to the goal",
    "plate-slide-side-v3": "slide the plate sideways to the goal",
    "push-v3": "push the object to the goal",
    "push-back-v3": "push the object backwards to the goal",
    "push-wall-v3": "push the object around the wall to the goal",
    "reach-v3": "reach the goal position",
    "reach-wall-v3": "reach the goal position behind the wall",
    "shelf-place-v3": "pick up the object and place it on the shelf",
    "stick-pull-v3": "use the stick to pull the object",
    "stick-push-v3": "use the stick to push the object",
    "sweep-v3": "sweep the object off the table",
    "sweep-into-v3": "sweep the object into the hole",
    "window-close-v3": "push the window closed",
    "window-open-v3": "push the window open",
}

ML45_TRAIN = [
    "assembly-v3",
    "basketball-v3",
    "button-press-topdown-v3",
    "button-press-topdown-wall-v3",
    "button-press-v3",
    "button-press-wall-v3",
    "coffee-button-v3",
    "coffee-pull-v3",
    "coffee-push-v3",
    "dial-turn-v3",
    "disassemble-v3",
    "door-close-v3",
    "door-open-v3",
    "drawer-close-v3",
    "drawer-open-v3",
    "faucet-close-v3",
    "faucet-open-v3",
    "hammer-v3",
    "handle-press-side-v3",
    "handle-press-v3",
    "handle-pull-side-v3",
    "handle-pull-v3",
    "lever-pull-v3",
    "peg-insert-side-v3",
    "peg-unplug-side-v3",
    "pick-out-of-hole-v3",
    "pick-place-v3",
    "pick-place-wall-v3",
    "plate-slide-back-side-v3",
    "plate-slide-back-v3",
    "plate-slide-side-v3",
    "plate-slide-v3",
    "push-back-v3",
    "push-v3",
    "push-wall-v3",
    "reach-v3",
    "reach-wall-v3",
    "shelf-place-v3",
    "soccer-v3",
    "stick-pull-v3",
    "stick-push-v3",
    "sweep-into-v3",
    "sweep-v3",
    "window-close-v3",
    "window-open-v3",
]

ML45_TEST = [
    "bin-picking-v3",
    "box-close-v3",
    "door-lock-v3",
    "door-unlock-v3",
    "hand-insert-v3",
]


class MultiCameraWrapper(gym.Wrapper):
    """Wrapper that renders multiple cameras and includes images in info dict."""

    def __init__(self, env: gym.Env, camera_names: list[str]):
        super().__init__(env)
        self.camera_names = camera_names

    def _render_cameras(self) -> dict[str, np.ndarray]:
        renderer = self.unwrapped.mujoco_renderer
        images = {}
        for cam_name in self.camera_names:
            viewer = renderer._get_viewer(render_mode="rgb_array")  # noqa: SLF001
            if len(renderer._viewers.keys()) >= 1:  # noqa: SLF001
                viewer.make_context_current()
            img = viewer.render(render_mode="rgb_array", camera_id=CAMERA_IDS[cam_name])
            images[cam_name] = img[::-1].copy()
        return images

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        info["cameras"] = self._render_cameras()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        info["cameras"] = self._render_cameras()
        return obs, reward, terminated, truncated, info


@dataclasses.dataclass
class PolicyArgs:
    config: str = "pi05_metaworld"
    dir: str = "checkpoints/pi05_metaworld/pi05_metaworld_test/5000/"


@dataclasses.dataclass
class Args:
    policy: PolicyArgs = dataclasses.field(default_factory=PolicyArgs)

    # Number of denoising steps for flow matching.
    num_steps: int = 10
    # Tasks to evaluate. If empty, uses --split to select.
    tasks: list[str] = dataclasses.field(default_factory=list)
    # ML45 split to use when --tasks is empty.
    split: Literal["train", "test"] = "train"
    # Number of parallel environments per task.
    num_envs: int = 15
    # Maximum steps per episode.
    max_steps: int = 300
    # Number of steps between re-planning.
    replan_steps: int = 10
    # Output directory for results.
    output_dir: str = "denoising_step_exp/results/ablation"

    width: int = 224
    height: int = 224
    policy_cameras: list[str] = dataclasses.field(default_factory=lambda: ["corner", "corner4", "gripperPOV"])
    seed: int = 69_420


def eval_task(task_name: str, policy, args: Args) -> dict:
    """Evaluate a single task and return success info."""
    prompt = TASK_TO_PROMPT[task_name]
    num_envs = args.num_envs

    env_fns = [
        lambda i=i: MultiCameraWrapper(
            gym.make(
                "Meta-World/MT1",
                env_name=task_name,
                seed=args.seed + i,
                width=args.width,
                height=args.height,
            ),
            args.policy_cameras,
        )
        for i in range(num_envs)
    ]
    env = gym.vector.AsyncVectorEnv(env_fns, context="spawn")

    try:
        obs, info = env.reset(seed=args.seed)
        camera_views = info["cameras"]
        success = np.zeros(num_envs, dtype=bool)
        total_reward = np.zeros(num_envs)
        action_plan = collections.deque()

        pbar = tqdm(range(args.max_steps), desc=task_name)
        for _step in pbar:
            if not action_plan:
                obs_dict = {
                    "observation/image": camera_views["corner4"],
                    "observation/wrist_image": camera_views["gripperPOV"],
                    "observation/state": obs.astype(np.float32)[..., :4],
                    "prompt": [prompt] * num_envs,
                }

                result = policy.infer(obs_dict)
                action_chunk = np.clip(result["actions"], -1.0, 1.0).astype(np.float32)

                for t in range(args.replan_steps):
                    action_plan.append(action_chunk[:, t, :])

            action = action_plan.popleft()
            obs, reward, terminated, truncated, info = env.step(action)
            camera_views = info["cameras"]
            total_reward += reward

            step_success = np.asarray(info.get("success", np.zeros(num_envs)), dtype=bool)
            success |= step_success
            if success.all():
                break

            pbar.set_postfix(reward=f"{total_reward.mean():.1f}", success=f"{success.mean():.0%}")

        success_rate = float(success.mean())
        per_env_success = success.tolist()
    finally:
        env.close()

    return {
        "task_name": task_name,
        "success_rate": success_rate,
        "per_env_success": per_env_success,
        "mean_reward": float(total_reward.mean()),
    }


def main(args: Args) -> None:
    # Determine task list
    if args.tasks:
        task_list = args.tasks
    elif args.split == "train":
        task_list = ML45_TRAIN
    else:
        task_list = ML45_TEST

    logger.info(f"Evaluating {len(task_list)} tasks with num_steps={args.num_steps}")

    # Load policy in-process with num_steps passed as sample_kwargs
    config = _config.get_config(args.policy.config)
    policy = _policy_config.create_trained_policy(
        config,
        args.policy.dir,
        sample_kwargs={"num_steps": args.num_steps},
    )
    logger.info(f"Policy loaded (num_steps={args.num_steps})")

    # Run evaluation
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / f"results_{args.num_steps}steps.json"

    per_task_results = {}
    for task_name in tqdm(task_list, desc=f"num_steps={args.num_steps}"):
        task_result = eval_task(task_name, policy, args)
        per_task_results[task_name] = task_result
        logger.info(f"[{task_name}] success_rate={task_result['success_rate']:.2f}")

        # Save incrementally
        summary = {
            "num_steps": args.num_steps,
            "num_envs": args.num_envs,
            "max_steps": args.max_steps,
            "replan_steps": args.replan_steps,
            "seed": args.seed,
            "checkpoint": args.policy.dir,
            "split": args.split,
            "tasks_evaluated": len(per_task_results),
            "mean_success_rate": float(np.mean([r["success_rate"] for r in per_task_results.values()])),
            "per_task": per_task_results,
        }
        with open(results_path, "w") as f:
            json.dump(summary, f, indent=2)

    mean_sr = summary["mean_success_rate"]
    logger.info(f"num_steps={args.num_steps}: mean_success_rate={mean_sr:.4f} ({mean_sr:.1%})")
    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = tyro.cli(Args)
    main(args)
