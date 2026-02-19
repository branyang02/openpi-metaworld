"""
MUJOCO_GL=egl uv run examples/metaworld/main.py
"""

import dataclasses
import logging
import math
import os
from typing import Literal

import gymnasium as gym
import imageio.v3 as iio
import metaworld  # noqa: F401
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
from tqdm import tqdm
import tyro

logger = logging.getLogger(__name__)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


# https://metaworld.farama.org/rendering/rendering/#render-from-a-specific-camera
CAMERA_IDS = {
    "topview": 0,
    "corner": 1,
    "corner2": 2,
    "corner3": 3,
    "behindGripper": 4,
    "gripperPOV": 5,
}


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000

    benchmark_name: Literal["MT1", "MT10", "MT50", "ML1", "ML10", "ML45"] = "ML45"
    env_name: str | None = None

    width: int = 224
    height: int = 224

    # Cameras to use for policy input
    policy_cameras: list[str] = dataclasses.field(default_factory=lambda: ["gripperPOV", "corner", "corner2"])
    # The camera used for rendering the video output (must be one of the policy cameras)
    render_camera: str = "corner"

    num_episodes: int = 2
    max_steps: int = 200
    fps: int = 24

    seed: int = 42


class MultiCameraVectorWrapper(gym.vector.VectorWrapper):
    """
    Gym wrapper to render multiple camera views at each step and include them in the info dict.

    info["cameras"]: list[dict[str, np.ndarray]] # camera_names -> image
        - list length = env vector size
        - images are (H, W, 3) uint8 RGB
    """

    def __init__(self, env: gym.vector.VectorEnv, camera_names: list[str]):
        super().__init__(env)
        self.camera_names = camera_names

    def _render_cameras_one(self, e) -> dict[str, np.ndarray]:
        renderer = e.unwrapped.mujoco_renderer
        images = {}
        for cam_name in self.camera_names:
            # HACK (branyang02): Very Very Very Hacky
            # Take a look at gymnasium.envs.muojoco.mujoco_rendering.MujocoRenderer.render()
            # Implemented solutions from:
            # https://github.com/Farama-Foundation/Metaworld/issues/448
            # https://github.com/Farama-Foundation/Gymnasium/issues/736
            viewer = renderer._get_viewer(render_mode="rgb_array")  # noqa: SLF001
            if len(renderer._viewers.keys()) >= 1:  # noqa: SLF001
                viewer.make_context_current()
            img = viewer.render(render_mode="rgb_array", camera_id=CAMERA_IDS[cam_name])
            images[cam_name] = img[::-1].copy()  # flip vertically
        return images

    def _render_all(self) -> list[dict[str, np.ndarray]]:
        return [self._render_cameras_one(e) for e in self.env.envs]

    def reset(self, **kwargs):
        obs, infos = self.env.reset(**kwargs)
        infos["cameras"] = self._render_all()
        return obs, infos

    def step(self, actions):
        obs, rewards, terms, truncs, infos = self.env.step(actions)
        infos["cameras"] = self._render_all()
        return obs, rewards, terms, truncs, infos


def tile_frames(frames: list[np.ndarray]) -> np.ndarray:
    """Arrange N frames into a grid image.

    Grid layout: cols = ceil(sqrt(N)), rows = ceil(N / cols).
    Empty slots are filled with black.
    """
    n = len(frames)
    h, w, c = frames[0].shape
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    grid = np.zeros((rows * h, cols * w, c), dtype=frames[0].dtype)
    for idx, frame in enumerate(frames):
        r, col = divmod(idx, cols)
        grid[r * h : (r + 1) * h, col * w : (col + 1) * w] = frame

    return grid


def make_env(
    benchmark_name: str,
    env_name: str | None,
    seed: int,
    width: int = 224,
    height: int = 224,
    camera_names: list[str] | None = None,
    vector_strategy: Literal["sync", "async"] = "sync",
) -> gym.vector.VectorEnv:
    """
    Environment creation notes:

    - MT1 and ML1 create a single environment.
    - MT10 and MT50 create 10 and 50 environments respectively, each with different tasks.
    - ML10 and ML45 create 5 environments from the *test set* of ML10 and ML45 respectively.
    These tasks are not seen during training.
    - We wrap the environments in a custom wrapper to render multiple camera views at each step

    Task breakdown:

    ML10 test tasks:
        0: SawyerDrawerOpenEnvV3
        1: SawyerDoorCloseEnvV3
        2: SawyerShelfPlaceEnvV3
        3: SawyerSweepIntoGoalEnvV3
        4: SawyerLeverPullEnvV3

    ML45 test tasks:
        0: SawyerBinPickingEnvV3
        1: SawyerBoxCloseEnvV3
        2: SawyerHandInsertEnvV3
        3: SawyerDoorLockEnvV3
        4: SawyerDoorUnlockEnvV3

    References:
    - Meta-World environments: https://meta-world.github.io/
    - Environment creation code adapted from:
    https://metaworld.farama.org/introduction/basic_usage/
    """

    # TODO(branyang02): should we support async vector envs?
    if vector_strategy == "async":
        raise NotImplementedError("Async vector environments are not supported yet!")

    # TODO(branyang02): should we support MT1 and ML1?
    if benchmark_name == "MT1":
        raise NotImplementedError("MT1 is not implemented yet")
        env = gym.make("Meta-World/MT1", env_name=env_name, seed=seed)
    if benchmark_name == "ML1":
        raise NotImplementedError("ML1 is not implemented yet")
        env = gym.make("Meta-World/ML1-test", env_name=env_name, seed=seed)

    benchmark_ids = {
        "MT10": "Meta-World/MT10",
        "MT50": "Meta-World/MT50",
        "ML10": "Meta-World/ML10-test",
        "ML45": "Meta-World/ML45-test",
    }
    if benchmark_name not in benchmark_ids:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")

    env = gym.make_vec(
        benchmark_ids[benchmark_name],
        vector_strategy=vector_strategy,
        seed=seed,
        width=width,
        height=height,
    )

    return MultiCameraVectorWrapper(env, camera_names)


def main(args: Args) -> None:
    # policy = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    # logger.info(f"Server metadata: {policy.get_server_metadata()}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    env = make_env(
        args.benchmark_name,
        args.env_name,
        args.seed,
        width=args.width,
        height=args.height,
        camera_names=args.policy_cameras,
    )
    num_envs = env.num_envs

    for episode in range(args.num_episodes):
        obs, info = env.reset(seed=args.seed + episode)
        camera_views = info["cameras"]
        total_reward = np.zeros(num_envs)
        success = np.zeros(num_envs, dtype=bool)

        video_path = os.path.join(OUTPUT_DIR, f"episode_{episode:03d}.mp4")
        with iio.imopen(video_path, "w", plugin="pyav") as video:
            video.init_video_stream("h264", fps=args.fps)

            pbar = tqdm(range(args.max_steps), desc=f"Episode {episode + 1}/{args.num_episodes}")
            for _step in pbar:
                frames = [cv[args.render_camera] for cv in camera_views]  # list of (H, W, 3)
                grid_frame = tile_frames(frames) if num_envs > 5 else np.concatenate(frames, axis=1)
                video.write_frame(grid_frame)

                # result = policy.infer(
                #     {
                #         "state": obs.astype(np.float32),
                #         **{f"image/{name}": camera_views[name] for name in args.policy_cameras},
                #     }
                # )
                # action = np.clip(result["actions"], -1.0, 1.0).astype(np.float32)

                action = env.action_space.sample()  # (5, 4)

                obs, reward, terminated, truncated, info = env.step(action)
                camera_views = info["cameras"]
                total_reward += reward
                success |= np.asarray(info.get("success", np.zeros(num_envs)), dtype=bool)
                pbar.set_postfix(reward=f"{total_reward.mean():.1f}", success=f"{success.mean():.0%}")

        logger.info(
            f"Episode {episode + 1}/{args.num_episodes}: "
            f"mean_reward={total_reward.mean():.2f}, success_rate={success.mean():.2f}, "
            f"video={video_path}"
        )

    env.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = tyro.cli(Args)
    main(args)
