"""LoRA training on an in-memory buffer of filtered rollout samples.

Thin wrapper around `scripts/train.py`'s `init_train_state` + `train_step` that swaps
the on-disk LeRobot data loader for our in-memory :class:`InferenceSampleDataset`.
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import functools
import importlib.util
import logging
import pathlib
from typing import Any

import etils.epath as epath
from flax.training import common_utils
import jax
import jax.numpy as jnp

from experiments.filtered_bc.dataset import build_training_dataset
from experiments.filtered_bc.envs.adapter import InferenceSample
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders

logger = logging.getLogger(__name__)


# ---- Reuse init_train_state + train_step from scripts/train.py ---------------------


def _import_train_script() -> Any:
    """Import scripts/train.py as a module (it's a script, not a regular package module)."""
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("openpi_train_script", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---- Data loader wiring ------------------------------------------------------------


def _make_data_loader(
    samples: Sequence[InferenceSample],
    train_config: _config.TrainConfig,
    *,
    data_sharding: jax.sharding.Sharding,
    shuffle: bool = True,
    seed: int = 0,
    skip_norm_stats: bool = False,
    base_ckpt_fallback: str | None = None,
) -> _data_loader.DataLoaderImpl:
    """Build a DataLoaderImpl from in-memory rollout samples."""
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    dataset = build_training_dataset(
        samples, train_config, skip_norm_stats=skip_norm_stats, base_ckpt_fallback=base_ckpt_fallback
    )

    # When the filtered-success pool is smaller than the configured batch size, cap
    # batch size to the dataset size so the TorchDataLoader can still yield batches.
    effective_batch = min(train_config.batch_size, len(dataset))
    if effective_batch < train_config.batch_size:
        logger.info(
            f"Capping batch_size {train_config.batch_size} -> {effective_batch} "
            f"(only {len(dataset)} successful samples available)"
        )
    local_batch_size = max(1, effective_batch // jax.process_count())
    torch_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=data_sharding,
        shuffle=shuffle,
        num_batches=None,
        num_workers=0,  # single-process; our dataset is tiny and already in RAM.
        seed=seed,
        framework="jax",
    )
    return _data_loader.DataLoaderImpl(data_config, torch_loader)


# ---- Main training entry point -----------------------------------------------------


def train_lora(
    train_config: _config.TrainConfig,
    samples: Sequence[InferenceSample],
    *,
    base_ckpt: str | None = None,
    num_train_steps: int | None = None,
    log_interval: int = 50,
    save_interval: int | None = None,
    skip_norm_stats: bool = False,
) -> training_utils.TrainState:
    """Run LoRA fine-tuning on the given in-memory samples. Returns the final TrainState.

    Parameters
    ----------
    train_config : TrainConfig
        Should typically be ``pi05_metaworld_low_mem_finetune`` (LoRA variant with matching
        freeze filter). The config is used both for model construction and as the optimizer
        + LR schedule specification.
    samples : Sequence[InferenceSample]
        Successful rollouts to train on (already filtered).
    base_ckpt : str, optional
        Path to the base checkpoint to initialize weights from, e.g. a finetuned
        ``pi05_metaworld`` checkpoint directory (expects a ``params/`` subdir). Overrides
        ``train_config.weight_loader``. For the filtered-BC baseline this is critical:
        the default ``pi05_base`` weight_loader points at the foundation-pretrained model
        (no action training), so we must override it to start from the same checkpoint
        that produced the rollouts.
    num_train_steps : int, optional
        Overrides ``train_config.num_train_steps`` if provided.
    save_interval : int, optional
        If set, writes a checkpoint every ``save_interval`` steps to
        ``train_config.checkpoint_dir``. If None, only the final step is saved.
    skip_norm_stats : bool
        Useful for unit tests. Real training MUST have norm stats available.
    """
    train_module = _import_train_script()

    if base_ckpt is not None:
        params_path = str(pathlib.Path(base_ckpt).resolve() / "params")
        logger.info(f"Overriding train_config.weight_loader to load from {params_path}")
        train_config = dataclasses.replace(
            train_config,
            weight_loader=_weight_loaders.CheckpointWeightLoader(params_path),
        )

    num_steps = num_train_steps if num_train_steps is not None else train_config.num_train_steps
    logger.info(f"Training for {num_steps} steps with batch_size={train_config.batch_size} on {len(samples)} samples.")

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(train_config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(train_config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Data loader.
    data_loader = _make_data_loader(
        samples,
        train_config,
        data_sharding=data_sharding,
        shuffle=True,
        seed=train_config.seed,
        skip_norm_stats=skip_norm_stats,
        base_ckpt_fallback=base_ckpt,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logger.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Train state (loads base checkpoint via train_config.weight_loader).
    train_state, train_state_sharding = train_module.init_train_state(train_config, init_rng, mesh, resume=False)
    jax.block_until_ready(train_state)
    logger.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    ptrain_step = jax.jit(
        functools.partial(train_module.train_step, train_config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    # Optional checkpoint manager (only if save_interval set).
    checkpoint_manager = None
    if save_interval is not None:
        checkpoint_manager, _ = _checkpoints.initialize_checkpoint_dir(
            train_config.checkpoint_dir,
            keep_period=train_config.keep_period,
            overwrite=True,
            resume=False,
        )

    infos: list[dict] = []
    for step in range(num_steps):
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % log_interval == 0 or step == num_steps - 1:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            logger.info(f"step {step:>6d}: {info_str}")
            infos = []
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            batch = next(data_iter)

        if checkpoint_manager is not None and ((step + 1) % save_interval == 0 or step == num_steps - 1):
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step + 1)

    if checkpoint_manager is not None:
        checkpoint_manager.wait_until_finished()

    return train_state


__all__ = ["train_lora"]
