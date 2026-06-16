"""Flow-DPO training on in-memory (pos, neg) rollout pairs.

Mirrors :mod:`experiments.filtered_bc.train` but:

1. Uses :class:`experiments.preference_bc.dataset.PreferencePairDataset` — each batch
   element is a ``pos/*``/``neg/*`` dict with both halves already transformed by the
   training data pipeline.
2. Maintains a **frozen reference model** = the trainable state's params at step 0
   (before any LoRA updates). Stored once outside the train step, passed in as a
   pure-dict argument so it doesn't accumulate gradients.
3. Substitutes the stock flow-matching loss with
   :func:`experiments.preference_bc.dpo_loss.flow_dpo_loss_from_mses`, which contrasts
   ``(MSE_theta(pos) - MSE_ref(pos))`` against ``(MSE_theta(neg) - MSE_ref(neg))``
   with paired noise (same ``rng`` used for all four ``compute_loss`` calls).

No other parts of the pi0.5 training loop change — same LoRA setup, same freeze
filter, same optimizer/LR schedule, same weight loader override.
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import importlib.util
import logging
import pathlib
from typing import Any

import etils.epath as epath
from flax import nnx
from flax.training import common_utils
import jax
import jax.numpy as jnp
import optax

from experiments.filtered_bc.envs.adapter import InferenceSample
from experiments.preference_bc.dataset import build_preference_dataset
from experiments.preference_bc.dpo_loss import flow_dpo_loss_components
import openpi.models as _models_pkg  # noqa: F401 — ensures openpi.models namespace is registered
from openpi.models import model as _model
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders

logger = logging.getLogger(__name__)


def _import_train_script() -> Any:
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("openpi_train_script", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---- DPO-aware data loader -------------------------------------------------------


class _PreferenceDataLoader:
    """Wrapper around a dict-emitting TorchDataLoader that yields the raw batch dict.

    Unlike :class:`openpi.training.data_loader.DataLoaderImpl`, we don't convert to a
    single ``(Observation, Actions)`` tuple here because the batch carries two halves
    (``pos/*`` and ``neg/*``). The split is done inside the train step after sharding.
    """

    def __init__(self, data_config: _config.DataConfig, torch_loader: _data_loader.TorchDataLoader):
        self._data_config = data_config
        self._torch_loader = torch_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        yield from self._torch_loader


def _make_pair_loader(
    pos_samples: Sequence[InferenceSample],
    neg_samples: Sequence[InferenceSample],
    train_config: _config.TrainConfig,
    *,
    data_sharding: jax.sharding.Sharding,
    shuffle: bool = True,
    seed: int = 0,
    skip_norm_stats: bool = False,
    base_ckpt_fallback: str | None = None,
) -> _PreferenceDataLoader:
    """Build a DataLoader that emits paired ``pos/*``/``neg/*`` dicts."""
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    dataset = build_preference_dataset(
        pos_samples,
        neg_samples,
        train_config,
        skip_norm_stats=skip_norm_stats,
        base_ckpt_fallback=base_ckpt_fallback,
    )

    effective_batch = min(train_config.batch_size, len(dataset))
    if effective_batch < train_config.batch_size:
        logger.info(
            f"Capping batch_size {train_config.batch_size} -> {effective_batch} (pair-pool has {len(dataset)} items)"
        )
    local_batch_size = max(1, effective_batch // jax.process_count())
    torch_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=data_sharding,
        shuffle=shuffle,
        num_batches=None,
        num_workers=0,
        seed=seed,
        framework="jax",
    )
    return _PreferenceDataLoader(data_config, torch_loader)


# ---- DPO train step --------------------------------------------------------------


def compute_dpo_components(
    model: Any,
    rng: jax.Array,
    obs_pos: Any,
    act_pos: Any,
    obs_neg: Any,
    act_neg: Any,
    mse_ref_pos: jax.Array,
    mse_ref_neg: jax.Array,
    *,
    beta: float,
) -> dict:
    """Run the trainable model on both halves and return the DPO components dict.

    Exposed at module scope so tests can call it with a fake model. The train step
    wraps this in a value_and_grad closure that closes over ``beta``.
    """
    mse_theta_pos = model.compute_loss(rng, obs_pos, act_pos, train=True)
    mse_theta_neg = model.compute_loss(rng, obs_neg, act_neg, train=True)
    return flow_dpo_loss_components(mse_theta_pos, mse_ref_pos, mse_theta_neg, mse_ref_neg, beta=beta)


def _split_pair_batch(batch_dict: dict) -> tuple[tuple, tuple]:
    """Split a pos/neg pair-dict into ``((obs_pos, act_pos), (obs_neg, act_neg))``."""
    pos_items = {k[4:]: v for k, v in batch_dict.items() if k.startswith("pos/")}
    neg_items = {k[4:]: v for k, v in batch_dict.items() if k.startswith("neg/")}
    pos = (_model.Observation.from_dict(pos_items), pos_items["actions"])
    neg = (_model.Observation.from_dict(neg_items), neg_items["actions"])
    return pos, neg


def make_dpo_train_step(train_config: _config.TrainConfig, beta: float):
    """Build a jit-friendly DPO train step closure bound to a specific (config, beta)."""

    def dpo_train_step(
        rng: jax.Array,
        state: training_utils.TrainState,
        ref_params: nnx.State,
        batch_dict: dict,
    ) -> tuple[training_utils.TrainState, dict]:
        (obs_pos, act_pos), (obs_neg, act_neg) = _split_pair_batch(batch_dict)
        train_rng = jax.random.fold_in(rng, state.step)

        # Reference forward (no gradient). Use the same rng as the trainable forward
        # so positives/negatives share their noise/time samples with their ref-model
        # counterparts. Paired noise is the whole point of Diffusion-DPO.
        ref_model = nnx.merge(state.model_def, ref_params)
        ref_model.eval()
        mse_ref_pos = jax.lax.stop_gradient(ref_model.compute_loss(train_rng, obs_pos, act_pos, train=False))
        mse_ref_neg = jax.lax.stop_gradient(ref_model.compute_loss(train_rng, obs_neg, act_neg, train=False))

        model = nnx.merge(state.model_def, state.params)
        model.train()

        def loss_fn(model, rng, obs_p, act_p, obs_n, act_n, ref_pos, ref_neg):
            components = compute_dpo_components(model, rng, obs_p, act_p, obs_n, act_n, ref_pos, ref_neg, beta=beta)
            return components["loss"], components

        diff_state = nnx.DiffState(0, train_config.trainable_filter)
        (_loss, components), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
            model, train_rng, obs_pos, act_pos, obs_neg, act_neg, mse_ref_pos, mse_ref_neg
        )

        params = state.params.filter(train_config.trainable_filter)
        updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
        new_params = optax.apply_updates(params, updates)

        nnx.update(model, new_params)
        new_full_params = nnx.state(model)

        new_state = dataclasses.replace(
            state,
            step=state.step + 1,
            params=new_full_params,
            opt_state=new_opt_state,
        )
        if state.ema_decay is not None:
            new_state = dataclasses.replace(
                new_state,
                ema_params=jax.tree.map(
                    lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new,
                    state.ema_params,
                    new_full_params,
                ),
            )

        return new_state, components

    return dpo_train_step


# ---- Main entrypoint -------------------------------------------------------------


def train_dpo(
    train_config: _config.TrainConfig,
    pos_samples: Sequence[InferenceSample],
    neg_samples: Sequence[InferenceSample],
    *,
    base_ckpt: str | None = None,
    num_train_steps: int | None = None,
    beta: float = 2000.0,
    log_interval: int = 50,
    skip_norm_stats: bool = False,
) -> training_utils.TrainState:
    """Run Flow-DPO training on in-memory (pos, neg) rollout samples.

    Returns the final train state. The merged params can be pulled out via
    ``state.params.to_pure_dict()``.
    """
    if len(pos_samples) == 0 or len(neg_samples) == 0:
        raise ValueError(
            f"train_dpo needs >=1 pos AND >=1 neg sample (got {len(pos_samples)} pos, {len(neg_samples)} neg)."
        )

    train_module = _import_train_script()

    if base_ckpt is not None:
        params_path = str(pathlib.Path(base_ckpt).resolve() / "params")
        logger.info(f"Overriding train_config.weight_loader to load from {params_path}")
        train_config = dataclasses.replace(
            train_config,
            weight_loader=_weight_loaders.CheckpointWeightLoader(params_path),
        )

    num_steps = num_train_steps if num_train_steps is not None else train_config.num_train_steps
    logger.info(
        f"Flow-DPO: {num_steps} steps, batch={train_config.batch_size}, beta={beta}, "
        f"{len(pos_samples)} pos x {len(neg_samples)} neg = {len(pos_samples) * len(neg_samples)} pairs."
    )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(train_config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(train_config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    data_loader = _make_pair_loader(
        pos_samples,
        neg_samples,
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

    train_state, train_state_sharding = train_module.init_train_state(train_config, init_rng, mesh, resume=False)
    jax.block_until_ready(train_state)
    logger.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    # Freeze initial params as the DPO reference. These params are already sharded
    # correctly; the reference forward pass inside the train step will reuse the
    # same model_def as the trainable one.
    ref_params = train_state.params

    dpo_step = make_dpo_train_step(train_config, beta=beta)
    # Note: we do NOT donate argnums=1 (state) because ref_params aliases state.params
    # at init, and donation of one would invalidate the buffer the other is reading.
    # Cost: one extra ~5-7 GB param copy held alongside the trainable one; within budget.
    ptrain_step = jax.jit(
        dpo_step,
        in_shardings=(replicated_sharding, train_state_sharding, train_state_sharding.params, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
    )

    infos: list[dict] = []
    for step in range(num_steps):
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, ref_params, batch)
        infos.append(info)
        if step % log_interval == 0 or step == num_steps - 1:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={float(v):.4f}" for k, v in reduced_info.items())
            logger.info(f"step {step:>6d}: {info_str}")
            infos = []
        batch = next(data_iter)

    jax.block_until_ready(train_state)
    return train_state


__all__ = ["train_dpo"]
