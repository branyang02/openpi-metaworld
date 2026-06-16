# Preference-BC baseline (Flow-DPO)

Second parametric baseline for the activation-steering paper. Uses a **30-rollout-per-task data budget** spent on a preference-optimization LoRA fine-tune that uses BOTH successful AND failed rollouts — directly mirroring how the paper's method uses `mean(success) − mean(failure)` in activation space.

Drops into the same envs as the filtered-BC baseline (MetaWorld, LIBERO, RoboCasa) via the shared `EnvAdapter` Protocol.

## Loss

Diffusion-DPO (Wallace et al. 2023) adapted to flow matching:

```
Δ_pos = MSE_θ(pos) − MSE_ref(pos)
Δ_neg = MSE_θ(neg) − MSE_ref(neg)
L = −E[ log σ( β · (Δ_neg − Δ_pos) ) ]
```

- `MSE_*` is the pi0.5 flow-matching per-sample loss: `mean((v − u_t)², axis=-1)` (shape `[B, H]`, reduced to scalar-per-pair).
- `π_ref` is the trainable state's params **at step 0** — a frozen snapshot of the base LoRA-init policy. No separate model to load.
- Paired noise: the same `rng` is folded into all four `compute_loss` calls per step, so positive and negative halves see the same `(t, ε)` per pair (key variance-reduction trick from the paper).

## Pipeline

Per task:

```
rollout (N=30) → partition into (positives, negatives) [no filter] →
  skip if all-success or all-failure →
Flow-DPO train 500 steps on cartesian pair pool (|pos| × |neg|) →
merge LoRA into base weights →
  MetaWorld: build PI0Pytorch in-process, eval
  LIBERO/RoboCasa: save merged ckpt, restart server, eval via subprocess client
```

## Files

```
experiments/preference_bc/
├── dataset.py                # PreferencePairDataset (cartesian pos × neg)
├── dpo_loss.py               # flow_dpo_loss_from_mses + logging helpers
├── train.py                  # train_dpo (forks filtered_bc/train.py, swaps loss)
├── run_preference_bc.py      # orchestrator, dispatches on --args.env
├── run_metaworld.sh          # 10-task MetaWorld curated subset sweep (matches filtered-BC)
├── run_libero.sh             # libero_10 sweep
├── run_robocasa.sh           # RoboCasa 7-task subset sweep
└── README.md
```

Reuses unchanged from filtered-BC:
- `experiments/filtered_bc/envs/` — EnvAdapter Protocol + 3 concrete adapters
- `experiments/filtered_bc/merge_save.py` — LoRA merge + disk serialization
- `experiments/filtered_bc/_serve_policy_nocompile.py` — subprocess server shim (skips `torch.compile` autotune)
- `examples/{libero_env,robocasa_env}/filtered_bc_client.py` — rollout clients (env's isolated venvs)

## Smoke tests

```bash
# MetaWorld
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl \
    XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform PYTHONUNBUFFERED=1 \
    uv run python -u -m experiments.preference_bc.run_preference_bc \
        --args.env metaworld --args.tasks reach-v3 \
        --args.num-rollouts 4 --args.num-train-steps 30 --args.batch-size 4 \
        --args.eval-num-episodes 3 --args.beta 200.0

# LIBERO
... --args.env libero --args.tasks libero_spatial:0 --args.replan-steps 5 --args.max-steps 220

# RoboCasa (pick a high-base-rate task for smoke)
... --args.env robocasa --args.tasks OpenStandMixerHead --args.replan-steps 5
```

## Full sweeps

```bash
bash experiments/preference_bc/run_metaworld.sh > experiments/preference_bc/logs/metaworld.log 2>&1 &
bash experiments/preference_bc/run_libero.sh    > experiments/preference_bc/logs/libero.log    2>&1 &
bash experiments/preference_bc/run_robocasa.sh  > experiments/preference_bc/logs/robocasa.log  2>&1 &
```

## Key hyperparameters

- `--args.beta` (DPO sharpness, default 2000.0). Start sweep at {200, 2000, 20000}.
- `--args.num-rollouts` (CLI default 15; sweep scripts run 30).
- `--args.num-train-steps` (default 500).
- `--args.batch-size` (default 8; pair-pool size is `|pos| × |neg|`).

## Fallback

If Flow-DPO proves unstable at the 30-rollout scale (training diverges / reward-accuracy never exceeds chance), swap `flow_dpo_loss_from_mses` for a hinge-style Margin-Repulsion-BC loss — single-line change in `train.py`.

## Tests

```bash
uv run pytest tests/preference_bc/
```

Covers: pair cartesian indexing, dataset error paths, DPO loss math (log 2 at θ=ref, monotonic in preference gap, saturation at ±∞).
