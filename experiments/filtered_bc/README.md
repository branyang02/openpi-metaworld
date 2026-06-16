# Filtered-BC baseline (pi0.5)

Parametric baseline for the activation-steering paper. Uses a **30-rollout-per-task data budget** spent on LoRA fine-tuning instead of an inference-time activation hook. Supports three simulators behind a common adapter interface: **MetaWorld**, **LIBERO**, **RoboCasa**.

Per task, the pipeline:

1. Rolls out N episodes with the base pi0.5 policy.
2. Filters to successful episodes.
3. LoRA fine-tunes (JAX, pi0.5 flow-matching loss) on the successful `(obs, action_chunk)` pairs.
4. Merges LoRA adapters into the base weights (numpy).
5. Evaluates the merged policy on held-out episodes.

Results are written incrementally to the `--args.results-json` path after every task so a mid-sweep crash still leaves partial data on disk.

## Execution modes

**MetaWorld** runs in-process: the base Policy is loaded in the same Python process, rollout + eval go through `AsyncVectorEnv` with a live `openpi.policies.policy.Policy`, and the merged model is rebuilt into `PI0Pytorch` on GPU for eval. One process, no subprocess orchestration.

**LIBERO** and **RoboCasa** run server-client: their env libraries live in separate venvs (`examples/libero_env/` on Python 3.8, `examples/robocasa_env/` on Python 3.11) that can't coexist with the training JAX stack. Per task the orchestrator:

1. Spawns a `scripts/serve_policy.py` subprocess (root venv, `--pytorch`) on a random free port.
2. Spawns `examples/{libero,robocasa}_env/filtered_bc_client.py` in that env's venv — it records `(obs, action_chunk)` per replan, pickles them.
3. After LoRA merge, writes the merged ckpt to a scratch dir via `save_merged_jax_checkpoint` and launches a *fresh* server pointing at it for eval.

## Installation

Uses the **root openpi venv** for the orchestrator + MetaWorld; LIBERO and RoboCasa use their own venvs for the rollout/eval clients.

```bash
git submodule update --init --recursive
GIT_LFS_SKIP_SMUDGE=1 uv sync                   # root
cd examples/libero_env && uv sync               # LIBERO (Python 3.8)
cd examples/robocasa_env && uv sync             # RoboCasa (Python 3.11)
```

## Files

```
experiments/filtered_bc/
├── envs/
│   ├── adapter.py      # EnvAdapter Protocol + shared dataclasses (InferenceSample, EpisodeRollout, ...)
│   ├── metaworld.py    # In-process adapter
│   ├── libero.py       # Subprocess adapter + server/client orchestration
│   └── robocasa.py     # Subprocess adapter + server/client orchestration
├── dataset.py          # InferenceSampleDataset + build_training_dataset (env-agnostic transform stack)
├── train.py            # train_lora (lifts init_train_state/train_step from scripts/train.py)
├── merge_save.py       # merge_lora_params, build_pytorch_model_from_merged, save_merged_jax_checkpoint
├── run_filtered_bc.py  # Orchestrator; dispatches on --args.env
├── make_report.py      # Renders results.json → markdown
├── run_metaworld.sh    # 10-task MetaWorld curated subset sweep
├── run_libero.sh       # Full LIBERO suite sweep (libero_10 by default)
└── run_robocasa.sh     # RoboCasa subset sweep (7 curated tasks by default)
```

Training configs (in `src/openpi/training/config.py`, all LoRA on PaliGemma + action expert, EMA off, vision tower trainable):

- `pi05_metaworld_low_mem_finetune`
- `pi05_libero_low_mem_finetune`
- `pi05_robocasa_low_mem_finetune`

## Usage

### Smoke tests (1 task each, short)

```bash
# MetaWorld
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl \
    XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform \
    PYTHONUNBUFFERED=1 \
    uv run python -u -m experiments.filtered_bc.run_filtered_bc \
        --args.env metaworld --args.tasks reach-v3 \
        --args.num-rollouts 3 --args.num-train-steps 30 --args.batch-size 4 \
        --args.eval-num-episodes 3

# LIBERO
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl \
    XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform \
    PYTHONUNBUFFERED=1 \
    uv run python -u -m experiments.filtered_bc.run_filtered_bc \
        --args.env libero --args.tasks libero_spatial:0 \
        --args.num-rollouts 2 --args.num-train-steps 20 --args.batch-size 2 \
        --args.eval-num-episodes 2 --args.max-steps 220 --args.replan-steps 5

# RoboCasa
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl \
    XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform \
    PYTHONUNBUFFERED=1 \
    uv run python -u -m experiments.filtered_bc.run_filtered_bc \
        --args.env robocasa --args.tasks CloseFridge \
        --args.num-rollouts 2 --args.num-train-steps 20 --args.batch-size 2 \
        --args.eval-num-episodes 2 --args.replan-steps 5
```

### Full sweeps

```bash
bash experiments/filtered_bc/run_metaworld.sh > experiments/filtered_bc/logs/metaworld.log 2>&1 &
bash experiments/filtered_bc/run_libero.sh    > experiments/filtered_bc/logs/libero.log    2>&1 &
bash experiments/filtered_bc/run_robocasa.sh  > experiments/filtered_bc/logs/robocasa.log  2>&1 &
```

Each script runs 30 rollouts × 500 LoRA steps × batch 8 × 30 eval episodes per task on the default task list (10-task curated subset for MetaWorld, `libero_10` for LIBERO, 7-task subset for RoboCasa). Override via env vars (`BASE_CKPT`, `RESULTS_JSON`, `LIBERO_SUITE`, `ROBOCASA_TASK_SET`).

### Report

```bash
uv run python -m experiments.filtered_bc.make_report \
    --args.results experiments/filtered_bc/results_metaworld.json \
    --args.out experiments/filtered_bc/report_metaworld.md
```

## What's trainable

Freeze filter `All(.*llm.*, Not(.*lora.*))` (the stock LoRA recipe), so per-env:

- **Frozen:** PaliGemma LM base weights, action-expert base weights.
- **Trainable:** LoRA adapters on PaliGemma (rank 16) and action expert (rank 32); SigLip vision tower; `action_in_proj`, `action_out_proj`, `time_mlp_in`, `time_mlp_out`.

A one-off experiment confirmed the vision tower should remain trainable — freezing it regressed `basketball-v3` from 93% to 33% on 15 eval episodes. This matches the upstream `pi0_libero_low_mem_finetune` recipe.

## Critical environment variables

Both must be set whenever the in-process MetaWorld path runs a train-then-eval cycle, or the subprocess path serves a merged ckpt:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
```

Without `ALLOCATOR=platform`, JAX keeps GPU memory pooled after `del train_state`, and subsequent GPU allocations OOM a 46 GB L40.

## Norm stats

The filtered-BC LoRA configs don't carry their own norm_stats — both `build_training_dataset` and `_build_policy_from_model` fall back to reading them from `<base_ckpt>/assets/<asset_id>/`. No manual symlink needed.

## Testing

Smoke runs (above) are the regression check.
