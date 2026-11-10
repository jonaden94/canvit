#!/bin/bash
# exp33 — ImageNet-1k FULL-MODEL finetune through the harness at the CURRENT code base.
# Source: exp22-fovi step-1900544 -- NEW. exp22-fovi never had a downstream run. Best val/scene_cos_norm_t9 = 0.853339 over all 238 val points (wandb run r64ck13l); converted with to_hf 2026-07-31. NOTE the pre-existing step-516096-hf export is NOT the best and must not be used.
#
# Recipe is exp25's, copied value-for-value (which is itself the original canvit_specialize
# TPU in1k finetune, batch-adapted for one A100 by the recipe's own sanctioned rule:
# batch 256->64, peak_lr 2.5e-5->6.25e-6, warmup 25k->100k, 100,080@256 -> 401,408@64).
# Everything else byte-identical to the TPU recipe: wd 1e-4, grad_clip 1.0,
# label_smoothing 0.1, n_timesteps 4, min_vp_scale 0.05, t0 = full scene.
#
# vs exp25 the ONLY changes are the pins (current code under test) and, for in1k-fovi-1901k,
# a source checkpoint exp25 never had. EVAL_POLICY=random (coarse-to-fine is uniform-only / OOD for a fixed-scale foveated model).
set -euo pipefail

# === ESSENTIALS ===
RUN_GROUP=caglar_exp33_in1k_finetune
RUN_NAME=in1k-fovi-1901k
ARRAY="${ARRAY:-0-48%1}"     # 49 jobs x 8192 = 401,408 steps (~20 epochs @ batch 64).
                             # Override when RESUMING a partially-done run: the array is a
                             # BUDGET, not a schedule (job_index comes from the checkpoint's
                             # resume_state, not SLURM_ARRAY_TASK_ID), so pass the number of
                             # chunks still owed, e.g. ARRAY=0-44%1.
TIME="${TIME:-0-02:00:00}"   # 2h, NOT 12h. <=2h lands in Grete's `2h` QOS, which
                             # schedules in minutes; a 12h request goes to `normal` and waited
                             # ~24-31h BETWEEN chunks. Measured chunk times: uniform arms
                             # 69-78 min (comfortable), foveated arms 93-109 min (only ~11 min
                             # of headroom -- a timeout costs that chunk, which is recoverable
                             # because no checkpoint is written, so the next task just redoes
                             # it). steps_per_job CANNOT be lowered to buy margin:
                             # _check_schedule_invariants refuses to resume if it changes.
MEM=128G
NGPU=1
TASK=in1k

# === config (exp25 recipe) ===
CFG_WANDB_PROJECT=caglar_exp33_in1k_finetune
CFG_MODEL_REPO=/mnt/vast-nhr/projects/nib00021/jonathan/repos/canvit/logs/jon_exp22_full_runs/exp22-fovi/checkpoints/step-1900544-hf
CFG_PROBE_REPO=canvit/dinov3-vitb16-lvd1689m-in1k-512x512-linear-clf-probe  # fused into the head (TPU parity)
CFG_MODE=finetune
CFG_BATCH_SIZE=64
CFG_PEAK_LR=6.25e-6
CFG_WEIGHT_DECAY=1e-4
CFG_WARMUP_STEPS=100000
CFG_MAX_STEPS=401408
CFG_STEPS_PER_JOB=8192
CFG_N_TIMESTEPS=4
CFG_GRAD_CLIP=1.0
CFG_LABEL_SMOOTHING=0.1
CFG_MIN_VP_SCALE=0.05
CFG_EVAL_POLICY=random
CFG_VAL_EVERY=10000
OPT_RESUME=True              # 49-job array must resume across tasks (in1k default is False)
EXTRA_ARGS="--cfg.train-start-full --cfg.foveated-scale.fixed-scale 2.0"
# =================

TRAIN_COMMIT=716051a
PYTORCH_COMMIT=d616b7b
FOVI_COMMIT=c399d3b

# Repo root, derived from this script's own location (slurm/runs/<group>/<run>.sh),
# so the run submits from YOUR clone rather than one hardcoded checkout.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
mkdir -p "logs/$RUN_GROUP/$RUN_NAME/log"
export TASK RUN_GROUP RUN_NAME NGPU EXTRA_ARGS TRAIN_COMMIT PYTORCH_COMMIT FOVI_COMMIT
for v in $(compgen -v); do [[ "$v" == CFG_* || "$v" == OPT_* ]] && export "$v"; done

sbatch \
    --gpus-per-node=A100:$NGPU --ntasks-per-node=$NGPU --mem=$MEM --time=$TIME \
    --array="$ARRAY" \
    --output="logs/$RUN_GROUP/$RUN_NAME/log/job-%A_%a.log" \
    --error="logs/$RUN_GROUP/$RUN_NAME/log/job-%A_%a.log" \
    --export=ALL slurm/harness_train.sbatch
