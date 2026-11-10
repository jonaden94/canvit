#!/bin/bash
# exp32 PHASE B — the x0.1 LR drop for exp32-uniform16.
#
# DO NOT SUBMIT until exp32-uniform16 has actually produced step-1441792.pt. Phase A's
# array is a budget, not a schedule (job_index comes from the checkpoint, so the step count
# advances only for jobs that succeed) -- check the file exists first. The guard below
# refuses to submit otherwise, which is the whole safety property: the drop point is a
# FILENAME, so it cannot fire early, late, or twice no matter how many array tasks failed.
#
# Mechanism (identical to exp22's lrdrop branches):
#   CFG_SEED_CKPT -> SEED mode: loads model weights + standardizers + model config; FRESH
#     optimizer/scheduler/step=0; NEW wandb run (parent untouched). Parent step k <-> k-1441792.
#   CFG_WARMUP_STEPS=0 -> pure ConstantLR: LR = peak_lr = 4e-5 from step 0 (= 4e-4 x 0.1).
#   --cfg.init-backbone-from-teacher is DROPPED even for the teacherinit arms: the seed
#     checkpoint already carries trained weights, so the flag would re-init the backbone
#     and then be immediately overwritten by the seed load.
#   Fresh Adam moments are the one unavoidable difference vs literally continuing the parent
#     -- SEED mode carries no optimizer state. Standard for LR-drop branch experiments.
#   Data: fresh run -> shard schedule restarts at job 0, so early shards repeat. Harmless.
#
# ARRAY 0-24%1: 25 x 8192 = 204,800 steps after the drop, per the owner's spec (the exp22
# branches instead ran to the parent's full 2,007,040-equivalent -- deliberately shorter here).
set -euo pipefail

# === ESSENTIALS ===
RUN_GROUP=caglar_exp32_pretrain_lrdrop
RUN_NAME=exp32-uniform16-lrdrop
ARRAY=0-24%1                  # 25 jobs x 8192 = 204,800 steps
TIME=0-02:00:00
MEM=128G
NGPU=1
TASK=distill

SEED_CKPT=/mnt/vast-nhr/projects/nib00021/jonathan/repos/canvit/logs/jon_exp32_pretrain_lrdrop/exp32-uniform16/checkpoints/step-1441792.pt
if [ ! -f "$SEED_CKPT" ]; then
    echo "REFUSING: $SEED_CKPT does not exist yet."
    echo "Phase A (exp32-uniform16) has not reached step 1441792."
    echo "Re-submit the remaining phase-A tasks first, then run this launcher."
    exit 1
fi

# === config (same as phase A, except the LR and no warmup) ===
CFG_WANDB_PROJECT=caglar_exp32_pretrain_lrdrop
CFG_SEED_CKPT=$SEED_CKPT
CFG_PEAK_LR=0.00004           # 4e-4 x 0.1
CFG_WARMUP_STEPS=0            # -> ConstantLR at 4e-5 from step 0
CFG_BATCH_SIZE_PER_GPU=64
CFG_STEPS_PER_JOB=8192
CFG_VAL_EVERY=8192
CFG_LOG_EVERY=512
CFG_NUM_WORKERS=4
EXTRA_ARGS="--cfg.model.patcher-name uniform"
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
    --gpus-per-node=A100:$NGPU \
    --ntasks-per-node=$NGPU \
    --mem=$MEM \
    --time=$TIME \
    --array="$ARRAY" \
    --output="logs/$RUN_GROUP/$RUN_NAME/log/job-%A_%a.log" \
    --error="logs/$RUN_GROUP/$RUN_NAME/log/job-%A_%a.log" \
    --export=ALL \
    slurm/harness_train.sbatch
