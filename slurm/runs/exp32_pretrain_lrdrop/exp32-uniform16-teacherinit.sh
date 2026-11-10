#!/bin/bash
# exp32 PHASE A — from-scratch replication of exp22-uniform16-teacherinit at the CURRENT code base.
#
# Replicates jon_exp22_full_runs/exp22-uniform16-teacherinit config-for-config. Every CFG_ value and
# every EXTRA_ARGS flag below was copied from that launcher; only the entry point (harness
# instead of the deleted canvit_pretrain.train), the pins, and normalizer_shards differ.
#
# KNOWN, ACCEPTED DIFFS vs exp22 (owner-confirmed 2026-07-31):
#   1. normalizer_shards = 4 (current default) vs exp22's effective 1 shard / all samples.
#      exp22's pin (fe24aa1) had no such knob -- only normalizer_max_samples=0. Pooling 4
#      shards is the more accurate estimate but shifts the target normalization, so the
#      LOSS SCALE is not identical to exp22's and curves will not overlay exactly.
#   2. Entry point: canvit_train.harness.run vs canvit_pretrain.train. Already A/B-verified
#      (exp23, exp26) -- that comparison is why this replication is worth running at all.
#   3. PYTORCH pin 3277048 -> 1f5121b. VERIFIED IRRELEVANT HERE: the diff touches only
#      data/ade20k.py, metrics.py, model/classification, model/segmentation and policy/ --
#      nothing in the distill training path (no model/pretraining, backbone, patcher,
#      attention, rope, standardizers). So pretraining numerics are unchanged.
#
# PHASE A = warmup(100k) -> constant 4e-4, exactly as exp22. It stops at step 630784;
# the x0.1 LR drop is a SEPARATE launcher (see exp32-uniform16-teacherinit-lrdrop.sh) that seeds from this
# run's step-630784.pt. There is no in-run LR-drop feature and the drop point is therefore
# a FILENAME, not a step comparison -- it cannot fire early if array tasks fail.
#
# ARRAY = 77 x 8192 = 630784 steps.
# IMPORTANT: the step count advances only for jobs that SUCCEED (job_index comes from the
# checkpoint's resume_state, not SLURM_ARRAY_TASK_ID). So this array is a BUDGET, not a
# schedule: if tasks die, re-submit the remainder until step-630784.pt exists, and only
# then launch the lrdrop phase.
set -euo pipefail

# === ESSENTIALS ===
RUN_GROUP=caglar_exp32_pretrain_lrdrop
RUN_NAME=exp32-uniform16-teacherinit
ARRAY=0-76%1
TIME=0-02:00:00
MEM=128G
NGPU=1
TASK=distill

# === config (copied from exp22; see header for the three accepted diffs) ===
CFG_WANDB_PROJECT=caglar_exp32_pretrain_lrdrop
CFG_PEAK_LR=0.0004
CFG_BATCH_SIZE_PER_GPU=64
CFG_STEPS_PER_JOB=8192
CFG_VAL_EVERY=8192            # validate once per job (= steps_per_job), as exp22
CFG_LOG_EVERY=512
CFG_NUM_WORKERS=4
EXTRA_ARGS="--cfg.model.patcher-name uniform --cfg.init-backbone-from-teacher"
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
