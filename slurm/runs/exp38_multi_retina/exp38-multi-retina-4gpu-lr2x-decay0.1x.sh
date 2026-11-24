#!/bin/bash
# exp38 MULTI-RETINA, STAGE 1 LR DECAY (canvit_claude/canvit/active_pretraining/01-plan.md, stage 1).
# exp38-multi-retina-4gpu-lr2x-s1 continued from its best val/scene_cos_norm_t9 checkpoint
# (step 360,448: 0.88243 mean over the 9 retinas, val_metric 0.93977) at 0.1x its LR: 8e-5,
# constant, no warmup. Everything else as exp38 (4-GPU DDP, 4 x 64, 9 retinas, same model flags).
# RESUME_FROM (not seed): the full state of exp38 at that step continues in this run's directory --
# weights, AdamW (unchanged), step counter (starts at 360,448), shard schedule (job 44 onward, as
# exp38's next array task would) -- and only the LR is this run's. W&B: a new run forked from
# exp38's run 33gy6d4a at step 360,448, so its curves continue exp38's.
# LENGTH. 10 x 8192 = 81,920 steps; pick the result by its best eval, not its last.
#
# Submit: bash slurm/runs/exp38_multi_retina/exp38-multi-retina-4gpu-lr2x-decay0.1x.sh
set -euo pipefail

# === ESSENTIALS ===
RUN_GROUP=jon_exp38_multi_retina
SEED="${SEED:-1}"
FROM_STEP=360448
RUN_NAME=exp38-multi-retina-4gpu-lr2x-s$SEED-decay0.1x-from360k
ARRAY="${ARRAY:-0-9%1}"       # 10 x 8192 = 81,920 steps
TIME=0-02:00:00
MEM=512G                      # 128G per GPU, as exp38
NGPU=4
TASK=distill

# === config ===
CFG_WANDB_PROJECT=jon_exp38_multi_retina
CFG_SEED=$SEED
CFG_PEAK_LR=0.00008           # 0.1 x exp38's 8e-4
CFG_WARMUP_STEPS=0            # no warmup (explicit; exp38's default is 100k)
CFG_BATCH_SIZE_PER_GPU=64
CFG_STEPS_PER_JOB=8192
CFG_VAL_EVERY=8192
CFG_LOG_EVERY=512
CFG_NUM_WORKERS=4
FROM_CKPT=/mnt/vast-nhr/projects/nib00021/jonathan/repos/canvit/logs/jon_exp38_multi_retina/exp38-multi-retina-4gpu-lr2x-s1/checkpoints/step-$FROM_STEP.pt
[ -f "$FROM_CKPT" ] || { echo "REFUSING: $FROM_CKPT missing"; exit 1; }
CFG_RESUME_FROM=$FROM_CKPT
RETINAS="0.75 65 6 False 0.75 80 7 True 0.75 92 8 True 2.2 65 6 False 2.2 80 7 True 2.2 92 8 True 7.7 65 6 False 7.7 80 7 True 7.7 92 8 True"
EXTRA_ARGS="--cfg.model.patcher-name foveated --cfg.model.foveated-patcher.fov 50 --cfg.model.foveated-patcher.retinas $RETINAS --cfg.model.foveated-patcher.arch-flag doubleres --cfg.model.foveated-patcher.rf-interp bilinear --cfg.model.foveated-patcher.conditioning.mode film --cfg.model.foveated-patcher.conditioning.film.fourier.num-features 256 --cfg.model.foveated-patcher.conditioning.film.fourier.sigma 4 --cfg.foveated-scale.fixed-scale 2.0 --cfg.init-backbone-from-teacher"
# =================

# main with RunSettings.resume_from (exp38 itself ran 15b4ed8; main is bit-identical for this geometry).
TRAIN_COMMIT="${TRAIN_COMMIT:?TRAIN_COMMIT=<main commit with resume_from>}"

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
mkdir -p "logs/$RUN_GROUP/$RUN_NAME/log"
export TASK RUN_GROUP RUN_NAME NGPU EXTRA_ARGS TRAIN_COMMIT
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
