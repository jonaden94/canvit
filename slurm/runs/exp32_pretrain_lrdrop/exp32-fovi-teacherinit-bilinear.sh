#!/bin/bash
# exp32 BILINEAR arm — does fovi's bilinear kernel readout beat its default nearest-cell one?
#
# A/B against exp32-fovi-teacherinit-s$SEED (jon_exp32_pretrain_lrdrop), the nearest arm.
# The config is exp32-fovi-teacherinit-seed.sh's, plus ONE flag:
#   --cfg.model.foveated-patcher.rf-interp bilinear
# Each sample then reads its 4 surrounding cells of fovi's kernel grid with bilinear weights
# instead of snapping to the nearest one. Same doubleres grid, same parameters.
#
# WHY THE NEW PINS DO NOT BREAK THE A/B (checked 2026-11-04). exp32 pinned 716051a +
# CanViT-PyTorch d616b7b + fovi c399d3b. Diffed against the pins below, the distill training
# path changed only in imports, renames, docstrings, off-by-default fields, and `validate`
# returning its series rather than logging it (same val/ keys). The resolved config differs
# from exp32's only in rf_interp. At one seed both arms start from bit-identical weights and
# read the same shard schedule, so s$SEED vs exp32-s$SEED is a paired comparison.
# Remaining differences: GPU nondeterminism (~0.005 between same-seed runs), and the IN21k
# workspace u25995-inet21k-feat2 (the expired `feat` re-staged; normalizer stats verified
# identical for exp37).
#
# READING IT (one seed, owner's choice). Gap vs exp32-s$SEED on val/scene_cos_raw:
#   > ~0.027 (exp32's seed spread at 188k), consistent along the curve -> real effect.
#   < ~0.005 -> no meaningful effect.   In between -> run more seeds.
#
# Seed 4: SEED=4 bash slurm/runs/exp32_pretrain_lrdrop/exp32-fovi-teacherinit-bilinear.sh
set -euo pipefail

# === ESSENTIALS ===
RUN_GROUP=jon_exp32_pretrain_lrdrop
SEED="${SEED:?set SEED to the exp32 seed to pair with (1..5)}"
RUN_NAME=exp32-fovi-teacherinit-bilinear-s$SEED
ARRAY=0-24%1                  # 25 x 8192 = 204800 steps, as exp32-fovi-teacherinit-s*
TIME=0-02:00:00
MEM=128G
NGPU=1
TASK=distill

# === config: exp32-fovi-teacherinit-seed.sh's, plus rf-interp bilinear ===
CFG_WANDB_PROJECT=jon_exp32_pretrain_lrdrop   # where exp32-s1..s5 are (renamed server-side in d0ef952); s4 first logged to exp32_pretrain_lrdrop by mistake
CFG_SEED=$SEED
CFG_PEAK_LR=0.0004
CFG_BATCH_SIZE_PER_GPU=64
CFG_STEPS_PER_JOB=8192
CFG_VAL_EVERY=8192
CFG_LOG_EVERY=512
CFG_NUM_WORKERS=4
EXTRA_ARGS="--cfg.model.patcher-name foveated --cfg.model.foveated-patcher.fov 35 --cfg.model.foveated-patcher.resolution 64 --cfg.model.foveated-patcher.cmf-a 0.5 --cfg.model.foveated-patcher.cart-patch-size 5 --cfg.model.foveated-patcher.arch-flag doubleres --cfg.model.foveated-patcher.conditioning.mode film --cfg.model.foveated-patcher.conditioning.film.fourier.num-features 256 --cfg.model.foveated-patcher.conditioning.film.fourier.sigma 4 --cfg.foveated-scale.fixed-scale 2.0 --cfg.init-backbone-from-teacher --cfg.model.foveated-patcher.rf-interp bilinear"
# =================

TRAIN_COMMIT=9ecc492          # adds FoveatedPatcherConfig.rf_interp
FOVI_COMMIT=ed9b431           # adds KNNConvLayer(rf_interp='bilinear')

# Repo root, derived from this script's own location (slurm/runs/<group>/<run>.sh),
# so the run submits from YOUR clone rather than one hardcoded checkout.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
mkdir -p "logs/$RUN_GROUP/$RUN_NAME/log"
export TASK RUN_GROUP RUN_NAME NGPU EXTRA_ARGS TRAIN_COMMIT FOVI_COMMIT
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
