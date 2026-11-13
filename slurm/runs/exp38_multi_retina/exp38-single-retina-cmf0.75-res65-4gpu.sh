#!/bin/bash
# exp38 SINGLE-RETINA CONTROL — exp38-multi-retina-4gpu.sh with only retina 0 of its nine
# (cmf_a 0.75, res 65, cps 6, force False; 115 patches, k 61), 4-GPU DDP.
#
# Identical to the multi-retina run (same pins, recipe, seed, length) except:
#   * one fixed retina, set by the top-level fields instead of `retinas`; every sample uses it
#     (multi-retina: ~1/9 of samples);
#   * no per-retina embedding in FiLM (a single-retina patcher has none);
#   * the retina's own doubleres kernel grid, 16 x 16 (multi-retina: one shared 22 x 22 grid,
#     sized for the densest retina). Owner's choice 2026-11-12: compare with the grid this
#     retina would naturally get.
#
# COMPARE at the same step: this run's `val_metric` (= scene_cos_raw) vs the multi-retina run's
# `cmf0.75_res65_cps6/scene_cos_raw` (its own `val_metric` is the mean over 9 retinas). Same 256
# val images. One seed each; exp32's seed spread was ~0.01 at 196k steps.
#
# Pins: the multi-retina run's 15b4ed8 + fovi 27b1bde, so the code is identical. Post-port main
# would behave the same (geometry gated bit-identical, the other changes are no-ops for distill).
#
# Submit: bash slurm/runs/exp38_multi_retina/exp38-single-retina-cmf0.75-res65-4gpu.sh
set -euo pipefail

# === ESSENTIALS ===
RUN_GROUP=jon_exp38_multi_retina
SEED="${SEED:-1}"
RUN_NAME=exp38-single-retina-cmf0.75-res65-4gpu-lr2x-s$SEED
ARRAY="${ARRAY:-0-23%1}"      # 24 x 8192 = 196608 steps, as the multi-retina run
TIME=0-02:00:00
MEM=512G
NGPU=4
TASK=distill

# === config ===
CFG_WANDB_PROJECT=jon_exp38_multi_retina
CFG_SEED=$SEED
CFG_PEAK_LR=0.0008            # exp37 lr2x
CFG_BATCH_SIZE_PER_GPU=64
CFG_STEPS_PER_JOB=8192
CFG_VAL_EVERY=8192
CFG_LOG_EVERY=512
CFG_NUM_WORKERS=4
EXTRA_ARGS="--cfg.model.patcher-name foveated --cfg.model.foveated-patcher.fov 50 --cfg.model.foveated-patcher.cmf-a 0.75 --cfg.model.foveated-patcher.resolution 65 --cfg.model.foveated-patcher.cart-patch-size 6 --cfg.model.foveated-patcher.no-force-patches-less-than-matched --cfg.model.foveated-patcher.arch-flag doubleres --cfg.model.foveated-patcher.rf-interp bilinear --cfg.model.foveated-patcher.conditioning.mode film --cfg.model.foveated-patcher.conditioning.film.fourier.num-features 256 --cfg.model.foveated-patcher.conditioning.film.fourier.sigma 4 --cfg.foveated-scale.fixed-scale 2.0 --cfg.init-backbone-from-teacher"
# =================

TRAIN_COMMIT=15b4ed8          # same as the multi-retina run
FOVI_COMMIT=27b1bde

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
