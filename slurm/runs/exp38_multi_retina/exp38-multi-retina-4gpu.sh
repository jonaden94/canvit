#!/bin/bash
# exp38 MULTI-RETINA — one foveated CanViT trained on 9 retinas at once, 4-GPU DDP.
#
# BASE. exp32-fovi-teacherinit (teacher-init backbone, FiLM fourier 256 / sigma 4, doubleres,
# fixed scale 2) with exp37 lr2x-s1's DDP recipe (4 GPUs x 64, peak LR 8e-4 = 4e-4 * sqrt(4),
# 100k-step warmup) and exp32-bilinear's rf-interp bilinear. Changed beyond that:
#   * fov 35 -> 50, and the single retina (res 64, cmf_a 0.5, cps 5) -> the 9 retinas of
#     fovi/notebooks/visual_search/proposed_patch_embedding.ipynb section 1:
#     cmf_a 0.75 / 2.2 / 7.7 x (res 65, cps 6, force False) / (80, 7, True) / (92, 8, True);
#     every one gives 115 patches, k ~60 / ~86 / ~117.
#   * one kernel tensor on the largest native grid (22 x 22), read bilinearly by every retina;
#     FiLM's (gamma, beta) depend on patch position AND retina (one learned vector per retina).
#   * each sample's retina is drawn uniformly per step and held for all its glimpses;
#     validation runs all 9 retinas (val_metric = their mean; per-retina keys <retina>/...).
#
# NO PAIRED BASELINE. Different fov, retinas and patch embedding, so no exp32/exp37 run is
# comparable per step; the same seed does not even give the same init (new parameters change
# the RNG order). What this run can show: it trains, every retina improves, and how far apart
# the retinas end up. Whether one retina loses from sharing needs a single-retina control at
# these settings (not launched).
#
# LENGTH. 24 x 8192 = 196608 steps, as exp37. Student step time measured 2026-11-06 (synthetic
# images, MIG slice, batch 16): 295 ms per 2-glimpse chunk vs 333 ms for exp37's retina
# (115 vs 140 patches), so a job should fit 2 h like exp37's (~94 min) plus ~4 min for 9x validation.
#
# Submit: bash slurm/runs/exp38_multi_retina/exp38-multi-retina-4gpu.sh
set -euo pipefail

# === ESSENTIALS ===
RUN_GROUP=jon_exp38_multi_retina
SEED="${SEED:-1}"
RUN_NAME=exp38-multi-retina-4gpu-lr2x-s$SEED
ARRAY="${ARRAY:-0-23%1}"      # 24 x 8192 = 196608 steps
TIME=0-02:00:00
MEM=512G                      # 128G per GPU, as exp37
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
RETINAS="0.75 65 6 False 0.75 80 7 True 0.75 92 8 True 2.2 65 6 False 2.2 80 7 True 2.2 92 8 True 7.7 65 6 False 7.7 80 7 True 7.7 92 8 True"
EXTRA_ARGS="--cfg.model.patcher-name foveated --cfg.model.foveated-patcher.fov 50 --cfg.model.foveated-patcher.retinas $RETINAS --cfg.model.foveated-patcher.arch-flag doubleres --cfg.model.foveated-patcher.rf-interp bilinear --cfg.model.foveated-patcher.conditioning.mode film --cfg.model.foveated-patcher.conditioning.film.fourier.num-features 256 --cfg.model.foveated-patcher.conditioning.film.fourier.sigma 4 --cfg.foveated-scale.fixed-scale 2.0 --cfg.init-backbone-from-teacher"
# =================

TRAIN_COMMIT=15b4ed8          # adds FoveatedPatcherConfig.retinas (branch multi-retina)
FOVI_COMMIT=27b1bde           # fovi HEAD 2026-11-06; its min-coverage k (53b750f) is what the 9 retinas' k were measured with

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
