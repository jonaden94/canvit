#!/bin/bash
# exp34 — ADE20K frozen-PROBE training through the harness at the CURRENT code base.
# Source: exp22-fovi-teacherinit-lrdrop-1196k step-155648 (best val/scene_cos_norm_t9; the source exp24/exp25 used)
#
# Recipe is exp24's = the original canvit_specialize ade20k probe, reproduced by the harness
# ade20k defaults (mIoU gate passed vs the specialize-derived standalone): frozen backbone
# (default preset = TrainSpec.probe), 40k steps, random-view training, n_timesteps 10,
# scene 512, canvas_grid 32, val resize squish.
#
# resize_mode=squish for ALL arms incl. foveated -- the protocol every earlier CanViT /
# specialize number was measured under. It distorts aspect ratio, so it is not the choice for
# a human-viewing comparison (center_crop preserves the geometry foveated sampling assumes),
# but it is the one that makes these numbers comparable to exp24 and to the published values.
#
# vs exp24 the ONLY changes are the pins and, for ade20k-fovi-1901k, a source exp24 never had.
set -euo pipefail

# === ESSENTIALS ===
RUN_GROUP=caglar_exp34_ade20k_probe
RUN_NAME=ade20k-fovi-ti-1196k
ARRAY=0-0%1
TIME=0-08:00:00
MEM=64G
NGPU=1                       # ade20k has supports_ddp=False
TASK=ade20k

# === config (exp24 recipe) ===
CFG_WANDB_PROJECT=caglar_exp34_ade20k_probe
CFG_MODEL_REPO=/mnt/vast-nhr/projects/nib00021/jonathan/repos/canvit/logs/jon_exp22_full_runs/exp22-fovi-teacherinit-lrdrop-1196k/checkpoints/step-155648-hf
CFG_RESIZE_MODE=squish
EXTRA_ARGS="--cfg.foveated-scale.fixed-scale 2.0"
# =================

TRAIN_COMMIT=716051a
PYTORCH_COMMIT=d616b7b
FOVI_COMMIT=c399d3b

# Repo root, derived from this script's own location (slurm/runs/<group>/<run>.sh),
# so the run submits from YOUR clone rather than one hardcoded checkout.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
mkdir -p "logs/$RUN_GROUP/$RUN_NAME/log"
export ADE20K_ROOT="${ADE20K_ROOT:-/mnt/vast-nhr/projects/nib00021/jonathan/datasets/zhoubolei--scene_parse_150/ADEChallengeData2016}"
export TASK RUN_GROUP RUN_NAME NGPU EXTRA_ARGS TRAIN_COMMIT PYTORCH_COMMIT FOVI_COMMIT
for v in $(compgen -v); do [[ "$v" == CFG_* || "$v" == OPT_* ]] && export "$v"; done

sbatch \
    --gpus-per-node=A100:$NGPU --ntasks-per-node=$NGPU --mem=$MEM --time=$TIME \
    --array="$ARRAY" \
    --output="logs/$RUN_GROUP/$RUN_NAME/log/job-%A_%a.log" \
    --error="logs/$RUN_GROUP/$RUN_NAME/log/job-%A_%a.log" \
    --export=ALL slurm/harness_train.sbatch
