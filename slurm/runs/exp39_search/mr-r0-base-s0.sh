#!/bin/bash
# exp39 base recipe on the NEW-geometry backbone (2026-11-20, owner's request): the search default
# PG policy (REINFORCE on the LayerNorm'd canvas tokens + cue, no BatchNorm = the simplest
# pg-lnfeat of the exp36_1/policy-algos ADE20K study) on COCO-Search18, with
#   - code: branch search-fixed-retina (main c45fff9 + --cfg.retina), i.e. after the fovi port;
#   - backbone: exp38-multi-retina-4gpu-lr2x-s1 step-253952 (pretrained on the ported geometry,
#     9 retinas, fov 50, view scale 2.0; its newest = best checkpoint, val_metric 0.9367),
#     used through ONE retina: cmf0.75_res65_cps6 (retina 0, the most foveated; val 0.9369).
#
# Recipe otherwise = sweep.sh's `base`: search defaults + Haoze's augmentations, fp32 on V100,
# 20000 steps, validation + figures every 500, checkpoint every 2000. NOT directly comparable to
# exp39's base-s0/1/2 (exp22 backbone: fov 35, cmf 0.5, res 64, old geometry; found_t3_policy
# 0.880-0.890): the backbone differs.
#
#   TRAIN_COMMIT=<search-fixed-retina commit> bash slurm/runs/exp39_search/mr-r0-base-s0.sh
#
# Time: exp39 runs took ~4.4 h on a V100; 6 h walltime.
set -euo pipefail

# === ESSENTIALS ===
RUN_GROUP=jon_exp39_search
SEED="${SEED:-0}"
RUN_NAME=mr-r0-base-s$SEED
TIME="${TIME:-0-06:00:00}"
MEM=64G
NGPU=1                       # search is single-GPU only
GPU="${GPU:-V100}"
PARTITION="${PARTITION:-jupyter}"   # the only partition with V100s
TASK=search

CFG_MODEL_REPO=/mnt/vast-nhr/projects/nib00021/jonathan/repos/canvit/logs/jon_exp38_multi_retina/exp38-multi-retina-4gpu-lr2x-s1/checkpoints/step-253952.pt

# === config (= sweep.sh base, + the retina) ===
CFG_SEED=$SEED
CFG_WANDB_PROJECT=jon_exp39_search
CFG_NUM_WORKERS=8
CFG_MAX_STEPS=20000
CFG_RETINA=cmf0.75_res65_cps6
OPT_CKPT_EVERY=2000
EXTRA_ARGS="--cfg.aug.color-jitter --cfg.aug.rotate --cfg.aug.affine --cfg.aug.autocontrast --cfg.aug.no-crop --cfg.no-amp"
# =================

TRAIN_COMMIT="${TRAIN_COMMIT:?set TRAIN_COMMIT to the search-fixed-retina commit to run}"

_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_SBATCH="$(cd "$_HERE/../.." && pwd)/harness_train.sbatch"
cd "$(git -C "$_HERE" rev-parse --path-format=absolute --git-common-dir)/.."
[ -e "logs/$RUN_GROUP/$RUN_NAME" ] && { echo "logs/$RUN_GROUP/$RUN_NAME exists -- refusing to reuse a run dir"; exit 1; }
mkdir -p "logs/$RUN_GROUP/$RUN_NAME/log"
export TASK RUN_GROUP RUN_NAME NGPU EXTRA_ARGS TRAIN_COMMIT
for v in $(compgen -v); do [[ "$v" == CFG_* || "$v" == OPT_* ]] && export "$v"; done

sbatch \
    --partition="$PARTITION" --gpus-per-node=$GPU:$NGPU --ntasks-per-node=$NGPU --mem=$MEM --time=$TIME \
    --output="logs/$RUN_GROUP/$RUN_NAME/log/job-%j.log" \
    --error="logs/$RUN_GROUP/$RUN_NAME/log/job-%j.log" \
    --export=ALL "$_SBATCH"
