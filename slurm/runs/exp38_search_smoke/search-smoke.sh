#!/bin/bash
# Visual search (COCO-Search18, TP + TA) SMOKE run: does the search task train and validate
# end to end on a GPU with a real pretrained foveated backbone? Not a training run -- 300
# steps, validation every 100 (random / greedy / policy fixations), figures every 100.
#
# Backbone: the exp22 foveated teacher-init ViT-B that the exp36 foveated policy runs used
# (pretrained at view scale 2.0, which is SearchConfig's default foveated_scale).
# Everything else is the search defaults: frozen backbone, task head + policy trained
# jointly, condition=both. Design: claude_dev/search/00-design.md.
#
# What to check in the log: train total_loss falls; every validation prints loss_t*/hit_t*
# and the presence metrics for all three fixation rules; PNGs appear under
# logs/$RUN_GROUP/$RUN_NAME/visualization/. No tracker (tracker=none): the numbers are in
# the job log only.
set -euo pipefail

# === ESSENTIALS ===
RUN_GROUP=jon_exp38_search_smoke
SEED="${SEED:-0}"
RUN_NAME=search-smoke-s$SEED
TIME="${TIME:-0-00:30:00}"
MEM=64G
NGPU=1                       # search is single-GPU only (supports_ddp=False)
TASK=search

CFG_MODEL_REPO=/mnt/vast-nhr/projects/nib00021/jonathan/repos/canvit/logs/jon_exp22_full_runs/exp22-fovi-teacherinit-lrdrop-1196k/checkpoints/step-155648.pt

# === config ===
CFG_SEED=$SEED
CFG_MAX_STEPS=300
CFG_WARMUP_STEPS=50
CFG_VAL_EVERY=100
CFG_VIZ_EVERY=100
CFG_LOG_EVERY=20
CFG_NUM_WORKERS=8
CFG_TRACKER=none
EXTRA_ARGS=""
# =================

TRAIN_COMMIT="${TRAIN_COMMIT:?set TRAIN_COMMIT to the search-v1 commit to run}"
FOVI_COMMIT=27b1bde

# Submit from the MAIN clone: harness_train.sbatch takes the venv (.venv-cu126) and the
# sibling repos (fovi) from the submit directory, and a git worktree has neither. The
# commit is pinned, so the main clone's checked-out branch does not matter. The sbatch
# script itself is THIS tree's copy (it is read at submit time).
_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_SBATCH="$(cd "$_HERE/../.." && pwd)/harness_train.sbatch"
cd "$(git -C "$_HERE" rev-parse --path-format=absolute --git-common-dir)/.."
mkdir -p "logs/$RUN_GROUP/$RUN_NAME/log"
export TASK RUN_GROUP RUN_NAME NGPU EXTRA_ARGS TRAIN_COMMIT FOVI_COMMIT
for v in $(compgen -v); do [[ "$v" == CFG_* || "$v" == OPT_* ]] && export "$v"; done

sbatch \
    --gpus-per-node=A100:$NGPU --ntasks-per-node=$NGPU --mem=$MEM --time=$TIME \
    --output="logs/$RUN_GROUP/$RUN_NAME/log/job-%j.log" \
    --error="logs/$RUN_GROUP/$RUN_NAME/log/job-%j.log" \
    --export=ALL "$_SBATCH"
