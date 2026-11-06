#!/bin/bash
# Visual search (COCO-Search18), FIRST TRAINING RUN. Class-cued search on a frozen
# pretrained foveated CanViT; task head + policy trained jointly. Target-present and
# target-absent trials (condition=both). Design and options: claude_dev/search/01-summary.md.
#
# Recipe = the search defaults, plus Haoze's augmentations (his use_advanced_augmentations:
# true => flip, colour jitter + greyscale, rotation, affine, autocontrast; no random crop).
# Everything else is the default: heatmap + 0.75 * focal presence loss on the last glimpse;
# PG policy with the terminal -KL reward, masked on target-absent trials; 20000 steps
# (~150 epochs at batch 16), validation + figures every 500.
#
# Backbone: the exp22 foveated teacher-init ViT-B that the exp36 policy runs used
# (retina fov 35, cmf_a 0.5, res 64; pretrained at view scale 2.0 = SearchConfig's default).
#
# V100. V100s exist only in the `jupyter` partition. V100 (sm_70) has no bfloat16, and the
# harness would fall back to float16 autocast WITHOUT a GradScaler, so this trains in fp32
# (--cfg.no-amp). For an A100: PARTITION=grete:shared GPU=A100 and drop --cfg.no-amp.
#
# ONE JOB, NOT AN ARRAY: resume is not tested for the search task, so the run must fit the
# walltime. TIME is set from the V100 smoke run's step time (exp38). Checkpoints every 2000
# steps keep a crashed run's progress on disk; best.pt follows hit_final_policy.
#
# Later variants (not this run): --cfg.supervise every; --rl.reward difference (needs
# supervise every).
set -euo pipefail

# === ESSENTIALS ===
RUN_GROUP=jon_exp39_search
SEED="${SEED:-0}"
RUN_NAME=search-v1-haozeaug-s$SEED
TIME="${TIME:?set TIME from the step time of the exp38 V100 smoke run}"
MEM=64G
NGPU=1                       # search is single-GPU only (supports_ddp=False)
GPU="${GPU:-V100}"
PARTITION="${PARTITION:-jupyter}"   # the only partition with V100s
TASK=search

CFG_MODEL_REPO=/mnt/vast-nhr/projects/nib00021/jonathan/repos/canvit/logs/jon_exp22_full_runs/exp22-fovi-teacherinit-lrdrop-1196k/checkpoints/step-155648.pt

# === config ===
CFG_SEED=$SEED
CFG_WANDB_PROJECT=jon_exp39_search
CFG_NUM_WORKERS=8
OPT_CKPT_EVERY=2000
EXTRA_ARGS="--cfg.aug.color-jitter --cfg.aug.rotate --cfg.aug.affine --cfg.aug.autocontrast --cfg.aug.no-crop --cfg.no-amp"
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
    --partition="$PARTITION" --gpus-per-node=$GPU:$NGPU --ntasks-per-node=$NGPU --mem=$MEM --time=$TIME \
    --output="logs/$RUN_GROUP/$RUN_NAME/log/job-%j.log" \
    --error="logs/$RUN_GROUP/$RUN_NAME/log/job-%j.log" \
    --export=ALL "$_SBATCH"
