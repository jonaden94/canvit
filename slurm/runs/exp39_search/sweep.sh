#!/bin/bash
# exp39 visual-search sweep on the V100 `jupyter` partition: 10 runs one after another per
# 48 h job (slurm/harness_sweep.sbatch), because that partition allows only 2 jobs per user.
#
#   TRAIN_COMMIT=<sha> bash slurm/runs/exp39_search/sweep.sh A      # job A: every config, seed 0
#   TRAIN_COMMIT=<sha> bash slurm/runs/exp39_search/sweep.sh B      # job B: seed replicates
#   TRAIN_COMMIT=<sha> bash slurm/runs/exp39_search/sweep.sh test   # 2 short runs, ~15 min
#
# Base recipe = the search defaults + Haoze's augmentations (his use_advanced_augmentations:
# flip, colour jitter + greyscale, rotation, affine, autocontrast; no random crop), fp32 on
# V100 (no bf16; the fp16 fallback has no GradScaler), 20000 steps (~74 epochs; the first
# exp39 run plateaued in localisation by ~6-10k and in presence by ~12k), validation and
# figures every 500, a checkpoint every 2000. Measured on that run: 0.63 s/step + ~55 s per
# validation => ~4.2 h per run, 10 runs ~42 h of the 48 h.
#
# Backbone: the exp22 foveated teacher-init ViT-B (retina fov 35, cmf_a 0.5, res 64;
# pretrained at view scale 2.0 = SearchConfig's default).
#
# Reading the results: one run's found has a standard error of ~0.02 (326 target-present
# val trials); base / sup-every / gain-reward have 3 seeds, so their spread is the noise floor.
set -euo pipefail

WHICH="${1:?usage: sweep.sh A|B|C|D|E|test}"
RUN_GROUP=jon_exp39_search
TASK=search
NGPU=1
MEM=64G
GPU=V100
PARTITION=jupyter

CFG_MODEL_REPO=/mnt/vast-nhr/projects/nib00021/jonathan/repos/canvit/logs/jon_exp22_full_runs/exp22-fovi-teacherinit-lrdrop-1196k/checkpoints/step-155648.pt
CFG_WANDB_PROJECT=jon_exp39_search
CFG_NUM_WORKERS=8
CFG_MAX_STEPS=20000
OPT_CKPT_EVERY=2000
EXTRA_ARGS="--cfg.aug.color-jitter --cfg.aug.rotate --cfg.aug.affine --cfg.aug.autocontrast --cfg.aug.no-crop --cfg.no-amp"

# per-run changes from the base recipe (appended after EXTRA_ARGS; tyro takes the last flag)
SUP="--cfg.supervise every"
GAIN="--cfg.supervise every --rl.reward difference"
AUG_DEFAULT="--cfg.aug.no-color-jitter --cfg.aug.no-rotate --cfg.aug.no-affine --cfg.aug.no-autocontrast --cfg.aug.crop"
AUG_NONE="--cfg.aug.no-color-jitter --cfg.aug.no-rotate --cfg.aug.no-affine --cfg.aug.no-autocontrast --cfg.aug.no-crop --cfg.aug.hflip-p 0"
PLR5="--rl.policy-lr 5e-4"
PLR10="--rl.policy-lr 1e-3"

case "$WHICH" in
A)  TIME=2-00:00:00
    SWEEP_SPEC="
base-s0              --cfg.seed 0
sup-every-s0         --cfg.seed 0 $SUP
gain-reward-s0       --cfg.seed 0 $GAIN
aug-default-s0       --cfg.seed 0 $AUG_DEFAULT
aug-none-s0          --cfg.seed 0 $AUG_NONE
policy-lr5e-4-s0     --cfg.seed 0 $PLR5
policy-lr1e-3-s0     --cfg.seed 0 $PLR10
nll-point-s0         --cfg.seed 0 --cfg.loss nll_point
shared-emb-s0        --cfg.seed 0 --cfg.embedding shared
head-lr3e-4-s0       --cfg.seed 0 --cfg.peak-lr 3e-4
" ;;
B)  TIME=2-00:00:00
    SWEEP_SPEC="
base-s1              --cfg.seed 1
base-s2              --cfg.seed 2
sup-every-s1         --cfg.seed 1 $SUP
sup-every-s2         --cfg.seed 2 $SUP
gain-reward-s1       --cfg.seed 1 $GAIN
gain-reward-s2       --cfg.seed 2 $GAIN
aug-default-s1       --cfg.seed 1 $AUG_DEFAULT
policy-lr5e-4-s1     --cfg.seed 1 $PLR5
policy-lr1e-3-s1     --cfg.seed 1 $PLR10
aug-none-s1          --cfg.seed 1 $AUG_NONE
" ;;
C|D)  # 2026-11-09, owner: replicate the only lead (head LR 3e-4, 1 seed so far) and the gain
    # reward's fixation effect (gain-reward vs sup-every differ ONLY in the reward; seeds 3-4
    # paired). One run of each kind per job, so either job alone is a balanced replicate.
    # A ran 10 runs in 44.4 h => ~4.4 h per run, 3 runs ~13.3 h.
    TIME=0-16:00:00
    if [ "$WHICH" = C ]; then HS=1; RS=3; else HS=2; RS=4; fi
    SWEEP_SPEC="
head-lr3e-4-s$HS       --cfg.seed $HS --cfg.peak-lr 3e-4
gain-reward-s$RS       --cfg.seed $RS $GAIN
sup-every-s$RS         --cfg.seed $RS $SUP
" ;;
E)  # 2026-11-10: job C (17206767) was cancelled by accident at step 17000 of its last run; that
    # run's local dir and W&B run (mlfn5a9g) were deleted, and it is redone here from scratch.
    TIME=0-06:00:00
    SWEEP_SPEC="
sup-every-s3         --cfg.seed 3 $SUP
" ;;
test)
    # Checks the wrapper: a run that fails at startup must not stop the next one; the second
    # run trains, validates, writes figures and eval.jsonl, and logs to a separate W&B
    # project (to look at the panel order) under its own group.
    TIME=0-00:30:00
    RUN_GROUP=jon_exp39_search_sweeptest
    CFG_WANDB_PROJECT=jon_exp39_search_sweeptest
    CFG_MAX_STEPS=120
    CFG_WARMUP_STEPS=20
    CFG_VAL_EVERY=60
    CFG_VIZ_EVERY=60
    OPT_CKPT_EVERY=0
    SWEEP_SPEC="
must-fail            --cfg.no-such-flag 1
short-gain-reward    --cfg.seed 0 $GAIN
" ;;
*)  echo "usage: sweep.sh A|B|C|D|E|test" >&2; exit 1 ;;
esac

if [ "${DRY_RUN:-0}" = 1 ]; then  # print what would run, submit nothing
    echo "EXTRA_ARGS=$EXTRA_ARGS"; printf '%s\n' "$SWEEP_SPEC"; exit 0
fi
TRAIN_COMMIT="${TRAIN_COMMIT:?set TRAIN_COMMIT to the search-v1 commit to run}"
FOVI_COMMIT=27b1bde

# Submit from the MAIN clone: harness_train.sbatch takes the venv (.venv-cu126) and the
# sibling repos (fovi) from the submit directory, and a git worktree has neither. The commit
# is pinned (the sweep reads harness_train.sbatch from TRAIN_COMMIT too), so the main
# clone's checked-out branch does not matter.
_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_SBATCH="$(cd "$_HERE/../.." && pwd)/harness_sweep.sbatch"
cd "$(git -C "$_HERE" rev-parse --path-format=absolute --git-common-dir)/.."
git cat-file -e "$TRAIN_COMMIT^{commit}" || { echo "unknown TRAIN_COMMIT $TRAIN_COMMIT" >&2; exit 1; }
mkdir -p "logs/$RUN_GROUP/sweep"
export TASK RUN_GROUP NGPU EXTRA_ARGS TRAIN_COMMIT FOVI_COMMIT SWEEP_SPEC
for v in $(compgen -v); do [[ "$v" == CFG_* || "$v" == OPT_* ]] && export "$v"; done

sbatch \
    --partition="$PARTITION" --gpus-per-node=$GPU:$NGPU --ntasks-per-node=$NGPU --mem=$MEM --time=$TIME \
    --job-name="exp39-sweep-$WHICH" \
    --output="logs/$RUN_GROUP/sweep/sweep-$WHICH-%j.log" \
    --error="logs/$RUN_GROUP/sweep/sweep-$WHICH-%j.log" \
    --export=ALL "$_SBATCH"
