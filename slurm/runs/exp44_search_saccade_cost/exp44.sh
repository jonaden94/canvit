#!/bin/bash
# exp44 (owner 2026-11-27): does a saccade cost make the search policy's scanpaths human-like (short
# saccades), and does the policy need to know where it looks for that? exp43's PG recipe 1:1 (frozen exp38
# step-360448, all 9 retinas, FiLM U-Nets 64/3/3, fractional per-glimpse reward with running norm, supervise
# every, target-present mask, AdamW 4e-4 constant for head and policy, batch 64, 5,000 steps, val every 500)
# plus two factors:
#   POS   nopos  = the policy sees the canvas only (as exp43)
#         efcopy = + the efference copy (--rl.efference-copy): one input channel, a Gaussian bump at the
#                  current fixation, so it knows which of the glimpses on its canvas is the current one
#   cost  BETA=0 (no cost flags) | BETA=0.3 (r = 1/32, half a policy cell) | BETA=0.3 R=0.06 (the strongest
#         retina's cortical magnification, Valentin) | BETA=1.0 (cost as strong as the task)
# at entropy target 4.0 and 6.0, seeds 1-2. Cost: Valentin's --rl.saccade-cost log (main 1b09497):
# target_t = z(r_t) - beta z_batch(log(1 + d_t / r)), d_t = saccade length in scene units (width 2), the
# first saccade (from t0's centre) pays, target-absent trials carry neither reward nor cost.
# The nopos BETA=0 arm IS exp43 (pg-s1/s2, pg-e6.0-s1/s2): bit-identical on this code (CPU parity probe,
# canvit_claude/canvit/active_pretraining/exp44_saccade_cost.md), so it is refused here, not rerun.
# Validation adds saccade_len_t<t>_<rule> and saccade_len_<rule> (policy / greedy / random).
# Submit: POS=efcopy BETA=0.3 [R=0.06] ENT=4.0 SEED=1 TRAIN_COMMIT=<sha> bash slurm/runs/exp44_search_saccade_cost/exp44.sh
set -euo pipefail

# === ESSENTIALS ===
POS="${POS:?POS=nopos|efcopy}"
BETA="${BETA:?BETA=0|0.3|1.0}"
R="${R:-}"                    # empty = the default r (1 / centers_per_axis = 1/32)
ENT="${ENT:-4.0}"
SEED="${SEED:-1}"
RUN_GROUP=jon_exp44_search_saccade_cost
RUN_NAME=exp44-pg-e$ENT-$POS-b$BETA${R:+-r$R}-s$SEED
if [ "$POS" = nopos ] && [ "$BETA" = 0 ]; then
  echo "REFUSING: POS=nopos BETA=0 is exp43 (exp43-search-pg[-e$ENT]-s$SEED), bit-identical; not rerun"; exit 1
fi
[ "$BETA" = 0 ] && [ -n "$R" ] && { echo "REFUSING: R without a cost (BETA=0)"; exit 1; }
LR=0.0004                     # task head AND policy (one AdamW, two groups)
MAX_STEPS="${MAX_STEPS:-5000}"
STEPS_PER_JOB="${STEPS_PER_JOB:-5000}"   # ~2 h 15 min on production A100s
N_JOBS=$(( (MAX_STEPS + STEPS_PER_JOB - 1) / STEPS_PER_JOB ))
ARRAY="${ARRAY:-0-${N_JOBS}%1}"          # one spare task: resumes after a time limit, else exits at once
TIME="${TIME:-0-03:00:00}"
MEM=64G
NGPU=1                        # search is single-GPU only
TASK=search

# === config (exp43's, 1:1) ===
CFG_MODEL_REPO=/mnt/vast-nhr/projects/nib00021/jonathan/repos/canvit/logs/jon_exp38_multi_retina/exp38-multi-retina-4gpu-lr2x-s1/checkpoints/step-360448.pt
[ -f "$CFG_MODEL_REPO" ] || { echo "REFUSING: $CFG_MODEL_REPO missing"; exit 1; }
CFG_WANDB_PROJECT=jon_exp44_search_saccade_cost
CFG_SEED=$SEED
CFG_RETINA=all
CFG_BATCH_SIZE=64
CFG_MAX_STEPS=$MAX_STEPS
CFG_VAL_EVERY=500
CFG_NUM_WORKERS=8
CFG_PEAK_LR=$LR
CFG_WEIGHT_DECAY=0.0001
CFG_WARMUP_STEPS=0
CFG_LR_SCHEDULE=constant
CFG_SUPERVISE=every
CFG_HEAD_BLOCK_LAYERS=3
CFG_HEAD_LEVELS=3
OPT_N_STEPS=$STEPS_PER_JOB
OPT_RESUME=True
AUG="--cfg.aug.color-jitter --cfg.aug.rotate --cfg.aug.affine --cfg.aug.autocontrast --cfg.aug.no-crop"
POLICY="--rl.block-layers 3 --rl.policy-levels 3 --rl.reward fractional --rl.reward-center running_mean --rl.reward-scale running_std --rl.policy-lr $LR --rl.policy-weight-decay 0.0001 --rl.policy-betas 0.9 0.999 --rl.policy-warmup-steps 0 --rl.policy-warmup-frac 0"
PG="--rl.objective pg --rl.entropy-target $ENT --rl.entropy-alpha-min 1e-6 --rl.entropy-alpha-max None --rl.entropy-alpha-init None --rl.entropy-update per_rollout --rl.alpha-lr 0.01"
case "$POS" in
  nopos)  POS_ARGS="" ;;
  efcopy) POS_ARGS="--rl.efference-copy" ;;
  *) echo "unknown POS $POS"; exit 1 ;;
esac
COST=""
if [ "$BETA" != 0 ]; then
  COST="--rl.saccade-cost log --rl.saccade-cost-weight $BETA${R:+ --rl.saccade-cost-r $R}"
fi
EXTRA_ARGS="$AUG $POLICY $PG $POS_ARGS $COST"
# =================

TRAIN_COMMIT="${TRAIN_COMMIT:?TRAIN_COMMIT=<main commit with --rl.efference-copy>}"

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
