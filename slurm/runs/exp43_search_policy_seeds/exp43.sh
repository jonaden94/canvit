#!/bin/bash
# exp43 (owner 2026-11-25): exp42's two policy arms on VISUAL SEARCH (COCO-Search18), as close to
# exp42 as the task allows, so the same policies are trained for two tasks (DINOv3 reconstruction,
# search). Arms: pg, qreg; seeds 1-5.
#   same as exp42: backbone exp38 step-360448 (frozen), ALL 9 retinas (one drawn per training
#     sample, every validation on each), policy = U-Net on the LayerNorm'd canvas, 3 levels,
#     3 blocks per stage, 32x32 = 1024 fixation cells; greedy per-glimpse reward = fractional drop
#     of the per-image task loss, running mean/std; policy AdamW lr 4e-4, betas 0.9/0.999,
#     wd 1e-4, no warmup, constant; PG entropy target 4.0, one dual step per rollout (alpha_lr
#     0.01, alpha 1e-6..10); QReg eps-greedy prime 0.5; batch 64; grad clip 1.0; A100 bf16.
#   search-specific: task head and policy trained jointly (no frozen task model exists), both
#     target-conditioned (FiLM), width 64 (small training set); task loss = heatmap KL on the
#     target-present trials + presence focal loss, supervised at EVERY glimpse (needed for the
#     per-glimpse reward); reward on target-present trials only; 4 glimpses; Haoze's
#     augmentations; head AdamW = the policy's (lr 4e-4 constant, betas 0.9/0.999, wd 1e-4, no
#     warmup); 10,000 steps, validation every 500 (random / greedy /
#     policy fixations, 9 retinas each; best.pt = found_t3_policy averaged over the retinas).
#   vs Valentin's model (exp39 mr-r0-base-s0): backbone 253,952 -> 360,448; one retina -> all 9;
#     head + policy 64/2/6 -> 64/3/3; terminal -> per-glimpse reward; supervise last -> every;
#     entropy 2.11 -> 4.0; policy and head optimizer -> exp42's; batch 16 -> 64; 20k -> 10k
#     steps; fp32 V100 -> bf16 A100.
#
# 2026-11-25 evening: + PG arms with entropy target 5.0 and 6.0 (ENT=5.0|6.0), else identical.
# Submit: [ENT=5.0] ARM=pg SEED=1 TRAIN_COMMIT=<sha> bash slurm/runs/exp43_search_policy_seeds/exp43.sh
set -euo pipefail

# === ESSENTIALS ===
ARM="${ARM:?ARM=pg|qreg}"
RUN_GROUP=jon_exp43_search_policy_seeds
SEED="${SEED:-1}"
ENT="${ENT:-4.0}"             # PG entropy target; 4.0 = the original arms (names unchanged)
SUFFIX=""; [ "$ENT" != "4.0" ] && SUFFIX="-e$ENT"
RUN_NAME=exp43-search-$ARM$SUFFIX-s$SEED
STEPS_PER_JOB="${STEPS_PER_JOB:?STEPS_PER_JOB=5000 (smoke 2026-11-25: 0.82 s per step, 110 s per 9-retina validation)}"
MAX_STEPS="${MAX_STEPS:-10000}"   # 2026-11-26 reruns: MAX_STEPS=4501 STEPS_PER_JOB=4501 (one job; LR constant, so = the first 4,500 steps of 10k; last validation at 4,500)
N_JOBS=$(( (MAX_STEPS + STEPS_PER_JOB - 1) / STEPS_PER_JOB ))
ARRAY="${ARRAY:-0-$((N_JOBS - 1))%1}"
TIME="${TIME:-0-02:00:00}"
MEM=64G
NGPU=1                        # search is single-GPU only
TASK=search

# === config ===
CFG_MODEL_REPO=/mnt/vast-nhr/projects/nib00021/jonathan/repos/canvit/logs/jon_exp38_multi_retina/exp38-multi-retina-4gpu-lr2x-s1/checkpoints/step-360448.pt
[ -f "$CFG_MODEL_REPO" ] || { echo "REFUSING: $CFG_MODEL_REPO missing"; exit 1; }
CFG_WANDB_PROJECT=jon_exp43_search_policy_seeds
CFG_SEED=$SEED
CFG_RETINA=all
CFG_BATCH_SIZE=64
CFG_MAX_STEPS=$MAX_STEPS
CFG_VAL_EVERY=500
CFG_NUM_WORKERS=8
CFG_PEAK_LR=0.0004            # task head = the policy's AdamW (exp42)
CFG_WEIGHT_DECAY=0.0001
CFG_WARMUP_STEPS=0
CFG_LR_SCHEDULE=constant   # head LR constant, like the policy (exp42)
CFG_SUPERVISE=every
CFG_HEAD_BLOCK_LAYERS=3
CFG_HEAD_LEVELS=3
OPT_N_STEPS=$STEPS_PER_JOB    # per array task; the next task resumes from the last checkpoint
OPT_RESUME=True
AUG="--cfg.aug.color-jitter --cfg.aug.rotate --cfg.aug.affine --cfg.aug.autocontrast --cfg.aug.no-crop"
POLICY="--rl.block-layers 3 --rl.policy-levels 3 --rl.reward fractional --rl.reward-center running_mean --rl.reward-scale running_std --rl.policy-lr 0.0004 --rl.policy-weight-decay 0.0001 --rl.policy-betas 0.9 0.999 --rl.policy-warmup-steps 0 --rl.policy-warmup-frac 0"
PG="--rl.objective pg --rl.entropy-target $ENT --rl.entropy-alpha-min 1e-6 --rl.entropy-alpha-max None --rl.entropy-alpha-init None --rl.entropy-update per_rollout --rl.alpha-lr 0.01"
# QReg has no entropy dual ascent: unset search's PG defaults for it
QREG="--rl.objective qreg --rl.entropy-alpha-init None --rl.entropy-alpha-min None --rl.entropy-alpha-max None --rl.entropy-update per_glimpse"
case "$ARM" in
  pg)   ARM_ARGS="$PG" ;;
  qreg) ARM_ARGS="$QREG" ;;
  *) echo "unknown ARM $ARM"; exit 1 ;;
esac
EXTRA_ARGS="$AUG $POLICY $ARM_ARGS"
# =================

TRAIN_COMMIT="${TRAIN_COMMIT:?TRAIN_COMMIT=<policy-tokens commit with --cfg.retina all + head_levels>}"

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
