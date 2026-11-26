#!/bin/bash
# exp40 ROUND 2 — which joint policy keeps the reconstruction (owner 2026-11-23). Seeded from exp38
# multi-retina step-352256, single GPU, 25 x 4096 = 102,400 steps. Differences to exp40 round 1
# (exp40-t3-{readonly,full}-s1, pin 28201af): ONE rollout branch (center start, then sampled from the
# policy), policy LR = task LR 4e-4 with the SAME 20,480-step warmup (round 1: 2e-4, no warmup),
# 1 readout head per policy token (round 1: 8), the U-Net scorer without its input BatchNorm.
#
# ARMS (owner's final table):
#   p1-e2.9    U-Net on the canvas (ln_feat, 32x32), policy gradient into the backbone, entropy 2.9
#   p1-e4.0    the same, entropy target 4.0 (~55 equally likely cells instead of ~18)
#   p2-e2.9    U-Net, stop-grad (no policy gradient into the backbone), entropy 2.9
#   t3iso-e2.9 3 isolated policy tokens (own weights, read the reconstruction path detached), 1 head
#   t3ro-e2.9  3 read-only policy tokens (shared weights, gradient into the backbone), 1 head
#   t3iso-taskopt-e2.9  t3iso-e2.9 with the policy group's AdamW = the task's (weight decay 1e-4,
#              betas 0.9/0.999 instead of the RL recipe's 1e-2, 0.9/0.95; owner 2026-11-23)
# Shared: PG, rl_weight 1.0, entropy dual ascent (alpha floor 1e-6), validation every chunk
# (fixation_grid + policy pass, 9 retinas).
#
# Submit: ARM=p1-e2.9 bash slurm/runs/exp40_policy_tokens/exp40-round2.sh
set -euo pipefail

# === ESSENTIALS ===
ARM="${ARM:?ARM=p1-e2.9|p1-e4.0|p2-e2.9|t3iso-e2.9|t3iso-taskopt-e2.9|t3ro-e2.9}"
RUN_GROUP=jon_exp40_policy_tokens
SEED="${SEED:-1}"
RUN_NAME=exp40-$ARM-s$SEED
ARRAY="${ARRAY:-0-24%1}"      # 25 x 4096 = 102,400 steps
TIME=0-02:00:00
MEM=128G
NGPU=1
TASK=distill
WARMUP=20480                  # task AND policy

# === config ===
CFG_WANDB_PROJECT=jon_exp40_policy_tokens
CFG_SEED=$SEED
CFG_PEAK_LR=0.0004            # exp38's 8e-4 / sqrt(4 GPUs)
CFG_WARMUP_STEPS=$WARMUP
CFG_BATCH_SIZE_PER_GPU=64
CFG_STEPS_PER_JOB=4096
CFG_VAL_EVERY=4096            # one validation per chunk
CFG_LOG_EVERY=128
CFG_NUM_WORKERS=4
SEED_CKPT=/mnt/vast-nhr/projects/nib00021/jonathan/repos/canvit/logs/jon_exp38_multi_retina/exp38-multi-retina-4gpu-lr2x-s1/checkpoints/step-352256.pt
[ -f "$SEED_CKPT" ] || { echo "REFUSING: $SEED_CKPT missing"; exit 1; }
CFG_SEED_CKPT=$SEED_CKPT
RETINAS="0.75 65 6 False 0.75 80 7 True 0.75 92 8 True 2.2 65 6 False 2.2 80 7 True 2.2 92 8 True 7.7 65 6 False 7.7 80 7 True 7.7 92 8 True"
MODEL_ARGS="--cfg.model.patcher-name foveated --cfg.model.foveated-patcher.fov 50 --cfg.model.foveated-patcher.retinas $RETINAS --cfg.model.foveated-patcher.arch-flag doubleres --cfg.model.foveated-patcher.rf-interp bilinear --cfg.model.foveated-patcher.conditioning.mode film --cfg.model.foveated-patcher.conditioning.film.fourier.num-features 256 --cfg.model.foveated-patcher.conditioning.film.fourier.sigma 4 --cfg.foveated-scale.fixed-scale 2.0"
COMMON="--preset joint --cfg.n-random-start-branches 0 --cfg.rl.objective pg --cfg.rl.no-select-bn-eval --cfg.rl.entropy-alpha-min 1e-6 --cfg.rl.policy-lr 0.0004 --cfg.rl.policy-warmup-steps $WARMUP"
UNET="--cfg.rl.feature-groups ln_feat --cfg.rl.centers-per-axis 32 --cfg.policy-scorer-input-norm none"
case "$ARM" in
  p1-e2.9)    ARM_ARGS="$UNET --spec.policy-grad-to-backbone True --cfg.rl.entropy-target 2.9" ;;
  p1-e4.0)    ARM_ARGS="$UNET --spec.policy-grad-to-backbone True --cfg.rl.entropy-target 4.0" ;;
  p2-e2.9)    ARM_ARGS="$UNET --spec.policy-grad-to-backbone False --cfg.rl.entropy-target 2.9" ;;
  t3iso-e2.9) ARM_ARGS="--cfg.model.n-policy-tokens 3 --cfg.model.policy-token-mode isolated --cfg.model.policy-heads 1 --spec.policy-grad-to-backbone False --cfg.rl.entropy-target 2.9" ;;
  t3iso-taskopt-e2.9) ARM_ARGS="--cfg.model.n-policy-tokens 3 --cfg.model.policy-token-mode isolated --cfg.model.policy-heads 1 --spec.policy-grad-to-backbone False --cfg.rl.entropy-target 2.9 --cfg.rl.policy-weight-decay 0.0001 --cfg.rl.policy-betas 0.9 0.999" ;;
  t3ro-e2.9)  ARM_ARGS="--cfg.model.n-policy-tokens 3 --cfg.model.policy-token-mode readonly --cfg.model.policy-heads 1 --spec.policy-grad-to-backbone True --cfg.rl.entropy-target 2.9" ;;
  *) echo "unknown ARM $ARM"; exit 1 ;;
esac
EXTRA_ARGS="$COMMON $ARM_ARGS $MODEL_ARGS"
# =================

TRAIN_COMMIT="${TRAIN_COMMIT:-376be36}"   # policy-tokens: viz 8056633 + tokens + isolated mode, on main 40c1927

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
