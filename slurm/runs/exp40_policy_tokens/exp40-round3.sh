#!/bin/bash
# exp40 ROUND 3 — joint policy as a CONTINUATION of exp38 (owner 2026-11-24). Round 2 restarted from
# exp38's weights with a fresh AdamW and a 20k warmup, so its early rise (low LR) and later drop (LR
# rising) confound the policy with the LR schedule. Here every run takes over exp38's AdamW state for
# the task group (exp_avg as is, exp_avg_sq x 3: measured ratio 1-GPU/4-GPU, median ~3 over the 236
# tensors) and trains at a constant task LR 4e-4 (exp38's 8e-4 / sqrt(4 GPUs)) from step 0, no warmup.
# The new policy group starts with a fresh AdamW, also without warmup (owner: no warmup anywhere).
# Seed: exp38 multi-retina
# step-352256, 1 GPU, 25 x 4096 = 102,400 steps.
#
# ARMS (owner's table):
#   control             exp38's recipe without a policy: center start + random start, random glimpses
#   p2-e4.0             U-Net on the canvas, stop-grad, entropy 4.0; one branch (center start, policy)
#   p2-e4.0-rb          p2-e4.0 + a second branch: center start, random glimpses, task loss only
#   t3iso-taskopt-e4.0     3 isolated policy tokens, 1 head, policy AdamW = task's, entropy 4.0
#   t3iso-taskopt-e4.0-rb  t3iso-taskopt-e4.0 + the same second branch
# Shared by the policy arms: PG, rl_weight 1.0, entropy dual ascent (alpha floor 1e-6), policy LR
# 4e-4. Validation every chunk (fixation_grid + policy pass, 9 retinas).
#
# Submit: ARM=control bash slurm/runs/exp40_policy_tokens/exp40-round3.sh
set -euo pipefail

# === ESSENTIALS ===
ARM="${ARM:?ARM=control|p2-e4.0|p2-e4.0-rb|t3iso-taskopt-e4.0|t3iso-taskopt-e4.0-rb}"
RUN_GROUP=jon_exp40_policy_tokens
SEED="${SEED:-1}"
RUN_NAME=exp40-r3-$ARM-s$SEED
ARRAY="${ARRAY:-0-24%1}"      # 25 x 4096 = 102,400 steps
TIME=0-02:00:00
MEM=128G
NGPU=1
TASK=distill
POLICY_WARMUP=0               # no warmup anywhere (owner 2026-11-24)

# === config ===
CFG_WANDB_PROJECT=jon_exp40_policy_tokens
CFG_SEED=$SEED
CFG_PEAK_LR=0.0004            # exp38's 8e-4 / sqrt(4 GPUs)
CFG_WARMUP_STEPS=0            # continuation: exp38's AdamW state is taken over
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
SEED_OPT="--cfg.seed-optimizer --cfg.seed-exp-avg-sq-scale 3.0"
POLICY="--preset joint --cfg.rl.objective pg --cfg.rl.no-select-bn-eval --cfg.rl.entropy-alpha-min 1e-6 --cfg.rl.policy-lr 0.0004 --cfg.rl.policy-warmup-steps $POLICY_WARMUP --cfg.rl.policy-warmup-frac 0 --cfg.rl.entropy-target 4.0 --spec.policy-grad-to-backbone False"
UNET="--cfg.rl.feature-groups ln_feat --cfg.rl.centers-per-axis 32 --cfg.policy-scorer-input-norm none"
ISO="--cfg.model.n-policy-tokens 3 --cfg.model.policy-token-mode isolated --cfg.model.policy-heads 1 --cfg.rl.policy-weight-decay 0.0001 --cfg.rl.policy-betas 0.9 0.999"
case "$ARM" in
  control)               ARM_ARGS="" ;;   # default preset, task's branches: FULL + RANDOM start (exp38)
  p2-e4.0)               ARM_ARGS="$POLICY $UNET --cfg.rl.second-branch none" ;;
  p2-e4.0-rb)            ARM_ARGS="$POLICY $UNET --cfg.rl.second-branch center_start" ;;
  t3iso-taskopt-e4.0)    ARM_ARGS="$POLICY $ISO --cfg.rl.second-branch none" ;;
  t3iso-taskopt-e4.0-rb) ARM_ARGS="$POLICY $ISO --cfg.rl.second-branch center_start" ;;
  *) echo "unknown ARM $ARM"; exit 1 ;;
esac
EXTRA_ARGS="$SEED_OPT $ARM_ARGS $MODEL_ARGS"
# =================

TRAIN_COMMIT="${TRAIN_COMMIT:?TRAIN_COMMIT=<policy-tokens commit with seed_optimizer + second_branch>}"

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
