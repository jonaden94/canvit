#!/bin/bash
# exp41 FROZEN MODEL, POLICY ONLY (owner 2026-11-24). The reconstruction model is exp38 multi-retina at
# its best val/scene_cos_norm_t9 checkpoint (step 360,448, before any LR decay) and stays frozen
# (--preset policy_only: no task loss, no backbone gradient); only the policy trains. Question: how the
# policy evolves on a fixed model (how long it improves, when it stagnates) and whether one policy
# formulation learns faster / better than another. The fixation-grid validation must stay constant
# (the model does not change); the policy validation is the result.
# One policy branch (center start, then policy glimpses), 1 GPU, 25 x 4096 = 102,400 steps,
# policy LR 4e-4 constant, no warmup. Optimizer for EVERY arm = the task's (owner 2026-11-25): AdamW
# betas (0.9, 0.999), weight decay 1e-4, LR 4e-4 (exp38's 8e-4 / sqrt(4 GPUs)), grad clip 1.0; for
# SAC the critics too (one policy group). Architectures as exp40 round 3:
#   p2-*     ConvNeXt U-Net on the canvas (ln_feat, 32 x 32 cells, no input norm)
#   t3iso-*  3 isolated policy tokens, 1 head
# Learning rules (all GREEDY: each choice is credited with the fractional reduction of the distill
# loss by the next glimpse). Trajectories as in CanViT training: chunks of 2 glimpses, extended with
# p = 0.5, backward + detach per chunk (policy_only keeps distill's own lengths).
#   pg     policy gradient, entropy dual ascent to 4.0 nats (alpha floor 1e-6)
#   qreg   QReg: the policy regresses the reward at the taken cell, epsilon-greedy rollout
#          (half the glimpses random), argmax deploy. Policy tokens: their readout's unbounded
#          per-cell values (PolicyTokenReadout.values), not the log-probabilities.
#   dqreg  dense QReg: QReg + 4 random extra cells glimpsed (no grad) per policy state
#   sac    SAC-Discrete: twin canvas U-Net critics (P2's architecture, for both actors),
#          gamma 0 = GREEDY like the other rules (the critics learn the next glimpse's gain; owner
#          2026-11-25: no long-term credit now), temperature steered to 4.0 nats, no replay
#
# Submit: ARM=p2-pg TRAIN_COMMIT=<sha> bash slurm/runs/exp41_frozen_policy/exp41.sh
set -euo pipefail

# === ESSENTIALS ===
ARM="${ARM:?ARM=p2-pg|t3iso-pg|p2-qreg|t3iso-qreg|p2-dqreg|t3iso-dqreg|p2-sac|t3iso-sac}"
RUN_GROUP=jon_exp41_frozen_policy
SEED="${SEED:-1}"
RUN_NAME=exp41-$ARM-s$SEED
ARRAY="${ARRAY:-0-24%1}"      # 25 x 4096 = 102,400 steps
TIME="${TIME:-0-02:00:00}"
MEM=128G
NGPU=1
TASK=distill

# === config ===
CFG_WANDB_PROJECT=jon_exp41_frozen_policy
CFG_SEED=$SEED
CFG_PEAK_LR=0.0004            # no trainable task group; kept equal to the policy LR
CFG_WARMUP_STEPS=0
CFG_BATCH_SIZE_PER_GPU=64
CFG_STEPS_PER_JOB="${STEPS_PER_JOB:-4096}"
CFG_VAL_EVERY=4096            # one validation per 4096 steps
CFG_LOG_EVERY=128
CFG_NUM_WORKERS=4
SEED_CKPT=/mnt/vast-nhr/projects/nib00021/jonathan/repos/canvit/logs/jon_exp38_multi_retina/exp38-multi-retina-4gpu-lr2x-s1/checkpoints/step-360448.pt
[ -f "$SEED_CKPT" ] || { echo "REFUSING: $SEED_CKPT missing"; exit 1; }
CFG_SEED_CKPT=$SEED_CKPT
RETINAS="0.75 65 6 False 0.75 80 7 True 0.75 92 8 True 2.2 65 6 False 2.2 80 7 True 2.2 92 8 True 7.7 65 6 False 7.7 80 7 True 7.7 92 8 True"
MODEL_ARGS="--cfg.model.patcher-name foveated --cfg.model.foveated-patcher.fov 50 --cfg.model.foveated-patcher.retinas $RETINAS --cfg.model.foveated-patcher.arch-flag doubleres --cfg.model.foveated-patcher.rf-interp bilinear --cfg.model.foveated-patcher.conditioning.mode film --cfg.model.foveated-patcher.conditioning.film.fourier.num-features 256 --cfg.model.foveated-patcher.conditioning.film.fourier.sigma 4 --cfg.foveated-scale.fixed-scale 2.0"
POLICY="--preset policy_only --cfg.rl.second-branch none --cfg.rl.no-select-bn-eval --cfg.rl.policy-lr 0.0004 --cfg.rl.policy-warmup-steps 0 --cfg.rl.policy-warmup-frac 0 --cfg.rl.policy-weight-decay 0.0001 --cfg.rl.policy-betas 0.9 0.999"
PG="--cfg.rl.objective pg --cfg.rl.entropy-alpha-min 1e-6 --cfg.rl.entropy-target 4.0"
QREG="--cfg.rl.objective qreg"
DQREG="--cfg.rl.objective qreg --cfg.rl.qreg-extra-candidates 4"
SAC="--cfg.rl.objective sac --cfg.rl.sac-target-entropy 4.0 --cfg.rl.sac-gamma 0"   # greedy, like the others
UNET="--cfg.rl.feature-groups ln_feat --cfg.rl.centers-per-axis 32 --cfg.policy-scorer-input-norm none"
ISO="--cfg.model.n-policy-tokens 3 --cfg.model.policy-token-mode isolated --cfg.model.policy-heads 1"
case "$ARM" in
  p2-pg)      ARM_ARGS="$PG $UNET" ;;
  t3iso-pg)   ARM_ARGS="$PG $ISO" ;;
  p2-qreg)    ARM_ARGS="$QREG $UNET" ;;
  t3iso-qreg) ARM_ARGS="$QREG $ISO" ;;
  p2-dqreg)   ARM_ARGS="$DQREG $UNET" ;;
  t3iso-dqreg) ARM_ARGS="$DQREG $ISO" ;;
  p2-sac)     ARM_ARGS="$SAC $UNET" ;;
  t3iso-sac)  ARM_ARGS="$SAC $ISO $UNET" ;;   # UNET = the critics' architecture
  *) echo "unknown ARM $ARM"; exit 1 ;;
esac
EXTRA_ARGS="$POLICY $ARM_ARGS $MODEL_ARGS"
# =================

TRAIN_COMMIT="${TRAIN_COMMIT:?TRAIN_COMMIT=<policy-tokens commit>}"

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
