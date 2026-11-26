#!/bin/bash
# exp42 (owner 2026-11-25): exp41's P2 arms (U-Net policy on the canvas, frozen exp38 step-360448,
# policy only) over 5 SEEDS, with the smaller 3-level U-Net (bottleneck 8x8 instead of 1x1). Question:
# across seeds, does the policy still improve after the first ~4k steps (exp41-s1 was flat after
# 4096)? Arms: pg, qreg; seeds 1-5; 20 x 4096 = 81,920 steps; validation every 4096.
# Everything else as exp41 (all greedy, distill's trajectory lengths, the task's AdamW for the
# policy: betas 0.9/0.999, wd 1e-4, LR 4e-4, grad clip 1.0). PG entropy control: see PG below.
#
# 2026-11-25 evening: + PG arms with entropy target 5.0 and 6.0 (ENT=5.0|6.0), else identical.
# Submit: [ENT=5.0] ARM=pg SEED=1 TRAIN_COMMIT=<sha> bash slurm/runs/exp42_policy_seeds/exp42.sh
set -euo pipefail

# === ESSENTIALS ===
ARM="${ARM:?ARM=pg|qreg}"
RUN_GROUP=jon_exp42_policy_seeds
SEED="${SEED:-1}"
ENT="${ENT:-4.0}"             # PG entropy target; 4.0 = the original arms (names unchanged)
SUFFIX=""; [ "$ENT" != "4.0" ] && SUFFIX="-e$ENT"
RUN_NAME=exp42-p2l3-$ARM$SUFFIX-s$SEED
ARRAY="${ARRAY:-0-9%1}"       # 10 x 4096 = 40,960 steps (owner 2026-11-25; was 20 x 4096)
TIME="${TIME:-0-02:00:00}"
MEM=128G
NGPU=1
TASK=distill

# === config ===
CFG_WANDB_PROJECT=jon_exp42_policy_seeds
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
# PG entropy control (2026-11-25, after exp41): ONE dual step per rollout on the rollout's mean
# entropy, step 0.01 in log(alpha) (the search runs' / FoAViS setting) instead of one step per
# glimpse at 0.05. exp41's per-glimpse controller swung alpha over 4-5 decades (T3-iso) and, for
# P2-PG, pinned it at the cap of 10 from ~12k on with the entropy stuck at ~3. Local check (P2,
# 3 levels, batch 16, 256 steps): entropy 6.9 -> 3.0 -> back to 3.3-4.2, alpha smooth 0.005 -> 0.014.
PG="--cfg.rl.objective pg --cfg.rl.entropy-alpha-min 1e-6 --cfg.rl.entropy-target $ENT --cfg.rl.entropy-update per_rollout --cfg.rl.alpha-lr 0.01"
QREG="--cfg.rl.objective qreg"
UNET="--cfg.rl.feature-groups ln_feat --cfg.rl.centers-per-axis 32 --cfg.policy-scorer-input-norm none --cfg.rl.policy-levels 3"
case "$ARM" in
  pg)   ARM_ARGS="$PG $UNET" ;;
  qreg) ARM_ARGS="$QREG $UNET" ;;
  *) echo "unknown ARM $ARM"; exit 1 ;;
esac
EXTRA_ARGS="$POLICY $ARM_ARGS $MODEL_ARGS"
# =================

TRAIN_COMMIT="${TRAIN_COMMIT:?TRAIN_COMMIT=<policy-tokens commit with policy_levels>}"

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
