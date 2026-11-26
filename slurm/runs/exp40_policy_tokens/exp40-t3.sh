#!/bin/bash
# exp40 POLICY TOKENS T3 — joint PG policy (3 policy tokens) trained with distillation, seeded from
# exp38 multi-retina step-352256, single GPU.
#
# QUESTION (owner 2026-11-23). In 512-step smokes without warmup, every joint-policy arm ended far below
# exp38 on the fixed fixation_grid validation (T3-readonly 0.9109, T3-full 0.8868 val_metric vs 0.9390 at
# the start; a no-policy control 0.9342). Do the token arms recover the pretrained quality when the
# restart uses LR warmup and runs long?
#
# RECIPE. exp38's per-GPU batch (64) on 1 GPU; peak LR 4e-4 = exp38's 8e-4 / sqrt(4) (exp38 scales LR
# with sqrt(#GPUs)), reached by a linear warmup of WARMUP steps, then constant. Joint PG with the policy
# gradient into the backbone, rl_weight 1.0 (harness default, no extra weighting), entropy dual ascent to
# a PLACEHOLDER target of 2.9 nats over the 32x32 cells (FoAViS's share 0.42 of log 1024; the real value
# comes from the owner + Valentin), alpha floor 1e-6 (FoAViS). Validation once per chunk (every 4096 steps): fixation_grid
# + the policy pass (argmax), all 9 retinas.
#
# Submit: MODE=readonly bash slurm/runs/exp40_policy_tokens/exp40-t3.sh   (or MODE=full)
set -euo pipefail

# === ESSENTIALS ===
MODE="${MODE:?MODE=readonly|full}"
RUN_GROUP=jon_exp40_policy_tokens
SEED="${SEED:-1}"
RUN_NAME=exp40-t3-$MODE-s$SEED
ARRAY="${ARRAY:-0-19%1}"      # 20 x 4096 = 81920 steps
TIME=0-02:00:00
MEM=128G
NGPU=1
TASK=distill
WARMUP="${WARMUP:-20480}"   # ~20k (owner 2026-11-23) = 5 chunks

# === config ===
CFG_WANDB_PROJECT=jon_exp40_policy_tokens
CFG_SEED=$SEED
CFG_PEAK_LR=0.0004
CFG_WARMUP_STEPS=$WARMUP
CFG_BATCH_SIZE_PER_GPU=64
CFG_STEPS_PER_JOB=4096        # x 64 = 64 shards of 4096; joint PG ~1 s/step at batch 64 -> ~70 min + 1 val
CFG_VAL_EVERY=4096          # one validation per chunk (standard)
CFG_LOG_EVERY=128
CFG_NUM_WORKERS=4
SEED_CKPT=/mnt/vast-nhr/projects/nib00021/jonathan/repos/canvit/logs/jon_exp38_multi_retina/exp38-multi-retina-4gpu-lr2x-s1/checkpoints/step-352256.pt
[ -f "$SEED_CKPT" ] || { echo "REFUSING: $SEED_CKPT missing"; exit 1; }
CFG_SEED_CKPT=$SEED_CKPT
RETINAS="0.75 65 6 False 0.75 80 7 True 0.75 92 8 True 2.2 65 6 False 2.2 80 7 True 2.2 92 8 True 7.7 65 6 False 7.7 80 7 True 7.7 92 8 True"
MODEL_ARGS="--cfg.model.patcher-name foveated --cfg.model.foveated-patcher.fov 50 --cfg.model.foveated-patcher.retinas $RETINAS --cfg.model.foveated-patcher.arch-flag doubleres --cfg.model.foveated-patcher.rf-interp bilinear --cfg.model.foveated-patcher.conditioning.mode film --cfg.model.foveated-patcher.conditioning.film.fourier.num-features 256 --cfg.model.foveated-patcher.conditioning.film.fourier.sigma 4 --cfg.foveated-scale.fixed-scale 2.0"
POLICY_ARGS="--preset joint --spec.policy-grad-to-backbone True --cfg.rl.objective pg --cfg.rl.no-select-bn-eval --cfg.rl.entropy-alpha-min 1e-6 --cfg.rl.entropy-target 2.9 --cfg.model.n-policy-tokens 3 --cfg.model.policy-token-mode $MODE"
EXTRA_ARGS="$POLICY_ARGS $MODEL_ARGS"
# =================

TRAIN_COMMIT="${TRAIN_COMMIT:-28201af}"   # branch policy-tokens (policy-val bd05489 + cdffba5 + tokens 4380f7b + donated_buffer fix)

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
