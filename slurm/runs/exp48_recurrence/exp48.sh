#!/bin/bash
# exp48 — single-block recurrence (bViT / reViT) in the CanViT backbone, distill from scratch.
#
# Three arms, identical except for the backbone trunk. Same dims everywhere (vitb16: width
# 768, 12 heads, MLP 3072, 12 block evaluations; canvas 8x128 = 1024), same recipe as
# jon_exp32_pretrain_lrdrop/exp32-uniform16 (random init, uniform patcher, lr 4e-4,
# warmup 100k, batch 64):
#
#   ARM=vit    standard backbone, 12 independent blocks           (baseline, same pin)
#   ARM=bvit   ONE block looped 12 times                           (Byra et al. 2605.10661)
#   ARM=revit  ONE block looped 12 times, FFN = 4 depth-routed experts (Bulat et al. 2610.12448)
#
# Random init on ALL arms: DINOv3 teacher-init is undefined for a recurrent backbone (it
# refuses), so the baseline must not get it either, or the comparison is unfair.
#
# DEFAULT = SMOKE: one 2 h job (8192 steps). It answers "does it run, memory, speed, no
# NaN, aux losses logged, checkpoint written" — NOT "which is better": warmup is 100k
# steps, so after 8192 steps the LR is still ~8% of peak. For a real comparison, re-run
# with a larger ARRAY (e.g. ARRAY=0-24%1 = 204,800 steps); it RESUMES the same run from
# its latest checkpoint, the array is a budget (see exp32 README). Compare arms on
# `val/in1k_tts_top1_t*` and `*_cos_raw` at matched steps, several seeds, best checkpoint.
#
# Usage:
#   ARM=vit   bash slurm/runs/exp48_recurrence/exp48.sh
#   ARM=bvit  bash slurm/runs/exp48_recurrence/exp48.sh
#   ARM=revit bash slurm/runs/exp48_recurrence/exp48.sh
#   ARM=bvit SEED=1 ARRAY=0-24%1 bash slurm/runs/exp48_recurrence/exp48.sh
#
# Baseline: ARM=vit s0 is the same config as jon_exp32_pretrain_lrdrop/exp32-uniform16
# (older code pin), whose logs reach 1.43M steps — no need to re-run it for seed 0.
# Smoke 2026-11-19 (job 17408686, ARM=bvit, 2 x 8192 steps, ~65 min/job): ran, resumed,
# no NaN; loss tracks exp32 within ~0.03 at 16k steps, val scene_cos_raw@8192 0.6630 vs
# 0.6633. Harmless shutdown noise: DataLoader workers fail to rmtree their pymp-* dirs
# on the NFS $TMPDIR ("Device or resource busy") after wandb has finished.
# NOTHING IS SUBMITTED by writing this file.
set -euo pipefail

ARM=${ARM:?set ARM=vit|bvit|revit}
SEED=${SEED:-0}

# === ESSENTIALS ===
RUN_GROUP=caglar_exp48_recurrence
RUN_NAME=exp48-uniform16-${ARM}-s${SEED}
ARRAY=${ARRAY:-0-0%1}        # smoke: 1 job x 8192 steps
TIME=0-02:00:00
MEM=128G
NGPU=1
TASK=distill

# === config (= exp32-uniform16, plus an explicit seed) ===
CFG_WANDB_PROJECT=caglar_exp48_recurrence
CFG_SEED=$SEED
CFG_PEAK_LR=0.0004
CFG_BATCH_SIZE_PER_GPU=64
CFG_STEPS_PER_JOB=8192
# 8192 = exp32-uniform16's cadence (once per job, at each job's start), so every val
# point lands on a step exp32 also validated. Keep it fixed for the life of a run.
CFG_VAL_EVERY=${VAL_EVERY:-8192}
CFG_LOG_EVERY=512
CFG_NUM_WORKERS=4
case "$ARM" in
    vit)   REC_ARGS="" ;;
    bvit)  REC_ARGS="--cfg.model.recurrence.enabled" ;;
    revit) REC_ARGS="--cfg.model.recurrence.enabled --cfg.model.recurrence.n-experts 4" ;;
    *) echo "unknown ARM=$ARM (vit|bvit|revit)" >&2; exit 1 ;;
esac
EXTRA_ARGS="--cfg.model.patcher-name uniform $REC_ARGS"
# =================

TRAIN_COMMIT=3e42f3b   # adds CanViTConfig.recurrence
FOVI_COMMIT=c399d3b
# (no PYTORCH_COMMIT: TRAIN_COMMIT pins model + trainer together)

# Repo root, derived from this script's own location (slurm/runs/<group>/<run>.sh).
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

# Guard: distill loads the GATED DINOv3 teacher at startup. Without a token every job
# dies with 401 after queueing — refuse here instead.
source .envrc.grete >/dev/null
if [ -z "${HF_TOKEN:-}" ] && [ ! -f "${HF_HOME:-$HOME/.cache/huggingface}/token" ]; then
    echo "No HF token (run 'hf auth login' with an account granted access to" >&2
    echo "facebook/dinov3-vitb16-pretrain-lvd1689m). Not submitting." >&2
    exit 1
fi

mkdir -p "logs/$RUN_GROUP/$RUN_NAME/log"
export TASK RUN_GROUP RUN_NAME NGPU EXTRA_ARGS TRAIN_COMMIT FOVI_COMMIT
for v in $(compgen -v); do [[ "$v" == CFG_* || "$v" == OPT_* ]] && export "$v"; done

sbatch \
    --gpus-per-node=A100:$NGPU \
    --ntasks-per-node=$NGPU \
    --mem=$MEM \
    --time=$TIME \
    --array="$ARRAY" \
    --exclude=ggpu104 \
    --job-name="$RUN_NAME" \
    --output="logs/$RUN_GROUP/$RUN_NAME/log/job-%A_%a.log" \
    --error="logs/$RUN_GROUP/$RUN_NAME/log/job-%A_%a.log" \
    --export=ALL \
    slurm/harness_train.sbatch
