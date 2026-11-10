#!/bin/bash
# FEATURE-ENGINEERING ABLATION on exp36's QReg/foveated recipe: does the scorer need
# the DELTA feature groups ("ent_delta", "feat_delta" -- "what changed since last
# glimpse") at all?
#
# Identical to exp36_policy_qreg_fovi/policy-qreg-fovi-s0.sh except
# --rl.feature-groups, which drops "ent_delta"/"feat_delta" and keeps: ent,
# cos_prev, cos_init, ln_feat. Caglar's own ablation (jonathan's "is all this
# feature engineering necessary" question, 2026-11-03) -- exp41 was just the next
# free slot, not an official campaign number.
#
# --rl.feature-groups already existed in JointPolicyConfig before any local change,
# so this is safe to pin to the same TRAIN_COMMIT as exp36.
#
# Compare eval/miou_t0..t4 and ce_mean against exp36's reference (miou_final=0.4493,
# ce_mean=0.694, 9000 steps, same seed/backbone/probe).
#
# Submit: bash slurm/runs/exp41_ablation_no_delta_fovi/policy-ablation-no-delta-fovi-s0.sh
set -euo pipefail

# === ESSENTIALS ===
RUN_GROUP=caglar_exp41_ablation_no_delta_fovi
SEED="${SEED:-0}"
RUN_NAME=exp41-ablation-no-delta-fovi-s$SEED
ARRAY=0-0%1
TIME="${TIME:-0-02:00:00}"
QOS="${QOS:-2h}"
MEM=64G
NGPU=1
TASK=ade20k

# === the two halves of the model (identical to exp36 -- same comparison basis) ===
_PROBE_RUN=/mnt/vast-nhr/projects/nib00021/jonathan/repos/canvit/logs/jon_exp34_ade20k_probe/ade20k-fovi-ti-1196k/checkpoints
CFG_MODEL_REPO=/mnt/vast-nhr/projects/nib00021/jonathan/repos/canvit/logs/jon_exp22_full_runs/exp22-fovi-teacherinit-lrdrop-1196k/checkpoints/step-155648.pt
CFG_PROBE_REPO=$_PROBE_RUN/best.pt

# === config (identical to exp36; only --rl.feature-groups is new) ===
CFG_WANDB_PROJECT=caglar_exp41_ablation_no_delta_fovi
CFG_SEED=$SEED
CFG_MAX_STEPS=9000
CFG_N_TIMESTEPS=5
CFG_BATCH_SIZE=16
CFG_CANVAS_GRID=32
CFG_EVAL_POLICY=policy
CFG_RESIZE_MODE=squish
CFG_VAL_EVERY=1000
CFG_LOG_EVERY=50
CFG_NUM_WORKERS=4
EXTRA_ARGS="--preset policy_only --cfg.no-augment --cfg.foveated-scale.fixed-scale 2.0 --rl.feature-groups ent cos_prev cos_init ln_feat"
# =================

TRAIN_COMMIT=7f01b26
PYTORCH_COMMIT=d616b7b
FOVI_COMMIT=c399d3b

_PROBE_DONE=$_PROBE_RUN/step-40000.pt
if [ ! -f "$_PROBE_DONE" ]; then
    echo "REFUSING: the probe run has not finished ($_PROBE_DONE is missing)."
    exit 1
fi

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
mkdir -p "logs/$RUN_GROUP/$RUN_NAME/log"
export ADE20K_ROOT="${ADE20K_ROOT:-/mnt/vast-nhr/projects/nib00021/jonathan/datasets/zhoubolei--scene_parse_150/ADEChallengeData2016}"
export TASK RUN_GROUP RUN_NAME NGPU EXTRA_ARGS TRAIN_COMMIT PYTORCH_COMMIT FOVI_COMMIT
for v in $(compgen -v); do [[ "$v" == CFG_* || "$v" == OPT_* ]] && export "$v"; done

sbatch \
    --gpus-per-node=A100:$NGPU --ntasks-per-node=$NGPU --mem=$MEM --time=$TIME \
    ${QOS:+--qos=$QOS} \
    --array="$ARRAY" \
    --output="logs/$RUN_GROUP/$RUN_NAME/log/job-%A_%a.log" \
    --error="logs/$RUN_GROUP/$RUN_NAME/log/job-%A_%a.log" \
    --export=ALL slurm/harness_train.sbatch
