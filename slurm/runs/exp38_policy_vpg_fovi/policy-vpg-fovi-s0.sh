#!/bin/bash
# ADE20K viewpoint POLICY (VPG objective) on our OWN foveated model: the exp22
# foveated-teacherinit backbone plus the ADE20K probe trained on it.
#
# This is a COPY of exp36_policy_qreg_fovi/policy-qreg-fovi-s0.sh with only the
# objective changed (--rl.objective vpg added to EXTRA_ARGS) and RUN_GROUP/
# RUN_NAME/CFG_WANDB_PROJECT repointed so it does not collide with the QReg run's
# checkpoints/logs. Same backbone, same probe, same seed, same every other
# setting as the QReg run (and the PG sibling, exp37) -- the only variable that
# differs is the RL objective, so eval/miou_t0..t4 are directly comparable
# across all three.
#
# NOT YET RUN ANYWHERE IN THIS PROJECT (checked docs/, claude_dev/unification/,
# and every policy launcher under slurm/runs/ on 2026-10-19): exp27/exp31/exp35/
# exp36 and all 21 published "qband" HF policy checkpoints are QReg. There is no
# PG or VPG reference number anywhere in this repo -- this run is the first one.
#
# See canvit/harness/policy/rl.py for what VPG actually is: the ``autoreg_tryout``
# REINFORCE port -- reward is TERMINAL (the final glimpse's segmentation quality,
# broadcast back over all 5 steps) rather than per-glimpse like QReg/PG, credit is
# assigned once at rollout end, and it uses a separately-trained value head as a
# baseline instead of QReg's reward-standardizer or PG's entropy-floor dual.
#
# --rl.no-select-bn-eval IS REQUIRED for objective=vpg -- canvit/harness/policy/
# build.py:120 REFUSES to build otherwise (raises ValueError): with the default
# select_bn_eval=True the glimpse is sampled from an eval-mode BatchNorm forward
# while log pi is differentiated from the train-mode scores, making REINFORCE's
# gradient off-policy and biased (measured: the two BN modes' argmax agrees only
# 8-14% on real rollout states). Do not drop this flag to "simplify" the command.
#
# Submit: bash slurm/runs/exp38_policy_vpg_fovi/policy-vpg-fovi-s0.sh
set -euo pipefail

# === ESSENTIALS ===
RUN_GROUP=caglar_exp38_policy_vpg_fovi
SEED="${SEED:-0}"
RUN_NAME=exp38-policy-vpg-fovi-s$SEED
ARRAY=0-0%1                  # single job: 9000 steps fits inside the walltime
TIME="${TIME:-0-02:00:00}"
# The fast `2h` QOS must be REQUESTED: a <=2h --time alone still lands in `normal`
# (verified -- the first submission showed QOS=normal despite --time=2:00:00), and
# `normal` has historically meant 24-31h of queueing. Set QOS= to fall back.
QOS="${QOS:-2h}"
MEM=64G
NGPU=1                       # ade20k is single-GPU only (supports_ddp=False)
TASK=ade20k

# === the two halves of the model (identical to exp36 -- same comparison basis) ===
_PROBE_RUN=/mnt/vast-nhr/projects/nib00021/jonathan/repos/canvit/logs/jon_exp34_ade20k_probe/ade20k-fovi-ti-1196k/checkpoints
CFG_MODEL_REPO=/mnt/vast-nhr/projects/nib00021/jonathan/repos/canvit/logs/jon_exp22_full_runs/exp22-fovi-teacherinit-lrdrop-1196k/checkpoints/step-155648.pt
CFG_PROBE_REPO=$_PROBE_RUN/best.pt   # the training checkpoint itself; no conversion

# === config (identical to exp36's QReg recipe; only the objective differs) ===
CFG_WANDB_PROJECT=caglar_exp38_policy_vpg_fovi
CFG_SEED=$SEED
CFG_MAX_STEPS=9000
CFG_N_TIMESTEPS=5
CFG_BATCH_SIZE=16
CFG_CANVAS_GRID=32           # matches the exp34 probe
CFG_EVAL_POLICY=policy
CFG_RESIZE_MODE=squish       # the measurement contract every earlier CanViT number used
CFG_VAL_EVERY=1000
CFG_LOG_EVERY=50
CFG_NUM_WORKERS=4
EXTRA_ARGS="--preset policy_only --cfg.no-augment --cfg.foveated-scale.fixed-scale 2.0 --rl.objective vpg --rl.no-select-bn-eval"
# =================

TRAIN_COMMIT=7f01b26
PYTORCH_COMMIT=d616b7b
FOVI_COMMIT=c399d3b

# Repo root, derived from this script's own location (slurm/runs/<group>/<run>.sh),
# so the run submits from YOUR clone rather than one hardcoded checkout.
# The probe must come from a FINISHED probe run; gate on the final step file, not
# best.pt (which appears from the first eval onward and could be a stale partial run).
_PROBE_DONE=$_PROBE_RUN/step-40000.pt          # ade20k max_steps
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
