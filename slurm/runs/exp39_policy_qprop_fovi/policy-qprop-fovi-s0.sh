#!/bin/bash
# ADE20K viewpoint POLICY (PG + Q-Prop critic) on our OWN foveated model: the exp22
# foveated-teacherinit backbone plus the ADE20K probe trained on it.
#
# This is a COPY of exp37_policy_pg_fovi/policy-pg-fovi-s0.sh with ONE flag added
# (--rl.qprop) and RUN_GROUP/RUN_NAME/CFG_WANDB_PROJECT repointed so it does not
# collide with the plain-PG run's checkpoints/logs. Same backbone, same probe, same
# seed, same every other setting as exp37 -- the only variable that differs is
# whether PG trains a second Q-value critic network alongside the actor, so
# eval/miou_t0..t4 are directly comparable to exp36 (QReg) / exp37 (PG) / exp38 (VPG).
#
# WHAT --rl.qprop ADDS (see canvit/harness/policy/rl.py's PG.qprop and
# canvit/harness/policy/build.py): a SECOND, fully independent ViewpointScorer
# (built identically to the actor's -- same grid/width/blocks) that predicts a
# Q-value (expected one-step reward) for every candidate glimpse location, trained
# by its own MSE-to-reward loss. The actor's own loss then gets an EXACT, zero-
# variance term (the critic's Q-values, weighted by the actor's own action
# probabilities, summed over all 1024 candidates -- no sampling needed since the
# grid is small and fully enumerable) PLUS the usual REINFORCE score-function term,
# but applied only to the RESIDUAL between the critic's prediction and the real
# measured reward for the action actually taken (small if the critic is accurate,
# so most of the actor's gradient signal is the noise-free analytic term instead of
# PG's usual fully-noisy single-sample signal).
#
# NOT YET RUN ANYWHERE IN THIS PROJECT: the qprop branch in rl.py's pg_loss has
# existed since the unification but was never reachable from any launcher/config
# (JointPolicyConfig had no `qprop` field, build_policy never passed one to PG(...)).
# Wired up 2026-10-21 -- see the local, uncommitted diff in canvit/harness/config.py,
# canvit/harness/policy/{build,joint}.py, canvit/{ade20k,in1k,distill}/task.py. Every
# change is additive and branches on `self.critic is not None` / `obj.qprop`, so
# qreg/vpg/plain-pg (exp36/37/38 and every other existing launcher) are byte-for-byte
# unaffected -- full `pytest canvit` run clean after the change.
#
# Submit: bash slurm/runs/exp39_policy_qprop_fovi/policy-qprop-fovi-s0.sh
set -euo pipefail

# === ESSENTIALS ===
RUN_GROUP=caglar_exp39_policy_qprop_fovi
SEED="${SEED:-0}"
RUN_NAME=exp39-policy-qprop-fovi-s$SEED
ARRAY=0-0%1                  # single job: 9000 steps fits inside the walltime
TIME="${TIME:-0-02:00:00}"
# The fast `2h` QOS must be REQUESTED: a <=2h --time alone still lands in `normal`
# (verified -- the first submission showed QOS=normal despite --time=2:00:00), and
# `normal` has historically meant 24-31h of queueing. Set QOS= to fall back.
QOS="${QOS:-2h}"
MEM=64G
NGPU=1                       # ade20k is single-GPU only (supports_ddp=False)
TASK=ade20k

# === the two halves of the model (identical to exp36/37/38 -- same comparison basis) ===
_PROBE_RUN=/mnt/vast-nhr/projects/nib00021/jonathan/repos/canvit/logs/jon_exp34_ade20k_probe/ade20k-fovi-ti-1196k/checkpoints
CFG_MODEL_REPO=/mnt/vast-nhr/projects/nib00021/jonathan/repos/canvit/logs/jon_exp22_full_runs/exp22-fovi-teacherinit-lrdrop-1196k/checkpoints/step-155648.pt
CFG_PROBE_REPO=$_PROBE_RUN/best.pt   # the training checkpoint itself; no conversion

# === config (identical to exp37's PG recipe; only --rl.qprop is new) ===
CFG_WANDB_PROJECT=caglar_exp39_policy_qprop_fovi
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
EXTRA_ARGS="--preset policy_only --cfg.no-augment --cfg.foveated-scale.fixed-scale 2.0 --rl.objective pg --rl.qprop"
# =================

# TRAIN_COMMIT deliberately NOT set (unlike exp36/37/38): --rl.qprop is a LOCAL,
# uncommitted change to canvit/harness/{config,policy/build,policy/joint}.py and the
# three task.py files -- all inside the `canvit` repo, which is what TRAIN_COMMIT
# pins. Pinning it would git-archive an old commit that predates this change, and
# run.py would reject the flag as unrecognized (confirmed: job 16795074 failed
# exactly this way against TRAIN_COMMIT=7f01b26). Omitting it runs the LIVE canvit
# checkout instead, which has the change. PYTORCH_COMMIT/FOVI_COMMIT stay pinned --
# neither CanViT-PyTorch nor fovi was touched. Re-add TRAIN_COMMIT once this is
# committed and verified.
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
