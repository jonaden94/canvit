#!/bin/bash
# FEATURE-ENGINEERING ADDITION on exp36's QReg/foveated recipe: combine ALL THREE new
# groups (rec_cls, local_feat, depth_feat) at once -- checking for any synergy the
# individual additions (exp44/45/46) might have missed on their own.
#
# Identical to exp36_policy_qreg_fovi/policy-qreg-fovi-s0.sh except --rl.feature-groups,
# which keeps exp36's six groups and APPENDS all three new ones. Each of the three was
# independently implemented/tested (StateEncoder gates each with its own `needs_*` flag
# and all three write into disjoint entries of the same `avail` dict), so combining them
# is not a new code path beyond what exp44/45/46 individually exercised -- still watched
# like a smoke test on submission, same discipline as the rest. Caglar's own numbering,
# exp47 was just the next free slot.
#
# TRAIN_COMMIT deliberately NOT set (uses the live checkout) -- same reason as
# exp44/45/46: these groups did not exist at TRAIN_COMMIT=7f01b26.
#
# Compare eval/miou_t0..t4 and ce_mean against exp36's reference (miou_final=0.4493,
# ce_mean=0.694, 9000 steps, same seed/backbone/probe).
#
# Submit: bash slurm/runs/exp47_addition_all_three_fovi/policy-addition-all-three-fovi-s0.sh
set -euo pipefail

# === ESSENTIALS ===
RUN_GROUP=caglar_exp47_addition_all_three_fovi
SEED="${SEED:-0}"
RUN_NAME=exp47-addition-all-three-fovi-s$SEED
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
CFG_WANDB_PROJECT=caglar_exp47_addition_all_three_fovi
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
EXTRA_ARGS="--preset policy_only --cfg.no-augment --cfg.foveated-scale.fixed-scale 2.0 --rl.feature-groups ent ent_delta cos_prev cos_init ln_feat feat_delta rec_cls local_feat depth_feat"
# =================

# TRAIN_COMMIT intentionally unset -- see header.
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
export TASK RUN_GROUP RUN_NAME NGPU EXTRA_ARGS PYTORCH_COMMIT FOVI_COMMIT
for v in $(compgen -v); do [[ "$v" == CFG_* || "$v" == OPT_* ]] && export "$v"; done

sbatch \
    --gpus-per-node=A100:$NGPU --ntasks-per-node=$NGPU --mem=$MEM --time=$TIME \
    ${QOS:+--qos=$QOS} \
    --array="$ARRAY" \
    --output="logs/$RUN_GROUP/$RUN_NAME/log/job-%A_%a.log" \
    --error="logs/$RUN_GROUP/$RUN_NAME/log/job-%A_%a.log" \
    --export=ALL slurm/harness_train.sbatch
