#!/bin/bash
# ADE20K viewpoint POLICY (QReg) on our OWN foveated model: the exp22 foveated-teacherinit
# backbone plus the ADE20K probe trained on it. The counterpart of the exp35 policy runs,
# which use the published backbone and probe instead.
#
# THE TWO HALVES. A policy run needs both, and neither alone is enough:
#   --cfg.model-repo   the frozen pretrained backbone
#   --cfg.probe-repo   the trained segmentation head -- it IS the reward model (the reward
#                      is the fraction of the probe`s CE a glimpse removes, so a random head
#                      makes the reward pure noise)
# Both flags take a training .pt, a local HF directory, or a Hub id, so the probe is passed
# as the checkpoint the probe run wrote. No conversion step.
#
# Both halves are the training checkpoints themselves. Nothing is exported first; the HF
# layout is for publishing, not for feeding one of our own runs into another.
#
# CANVAS_GRID 32 IS NOT OPTIONAL. The probe was trained at canvas_grid 32, so a policy run
# at another grid feeds the reward model a canvas resolution it never saw and the reward
# degrades silently. The exp35 recipe uses 64 because ITS probe was trained at 64. Match the
# probe you are actually using.
#
# FOVEATED SCALE 2.0 IS NOT OPTIONAL EITHER. A foveated backbone derives its fixation window
# as fix_size = scale * H, so a rollout at a scale it never saw makes EVERY glimpse out of
# distribution. It does not crash -- mIoU just falls as glimpses accumulate.
#
# RAN 2026-08-31, ten seeds (jobs 15654175-15654184), and it worked: miou t4 = 0.450 +- 0.003
# against 0.428 for random viewpoints, with eval/miou_t0 = 0.3768 on every seed. Results and
# what they do and do not establish: docs/q_policy_foveated.md.
#
# THE PATHS BELOW were written as repos/CanViT-train/... and were repointed to repos/canvit
# on 2026-09-09, after the 2026-09-03 rename left the old prefix dangling and this script
# stopping at its own probe-completion guard. They are absolute on purpose: the checkpoints
# exist in exactly one place, mode 640 under group HPC_nib00021, so any project member can
# read them without copying -- and `logs/` is gitignored, so a path derived from your own
# clone would find nothing.
#
# The recipe is the exp35 policy arm with this backbone+probe swapped in and the grid matched
# to the probe. That arm runs against the published UNIFORM backbone; this one is foveated,
# so the policy`s action space becomes the fixation grid rather than the safe-box grid.
# eval/miou_t0 is the pre-policy glimpse -- for a foveated model at a FIXED scale that is the
# CENTRED foveation, not a full-image view -- so it should match a probe-only eval of the
# same pair before later timesteps are trusted.
#
# WHAT THE POLICY CHOOSES HERE. With foveated_scale.mode=fixed every glimpse is the same
# foveation pattern and only the fixation centre changes, t0 included; the window is
# fix_size = scale * H, so at 2.0 it spans twice the image side. The policy picks where to
# spend resolution, not what is visible -- hence an action space of centres with no scale
# dimension. (per_rollout/per_glimpse differ: there t0 falls back to scale 1, a true
# full-image anchor.)
#
# Full procedure: docs/q_policy_foveated.md
# 10 seeds: for s in 0 1 2 3 4 5 6 7 8 9; do SEED=$s bash slurm/runs/exp36_policy_qreg_fovi/policy-qreg-fovi-s0.sh; done
set -euo pipefail

# === ESSENTIALS ===
RUN_GROUP=caglar_exp36_policy_qreg_fovi
SEED="${SEED:-0}"
RUN_NAME=exp36-policy-qreg-fovi-s$SEED
ARRAY=0-0%1                  # single job: 9000 steps fits inside the walltime
# exp35's ten seeds measured 01:14 each at canvas 64; this runs at canvas 32, so 2h
# leaves ~45min headroom. NOTE the ade20k task checkpoints only at the END, so a
# timeout loses the run rather than resuming it.
TIME="${TIME:-0-02:00:00}"
# The fast `2h` QOS must be REQUESTED: a <=2h --time alone still lands in `normal`
# (verified -- the first submission showed QOS=normal despite --time=2:00:00), and
# `normal` has historically meant 24-31h of queueing. Set QOS= to fall back.
QOS="${QOS:-2h}"
MEM=64G
NGPU=1                       # ade20k is single-GPU only (supports_ddp=False)
TASK=ade20k

# === the two halves of the model ===
_PROBE_RUN=/mnt/vast-nhr/projects/nib00021/jonathan/repos/canvit/logs/jon_exp34_ade20k_probe/ade20k-fovi-ti-1196k/checkpoints
CFG_MODEL_REPO=/mnt/vast-nhr/projects/nib00021/jonathan/repos/canvit/logs/jon_exp22_full_runs/exp22-fovi-teacherinit-lrdrop-1196k/checkpoints/step-155648.pt
CFG_PROBE_REPO=$_PROBE_RUN/best.pt   # the training checkpoint itself; no conversion

# === config (exp31 lossfix recipe; only the grid follows the probe) ===
CFG_WANDB_PROJECT=caglar_exp36_policy_qreg_fovi
CFG_SEED=$SEED
CFG_MAX_STEPS=9000           # 9000, not 8000: the loop evaluates on step % val_every == 0
                             # and never reaches max_steps, so an 8000-step run's last eval
                             # is at 7000. Matches the exp35 arm this is compared against.
CFG_N_TIMESTEPS=5
CFG_BATCH_SIZE=16
CFG_CANVAS_GRID=32           # matches the exp34 probe -- see header
CFG_EVAL_POLICY=policy
CFG_RESIZE_MODE=squish       # the measurement contract every earlier CanViT number used
CFG_VAL_EVERY=1000
CFG_LOG_EVERY=50
CFG_NUM_WORKERS=4
EXTRA_ARGS="--preset policy_only --cfg.no-augment --cfg.foveated-scale.fixed-scale 2.0"
# =================

TRAIN_COMMIT=7f01b26
PYTORCH_COMMIT=d616b7b
FOVI_COMMIT=c399d3b

# Repo root, derived from this script's own location (slurm/runs/<group>/<run>.sh),
# so the run submits from YOUR clone rather than one hardcoded checkout.
# The probe must come from a FINISHED probe run. `best.pt` is not a completion marker -- it
# appears at the first evaluation (step 500) and is rewritten whenever the metric improves,
# so testing for it would happily hand the policy a probe a few hundred steps old. Gate on
# the final step checkpoint instead, the same way the pretrain LR-drop launchers gate on
# their drop-step file.
_PROBE_DONE=$_PROBE_RUN/step-40000.pt          # ade20k max_steps
if [ ! -f "$_PROBE_DONE" ]; then
    echo "REFUSING: the probe run has not finished ($_PROBE_DONE is missing)."
    echo "  best.pt may already exist, but it would be an early checkpoint."
    echo "  Wait for exp34's ade20k-fovi-ti-1196k arm, or point _PROBE_RUN at a finished"
    echo "  ade20k run -- only its head.* weights are read, so any finished run works if"
    echo "  --cfg.canvas-grid matches the grid that probe was trained at."
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
