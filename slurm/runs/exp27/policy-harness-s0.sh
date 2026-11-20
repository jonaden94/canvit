#!/bin/bash
# exp27 ARM B — the SAME recipe through the UNIFIED HARNESS.
#
#   python -m canvit_train.harness.run ade20k --preset policy_only
#
# This is the run the whole session was building toward: the first production-scale
# test of the harness's policy path. `canvit_claude/canvit/unification/14-parity-coverage.md` calls
# the joint policy path "the substrate for all CanViT-RL work" whose first real run
# doubles as its production gate — this is that run.
#
# Judged against ARM A (exp27-policy-oldloop-s0), not against the published band,
# so the comparison carries no hardware or stack confound.
#
# CONFIG MAPPING — rl_train -> harness (canvit_claude/canvit/unification/15 SS A). Everything below
# is a deliberate match; anything NOT listed is already identical by default
# (model_repo, scene_size 512, batch_size 16, mode=frozen,
# and the whole JointPolicyConfig: qreg, scales (0.5,0.25), centers_per_axis 16,
# width 128, block_layers 3, prime_on_policy 0.5, dueling, lr 2e-4, wd 1e-2,
# betas (0.9,0.95), warmup_frac 0.125, target_momentum 0.997).
#
#   rl_train train_horizon=4  ->  --cfg.n-timesteps 5   (t0 full + 4 policy glimpses)
#   rl_train max_steps  = 640_000 // (16 * (1+4)) = 8000
#   rl_train probe      = probe-ade20k-40k-s512-c{canvas_grid}-in21k
#   rl_train NO augment ->  --cfg.no-augment
#   deploy eval + CE-based best.pt  ->  --cfg.eval-policy policy
#   squish resize       ->  CFG_RESIZE_MODE=squish   (see below — NOT the harness default)
#
# RESIZE MODE IS THE MEASUREMENT CONTRACT, not a tuning knob. CanViT-PyTorch-RL
# squish-resizes image AND mask to scene_size everywhere (its dataset class is named
# Ade20kSquish; config.py's docstring calls it "the measurement contract every entry
# point builds on"), and the qband band + EG-C2F baselines exist ONLY under squish.
# `Ade20kConfig.resize_mode` defaults to center_crop — the right default for NEW work
# (aspect-preserving, matches pretraining) but NOT band-comparable: the first exp27
# attempt ran center_crop on both arms and came out 0.016 CE "better" than the band,
# ~20x the band's own 0.0007 seed spread. Set it explicitly here, forever.
#
# THE ONE KNOWN REMAINING DIFFERENCE (doc 15 SS A gap #5, owner-deferred): the
# reward is scored at the probe's native 64x64 grid here, vs score_res=128 in
# rl_train. The reward FORMULA is identical. If this arm misses arm A, that is the
# first thing to suspect.
#
# NOTE on --cfg.no-augment: it must live in EXTRA_ARGS, NOT as CFG_AUGMENT=False.
# tyro renders bools as paired flags (--cfg.augment / --cfg.no-augment), while the
# launcher's CFG_FOO_BAR mapping emits `--cfg.foo-bar VALUE` — which cannot express
# a flag. Caught by a local smoke before submitting.
#
# NOTHING IS SUBMITTED by writing this file.
set -euo pipefail

# === ESSENTIALS ===
RUN_GROUP=exp27
SEED="${SEED:-0}"            # SEED=1 bash <this> for a second seed
RUN_NAME=exp27-policy-harness-s$SEED
ARRAY=0-0%1                  # single job: 8000 steps fits well inside the walltime
TIME=0-04:00:00
MEM=64G
NGPU=1                       # ade20k has supports_ddp=False
TASK=ade20k

# === config ===
CFG_WANDB_PROJECT=exp27
CFG_SEED=$SEED
CFG_MAX_STEPS=8000
CFG_N_TIMESTEPS=5
CFG_BATCH_SIZE=16
CFG_CANVAS_GRID=64
CFG_PROBE_REPO=canvit/probe-ade20k-40k-s512-c64-in21k
CFG_EVAL_POLICY=policy
CFG_RESIZE_MODE=squish       # THE MEASUREMENT CONTRACT — see the note below. Not optional.
CFG_VAL_EVERY=1000           # rl_train evaluates on the same cadence (9 evals/run)
CFG_LOG_EVERY=50
CFG_NUM_WORKERS=4
EXTRA_ARGS="--preset policy_only --cfg.no-augment"   # mode (b) + score_res 128 are DEFAULTS as of 4428e34
# =================

# cea4dee: shared eval-viewpoint knob (so a policy run is validated by DEPLOYING the
# policy instead of on random glimpses) + the recipe knobs (betas, LR ramp, augment,
# probe head in frozen mode). 4db7c3f: the squish protocol fix above.
PRETRAIN_COMMIT=4428e34
PYTORCH_COMMIT=1f5121b
FOVI_COMMIT=c399d3b

# NOT in env.sh — every exp24 ade20k script exports it explicitly, and without it
# _default_ade20k_root() silently falls back to /datasets/ADE20k/... which does not
# exist here.
export ADE20K_ROOT=/mnt/vast-nhr/projects/nib00021/jonathan/datasets/zhoubolei--scene_parse_150/ADEChallengeData2016

# Repo root, derived from this script's own location (slurm/runs/<group>/<run>.sh),
# so the run submits from YOUR clone rather than one hardcoded checkout.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
mkdir -p "logs/$RUN_GROUP/$RUN_NAME/log"
export TASK RUN_GROUP RUN_NAME NGPU EXTRA_ARGS ADE20K_ROOT PRETRAIN_COMMIT PYTORCH_COMMIT FOVI_COMMIT
for v in $(compgen -v); do [[ "$v" == CFG_* ]] && export "$v"; done

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
