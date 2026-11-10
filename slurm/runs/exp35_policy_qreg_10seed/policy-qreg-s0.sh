#!/bin/bash
# exp35 — ADE20K viewpoint-POLICY (QReg) through the HARNESS, 10 seeds, at the CURRENT code base.
#
# Recipe = exp27's ARM "lossfix" (slurm/runs/exp27/policy-lossfix-s0.sh) verbatim; only the
# pins and the run group change. That arm is the right base because it is the HARNESS path
# at pin bc0b16b, i.e. AFTER the one real harness/rl_train divergence was fixed (the policy
# gradient was 0.8x). After that fix the two training paths are bit-identical, so this is
# "our own ported q policy", which is what is being re-verified here.
#
# NOT based on exp27's bneval / pooled / oldloop arms: despite their names those submit
# slurm/archive/ade20k/train_policy.sbatch, i.e. the DELETED `canvit_pretrain.ade20k.rl_train`
# reference implementation. They are the reference the harness was judged against, not the
# harness itself. (This is easy to get wrong -- bneval's header is inherited text from the
# old-loop arm and reads as if it were the reference.)
#
# THE FROZEN BACKBONE. The owner asked to take the backbone out of one of the published
# ade20k-policy HF checkpoints. That is neither possible nor necessary:
#   * Not possible -- those checkpoints contain NO backbone. `best.pt` holds only
#     `net_state` (452 tensors, 5.68M params, 22.7MB safetensors) = the ViewpointScorer.
#     Policy training freezes the backbone, so it is never saved.
#   * Not necessary -- every one of the 21 published policy checkpoints records
#     model_repo = canvit/canvitb16-add-vpe-pretrain-g128px-s512px-in21k-dv3b16-2026-02-02,
#     and that string is ALREADY `Ade20kConfig.model_repo`'s default (ade20k/config.py:48).
#   So the owner's premise -- one identical backbone shared by all published policies -- is
#   correct and verified; it just needs no flag. CFG_MODEL_REPO is deliberately NOT set.
#   That backbone is the published Feb-2026 pretrain, NOT any exp22/exp28 model, so exp35 is
#   independent of exp28/exp29/exp30 -- and directly comparable to exp27.
#
# 10 seeds: for s in 0 1 2 3 4 5 6 7 8 9; do SEED=$s bash slurm/runs/exp35_policy_qreg_10seed/policy-qreg-s0.sh; done
set -euo pipefail

# === ESSENTIALS ===
RUN_GROUP=caglar_exp35_policy_qreg_10seed
SEED="${SEED:-0}"
RUN_NAME=exp35-policy-qreg-s$SEED
ARRAY=0-0%1                  # single job: 8000 steps fits well inside the walltime
TIME="${TIME:-0-06:00:00}"   # exp27 used 4h; widened -- mode (b) costs ~9% and nodes vary
MEM=64G
NGPU=1                       # ade20k has supports_ddp=False
TASK=ade20k

# === config (exp27 lossfix recipe; everything else must come from the defaults, or this
#     stops being the ported reference) ===
CFG_WANDB_PROJECT=caglar_exp35_policy_qreg_10seed
CFG_SEED=$SEED
CFG_MAX_STEPS=9000           # 9000, not 8000: the loop evaluates when step % val_every == 0
                             # and never reaches max_steps itself, so an 8000-step run's last
                             # eval is at 7000. One extra 1000-step block buys the eval AT
                             # 8000, i.e. the full-length result this recipe is judged on.
CFG_N_TIMESTEPS=5
CFG_BATCH_SIZE=16
CFG_CANVAS_GRID=64
CFG_PROBE_REPO=canvit/probe-ade20k-40k-s512-c64-in21k
CFG_EVAL_POLICY=policy
CFG_RESIZE_MODE=squish       # THE MEASUREMENT CONTRACT. Not optional -- it has silently
                             # regressed once before (1a0b452 defaulted it to center_crop and
                             # an arm landed 0.016 "better" than the band, ~20x the seed spread).
CFG_VAL_EVERY=1000           # rl_train evaluates on the same cadence (9 evals/run)
CFG_LOG_EVERY=50
CFG_NUM_WORKERS=4
EXTRA_ARGS="--preset policy_only --cfg.no-augment"   # objective=qreg, mode (b) and score_res 128 are DEFAULTS
# =================

TRAIN_COMMIT=716051a
PYTORCH_COMMIT=d616b7b
FOVI_COMMIT=c399d3b

# Repo root, derived from this script's own location (slurm/runs/<group>/<run>.sh),
# so the run submits from YOUR clone rather than one hardcoded checkout.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
mkdir -p "logs/$RUN_GROUP/$RUN_NAME/log"
export ADE20K_ROOT="${ADE20K_ROOT:-/mnt/vast-nhr/projects/nib00021/jonathan/datasets/zhoubolei--scene_parse_150/ADEChallengeData2016}"
export TASK RUN_GROUP RUN_NAME NGPU EXTRA_ARGS TRAIN_COMMIT PYTORCH_COMMIT FOVI_COMMIT
for v in $(compgen -v); do [[ "$v" == CFG_* || "$v" == OPT_* ]] && export "$v"; done

sbatch \
    --gpus-per-node=A100:$NGPU --ntasks-per-node=$NGPU --mem=$MEM --time=$TIME \
    --array="$ARRAY" \
    --output="logs/$RUN_GROUP/$RUN_NAME/log/job-%A_%a.log" \
    --error="logs/$RUN_GROUP/$RUN_NAME/log/job-%A_%a.log" \
    --export=ALL slurm/harness_train.sbatch
