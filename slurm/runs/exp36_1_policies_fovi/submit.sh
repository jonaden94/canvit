#!/bin/bash
# exp36_1: on-policy PG / Q-Prop vs QReg on V100s (jupyter partition), 8 seeds per arm.
#
#   bash slurm/runs/exp36_1_policies_fovi/submit.sh smoke   # 1 job: 300 steps of qprop + qreg
#   bash slurm/runs/exp36_1_policies_fovi/submit.sh full    # 2 jobs x 16 runs (8 seeds), 48 h each
#
# Every arm is exp36's recipe (exp36_policy_qreg_fovi/policy-qreg-fovi-s0.sh): the exp22
# foveated-teacherinit backbone, its exp34 ADE20K probe, canvas grid 32, fixed view scale
# 2.0, 9000 steps, batch 16, 5 glimpses. What differs is listed in queue.sbatch.
#
# PINS. canvit f3d6e6b is the first commit that can run --rl.qprop (Caglar's branch head;
# exp36's 7f01b26 rejects the flag). ALL arms pin it -- the QReg arm included -- so the four
# arms differ only in their flags. fovi c399d3b is exp36's. PYTORCH_COMMIT is deliberately
# unset: f3d6e6b is post-core-merge, where it has no effect.
#
# NUMERICS ON V100. No bf16 on sm_70: harness/run.py runs the training forward in fp16
# instead. No GradScaler is needed for THESE runs: policy_only runs the frozen backbone
# under no_grad, and the scorer/critic run outside autocast in fp32, so no gradient passes
# through fp16. Eval keeps its hardcoded bf16 autocast (emulated on V100). The qreg arm is
# the control against exp36's 10 A100 seeds (t4 0.4500 +- 0.0033).
set -euo pipefail
MODE="${1:?usage: submit.sh smoke|full}"

_PROBE_RUN=/mnt/vast-nhr/projects/nib00021/jonathan/repos/canvit/logs/jon_exp34_ade20k_probe/ade20k-fovi-ti-1196k/checkpoints
[ -f "$_PROBE_RUN/step-40000.pt" ] || { echo "REFUSING: probe run not finished"; exit 1; }

export TASK=ade20k NGPU=1
export TRAIN_COMMIT=f3d6e6b FOVI_COMMIT=c399d3b
export CFG_MODEL_REPO=/mnt/vast-nhr/projects/nib00021/jonathan/repos/canvit/logs/jon_exp22_full_runs/exp22-fovi-teacherinit-lrdrop-1196k/checkpoints/step-155648.pt
export CFG_PROBE_REPO=$_PROBE_RUN/best.pt
export CFG_N_TIMESTEPS=5 CFG_BATCH_SIZE=16 CFG_CANVAS_GRID=32 CFG_EVAL_POLICY=policy
export CFG_RESIZE_MODE=squish CFG_LOG_EVERY=50 CFG_NUM_WORKERS=4

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export ADE20K_ROOT="${ADE20K_ROOT:-/mnt/vast-nhr/projects/nib00021/jonathan/datasets/zhoubolei--scene_parse_150/ADEChallengeData2016}"
_Q=slurm/runs/exp36_1_policies_fovi/queue.sbatch

case "$MODE" in
    smoke)
        # Environment + fp16 + timing check: 300 steps, ONE full eval (at step 0), no wandb.
        export RUN_GROUP=jon_exp36_1_smoke RUN_PREFIX=exp36_1-smoke
        export CFG_MAX_STEPS=300 CFG_VAL_EVERY=1000 CFG_VIZ_EVERY=0 CFG_TRACKER=none
        export QUEUE="qprop:0 qreg:0" RUN_TIMEOUT=2h
        mkdir -p "logs/$RUN_GROUP"
        sbatch --time=0-03:00:00 --job-name=exp36_1-smoke \
            --output="logs/$RUN_GROUP/queue-%j.log" --export=ALL "$_Q"
        ;;
    full)
        export RUN_GROUP=jon_exp36_1_policies_fovi RUN_PREFIX=exp36_1-v100
        export CFG_WANDB_PROJECT=jon_exp36_1_policies_fovi
        export CFG_MAX_STEPS=9000 CFG_VAL_EVERY=1000
        # 8 seeds x 4 arms = 32 runs; job A takes the even seeds, job B the odd ones (16
        # runs each, ~40 h at the smoke test's V100 speeds). Seed-major: every seed runs all
        # four arms before the next seed starts, and the arm order rotates by one per seed
        # (job B offset by two), so a job cut off at 48 h loses at most one run per arm and
        # the two jobs lose different arms.
        ARMS=(qreg qprop pg qprop-lnfeat)
        queue_for() {  # $1 = rotation offset, rest = seeds
            local off=$1 k=0 q="" s i; shift
            for s in "$@"; do
                for i in 0 1 2 3; do q+="${ARMS[$(( (i + k + off) % 4 ))]}:$s "; done
                k=$((k + 1))
            done
            echo "${q% }"
        }
        QA="$(queue_for 0 0 2 4 6)"
        QB="$(queue_for 2 1 3 5 7)"
        mkdir -p "logs/$RUN_GROUP"
        for q in "$QA" "$QB"; do
            QUEUE="$q" sbatch --time=2-00:00:00 --job-name=exp36_1-v100 \
                --output="logs/$RUN_GROUP/queue-%j.log" --export=ALL "$_Q"
        done
        ;;
    *) echo "usage: submit.sh smoke|full"; exit 1 ;;
esac
