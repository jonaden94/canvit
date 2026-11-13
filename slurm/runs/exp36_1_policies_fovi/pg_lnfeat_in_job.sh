#!/bin/bash
# exp36_1 addendum: PG with canvas-only input (`ln_feat`), seeds 0-4 -- the fifth arm next to
# qreg / qprop / pg / qprop-lnfeat, same recipe, pins, W&B project and run group.
#
# Runs INSIDE an already-running V100 allocation (the owner's 48 h interactive job) as an
# overlap step, once that job's own work has written WAIT_FILE; the 5 runs go one after
# another through slurm/harness_sweep.sbatch (harness_train.sbatch taken from TRAIN_COMMIT).
# Start it detached on a login node -- the srun client must outlive this shell:
#
#   setsid nohup bash slurm/runs/exp36_1_policies_fovi/pg_lnfeat_in_job.sh JOBID WAIT_FILE \
#       > logs/jon_exp36_1_policies_fovi/pg-lnfeat-in-job.log 2>&1 < /dev/null &
set -euo pipefail
JOB="${1:?usage: pg_lnfeat_in_job.sh JOBID [WAIT_FILE]}"
WAIT_FILE="${2:-}"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
log() { echo "[$(date '+%F %T')] $*"; }

# --- identical to submit.sh full (exp36_1) ---
_PROBE_RUN=/mnt/vast-nhr/projects/nib00021/jonathan/repos/canvit/logs/jon_exp34_ade20k_probe/ade20k-fovi-ti-1196k/checkpoints
export TASK=ade20k NGPU=1
export TRAIN_COMMIT=f3d6e6b FOVI_COMMIT=c399d3b
export CFG_MODEL_REPO=/mnt/vast-nhr/projects/nib00021/jonathan/repos/canvit/logs/jon_exp22_full_runs/exp22-fovi-teacherinit-lrdrop-1196k/checkpoints/step-155648.pt
export CFG_PROBE_REPO=$_PROBE_RUN/best.pt
export CFG_N_TIMESTEPS=5 CFG_BATCH_SIZE=16 CFG_CANVAS_GRID=32 CFG_EVAL_POLICY=policy
export CFG_RESIZE_MODE=squish CFG_LOG_EVERY=50 CFG_NUM_WORKERS=4
export ADE20K_ROOT="${ADE20K_ROOT:-/mnt/vast-nhr/projects/nib00021/jonathan/datasets/zhoubolei--scene_parse_150/ADEChallengeData2016}"
export RUN_GROUP=jon_exp36_1_policies_fovi CFG_WANDB_PROJECT=jon_exp36_1_policies_fovi
export CFG_MAX_STEPS=9000 CFG_VAL_EVERY=1000
# queue.sbatch's `pg` arm + its `qprop-lnfeat` feature restriction
export EXTRA_ARGS="--preset policy_only --cfg.no-augment --cfg.foveated-scale.fixed-scale 2.0 --rl.objective pg --rl.no-select-bn-eval --rl.feature-groups ln_feat"
export SWEEP_SPEC="$(for s in 0 1 2 3 4; do echo "exp36_1-v100-pg-lnfeat-s$s --cfg.seed $s"; done)"

if [ -n "$WAIT_FILE" ]; then
    log "waiting for $WAIT_FILE (job $JOB)"
    until [ -e "$WAIT_FILE" ]; do
        [ -n "$(squeue -h -j "$JOB" -t R 2>/dev/null)" ] || { log "job $JOB not running -- abort"; exit 1; }
        sleep 60
    done
fi
log "starting sweep in job $JOB:"; echo "$SWEEP_SPEC"
# SLURM_OVERLAP=1 reaches the nested `srun` in harness_train.sbatch (tested: it sees the GPU)
export SLURM_OVERLAP=1
# A batch job sets this, a step started from a login node does not -- and harness_train.sbatch
# reads it under `set -u` (WORLD_SIZE); without it the run dies before python starts.
export SLURM_NTASKS_PER_NODE=1
rc=0
srun --jobid="$JOB" --overlap --cpus-per-task=8 -n1 bash slurm/harness_sweep.sbatch || rc=$?
log "sweep finished, exit $rc"
