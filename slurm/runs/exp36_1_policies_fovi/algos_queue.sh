#!/bin/bash
# policy-algos (2026-11-13): a file-based WORK QUEUE of exp36_1-setup runs, drained by any number
# of workers, each an srun --overlap step inside an already-running V100 job. Arms and their
# flags are documented in algos_in_job.sh (same five arms, same env); this script only replaces
# that one's fixed per-job seed list with a shared queue, so every GPU (and every concurrent
# worker on one GPU) keeps pulling the next run until none is left.
#
#   bash algos_queue.sh init QDIR "SEEDS"           one file per run in QDIR/todo, seed-major
#                                                   with a rotating arm order (as exp36_1)
#   TRAIN_COMMIT=<commit> setsid nohup bash algos_queue.sh work JOBID QDIR [WAIT] > LOG 2>&1 &
#
# A run is CLAIMED by an atomic `mv todo/X -> claimed/X.<worker>`; finished runs land in
# done/ (exit 0) or failed/. WAIT as in algos_in_job.sh ("log:<file>:<text>" or a file path).
set -euo pipefail
CMD="${1:?usage: algos_queue.sh init QDIR SEEDS | work JOBID QDIR [WAIT]}"
MAIN=/mnt/vast-nhr/projects/nib00021/jonathan/repos/canvit
log() { echo "[$(date '+%F %T')] $*"; }
ARMS=(qreg-lnfeat a2c-lnfeat ppo-lnfeat qtd-lnfeat qreg-dense4-lnfeat)
arm_args() {
    case "$1" in
        qreg-lnfeat)        echo "" ;;
        a2c-lnfeat)         echo "--rl.objective ac --rl.no-select-bn-eval" ;;
        ppo-lnfeat)         echo "--rl.objective ac --rl.no-select-bn-eval --rl.ppo-epochs 2" ;;
        qtd-lnfeat)         echo "--rl.objective qtd" ;;
        qreg-dense4-lnfeat) echo "--rl.qreg-extra-candidates 4" ;;
    esac
}

if [ "$CMD" = init ]; then
    QDIR="${2:?}"; SEEDS="${3:?}"
    mkdir -p "$QDIR"/{todo,claimed,done,failed}
    n=0 k=0
    for s in $SEEDS; do
        for i in 0 1 2 3 4; do
            arm=${ARMS[$(( (i + k) % 5 ))]}
            printf -v f '%s/todo/%03d_%s-s%s' "$QDIR" "$n" "$arm" "$s"
            echo "exp36_1-v100-$arm-s$s $(arm_args "$arm") --cfg.seed $s" > "$f"
            n=$((n + 1))
        done
        k=$((k + 1))
    done
    log "queued $n runs in $QDIR/todo"; ls "$QDIR/todo"
    exit 0
fi

[ "$CMD" = work ] || { echo "unknown command $CMD"; exit 1; }
JOB="${2:?}"; QDIR="${3:?}"; WAIT="${4:-}"
: "${TRAIN_COMMIT:?TRAIN_COMMIT must be the policy-algos commit to pin}"
cd "$MAIN"
git cat-file -e "$TRAIN_COMMIT^{commit}"
WORKER="$(hostname -s).$$"

# --- identical to exp36_1's submit.sh full (and algos_in_job.sh) ---
_PROBE_RUN=$MAIN/logs/jon_exp34_ade20k_probe/ade20k-fovi-ti-1196k/checkpoints
export TASK=ade20k NGPU=1 TRAIN_COMMIT FOVI_COMMIT=c399d3b
export CFG_MODEL_REPO=$MAIN/logs/jon_exp22_full_runs/exp22-fovi-teacherinit-lrdrop-1196k/checkpoints/step-155648.pt
export CFG_PROBE_REPO=$_PROBE_RUN/best.pt
export CFG_N_TIMESTEPS=5 CFG_BATCH_SIZE=16 CFG_CANVAS_GRID=32 CFG_EVAL_POLICY=policy
export CFG_RESIZE_MODE=squish CFG_LOG_EVERY=50 CFG_NUM_WORKERS=4
export ADE20K_ROOT="${ADE20K_ROOT:-/mnt/vast-nhr/projects/nib00021/jonathan/datasets/zhoubolei--scene_parse_150/ADEChallengeData2016}"
export RUN_GROUP=jon_exp36_1_policies_fovi CFG_WANDB_PROJECT=jon_exp36_1_policies_fovi
export CFG_MAX_STEPS=9000 CFG_VAL_EVERY=1000
export EXTRA_ARGS="--preset policy_only --cfg.no-augment --cfg.foveated-scale.fixed-scale 2.0 --rl.feature-groups ln_feat"
export SLURM_OVERLAP=1 SLURM_NTASKS_PER_NODE=1   # see pg_lnfeat_in_job.sh
# Two workers may share one 32 GB V100 (measured peaks: ppo 15.2 GB, dense qreg 12.1 GB,
# reserved); expandable segments cut fragmentation. No effect on numerics.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

job_running() { [ -n "$(squeue -h -j "$JOB" -t R 2>/dev/null)" ]; }
if [ -n "$WAIT" ]; then
    log "worker $WORKER waiting for: $WAIT (job $JOB)"
    while :; do
        job_running || { log "job $JOB not running -- abort"; exit 1; }
        case "$WAIT" in
            log:*) f="${WAIT#log:}"; txt="${f#*:}"; f="${f%%:*}"; grep -qF "$txt" "$f" 2>/dev/null && break ;;
            *) [ -e "$WAIT" ] && break ;;
        esac
        sleep 60
    done
fi

log "worker $WORKER: job $JOB, commit $TRAIN_COMMIT, queue $QDIR"
while job_running; do
    next=$(ls "$QDIR/todo" 2>/dev/null | head -1)
    [ -n "$next" ] || { log "queue empty -- worker $WORKER done"; exit 0; }
    mv "$QDIR/todo/$next" "$QDIR/claimed/$next.$WORKER" 2>/dev/null || continue   # lost the race
    export SWEEP_SPEC="$(cat "$QDIR/claimed/$next.$WORKER")"
    log "claimed $next: $SWEEP_SPEC"
    rc=0
    srun --jobid="$JOB" --overlap --cpus-per-task="${WORKER_CPUS:-8}" -n1 bash slurm/harness_sweep.sbatch < /dev/null || rc=$?
    if [ "$rc" -eq 0 ]; then mv "$QDIR/claimed/$next.$WORKER" "$QDIR/done/$next"
    elif [ "${next#r_}" = "$next" ]; then  # first failure (e.g. two evals' OOM): retry ONCE, at the end
        mv "$QDIR/claimed/$next.$WORKER" "$QDIR/todo/r_$next"
        # move the dead run's dir aside, or the retry would RESUME from its checkpoints
        run_name=$(awk '{print $1; exit}' "$QDIR/todo/r_$next")
        mv "$MAIN/logs/$RUN_GROUP/$run_name" "$MAIN/logs/$RUN_GROUP/$run_name.failed-$(date +%s)" 2>/dev/null || true
    else mv "$QDIR/claimed/$next.$WORKER" "$QDIR/failed/$next"; fi
    log "finished $next exit $rc"
done
log "job $JOB no longer running -- worker $WORKER stops"
