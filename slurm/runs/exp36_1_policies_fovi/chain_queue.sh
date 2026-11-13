#!/bin/bash
# A SELF-RENEWING work queue of harness runs on 48 h V100 jobs (owner's request 2026-11-13:
# "automatically start a new V100 48 h job when the current one is about to run out").
#
#   bash chain_queue.sh init QDIR ENVFILE SPECFILE   make the queue: ENVFILE = the exports
#                                                    every run shares (TRAIN_COMMIT, CFG_*,
#                                                    EXTRA_ARGS, RUN_GROUP, NWORKERS,
#                                                    RUN_BUDGET_MIN ...), SPECFILE = one
#                                                    "<run_name> <extra args>" line per run
#   bash chain_queue.sh submit QDIR                  submit the queue's ONE job
#
# The job keeps its jupyter slot until the queue is drained: when it runs short of walltime
# with work left, it REQUEUES ITSELF (`scontrol requeue`, same job id, a fresh 2-day limit
# when it restarts). Why not "submit a successor": the V100 `jupyter` partition's QOS
# (interactive) allows only 2 SUBMITTED jobs per user and the owner keeps one for
# interactive work, so a pending successor would need a third slot (and a keeper outside
# the jobs would need a node to live on -- CPU partitions here are booked out for days).
#
# Each start of the job (chain_queue_job.sbatch -> `chain_queue.sh job`):
#   1. puts back into todo/ every run still claimed by an earlier start of this job or by a
#      job that is gone (its run dir is moved aside, so the rerun starts clean, no resume);
#   2. runs NWORKERS workers. A worker claims a run only while the job has at least
#      RUN_BUDGET_MIN minutes left, so no run is cut by the walltime;
#   3. when the workers stop with work left (and no QDIR/STOP), requeues itself.
# A run that fails is retried once at the end of the queue (todo/r_<name>), then failed/.
# Stop a chain: `touch QDIR/STOP` (the current start still finishes its runs) or scancel.
#
# init snapshots this script and the sbatch into QDIR, so editing the repo copies never
# changes a running chain.
set -euo pipefail
CMD="${1:?usage: chain_queue.sh init QDIR ENVFILE SPECFILE | submit QDIR | job QDIR}"
Q="${2:?QDIR}"
MAIN=/mnt/vast-nhr/projects/nib00021/jonathan/repos/canvit
log() { echo "[$(date '+%F %T')] $*"; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$CMD" = init ]; then
    ENVFILE="${3:?ENVFILE}"; SPECFILE="${4:?SPECFILE}"
    [ -e "$Q" ] && { echo "$Q exists"; exit 1; }
    mkdir -p "$Q"/{todo,claimed,done,failed}
    cp "$ENVFILE" "$Q/env.sh"
    cp "$HERE/chain_queue.sh" "$HERE/chain_queue_job.sbatch" "$Q/"
    ( source "$Q/env.sh"; : "${TRAIN_COMMIT:?}" "${RUN_GROUP:?}"; git -C "$MAIN" cat-file -e "$TRAIN_COMMIT^{commit}" )
    n=0
    while IFS= read -r line; do
        line="${line%%#*}"; read -r name _ <<< "$line" || true
        [ -z "${name:-}" ] && continue
        printf -v f '%s/todo/%03d_%s' "$Q" "$n" "$name"
        echo "$line" > "$f"; n=$((n + 1))
    done < "$SPECFILE"
    log "queued $n runs in $Q/todo"; ls "$Q/todo"
    exit 0
fi

# sbatch from inside a job would inherit the job's SLURM_*/SBATCH_* variables; strip them.
clean_sbatch() { env $(env | grep -oE '^(SLURM|SBATCH)_[A-Za-z0-9_]*' | sed 's/^/-u /') sbatch "$@"; }
submit() {  # -> job id
    (cd "$MAIN" && clean_sbatch --parsable --export=ALL,Q="$Q" -J "chain-$(basename "$Q")" \
        -o "$Q/job-%j.out" "$Q/chain_queue_job.sbatch")
}

if [ "$CMD" = submit ]; then
    jid=$(submit); log "submitted chain job $jid for $Q"; exit 0
fi

[ "$CMD" = job ] || { echo "unknown command $CMD"; exit 1; }
JOB="${SLURM_JOB_ID:?job runs inside a Slurm job}"
cd "$MAIN"
source "$Q/env.sh"
: "${TRAIN_COMMIT:?}" "${RUN_GROUP:?}" "${RUN_BUDGET_MIN:?}"
export TRAIN_COMMIT RUN_GROUP
log "chain job $JOB (start $((${SLURM_RESTART_COUNT:-0} + 1))) on $(hostname): $(nvidia-smi -L | tr '\n' ' ') queue $Q commit $TRAIN_COMMIT"

# 1. stale claims (claimed/<entry>@<job>.<worker>): this job's workers have not started yet,
#    so a claim tagged with this job id is from an earlier start of it
for c in "$Q"/claimed/*; do
    [ -e "$c" ] || continue
    base=$(basename "$c"); entry="${base%@*}"; cj="${base##*@}"; cj="${cj%%.*}"
    if [ "$cj" = "$JOB" ] || [ -z "$(squeue -h -j "$cj" 2>/dev/null)" ]; then
        name=$(awk '{print $1; exit}' "$c")
        mv "$MAIN/logs/$RUN_GROUP/$name" "$MAIN/logs/$RUN_GROUP/$name.cut-$(date +%s)" 2>/dev/null || true
        mv "$c" "$Q/todo/$entry"; log "requeued $entry (claimed by job $cj, which is gone or restarted)"
    fi
done

minutes_left() {  # %L: [D-]HH:MM:SS | MM:SS
    local t="" d=0 h=0 m=0 try
    for try in 1 2 3 4 5; do  # a transient squeue failure must not end the job early
        t=$(squeue -h -j "$JOB" -o %L 2>/dev/null || true); [ -n "$t" ] && break; sleep 60
    done
    [ -n "$t" ] || { echo 0; return; }
    [[ "$t" == *-* ]] && { d=${t%%-*}; t=${t#*-}; }
    IFS=: read -r a b c <<< "$t"
    if [ -n "${c:-}" ]; then h=$a; m=$b; else m=$a; fi
    echo $(( 10#$d * 1440 + 10#$h * 60 + 10#$m ))
}

# 2. workers
worker() {
    local w=$1 next rc name
    while :; do
        local left; left=$(minutes_left)
        [ "$left" -ge "$RUN_BUDGET_MIN" ] || { log "w$w: $left min left < $RUN_BUDGET_MIN -- stop"; return; }
        next=$(ls "$Q/todo" | head -1 || true)
        [ -n "$next" ] || { log "w$w: queue empty"; return; }
        mv "$Q/todo/$next" "$Q/claimed/$next@$JOB.$w" 2>/dev/null || continue
        log "w$w: claimed $next ($left min left): $(cat "$Q/claimed/$next@$JOB.$w")"
        rc=0
        SWEEP_SPEC="$(cat "$Q/claimed/$next@$JOB.$w")" bash slurm/harness_sweep.sbatch < /dev/null || rc=$?
        if [ "$rc" -eq 0 ]; then mv "$Q/claimed/$next@$JOB.$w" "$Q/done/$next"
        elif [ "${next#r_}" = "$next" ]; then
            name=$(awk '{print $1; exit}' "$Q/claimed/$next@$JOB.$w")
            mv "$MAIN/logs/$RUN_GROUP/$name" "$MAIN/logs/$RUN_GROUP/$name.failed-$(date +%s)" 2>/dev/null || true
            mv "$Q/claimed/$next@$JOB.$w" "$Q/todo/r_$next"
        else mv "$Q/claimed/$next@$JOB.$w" "$Q/failed/$next"; fi
        log "w$w: finished $next exit $rc"
    done
}
for w in $(seq 1 "${NWORKERS:-1}"); do
    worker "$w" > "$Q/worker-$JOB-$w.log" 2>&1 &
    [ "$w" -lt "${NWORKERS:-1}" ] && sleep "${WORKER_STAGGER_S:-900}"
done
wait

log "chain job $JOB: $(ls "$Q/done" | wc -l) done, $(ls "$Q/failed" | wc -l) failed, $(ls "$Q/todo" | wc -l) todo"
# 3. work left -> requeue this job (same id; it restarts with a fresh time limit)
if [ "$(ls "$Q/todo" | wc -l)" -gt 0 ] && [ ! -e "$Q/STOP" ]; then
    log "work left -- requeueing job $JOB"
    scontrol requeue "$JOB"
    sleep 600   # the requeue terminates this script
fi
log "queue drained (or STOP) -- chain ends"
