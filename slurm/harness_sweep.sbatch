#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH -A nib00021
#SBATCH -C inet                  # required for HuggingFace Hub access
#
# Several harness runs ONE AFTER ANOTHER inside one allocation: for partitions that allow
# only a few jobs per user (the V100 `jupyter` partition: 2), where an array is not an
# option. Each run is an ordinary harness_train.sbatch run -- same pinning, same env, same
# per-run logs -- only the per-run variables differ.
#
# Env (set by the launcher, read at submit time):
#   SWEEP_SPEC     newline-separated lines "<run_name> <extra args...>" (blank / # lines skipped);
#                  the extra args are appended to EXTRA_ARGS for that run
#   TRAIN_COMMIT   REQUIRED: harness_train.sbatch itself is taken from this commit, so the
#                  runs use the launcher script of the code they run
#   RUN_GROUP, TASK, NGPU, EXTRA_ARGS, CFG_*, OPT_*, FOVI_COMMIT  as for harness_train.sbatch
#
# A failed run is recorded and the next one starts; the job's exit code is non-zero if any
# run failed. A run cut off by the walltime is lost (no resume), so order SWEEP_SPEC by
# priority.
set -u
cd "${SLURM_SUBMIT_DIR:-$PWD}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }
: "${SWEEP_SPEC:?SWEEP_SPEC must be set}"
: "${TRAIN_COMMIT:?TRAIN_COMMIT must be set (harness_train.sbatch is taken from it)}"
: "${RUN_GROUP:?RUN_GROUP must be set}"

_HARNESS="${TMPDIR:-/tmp}/harness_train.${SLURM_JOB_ID}.sbatch"
git -C "$PWD" show "$TRAIN_COMMIT:slurm/harness_train.sbatch" > "$_HARNESS" || {
    log "ERROR: cannot read slurm/harness_train.sbatch at $TRAIN_COMMIT"; exit 1; }
_BASE_EXTRA="${EXTRA_ARGS:-}"

log "=== sweep: job $SLURM_JOB_ID, group $RUN_GROUP, commit $TRAIN_COMMIT, host $(hostname) ==="
_SUMMARY=()
_FAILED=0
while IFS= read -r line; do
    line="${line%%#*}"
    read -r name args <<< "$line"
    [ -z "${name:-}" ] && continue
    mkdir -p "logs/$RUN_GROUP/$name/log"
    out="logs/$RUN_GROUP/$name/log/job-${SLURM_JOB_ID}.log"
    log "--- run $name: ${args:-<no extra args>}  (log: $out)"
    t0=$(date +%s)
    # stdin from /dev/null: srun reads stdin and would otherwise swallow the rest of SWEEP_SPEC
    # (the loop's input), ending the sweep after the first run (job 17138202)
    RUN_NAME="$name" EXTRA_ARGS="$_BASE_EXTRA ${args:-}" bash "$_HARNESS" > "$out" 2>&1 < /dev/null
    rc=$?
    mins=$(( ($(date +%s) - t0) / 60 ))
    log "--- run $name: exit $rc after ${mins} min"
    _SUMMARY+=("$name exit=$rc ${mins}min")
    [ "$rc" -ne 0 ] && _FAILED=$((_FAILED + 1))
done <<< "$SWEEP_SPEC"

log "=== sweep done: ${#_SUMMARY[@]} runs, $_FAILED failed ==="
for s in "${_SUMMARY[@]}"; do log "    $s"; done
rm -f "$_HARNESS"
[ "$_FAILED" -eq 0 ]
