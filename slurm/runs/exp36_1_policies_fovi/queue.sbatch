#!/bin/bash
#SBATCH -p jupyter
#SBATCH -A nib00021
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=V100:1
#SBATCH --mem=64G
#SBATCH -C inet                  # as harness_train.sbatch: HuggingFace Hub access
#
# Runs a QUEUE of ADE20K policy runs ONE AFTER ANOTHER on a single GPU, inside one
# allocation. Each entry is "ARM:SEED"; each run is the unchanged slurm/harness_train.sbatch
# (pinning, env, CFG_/EXTRA_ARGS handling) invoked as a child process -- that script ends in
# `exec srun`, which replaces only the child shell, so the loop continues afterwards.
# harness_train.sbatch is taken from TRAIN_COMMIT (as slurm/harness_sweep.sbatch does), not
# from the working tree, so edits to the clone cannot reach a queued or running job.
# (Changed 2026-11-12, after jobs 17232461/2 were submitted: those still read the clone's.)
# A failed run is logged and the queue moves on. Submitted by submit.sh, which sets
# QUEUE, RUN_GROUP, the pins and every CFG_* shared by the arms.
#
# The arms (all: exp36's recipe; the PG-family arms are made ON-POLICY by sampling the
# glimpse from the same train-mode scores whose log-prob is trained):
#   qreg          exp36 QReg, unchanged (argmax selection is unaffected by select_bn_eval)
#   qprop         PG + Q-Prop critic, on-policy
#   pg            the same without the critic
#   qprop-lnfeat  qprop with the canvas (ln_feat) as the ONLY input, for actor and critic
set -u
cd "${SLURM_SUBMIT_DIR:?}"

BASE="--preset policy_only --cfg.no-augment --cfg.foveated-scale.fixed-scale 2.0"
ONPOL="--rl.no-select-bn-eval"
arm_args() {
    case "$1" in
        qreg)         echo "$BASE" ;;
        qprop)        echo "$BASE --rl.objective pg --rl.qprop $ONPOL" ;;
        pg)           echo "$BASE --rl.objective pg $ONPOL" ;;
        qprop-lnfeat) echo "$BASE --rl.objective pg --rl.qprop $ONPOL --rl.feature-groups ln_feat" ;;
        *) return 1 ;;
    esac
}

echo "[$(date '+%F %T')] queue on $(hostname): ${QUEUE:?}"
_HARNESS="${TMPDIR:-/tmp}/harness_train.${SLURM_JOB_ID}.sbatch"
git show "${TRAIN_COMMIT:?}:slurm/harness_train.sbatch" > "$_HARNESS" || {
    echo "ERROR: cannot read slurm/harness_train.sbatch at $TRAIN_COMMIT"; exit 1; }
for entry in $QUEUE; do
    arm="${entry%%:*}"; seed="${entry##*:}"
    if ! EXTRA_ARGS="$(arm_args "$arm")"; then
        echo "[$(date '+%F %T')] SKIP unknown arm '$arm'"; continue
    fi
    export EXTRA_ARGS CFG_SEED="$seed" RUN_NAME="${RUN_PREFIX:?}-${arm}-s${seed}"
    log_file="logs/$RUN_GROUP/$RUN_NAME/log/job-${SLURM_JOB_ID}.log"
    mkdir -p "$(dirname "$log_file")"
    echo "[$(date '+%F %T')] START $RUN_NAME  ($EXTRA_ARGS)  -> $log_file"
    timeout "${RUN_TIMEOUT:-6h}" bash "$_HARNESS" > "$log_file" 2>&1
    echo "[$(date '+%F %T')] END   $RUN_NAME  exit=$?"
done
rm -f "$_HARNESS"
echo "[$(date '+%F %T')] queue done"
