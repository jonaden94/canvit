#!/bin/bash
#SBATCH -p jupyter
#SBATCH -A nib00021
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=V100:1
#SBATCH --mem=96G
#SBATCH -C inet
#SBATCH --time=2-00:00:00
#
# policy-algos: one V100 job whose batch script runs TWO algos_queue.sh workers on its own GPU
# (measured: two policy_only runs on one V100 give ~1.5x the throughput of one; peak reserved
# memory 15.2 GB per run). The workers live as long as this job; the job ends when the queue
# is empty. Env: Q (queue dir holding the algos_queue.sh snapshot), TRAIN_COMMIT.
cd "${SLURM_SUBMIT_DIR:?}"
source "$HOME/.bashrc"
set -u
: "${Q:?}" "${TRAIN_COMMIT:?}"
echo "[$(date '+%F %T')] $(hostname) $(nvidia-smi -L)  queue $Q  commit $TRAIN_COMMIT"
for w in 1 2; do
    WORKER_CPUS=8 TRAIN_COMMIT="$TRAIN_COMMIT" bash "$Q/algos_queue.sh" work "$SLURM_JOB_ID" "$Q" \
        > "$Q/worker-$SLURM_JOB_ID-$w.log" 2>&1 &
    [ "$w" = 1 ] && sleep 900   # desynchronize the two workers' evals (the memory peaks)
done
wait
echo "[$(date '+%F %T')] queue drained"
