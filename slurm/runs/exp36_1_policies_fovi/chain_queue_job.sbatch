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
#SBATCH --requeue
#SBATCH --open-mode=append
#
# The job of a self-renewing queue (it requeues itself while work is left); all logic is in chain_queue.sh (`job`). Env: Q =
# the queue dir, which holds the snapshots of this file and chain_queue.sh plus env.sh.
cd "${SLURM_SUBMIT_DIR:?}"
source "$HOME/.bashrc"
set -u
: "${Q:?}"
bash "$Q/chain_queue.sh" job "$Q"
