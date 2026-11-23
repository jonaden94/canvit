#!/bin/bash
#SBATCH -p jupyter
#SBATCH -A nib00021
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=V100:1
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#
# Held-out (A/B) oracle test on all 2000 val images, then SLEEP until the time limit: the owner
# uses the remaining allocation as an interactive session (srun --jobid=<id> --overlap --pty bash).
# Runs, in order: ce + random at --split-block 64, then at 32 (~3.5 h per ce run on a V100),
# then `merge` into the log. A failing step is logged and skipped, never ends the job.
# Submitted by submit_split.sh, which copies oracle.py into $OUT at submit time.
cd "${SLURM_SUBMIT_DIR:?}"
source "$HOME/.bashrc"
set -u
source slurm/env.sh

: "${OUT:?}"
TRAIN_COMMIT=f3d6e6b   # exp36_1's canvit pin
FOVI_COMMIT=c399d3b    # exp36/exp36_1's fovi pin: the geometry the exp22 backbone trained on
code="${TMPDIR:?}/oracle_src"
mkdir -p "$code/canvit" "$code/fovi"
git archive "$TRAIN_COMMIT" | tar -x -C "$code/canvit"
git -C ../fovi archive "$FOVI_COMMIT" | tar -x -C "$code/fovi"
echo "[$(date '+%F %T')] $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader)  canvit $TRAIN_COMMIT  fovi $FOVI_COMMIT"

run() {
    env PYTHONSAFEPATH=1 PYTHONPATH="$code/canvit:$code/fovi" \
        .venv-cu126/bin/python "$OUT/oracle.py" run --out "$OUT" "$@" \
        || echo "[$(date '+%F %T')] FAILED: $*"
}
for blk in 64 32; do
    run --criterion ce --split-block "$blk"
    run --criterion random --split-block "$blk" --log-every 200
done
.venv-cu126/bin/python "$OUT/oracle.py" merge "$OUT" | tee "$OUT/RESULTS.txt"
echo "[$(date '+%F %T')] experiments done; sleeping until the time limit (interactive use)"
touch "$OUT/DONE"
sleep infinity
