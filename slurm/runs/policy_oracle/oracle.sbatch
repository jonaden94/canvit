#!/bin/bash
#SBATCH -p jupyter
#SBATCH -A nib00021
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=V100:1
#SBATCH --mem=64G
#
# One shard of the greedy GT oracle (oracle.py), then the random-grid reference on the
# same images. Submitted by submit.sh, which sets OUT/START/END and copies oracle.py into
# $OUT at SUBMIT time -- the job runs that copy, so editing the clone cannot change it.
# The code it imports is pinned like harness_train.sbatch pins it: git archive of the
# exp36_1 commits into $TMPDIR, prepended to PYTHONPATH with PYTHONSAFEPATH=1.
cd "${SLURM_SUBMIT_DIR:?}"
source "$HOME/.bashrc"
set -eu
source slurm/env.sh

: "${OUT:?}" "${START:?}" "${END:?}"
TRAIN_COMMIT=f3d6e6b   # exp36_1's canvit pin (Caglar's branch head)
FOVI_COMMIT=c399d3b    # exp36/exp36_1's fovi pin: the geometry the exp22 backbone trained on
code="${TMPDIR:?}/oracle_src"
mkdir -p "$code/canvit" "$code/fovi"
git archive "$TRAIN_COMMIT" | tar -x -C "$code/canvit"
git -C ../fovi archive "$FOVI_COMMIT" | tar -x -C "$code/fovi"
echo "[$(date '+%F %T')] $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader)  canvit $TRAIN_COMMIT  fovi $FOVI_COMMIT  images [$START, $END)"

run() {
    env PYTHONSAFEPATH=1 PYTHONPATH="$code/canvit:$code/fovi" \
        .venv-cu126/bin/python "$OUT/oracle.py" run --out "$OUT" --start "$START" --end "$END" "$@"
}
run --criterion ce
run --criterion random --log-every 200
echo "[$(date '+%F %T')] done"
