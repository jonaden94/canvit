#!/bin/bash
# CPU stage of the human-scanpath evaluation: scores the saved GPU-shard predictions of one checkpoint
# (scripts/scanpath/eval_checkpoint.sh with SHARDS=N) on a whole CPU node of standard96s:test (192 threads,
# 515 GB, <= 1 h, starts at once; skill grete-cluster §1b). pysaliency's per-fixation scoring runs in PROCESSES
# forked workers.
#   cd <canvit clone> && sbatch --export=ALL,SHARDS=4 slurm/runs/scanpath/score_cpu.sbatch <best.pt> <name> cat2000
# Runs the clone's working tree (no commit pinning): do not edit canvit/scanpath while such jobs run.
# Written by Claude (AI, Claude Opus 5.5), 2026-11-23.
#SBATCH --job-name=scanpath_score
#SBATCH --account=nim00018
#SBATCH --partition=standard96s:test
#SBATCH --time=00:55:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=240G
#SBATCH --output=logs/scanpath_eval/slurm/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source ./.envrc
export TMPDIR=/mnt/lustre-grete/tmp/$USER
: "${SHARDS:?set SHARDS=N (the number of GPU shards to score)}"
PROCESSES=${PROCESSES:-48} NUM_WORKERS=0 bash scripts/scanpath/eval_checkpoint.sh "$@"
