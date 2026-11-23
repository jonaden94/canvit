#!/bin/bash
# Human-scanpath evaluation of one checkpoint on SLURM: runs scripts/scanpath/eval_checkpoint.sh (same arguments
# and env), from the clone it is submitted in. Lane: one full A100 on grete:shared with --qos=2h (skill grete-cluster):
# keep each job < 2 h -- big datasets as SHARD="i N" jobs, then one SHARDS=N scoring job.
#   cd <canvit clone> && mkdir -p logs/scanpath_eval/slurm
#   sbatch slurm/runs/scanpath/eval_checkpoint.sbatch <best.pt> <name> coco_search18:tp coco_search18:ta
#   for i in 0 1 2 3; do sbatch --export=ALL,SHARD="$i 4" slurm/runs/scanpath/eval_checkpoint.sbatch <best.pt> <name> cat2000; done
#   SHARDS=4 PROCESSES=6 bash scripts/scanpath/eval_checkpoint.sh <best.pt> <name> cat2000   # CPU, after the shards
# Personal paths come from .envrc (dataset roots, LOGS_DIR); account nim00018 (normative_modelling/CLAUDE.md).
# Runs the clone's working tree (no commit pinning): do not edit canvit/scanpath while such jobs run.
# Written by Claude (AI, Claude Opus 5.5), 2026-11-23.
#SBATCH --job-name=scanpath_eval
#SBATCH --account=nim00018
#SBATCH --partition=grete:shared
#SBATCH --qos=2h
#SBATCH --gres=gpu:A100:1
#SBATCH --time=01:55:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/scanpath_eval/slurm/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source ./.envrc
export TMPDIR=/mnt/lustre-grete/tmp/$USER
nvidia-smi -L
NUM_WORKERS=${NUM_WORKERS:-6} PROCESSES=${PROCESSES:-6} bash scripts/scanpath/eval_checkpoint.sh "$@"
