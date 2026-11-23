#!/bin/bash
# One retina of a multi-retina search checkpoint: Jonathan's search eval (harness.evaluate search, CS18 val) and the
# human-scanpath eval on COCO-Search18 TP + TA val (scripts/scanpath/eval_checkpoint.sh, RETINA=<retina>).
# Head and policy were trained through ONE retina (the run's --cfg.retina): every other retina is out of
# distribution for them (a proper test needs a policy per retina).
#   cd <canvit clone> && for r in <retina names>; do
#     sbatch --job-name=retina_$r slurm/runs/scanpath/retina_sweep.sbatch <best.pt> <name> $r; done
# Records: $LOGS_DIR/scanpath_eval/<name>/retina/<retina>/{search_validation,coco_search18_{tp,ta}_validation}.json
# Lane: one A100 on grete:shared with --qos=2h (skill grete-cluster). Runs the clone's working tree.
# Written by Claude (AI, Claude Opus 5.5), 2026-11-23.
#SBATCH --job-name=retina_sweep
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
CKPT=${1:?checkpoint}; NAME=${2:?name}; RET=${3:?retina}
OUT="$LOGS_DIR/scanpath_eval/$NAME/retina/$RET"; mkdir -p "$OUT"
nvidia-smi -L; git rev-parse --short HEAD
if [ ! -f "$OUT/search_validation.json" ]; then
  .venv-cu126/bin/python -m canvit.harness.evaluate search --opts.ckpt "$CKPT" --cfg.retina "$RET" \
    --cfg.num-workers 6 --opts.out "$OUT/search_validation.json.tmp" > "$OUT/search_validation.log" 2>&1
  mv "$OUT/search_validation.json.tmp" "$OUT/search_validation.json"
fi
RETINA=$RET NUM_WORKERS=6 PROCESSES=6 bash scripts/scanpath/eval_checkpoint.sh "$CKPT" "$NAME/retina/$RET" \
  coco_search18:tp coco_search18:ta
