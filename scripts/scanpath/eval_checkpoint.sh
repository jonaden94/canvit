#!/bin/bash
# Score one CanViT checkpoint on the human scanpath datasets (canvit.scanpath; design notes outside the repo:
# canvit_claude/scanpath/00-design.md). One record per dataset/split, so new checkpoints drop in:
#   $LOGS_DIR/scanpath_eval/<name>/<dataset>_<split>.{json,log,npz}   (+ _train<N>.npz, samples_<...>.npz)
#   $LOGS_DIR/scanpath_eval/baselines/centre_bias_fit_<dataset>.json  (fitted once, shared by every model)
# Usage (GPU; existing records are skipped):
#   source .envrc && bash scripts/scanpath/eval_checkpoint.sh <best.pt> <name> [dataset[:condition] ...]
#   default datasets: coco_search18:tp coco_search18:ta coco_freeview mit1003 daemons figrim cat2000
# Env: PROCESSES (CPU workers for pysaliency's scoring, default 1), NUM_WORKERS (loader, default 2),
#      TRAIN_SCANPATHS (training subset run through the model, default 2000).
# On SLURM: slurm/runs/scanpath/eval_checkpoint.sbatch.
# Written by Claude (AI, Claude Opus 5.5), 2026-11-23.
set -euo pipefail
cd "$(dirname "$0")/../.."
CKPT=${1:?checkpoint}; NAME=${2:?name}; shift 2
SETS=("$@"); [ ${#SETS[@]} -gt 0 ] || SETS=(coco_search18:tp coco_search18:ta coco_freeview mit1003 daemons figrim cat2000)
ROOT="${SCANPATH_RESULTS_DIR:-${LOGS_DIR:?set LOGS_DIR (source .envrc) or SCANPATH_RESULTS_DIR}/scanpath_eval}"
OUT="$ROOT/$NAME"; mkdir -p "$OUT" "$ROOT/baselines"
PY=.venv-cu126/bin/python
N=${TRAIN_SCANPATHS:-2000}
for s in "${SETS[@]}"; do
  ds=${s%%:*}; cond=both; [ "$ds" != "$s" ] && cond=${s#*:}
  split=$($PY -c "from canvit.scanpath.data import eval_part; print(eval_part('$ds'))")
  has_train=$($PY -c "from canvit.scanpath.data import DATASETS; print(int('train' in DATASETS['$ds']))")
  [ "$cond" != both ] && split="${cond}_$split"
  stem="${ds}_${split}"
  [ -f "$OUT/$stem.json" ] && { echo "skip $OUT/$stem.json"; continue; }
  train=(--cfg.train-scanpaths "$N")
  [ "$has_train" = 1 ] && train+=(--cfg.save-train-predictions "$OUT/${stem}_train$N.npz")
  echo "$(date '+%F %H:%M') $stem"
  $PY -m canvit.harness.evaluate scanpath --cfg.dataset "$ds" --cfg.condition "$cond" \
    --cfg.model canvit_policy --cfg.ckpt "$CKPT" --cfg.centre-bias-fit "$ROOT/baselines/centre_bias_fit_$ds.json" \
    --cfg.save-predictions "$OUT/$stem.npz" --cfg.save-samples "$OUT/samples_$stem.npz" "${train[@]}" \
    --cfg.processes "${PROCESSES:-1}" --cfg.num-workers "${NUM_WORKERS:-2}" --out "$OUT/$stem.json.tmp" \
    > "$OUT/$stem.log" 2>&1 && mv "$OUT/$stem.json.tmp" "$OUT/$stem.json"
done
echo "$(date '+%F %H:%M') done"
