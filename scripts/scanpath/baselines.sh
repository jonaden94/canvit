#!/bin/bash
# Reference lines for the scanpath evaluation (canvit_claude/scanpath/00-design.md §4.1, outside the repo): pysaliency's
# uniform model and centre bias on every dataset's evaluated split, scored exactly as the models are.
#   $LOGS_DIR/scanpath_eval/baselines/{uniform,centre_bias}_<dataset>_<split>.{json,log}
# The centre bias's bandwidth/eps is fitted once per dataset (baselines/centre_bias_fit_<dataset>.json, shared with
# every model; fitted here if missing).
#   source .envrc && PROCESSES=6 bash scripts/scanpath/baselines.sh [dataset[:condition] ...]   # CPU only
# Written by Claude (AI, Claude Opus 5.5), 2026-11-20; per-dataset layout 2026-11-23.
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT="${SCANPATH_RESULTS_DIR:-${LOGS_DIR:?set LOGS_DIR (source .envrc) or SCANPATH_RESULTS_DIR}/scanpath_eval}"
OUT="$ROOT/baselines"; mkdir -p "$OUT"
PY=.venv-cu126/bin/python
SETS=("$@"); [ ${#SETS[@]} -gt 0 ] || SETS=(coco_search18:tp coco_search18:ta coco_freeview mit1003 daemons figrim cat2000)
for s in "${SETS[@]}"; do
  ds=${s%%:*}; cond=both; [ "$ds" != "$s" ] && cond=${s#*:}
  split=$($PY -c "from canvit.scanpath.data import eval_part; print(eval_part('$ds'))")
  [ "$cond" != both ] && split="${cond}_$split"
  for model in uniform centre_bias; do
    f="$OUT/${model}_${ds}_${split}.json"
    [ -f "$f" ] && { echo "skip $f"; continue; }
    echo "$(date '+%F %H:%M') $model $ds $split"
    $PY -m canvit.harness.evaluate scanpath --cfg.dataset "$ds" --cfg.condition "$cond" --cfg.model "$model" \
      --cfg.centre-bias-fit "$OUT/centre_bias_fit_$ds.json" --cfg.device cpu --cfg.num-workers 0 \
      --cfg.processes "${PROCESSES:-1}" --out "$f.tmp" > "${f%.json}.log" 2>&1 && mv "$f.tmp" "$f"
  done
done
echo "$(date '+%F %H:%M') done"
