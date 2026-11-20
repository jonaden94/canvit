#!/bin/bash
# Reference lines for the scanpath evaluation (claude_dev/scanpath/00-design.md §4.1): pysaliency's uniform model
# and centre bias on COCO-Search18 split1 validation, target present and absent. The centre bias's bandwidth/eps is
# fitted ONCE on the training split (written to $OUT/centre_bias_fit.json) and reused by every run.
#   source .envrc && bash scripts/scanpath/baselines.sh            # ~1 h on one CPU
# Written by Claude (AI, Claude Opus 5.5), 2026-11-20.
set -euo pipefail
cd "$(dirname "$0")/../.."
# Results: $SCANPATH_RESULTS_DIR, else $LOGS_DIR/scanpath_eval (logs/ is gitignored, as for every run).
OUT="${SCANPATH_RESULTS_DIR:-${LOGS_DIR:?set LOGS_DIR (source .envrc) or SCANPATH_RESULTS_DIR}/scanpath_eval}"
PY=.venv-cu126/bin/python
mkdir -p "$OUT"

if [ ! -f "$OUT/centre_bias_fit.json" ]; then
  $PY - "$OUT/centre_bias_fit.json" <<'EOF'
import json, logging, sys
import numpy as np
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
from canvit.scanpath.evaluate import ScanpathConfig, load_coco_search18
from canvit.scanpath.models import fit_centre_bias
st, ft, _, _ = load_coco_search18(ScanpathConfig())
fit = fit_centre_bias(st, ft, n_images=60, bandwidths=(0.01, 0.016, 0.025, 0.04, 0.063),
                      epss=(1e-7, 1e-4))
json.dump(fit, open(sys.argv[1], "w"), indent=1)
print("centre bias fit:", fit)
EOF
fi
BW=$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['bandwidth'])" "$OUT/centre_bias_fit.json")
EPS=$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['eps'])" "$OUT/centre_bias_fit.json")

for cond in tp ta; do
  for model in uniform centre_bias; do
    f="$OUT/${model}_${cond}_validation.json"
    [ -f "$f" ] && { echo "skip $f"; continue; }
    echo "$(date '+%H:%M') $model $cond"
    $PY -m canvit.harness.evaluate scanpath --cfg.model "$model" --cfg.condition "$cond" \
      --cfg.centre-bias-bandwidth "$BW" --cfg.centre-bias-eps "$EPS" --cfg.device cpu --cfg.num-workers 0 \
      --out "$f" > "${f%.json}.log" 2>&1
  done
done
echo "$(date '+%H:%M') done"
