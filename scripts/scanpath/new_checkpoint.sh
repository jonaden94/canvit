#!/bin/bash
# One command for a new checkpoint: register it in $LOGS_DIR/scanpath_eval/checkpoints.json and score it on the focus
# set (COCO-Search18 TP + TA validation and official test, COCO-FreeView validation; temperature tau included, from
# 2000 training scanpaths), then render its one-page summary (scripts/scanpath/figures.py --summary).
#   cd <canvit clone> && source .envrc && mkdir -p logs/scanpath_eval/slurm
#   bash scripts/scanpath/new_checkpoint.sh <best.pt> <name> [register_checkpoint.py options: --label, --group, --notes]
# Submits (SLURM, account nim00018, skill grete-cluster):
#   3-4 GPU jobs (one A100 each, grete:shared, --qos=2h, < 2 h each; slurm/runs/scanpath/eval_checkpoint.sbatch):
#     CS18 TP + TA validation | CS18 TP + TA test | COCO-FreeView val | the search eval (search checkpoints only)
#   1 CPU job after CS18 val + FreeView succeed (standard96s:test, < 1 h): the post-hoc saccade-cost fit
#     (canvit.scanpath.posthoc.saccade_cost, helper_2) -> $OUT/saccade_cost.json
#   1 CPU job after all of them succeed (the cost job may fail) (standard96s:test, < 1 h): CS18 TP validation cut at the first target fixation
#     (CUT=1, comparable with the test split) and the summary -> $LOGS_DIR/scanpath_eval/<name>/summaries/<slug>.*
# Finished records are skipped, so re-running after a failure redoes only what is missing. The jobs run the clone's
# working tree: do not edit canvit/scanpath until they are done. DRY=1: print the sbatch commands only.
# Written by Claude (AI, Claude Opus 5.5), 2026-11-24.
set -euo pipefail
cd "$(dirname "$0")/../.."
CKPT=${1:?checkpoint}; NAME=${2:?name}; shift 2
PY=.venv-cu126/bin/python
ROOT="${SCANPATH_RESULTS_DIR:-${LOGS_DIR:?set LOGS_DIR (source .envrc)}/scanpath_eval}"
OUT="$ROOT/$NAME"; mkdir -p "$OUT" logs/scanpath_eval/slurm
run() { if [ -n "${DRY:-}" ]; then echo "DRY: $*" >&2; echo 0; else "$@" | awk '{print $NF}'; fi; }

$PY scripts/scanpath/register_checkpoint.py "$CKPT" "$NAME" "$@"
task=$($PY -c "import json,sys; print(next(e['task'] for e in json.load(open(sys.argv[1])) if e['name'] == sys.argv[2]))" \
  "$ROOT/checkpoints.json" "$NAME")

S=slurm/runs/scanpath/eval_checkpoint.sbatch
G=(--account=nim00018 --partition=grete:shared --qos=2h --gres=gpu:A100:1 --time=01:55:00 --cpus-per-task=8 --mem=64G
   --output="logs/scanpath_eval/slurm/%x-%j.out")
deps=""
if [ "$task" = search ] && [ ! -f "$OUT/search_validation.json" ]; then  # Jonathan's search eval (found within 3 fixations)
  j0=$(run sbatch --job-name="search_$NAME" "${G[@]}" --wrap="source ./.envrc && $PY -m canvit.harness.evaluate search \
--opts.ckpt $CKPT --cfg.num-workers 6 --opts.out $OUT/search_validation.json.tmp > $OUT/search_validation.log 2>&1 \
&& mv $OUT/search_validation.json.tmp $OUT/search_validation.json")
  deps=":$j0"
fi
j1=$(run sbatch --job-name="val_$NAME" $S "$CKPT" "$NAME" coco_search18:tp coco_search18:ta)
j2=$(run sbatch --job-name="test_$NAME" --export=ALL,PART=test $S "$CKPT" "$NAME" coco_search18:tp coco_search18:ta)
j3=$(run sbatch --job-name="fv_$NAME" $S "$CKPT" "$NAME" coco_freeview)
cost=""
if [ ! -f "$OUT/saccade_cost.json" ]; then  # helper_2's post-hoc saccade-cost fit (val npz only); the summary shows it
  j5=$(run sbatch --job-name="cost_$NAME" --account=nim00018 --partition=standard96s:test --time=00:59:00 \
    --cpus-per-task=16 --mem=64G --dependency="afterok:$j1:$j3" --output="logs/scanpath_eval/slurm/%x-%j.out" \
    --wrap="source ./.envrc && $PY -m canvit.scanpath.posthoc.saccade_cost $OUT --processes 6")
  cost=",afterany:$j5"  # the summary renders even if the cost fit fails (its panel then says "not run")
fi
j4=$(run sbatch --job-name="sum_$NAME" --account=nim00018 --partition=standard96s:test --time=00:59:00 \
  --cpus-per-task=16 --mem=64G --dependency="afterok:$j1:$j2:$j3$deps$cost" --output="logs/scanpath_eval/slurm/%x-%j.out" \
  --wrap="source ./.envrc && CUT=1 PROCESSES=16 bash scripts/scanpath/eval_checkpoint.sh $CKPT $NAME coco_search18:tp \
&& $PY scripts/scanpath/figures.py --out $OUT --summary $NAME --png")
echo "$NAME ($task): GPU jobs ${j0:-} $j1 $j2 $j3, saccade-cost job ${j5:-skipped}, summary job $j4"
