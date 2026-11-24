#!/bin/bash
# Score one CanViT checkpoint on the human scanpath datasets (canvit.scanpath; design notes outside the repo:
# canvit_claude/scanpath/00-design.md). One record per dataset/split, so new checkpoints drop in:
#   $LOGS_DIR/scanpath_eval/<name>/<dataset>_<split>.{json,log,npz}   (+ _train<N>.npz, samples_<...>.npz)
#   $LOGS_DIR/scanpath_eval/baselines/centre_bias_fit_<dataset>.json  (fitted once, shared by every model)
# Usage (GPU; existing records are skipped):
#   source .envrc && bash scripts/scanpath/eval_checkpoint.sh <best.pt> <name> [dataset[:condition] ...]
#   default datasets: coco_search18:tp coco_search18:ta coco_freeview mit1003 daemons figrim cat2000
# Big datasets in two stages (each GPU job < 2 h):
#   SHARD="i N" ...  GPU: predictions of block i of N only -> <stem>.shard<i>of<N>.npz (shard 0 also the
#                    training-subset predictions and the samples); one call per i.
#   SHARDS=N ...     CPU: score the N shard files (no GPU), with PROCESSES workers -> <stem>.json.
# Env: PROCESSES (CPU workers for pysaliency's scoring, default 1), NUM_WORKERS (loader, default 2), BATCH_SIZE (16),
#      TRAIN_SCANPATHS (training subset run through the model, default 2000),
#      DISPLAY_FRAME: unset = the STANDARD, the display shape of the checkpoint's training data (for the search
#      checkpoints 1680x1050: COCO-Search18 content only ever occupied the scene rows of that box; owner decision
#      2026-11-23); "scene" = the comparison rows (stimulus fills the scene), written as <stem>_scene.*
#      PART=test: COCO-Search18's official test split instead of data.eval_part (a second evaluation).
#      CUT=1: COCO-Search18 TP only, CPU: score the saved <stem>.npz with every scanpath cut at its first target
#      fixation (--cfg.cut-at-target; the test split's format) -> <stem>_cut.json. Run after the uncut record.
#      RETINA: a multi-retina backbone's retina other than the trained one (--cfg.retina; head and policy were
#      trained through one retina, so others are out of distribution for them); give it its own <name>.
# On SLURM: slurm/runs/scanpath/eval_checkpoint.sbatch (same arguments).
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
  split=${PART:-$($PY -c "from canvit.scanpath.data import eval_part; print(eval_part('$ds'))")}
  has_train=$($PY -c "from canvit.scanpath.data import DATASETS; print(int('train' in DATASETS['$ds']))")
  split_part=$split; [ "$cond" != both ] && split="${cond}_$split"
  stem="${ds}_${split}"
  frame=()
  if [ -n "${DISPLAY_FRAME:-}" ]; then
    frame=(--cfg.display-frame "$DISPLAY_FRAME"); [ "$DISPLAY_FRAME" = scene ] && stem="${stem}_scene"
  fi
  base=$stem; [ -n "${CUT:-}" ] && stem="${stem}_cut"
  retina=(); [ -n "${RETINA:-}" ] && retina=(--cfg.retina "$RETINA")
  train=(--cfg.train-scanpaths "$N")
  [ "$has_train" = 1 ] && train+=(--cfg.save-train-predictions "$OUT/${base}_train$N.npz")
  common=(--cfg.dataset "$ds" --cfg.condition "$cond" --cfg.part "$split_part" --cfg.model canvit_policy --cfg.ckpt "$CKPT" "${frame[@]}" "${retina[@]}"
          --cfg.num-workers "${NUM_WORKERS:-2}" --cfg.batch-size "${BATCH_SIZE:-16}" "${train[@]}")
  if [ -n "${SHARD:-}" ]; then  # GPU stage of one block
    read -r i n <<< "$SHARD"; tag="$stem.shard${i}of$n"
    [ -f "$OUT/$tag.npz" ] && { echo "skip $OUT/$tag.npz"; continue; }
    echo "$(date '+%F %H:%M') $tag"
    $PY -m canvit.harness.evaluate scanpath "${common[@]}" --cfg.shard "$i" "$n" \
      --cfg.save-predictions "$OUT/$tag.npz.tmp.npz" --cfg.save-samples "$OUT/samples_$stem.npz" \
      --out "$OUT/$tag.json" > "$OUT/$tag.log" 2>&1 && mv "$OUT/$tag.npz.tmp.npz" "$OUT/$tag.npz"
    continue
  fi
  [ -f "$OUT/$stem.json" ] && { echo "skip $OUT/$stem.json"; continue; }
  echo "$(date '+%F %H:%M') $stem"
  if [ -n "${CUT:-}" ]; then  # CPU: the uncut predictions, scored on the cut scanpaths
    run=(--cfg.predictions "$OUT/$base.npz" --cfg.device cpu --cfg.cut-at-target)
  elif [ -n "${SHARDS:-}" ]; then  # CPU stage: score the shards' predictions
    preds=(); for ((i = 0; i < SHARDS; i++)); do preds+=("$OUT/$stem.shard${i}of$SHARDS.npz"); done
    run=(--cfg.predictions "${preds[@]}" --cfg.device cpu)
  else
    run=(--cfg.save-predictions "$OUT/$stem.npz" --cfg.save-samples "$OUT/samples_$stem.npz")
  fi
  $PY -m canvit.harness.evaluate scanpath "${common[@]}" "${run[@]}" \
    --cfg.centre-bias-fit "$ROOT/baselines/centre_bias_fit_$ds.json" --cfg.processes "${PROCESSES:-1}" \
    --out "$OUT/$stem.json.tmp" > "$OUT/$stem.log" 2>&1 && mv "$OUT/$stem.json.tmp" "$OUT/$stem.json"
done
echo "$(date '+%F %H:%M') done"
