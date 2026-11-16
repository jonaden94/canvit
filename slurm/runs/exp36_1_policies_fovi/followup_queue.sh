#!/bin/bash
# exp36_1 follow-up (owner's go 2026-11-16): is the ~0.452 t4 plateau a DATA limit or a
# SUPERVISION limit? Training arms (step 1, the train-vs-val eval + oracle on training images,
# runs separately: claude_dev/policy_algos/README.md):
#
#   pg-lnfeat-nobn-aug      pg-lnfeat-nobn + ADE20K train augmentation (dinov3 recipe: rescale
#                           0.5-2, 512 crop, flip, photometric); val unchanged (squish)
#   pg-lnfeat-nobn-frac50   pg-lnfeat-nobn trained on a fixed 50 % of the training images
#   pg-lnfeat-nobn-frac25   ... 25 % (nested in the 50 %)
#     (anchor for all three: pg-lnfeat-nobn s0-4, same flags without the change)
#   qreg-dense16-lnfeat     dense QReg, 16 extra random candidates per state (anchor: dense4)
#     (dense32 dropped: OOM on a 32 GB V100 at batch 16 -- 512 glimpses in one forward -- and
#      ~14 h/run even if chunked; dense16 measured 2.6 s/step, ~7 h/run, 22 GB peak)
#
# Subsets: logs/jon_exp36_1_policies_fovi/ade20k_subsets/{frac50,frac25} (symlink roots, README
# there). One run per V100 on a self-renewing chain (chain_queue.sh).
#
#   TRAIN_COMMIT=<policy-algos commit> bash followup_queue.sh
set -euo pipefail
: "${TRAIN_COMMIT:?}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN=/mnt/vast-nhr/projects/nib00021/jonathan/repos/canvit
Q=$MAIN/logs/jon_exp36_1_policies_fovi/chain_followup
S=$MAIN/logs/jon_exp36_1_policies_fovi/ade20k_subsets
TMP=$(mktemp -d)

_PROBE_RUN=$MAIN/logs/jon_exp34_ade20k_probe/ade20k-fovi-ti-1196k/checkpoints
cat > "$TMP/env.sh" <<EOF
# --- identical to exp36_1's submit.sh full (and pg_arch_ablation.sh) ---
export TASK=ade20k NGPU=1 TRAIN_COMMIT=$TRAIN_COMMIT FOVI_COMMIT=c399d3b
export CFG_MODEL_REPO=$MAIN/logs/jon_exp22_full_runs/exp22-fovi-teacherinit-lrdrop-1196k/checkpoints/step-155648.pt
export CFG_PROBE_REPO=$_PROBE_RUN/best.pt
export CFG_N_TIMESTEPS=5 CFG_BATCH_SIZE=16 CFG_CANVAS_GRID=32 CFG_EVAL_POLICY=policy
export CFG_RESIZE_MODE=squish CFG_LOG_EVERY=50 CFG_NUM_WORKERS=4
export ADE20K_ROOT="\${ADE20K_ROOT:-/mnt/vast-nhr/projects/nib00021/jonathan/datasets/zhoubolei--scene_parse_150/ADEChallengeData2016}"
export RUN_GROUP=jon_exp36_1_policies_fovi CFG_WANDB_PROJECT=jon_exp36_1_policies_fovi
export CFG_MAX_STEPS=9000 CFG_VAL_EVERY=1000
export EXTRA_ARGS="--preset policy_only --cfg.no-augment --cfg.foveated-scale.fixed-scale 2.0 --rl.feature-groups ln_feat"
# --- chain ---
export NWORKERS=1 RUN_BUDGET_MIN=${RUN_BUDGET_MIN:-450}   # dense16 ~7 h
EOF

PG="--rl.objective pg --rl.no-select-bn-eval --rl.policy-input-norm none"
arm_args() {
    case "$1" in
        pg-lnfeat-nobn-aug)    echo "$PG --cfg.augment" ;;
        pg-lnfeat-nobn-frac50) echo "$PG --cfg.ade20k-root $S/frac50" ;;
        pg-lnfeat-nobn-frac25) echo "$PG --cfg.ade20k-root $S/frac25" ;;
        qreg-dense16-lnfeat)   echo "--rl.objective qreg --rl.qreg-extra-candidates 16" ;;
    esac
}
# seed-major so every arm has some seeds early
for s in 0 1 2 3 4; do
    for a in pg-lnfeat-nobn-aug pg-lnfeat-nobn-frac25 pg-lnfeat-nobn-frac50 qreg-dense16-lnfeat; do
        echo "exp36_1-v100-$a-s$s $(arm_args "$a") --cfg.seed $s"
    done
done > "$TMP/spec.txt"

bash "$HERE/chain_queue.sh" init "$Q" "$TMP/env.sh" "$TMP/spec.txt"
rm -rf "$TMP"
# the jupyter QOS allows 2 submitted jobs per user; when both are taken, submit later by hand
bash "$Q/chain_queue.sh" submit "$Q" || echo "submit failed (jupyter slots full?) -- later: bash $Q/chain_queue.sh submit $Q"
