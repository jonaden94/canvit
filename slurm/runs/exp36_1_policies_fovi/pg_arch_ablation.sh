#!/bin/bash
# exp36_1 scorer-ARCHITECTURE ablation on pg-lnfeat (owner's request 2026-11-13). Base = the
# pg-lnfeat arm exactly (on-policy PG, --rl.no-select-bn-eval, canvas-only ln_feat input); each
# arm changes ONE thing about the ViewpointScorer:
#
#   pg-lnfeat-lev4   U-Net pools 32 -> 4x4 (levels 4) instead of -> 1x1 (levels 6)    3.63M params
#   pg-lnfeat-lev3   U-Net pools 32 -> 8x8 (levels 3); 7x7 kernels already ~global     2.76M
#   pg-lnfeat-local  no U-Net: each score reads only its own canvas cell               0.18M
#   pg-lnfeat-nobn   no input BatchNorm (ln_feat is LayerNormed per token already),
#                    so the scorer has no BatchNorm and no train/eval difference      5.37M
#   (pg-lnfeat itself: levels 6 + BatchNorm, 5.37M -- already run, seeds 0-4)
#
# 5 seeds each, one run at a time per V100 (pg-lnfeat alone reserves ~19 GB; two would risk
# the OOM that hit the AC arms), on a self-renewing chain of 48 h V100 jobs (chain_queue.sh: one job that
# requeues itself while runs are left).
#
#   TRAIN_COMMIT=<policy-algos commit> bash pg_arch_ablation.sh
set -euo pipefail
: "${TRAIN_COMMIT:?}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN=/mnt/vast-nhr/projects/nib00021/jonathan/repos/canvit
Q=$MAIN/logs/jon_exp36_1_policies_fovi/chain_pg_arch
TMP=$(mktemp -d)

_PROBE_RUN=$MAIN/logs/jon_exp34_ade20k_probe/ade20k-fovi-ti-1196k/checkpoints
cat > "$TMP/env.sh" <<EOF
# --- identical to exp36_1's submit.sh full (and pg_lnfeat_in_job.sh) ---
export TASK=ade20k NGPU=1 TRAIN_COMMIT=$TRAIN_COMMIT FOVI_COMMIT=c399d3b
export CFG_MODEL_REPO=$MAIN/logs/jon_exp22_full_runs/exp22-fovi-teacherinit-lrdrop-1196k/checkpoints/step-155648.pt
export CFG_PROBE_REPO=$_PROBE_RUN/best.pt
export CFG_N_TIMESTEPS=5 CFG_BATCH_SIZE=16 CFG_CANVAS_GRID=32 CFG_EVAL_POLICY=policy
export CFG_RESIZE_MODE=squish CFG_LOG_EVERY=50 CFG_NUM_WORKERS=4
export ADE20K_ROOT="\${ADE20K_ROOT:-/mnt/vast-nhr/projects/nib00021/jonathan/datasets/zhoubolei--scene_parse_150/ADEChallengeData2016}"
export RUN_GROUP=jon_exp36_1_policies_fovi CFG_WANDB_PROJECT=jon_exp36_1_policies_fovi
export CFG_MAX_STEPS=9000 CFG_VAL_EVERY=1000
export EXTRA_ARGS="--preset policy_only --cfg.no-augment --cfg.foveated-scale.fixed-scale 2.0 --rl.objective pg --rl.no-select-bn-eval --rl.feature-groups ln_feat"
# --- chain ---
export NWORKERS=1 RUN_BUDGET_MIN=120   # pg-lnfeat took 79 min solo
EOF

ARMS=(lev4 lev3 local nobn)
arm_args() {
    case "$1" in
        lev4)  echo "--rl.policy-levels 4" ;;
        lev3)  echo "--rl.policy-levels 3" ;;
        local) echo "--rl.policy-readout local" ;;
        nobn)  echo "--rl.policy-input-norm none" ;;
    esac
}
k=0
for s in 0 1 2 3 4; do   # seed-major, rotating arm order
    for i in 0 1 2 3; do
        a=${ARMS[$(( (i + k) % 4 ))]}
        echo "exp36_1-v100-pg-lnfeat-$a-s$s $(arm_args "$a") --cfg.seed $s"
    done
    k=$((k + 1))
done > "$TMP/spec.txt"

bash "$HERE/chain_queue.sh" init "$Q" "$TMP/env.sh" "$TMP/spec.txt"
rm -rf "$TMP"
# the jupyter QOS allows 2 submitted jobs per user; when both are taken, submit later by hand
bash "$Q/chain_queue.sh" submit "$Q" || echo "submit failed (jupyter slots full?) -- later: bash $Q/chain_queue.sh submit $Q"
