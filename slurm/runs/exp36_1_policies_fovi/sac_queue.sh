#!/bin/bash
# exp36_1 SAC-Discrete arms (owner's request 2026-11-17; Christodoulou 2019, arXiv 1910.07207):
#
#   sac-lnfeat-nobn        --rl.objective sac, the paper's target entropy 0.98*log|A| (5.43 nats of 5.55)
#   sac-ent1-lnfeat-nobn   ... target entropy 1.0 nat = pg-lnfeat's own entropy target
#
# Comparable to pg-lnfeat(-nobn): the actor IS pg-lnfeat's scorer (5.37M, canvas-only ln_feat
# input, no input BatchNorm -- shown not to matter), sampled on-policy and deployed by argmax,
# same data, steps, LR schedule, eval and seeds 0-4. SAC adds two soft-Q critics of the actor's
# size + Polyak targets (like A2C's critic). No replay buffer: one update per rollout, as PG.
# Measured: 1.3 s/step (~3.5 h/run), 21.5 GB peak -> one run per V100.
#
#   TRAIN_COMMIT=<policy-algos commit> bash sac_queue.sh
set -euo pipefail
: "${TRAIN_COMMIT:?}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN=/mnt/vast-nhr/projects/nib00021/jonathan/repos/canvit
Q=$MAIN/logs/jon_exp36_1_policies_fovi/chain_sac
TMP=$(mktemp -d)

_PROBE_RUN=$MAIN/logs/jon_exp34_ade20k_probe/ade20k-fovi-ti-1196k/checkpoints
cat > "$TMP/env.sh" <<EOF
# --- identical to exp36_1's submit.sh full (and followup_queue.sh) ---
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
export NWORKERS=1 RUN_BUDGET_MIN=${RUN_BUDGET_MIN:-240}   # SAC ~3.5 h
EOF

SAC="--rl.objective sac --rl.no-select-bn-eval --rl.policy-input-norm none"
arm_args() {
    case "$1" in
        sac-lnfeat-nobn)      echo "$SAC" ;;
        sac-ent1-lnfeat-nobn) echo "$SAC --rl.sac-target-entropy 1.0" ;;
    esac
}
for s in 0 1 2 3 4; do  # seed-major
    for a in sac-ent1-lnfeat-nobn sac-lnfeat-nobn; do
        echo "exp36_1-v100-$a-s$s $(arm_args "$a") --cfg.seed $s"
    done
done > "$TMP/spec.txt"

bash "$HERE/chain_queue.sh" init "$Q" "$TMP/env.sh" "$TMP/spec.txt"
rm -rf "$TMP"
# the jupyter QOS allows 2 submitted jobs per user; when both are taken, submit later by hand
bash "$Q/chain_queue.sh" submit "$Q" || echo "submit failed (jupyter slots full?) -- later: bash $Q/chain_queue.sh submit $Q"
