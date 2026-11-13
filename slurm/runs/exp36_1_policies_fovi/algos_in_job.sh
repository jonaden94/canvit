#!/bin/bash
# policy-algos (2026-11-13): new policy-learning algorithms on exp36_1's exact setup, canvas-only
# input (`ln_feat`), logged next to the exp36_1 arms (same W&B project / run group).
#
#   qreg-lnfeat         QReg, the recipe -- the baseline the QReg-family arms below need
#   a2c-lnfeat          actor-critic: GAE(0.7, 1.0) over the glimpses, learned V(s) (rl.AC)
#   ppo-lnfeat          the same + PPO-clip 0.2, 2 passes per rollout (AdaptiveNN defaults)
#   qtd-lnfeat          multi-step Q-learning: double-Q bootstrapped target, Polyak target net
#   qreg-dense4-lnfeat  QReg + 4 extra random cells glimpsed and regressed per state
#
# Runs INSIDE an already-running V100 allocation as an overlap step (like
# pg_lnfeat_in_job.sh, whose two env fixes it shares), one run after another via the main
# clone's slurm/harness_sweep.sbatch; seed-major with a rotating arm order, so a walltime
# cut loses about one run per arm. Start it detached on a login node:
#
#   TRAIN_COMMIT=<policy-algos commit> setsid nohup bash .../algos_in_job.sh JOBID "SEEDS" [WAIT] \
#       > LOG 2>&1 < /dev/null &
#
# WAIT (optional): a file that must exist, or "log:<file>:<text>" -- a log that must contain <text>.
set -euo pipefail
JOB="${1:?usage: algos_in_job.sh JOBID \"SEEDS\" [WAIT]}"
SEEDS="${2:?seeds, e.g. \"0 1 2\"}"
WAIT="${3:-}"
: "${TRAIN_COMMIT:?TRAIN_COMMIT must be the policy-algos commit to pin}"
MAIN=/mnt/vast-nhr/projects/nib00021/jonathan/repos/canvit   # the job's submit dir: logs/, .git, harness_sweep
cd "$MAIN"
log() { echo "[$(date '+%F %T')] $*"; }
git cat-file -e "$TRAIN_COMMIT^{commit}"

# --- identical to exp36_1's submit.sh full ---
_PROBE_RUN=$MAIN/logs/jon_exp34_ade20k_probe/ade20k-fovi-ti-1196k/checkpoints
export TASK=ade20k NGPU=1 TRAIN_COMMIT FOVI_COMMIT=c399d3b
export CFG_MODEL_REPO=$MAIN/logs/jon_exp22_full_runs/exp22-fovi-teacherinit-lrdrop-1196k/checkpoints/step-155648.pt
export CFG_PROBE_REPO=$_PROBE_RUN/best.pt
export CFG_N_TIMESTEPS=5 CFG_BATCH_SIZE=16 CFG_CANVAS_GRID=32 CFG_EVAL_POLICY=policy
export CFG_RESIZE_MODE=squish CFG_LOG_EVERY=50 CFG_NUM_WORKERS=4
export ADE20K_ROOT="${ADE20K_ROOT:-/mnt/vast-nhr/projects/nib00021/jonathan/datasets/zhoubolei--scene_parse_150/ADEChallengeData2016}"
export RUN_GROUP=jon_exp36_1_policies_fovi CFG_WANDB_PROJECT=jon_exp36_1_policies_fovi
export CFG_MAX_STEPS=9000 CFG_VAL_EVERY=1000
export EXTRA_ARGS="--preset policy_only --cfg.no-augment --cfg.foveated-scale.fixed-scale 2.0 --rl.feature-groups ln_feat"

ARMS=(qreg-lnfeat a2c-lnfeat ppo-lnfeat qtd-lnfeat qreg-dense4-lnfeat)
arm_args() {
    case "$1" in
        qreg-lnfeat)        echo "" ;;
        a2c-lnfeat)         echo "--rl.objective ac --rl.no-select-bn-eval" ;;
        ppo-lnfeat)         echo "--rl.objective ac --rl.no-select-bn-eval --rl.ppo-epochs 2" ;;
        qtd-lnfeat)         echo "--rl.objective qtd" ;;
        qreg-dense4-lnfeat) echo "--rl.qreg-extra-candidates 4" ;;
    esac
}
SPEC="" k=0
for s in $SEEDS; do
    for i in 0 1 2 3 4; do
        arm=${ARMS[$(( (i + k) % 5 ))]}
        SPEC+="exp36_1-v100-$arm-s$s $(arm_args "$arm") --cfg.seed $s"$'\n'
    done
    k=$((k + 1))
done
export SWEEP_SPEC="$SPEC"

if [ -n "$WAIT" ]; then
    log "waiting for: $WAIT (job $JOB)"
    while :; do
        [ -n "$(squeue -h -j "$JOB" -t R 2>/dev/null)" ] || { log "job $JOB not running -- abort"; exit 1; }
        case "$WAIT" in
            log:*) f="${WAIT#log:}"; txt="${f#*:}"; f="${f%%:*}"; grep -qF "$txt" "$f" 2>/dev/null && break ;;
            *) [ -e "$WAIT" ] && break ;;
        esac
        sleep 60
    done
fi
log "starting sweep in job $JOB, commit $TRAIN_COMMIT:"; printf '%s' "$SWEEP_SPEC"
export SLURM_OVERLAP=1 SLURM_NTASKS_PER_NODE=1   # see pg_lnfeat_in_job.sh
rc=0
srun --jobid="$JOB" --overlap --cpus-per-task=8 -n1 bash slurm/harness_sweep.sbatch || rc=$?
log "sweep finished, exit $rc"
