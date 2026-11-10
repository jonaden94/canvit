#!/bin/bash
# exp32 SEED SPREAD — how much does a foveated teacher-init pretrain move between seeds?
#
# WHY. exp32-fovi-teacherinit (seed 0) runs ~0.019 below exp22-fovi-teacherinit in
# val/scene_cos_raw at matched steps under a matched eval convention. Three questions were
# ruled out for that gap (eval scale, canvit_pytorch pin, optimizer grouping) and one
# candidate survives with weak support (normalizer_shards, accepted diff #1 in
# exp32-fovi-teacherinit.sh). Before chasing a cause we need the null: the seed-to-seed
# spread of THIS config, which nobody has measured.
#
# The old loop never called torch.manual_seed, so each of its runs drew a different init --
# its three foveated samples span 0.022 in scene_cos_raw at step 65536. The harness DOES
# seed (harness/run.py:204, before build_model), so its runs at one seed agree to ~0.005.
# That 0.005 is nondeterminism, NOT init spread, and using it as the noise floor is what
# made the gap look systematic. This run group measures the real thing.
#
# WHAT CFG_SEED MOVES. torch.manual_seed(seed + rank) before build_model (random init of
# every non-teacher-initialised parameter) AND the webdataset shard schedule. It does NOT
# move the normalizer statistics: since fd83930 the stats pool the first n SORTED shards,
# so the distillation targets are seed-independent and identical to exp32-fovi-teacherinit's.
#
# SEEDS 1..5. Seed 0 is exp32-fovi-teacherinit itself -- already running, so it is a free
# sixth sample. Do not re-run it here; it would collide with that run's directory.
#
# LENGTH. 25 x 8192 = 204800 steps, not exp32's full 1130496 phase A. The gap is fully open
# by step ~40k (-0.025) and narrows after, so 200k spans the whole interesting region and
# overlaps every other foveated run we have.
#
# READING IT. Compare on train/full/scene_cos_raw, NOT the val curve: exp22's val is split
# at step 139264 by the d2f7b50 eval-scale fix and reads ~0.029 low before it. Reference
# points at step 40960 -- old loop: 0.8391 / 0.8474 / 0.8662 (n=3, spread 0.027);
# harness seed 0: 0.8286 (exp32), 0.8321 (exp28), 0.8412 (exp26, 1-shard normalizer).
# If seeds 1-5 spread over ~0.02 and straddle the old-loop values, the gap is the seed
# lottery and there is nothing to fix.
#
# 5 seeds: for s in 1 2 3 4 5; do SEED=$s bash slurm/runs/exp32_pretrain_lrdrop/exp32-fovi-teacherinit-seed.sh; done
set -euo pipefail

# === ESSENTIALS ===
RUN_GROUP=caglar_exp32_pretrain_lrdrop
SEED="${SEED:?set SEED=1..5 (seed 0 is exp32-fovi-teacherinit itself)}"
RUN_NAME=exp32-fovi-teacherinit-s$SEED
ARRAY=0-24%1                  # 25 x 8192 = 204800 steps
TIME=0-02:00:00
MEM=128G
NGPU=1
TASK=distill

# === config: byte-identical to exp32-fovi-teacherinit.sh except CFG_SEED ===
CFG_WANDB_PROJECT=caglar_exp32_pretrain_lrdrop
CFG_SEED=$SEED
CFG_PEAK_LR=0.0004
CFG_BATCH_SIZE_PER_GPU=64
CFG_STEPS_PER_JOB=8192
CFG_VAL_EVERY=8192            # validate once per job (= steps_per_job), as exp22
CFG_LOG_EVERY=512
CFG_NUM_WORKERS=4
EXTRA_ARGS="--cfg.model.patcher-name foveated --cfg.model.foveated-patcher.fov 35 --cfg.model.foveated-patcher.resolution 64 --cfg.model.foveated-patcher.cmf-a 0.5 --cfg.model.foveated-patcher.cart-patch-size 5 --cfg.model.foveated-patcher.arch-flag doubleres --cfg.model.foveated-patcher.conditioning.mode film --cfg.model.foveated-patcher.conditioning.film.fourier.num-features 256 --cfg.model.foveated-patcher.conditioning.film.fourier.sigma 4 --cfg.foveated-scale.fixed-scale 2.0 --cfg.init-backbone-from-teacher"
# =================

TRAIN_COMMIT=716051a
PYTORCH_COMMIT=d616b7b
FOVI_COMMIT=c399d3b

if [ "$SEED" = "0" ]; then
    echo "REFUSING: seed 0 is exp32-fovi-teacherinit, which is already running."
    echo "  Use it as the sixth sample; re-running it here would clobber that run dir."
    exit 1
fi

# Repo root, derived from this script's own location (slurm/runs/<group>/<run>.sh),
# so the run submits from YOUR clone rather than one hardcoded checkout.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
mkdir -p "logs/$RUN_GROUP/$RUN_NAME/log"
export TASK RUN_GROUP RUN_NAME NGPU EXTRA_ARGS TRAIN_COMMIT PYTORCH_COMMIT FOVI_COMMIT
for v in $(compgen -v); do [[ "$v" == CFG_* || "$v" == OPT_* ]] && export "$v"; done

sbatch \
    --gpus-per-node=A100:$NGPU \
    --ntasks-per-node=$NGPU \
    --mem=$MEM \
    --time=$TIME \
    --array="$ARRAY" \
    --output="logs/$RUN_GROUP/$RUN_NAME/log/job-%A_%a.log" \
    --error="logs/$RUN_GROUP/$RUN_NAME/log/job-%A_%a.log" \
    --export=ALL \
    slurm/harness_train.sbatch
