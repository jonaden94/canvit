#!/bin/bash
# exp37 DDP — exp32-fovi-teacherinit-s1..s3 re-run on 4 GPUs, at two peak LRs.
#
# WHAT CHANGES vs exp32-fovi-teacherinit-seed.sh. Only the GPU count (4, so global batch
# 256 at 64 per GPU) and the peak LR. Everything else is the same, including the three
# commit pins, so the code is the exact code exp32 ran.
#
# LR. Two arms on exp32's 4e-4, for a 4x larger global batch:
#   LR_MULT=2 -> 8e-4    square-root scaling (4e-4 * sqrt(4))
#   LR_MULT=3 -> 1.2e-3  more aggressive, between square-root and linear (4x = 1.6e-3,
#                        what exp17's 4-GPU run used)
# The warmup stays 100k STEPS (warmup_constant), as in exp32.
#
# SEEDS. torch.manual_seed(seed + rank) runs before build_model and rank 0's parameters are
# then broadcast to every rank, so seed s here starts from the SAME init as exp32 seed s.
# The shard schedule depends on seed, job index, rank and world size only, so the two LR
# arms at one seed also see the same data: compare the arms per seed (paired).
#
# LENGTH. 24 x 8192 = 196608 steps. The exp32 seeds ran 25 jobs (204800) but keep
# step-196608.pt and log val every 8192, so they can be compared at 196608.
#
# READING IT. At a fixed step these runs have seen 4x exp32's images. Compare to exp32 at
# equal steps (~equal wall time) AND at equal images.
#
# for m in 2 3; do for s in 1 2 3; do LR_MULT=$m SEED=$s bash slurm/runs/exp37_pretrain_ddp/exp37-fovi-teacherinit-ddp.sh; done; done
set -euo pipefail

# === ESSENTIALS ===
RUN_GROUP=jon_exp37_pretrain_ddp
SEED="${SEED:?set SEED=1..3}"
LR_MULT="${LR_MULT:?set LR_MULT=2 or 3 (multiple of the exp32 LR 4e-4)}"
case "$LR_MULT" in
    2) CFG_PEAK_LR=0.0008 ;;        # 4e-4 * 2 = square-root scaling for 4 GPUs
    3) CFG_PEAK_LR=0.0012 ;;        # 4e-4 * 3
    *) echo "REFUSING: LR_MULT must be 2 or 3, got '$LR_MULT'"; exit 1 ;;
esac
NGPU=4
RUN_NAME=exp37-fovi-teacherinit-4gpu-lr${LR_MULT}x-s$SEED
ARRAY="${ARRAY:-0-23%1}"      # 24 x 8192 = 196608 steps
TIME=0-02:00:00
MEM=512G                      # 128G per GPU, as exp32's 1-GPU runs
TASK=distill

# === config: identical to exp32-fovi-teacherinit-seed.sh except NGPU and CFG_PEAK_LR ===
CFG_WANDB_PROJECT=jon_exp37_pretrain_ddp
CFG_SEED=$SEED
CFG_BATCH_SIZE_PER_GPU=64
CFG_STEPS_PER_JOB=8192
CFG_VAL_EVERY=8192            # validate once per job (= steps_per_job), as exp32
CFG_LOG_EVERY=512
CFG_NUM_WORKERS=4
EXTRA_ARGS="--cfg.model.patcher-name foveated --cfg.model.foveated-patcher.fov 35 --cfg.model.foveated-patcher.resolution 64 --cfg.model.foveated-patcher.cmf-a 0.5 --cfg.model.foveated-patcher.cart-patch-size 5 --cfg.model.foveated-patcher.arch-flag doubleres --cfg.model.foveated-patcher.conditioning.mode film --cfg.model.foveated-patcher.conditioning.film.fourier.num-features 256 --cfg.model.foveated-patcher.conditioning.film.fourier.sigma 4 --cfg.foveated-scale.fixed-scale 2.0 --cfg.init-backbone-from-teacher"
# =================

# Same pins as exp32 (pre-merge, so PYTORCH_COMMIT is load-bearing here).
TRAIN_COMMIT=716051a
PYTORCH_COMMIT=d616b7b
FOVI_COMMIT=c399d3b

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
