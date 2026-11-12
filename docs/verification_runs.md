# Verification campaign: exp32–exp35

Four groups of runs that exercise, at full scale, all three training objectives plus the
viewpoint policy. Together they cover every task the trainer supports, on real data, for as
long as a production run — the evidence that the stack works end to end rather than only in
unit tests.

| group | what it trains | runs | judged on | status |
|---|---|---|---|---|
| `jon_exp32_pretrain_lrdrop` | `distill` pretraining from scratch | 4 (+5 seeds) | `val/scene_cos_raw_t9` | **complete** |
| `jon_exp33_in1k_finetune` | `in1k` full finetunes of four pretrained backbones | 4 | top-1 | **complete** |
| `jon_exp34_ade20k_probe` | `ade20k` frozen segmentation probes on the same four | 4 | CE and mIoU | **complete** |
| `jon_exp35_policy_qreg_10seed` | ADE20K viewpoint policy (Q-regression), 10 seeds | 10 | CE and mIoU | **complete** |

Each group below has the same two sections: **Setup** (what is run and how) and **Results**.

All four groups have produced their results. exp32 ran at constant LR throughout: the ×0.1
decay phase its group name refers to was never run here, so where that section compares
against `jon_exp22_full_runs` — the only campaign that completed a decay — it says so
explicitly.

## Shared procedure

Every launcher is a small self-documenting script. Submit by running it:

```bash
bash slurm/runs/<group>/<run>.sh
```

Each group below lists the exact commands for its own arms and the checkpoints they read.
Each also has a `slurm/runs/<group>/README.md` with the rest of the per-arm detail — job
ids and the traps specific to that group. This document is the overview.

### Running these as another member of project `nib00021`

Everything these runs *read* is shared and needs no copying: the source checkpoints are mode
640 under group `HPC_nib00021` with group-traversable parents, and so is every dataset root
`.envrc.grete` names. Four things are yours alone:

1. **Point `LOGS_DIR` and `WANDB_DIR` at a directory you own.** The values in
   `.envrc.grete` are the owner's and are group-readable but not group-writable, so a run
   that inherits them fails when it creates its run directory. `README.md` § *First-time
   setup for a new member of project `nib00021`* covers this and the next item.
2. **Log in to HuggingFace and W&B** (`hf auth login`, `wandb login`).
3. **Build the venv:** `UV_PROJECT_ENVIRONMENT=.venv-cu126 uv sync --no-group cuda --group
   cu126`. Jobs run in `.venv-cu126` and the launcher refuses to start without it.
4. **For exp32 only: access to the DINOv3 teacher, which is gated.**
   `facebook/dinov3-vitb16-pretrain-lvd1689m` is **`gated: manual`** on the Hub. **exp32
   will not run unless you are signed in to a HuggingFace account that has been granted
   access to that model** — request it on the model page, wait for Meta to approve, then
   `hf auth login` so `HF_TOKEN` is set (`.envrc.grete` picks it up from
   `$HF_HOME/token`). Without it the run fails at startup, when it reads the teacher's
   width. Every exp32 arm needs the teacher: the two `teacherinit` arms to initialise the
   backbone, and all four for validation, which computes teacher features on the fly.

   The other four groups need no gated model. The `canvit/*` checkpoints exp33 and exp35
   pull are public (verified 2026-09-09), so they need no token at all.

The checkpoint paths in the launchers are absolute for that reason: `logs/` is gitignored,
so each artifact exists in exactly one place, and a path derived from your own clone would
find nothing.

All launchers pin `TRAIN_COMMIT` / `PYTORCH_COMMIT` / `FOVI_COMMIT`, so each job runs a
frozen `git archive` snapshot and its code is immune to later edits of the clones (the
environment setup and the venv are read at job start, see `slurm/README.md`). Pinning resolves
the three repos as siblings of your clone (`canvit/`, `fovi/`, `CanViT-PyTorch/`), so it
needs all three present. **Current code runs all four recipes too** — verified 2026-09-09,
one short foreground run per task from these exact source checkpoints — so dropping the
pins is a supported way to run them; expect numbers close to the ones below, not identical.

Arrays are a **budget, not a schedule**: the job index comes from the checkpoint's resume
state rather than `SLURM_ARRAY_TASK_ID`, so the step count advances only for jobs that
succeed. If tasks die, resubmit the remainder — nothing is lost and nothing double-counts.
Resubmitting also continues the same wandb run, because the checkpoint carries its run id.

**Ask for a short wall clock.** Every launcher here requests 2 h or less, and that is the
reason the long runs are chunked into arrays at all: a short job gets scheduled sooner than
a long one, so many 2 h chunks move through the queue faster than one 12 h request. Keep it
that way when you write a new launcher — raise `--time` only when a single chunk genuinely
does not fit, and shrink `CFG_STEPS_PER_JOB` in a *new* run group instead if you can (it
cannot be lowered on a run already in progress: `_check_schedule_invariants` refuses to
resume if it changes).

### Without SLURM

A launcher wraps one command: `harness_train.sbatch` turns each `CFG_FOO_BAR` into
`--cfg.foo-bar` and each `OPT_FOO_BAR` into `--opts.foo-bar`, adds the run identity and the
data paths, and runs the trainer. So any arm can be run in the foreground instead — the
quickest way to check a setup before committing a job to the queue. exp32's foveated
teacher-init arm, as two steps with evaluation off:

```bash
python -m canvit.harness.run distill \
  --cfg.run-group my_smoke --cfg.run-name exp32-fovi-teacherinit \
  --cfg.logs-dir "$LOGS_DIR" --cfg.tracker none \
  --cfg.webdataset-dir "$WEBDATASET_DIR" --cfg.val-dir "$VAL_DIR" \
  --cfg.val-index-dir "$VAL_INDEX_DIR" \
  --cfg.peak-lr 0.0004 --cfg.batch-size-per-gpu 64 --cfg.steps-per-job 8192 \
  --cfg.model.patcher-name foveated --cfg.model.foveated-patcher.fov 35 \
  --cfg.model.foveated-patcher.resolution 64 --cfg.model.foveated-patcher.cmf-a 0.5 \
  --cfg.model.foveated-patcher.cart-patch-size 5 \
  --cfg.model.foveated-patcher.arch-flag doubleres \
  --cfg.model.foveated-patcher.conditioning.mode film \
  --cfg.model.foveated-patcher.conditioning.film.fourier.num-features 256 \
  --cfg.model.foveated-patcher.conditioning.film.fourier.sigma 4 \
  --cfg.foveated-scale.fixed-scale 2.0 --cfg.init-backbone-from-teacher \
  --opts.n-steps 2 --opts.eval-every 0
```

The other three groups translate the same way, from the `CFG_`/`OPT_`/`EXTRA_ARGS` block at
the top of their launcher. Three things to know:

- **`--cfg.run-group` is required.** It names the experiment and fixes where every artifact
  goes (`$LOGS_DIR/<run_group>/<run_name>/`, holding `checkpoints/` and `visualization/`).
- **distill has no `max_steps`** — it is array-shaped, so a foreground run needs
  `--opts.n-steps <N>` or it stops after `steps_per_job`. ade20k and in1k take
  `--cfg.max-steps` and run their whole schedule in one process.
- **`--opts.eval-every 0` turns evaluation off.** Otherwise a run evaluates at step 0, which
  for in1k and distill means a full 50k-image val pass before the first update.

Use a launcher for anything real: the 2 h chunking, resume across array tasks and the
shard-schedule invariant are what let a 1.4M-step pretrain finish at all.

---

## exp32 — pretraining from scratch

### Setup

Four pretrains from scratch: uniform / foveated patcher × with and without teacher init.
Warmup 100k → constant 4e-4 for the whole run. No source checkpoint — the only inputs are
the IN21k shards (`$WEBDATASET_DIR`) and the val image folder (`$VAL_DIR`).

```bash
bash slurm/runs/exp32_pretrain_lrdrop/exp32-uniform16-teacherinit.sh   # 77 x 8192 =   630,784
bash slurm/runs/exp32_pretrain_lrdrop/exp32-uniform16.sh               # 176 x 8192 = 1,441,792
bash slurm/runs/exp32_pretrain_lrdrop/exp32-fovi-teacherinit.sh        # 138 x 8192 = 1,130,496
bash slurm/runs/exp32_pretrain_lrdrop/exp32-fovi.sh                    # 245 x 8192 = 2,007,040
```

`--cfg.init-backbone-from-teacher` is what the `teacherinit` arms add; the foveated arms add
the patcher block and `--cfg.foveated-scale.fixed-scale 2.0`. Both are in `EXTRA_ARGS` at
the top of each launcher.

**A ×0.1 LR decay is not part of this.** These four are the constant-LR runs, which is all
that is needed to see the stack train end to end. A decay phase does improve the final
number — see the exp22 comparison in the results below, and `exp32-<arm>-lrdrop.sh` if you
want to run one — but it is a second campaign seeded from one of these runs' checkpoints,
not a flag. Its `CFG_SEED_CKPT` names an exact `step-<N>.pt`, so it refuses to submit until
that file exists (two of the three currently do refuse: see the results).

**Judge on `val/scene_cos_raw_t9`** — the raw scene cosine at the last of 10 eval glimpses.
This is the same scalar the trainer logs as `eval/val_metric` and selects `best.pt` on.

**Evaluation policy** is `auto`, which resolves per patcher: uniform arms validate under
coarse-to-fine, foveated arms under a fixation grid at their training scale (2.0).

### Results

All four arms finished. Best `val/scene_cos_raw_t9`, and where each stopped:

| arm | reached / target | best | at step | final | drift |
|---|---|---|---|---|---|
| `exp32-uniform16-teacherinit` | 622,592 / 630,784 | **0.9390** | 614,400 | 0.9390 | +0.0000 |
| `exp32-uniform16` | 1,433,600 / 1,441,792 | **0.9186** | 1,425,408 | 0.9186 | +0.0000 |
| `exp32-fovi` | 2,007,040 / 2,007,040 | **0.9258** | 1,916,928 | 0.9256 | −0.0002 |
| `exp32-fovi-teacherinit` | 1,130,496 / 1,130,496 | **0.9182** | 729,088 | 0.8750 | **−0.0432** |

**`exp32-fovi-teacherinit` degraded in the second half.** It peaked at step 729k and fell
0.043 by 1.12M while the other three arms drifted ≤0.0002. That anomaly is unexplained, and
worth diagnosing before this arm is used as anything but a smoke test — including as the
seed of a decay run, which from step-1130496 would start from the degraded state, not the
peak.

The two `uniform16` arms are 8192 steps short of their target because both lost their final
array task to a node fault (`ggpu150`, `NVML: GPU is lost`, 2026-08-06 09:29). Their curves
had long flattened, so the numbers stand; a re-run needs no special handling.

**Comparison with `jon_exp22_full_runs`, which did run a decay phase.** This is where the
LR decay left out of the setup above shows up:

| arm | exp32 (constant LR) | exp22 (constant LR) | exp22 after decay |
|---|---|---|---|
| `uniform16-teacherinit` | 0.9390 @ 614k | 0.9398 @ 639k | 0.9477 (4e-5), 0.9481 (+4e-6) |
| `uniform16` | 0.9186 @ 1,425k | 0.9199 @ 1,516k | 0.9262 (4e-5) |
| `fovi` | 0.9258 @ 1,999k | 0.9248 @ 1,942k | *no decay was run* |
| `fovi-teacherinit` | see above | 0.9362 @ 1,196k | 0.9423 (4e-5) |

Three arms reproduce exp22's constant-LR phase within 0.001–0.002 — which is the point of
the campaign. **The decay is worth +0.006–0.008 wherever it was run**, so the exp32 column
is a floor: these recipes reach a little higher than the numbers here if you add a decay
phase, and they are not directly comparable to exp22's decayed column.

**Differences between exp22 and exp32** — why these are reference numbers, not targets:

1. **Normalizer statistics.** exp32 pools the first 4 sorted shards; exp22 used a single
   shard (`shard-001751`, 4096 samples). Different standardization of the DINOv3 targets,
   so the loss scale is not identical and curves do not overlay exactly.
2. **Decay schedule.** exp22's `uniform16-teacherinit` got two drops (4e-5 then 4e-6) and
   its `fovi` arm none; exp32 ran none at all.
3. **Seeding.** The older trainer never called `torch.manual_seed`, so each exp22 run drew
   an unreproducible random init. exp32 seeds before `build_model`.

### Seed-spread runs

`exp32-fovi-teacherinit-s1` … `-s5`: five more runs of the foveated teacher-init config,
identical to `exp32-fovi-teacherinit` in every setting except `CFG_SEED`, 204,800 steps each.

```bash
for s in 1 2 3 4 5; do SEED=$s bash slurm/runs/exp32_pretrain_lrdrop/exp32-fovi-teacherinit-seed.sh; done
```

`CFG_SEED` moves the random init (`torch.manual_seed(seed + rank)` runs before
`build_model`) and the webdataset shard schedule. It does **not** move the normalizer
statistics, which pool the first 4 sorted shards and are identical across seeds. Seed 0 is
`exp32-fovi-teacherinit` itself, so the group yields six samples; the launcher refuses
`SEED=0` to protect that run's directory.

They exist to measure how much a foveated pretrain moves between seeds, which nothing in
this stack had quantified.

**Result.** `val/scene_cos_raw_t9` at step 188,416, the last point all six share (s2 stopped
one chunk early on a 2 h timeout):

| s0 | s1 | s2 | s3 | s4 | s5 |
|---|---|---|---|---|---|
| 0.8922 | 0.9075 | 0.9175 | 0.9075 | 0.9196 | 0.9164 |

mean **0.9101**, sd **0.0102**, spread **0.0274**. `exp22-fovi-teacherinit` scores 0.9070 at
the same step — **inside the range and just below the harness mean**.

This settles a question that was open for a month. `exp32-fovi-teacherinit` (seed 0) had
appeared to run ~0.019 below its exp22 counterpart, which looked like an implementation
deficit in the foveated path. It is not: seed 0 is simply the lowest of six draws, and the
seed-to-seed spread of this configuration (0.027) is larger than the gap that prompted the
investigation. Nothing needs fixing.

The corollary is a measurement rule: **a single foveated pretrain cannot resolve a
difference below ~0.03 in this metric.** Two configurations closer together than that are
indistinguishable without several seeds each.

---

## exp33 — ImageNet-1k full finetunes

### Setup

```bash
bash slurm/runs/exp33_in1k_finetune/in1k-uni16ti-803k.sh
bash slurm/runs/exp33_in1k_finetune/in1k-uni16-1516k.sh
bash slurm/runs/exp33_in1k_finetune/in1k-fovi-ti-1196k.sh
bash slurm/runs/exp33_in1k_finetune/in1k-fovi-1901k.sh
```

**Sources.** One exp22 pretrain per arm, as `--cfg.model-repo`, all under
`$EXP22 = /mnt/vast-nhr/projects/nib00021/jonathan/repos/canvit/logs/jon_exp22_full_runs`:

| arm | `--cfg.model-repo` |
|---|---|
| `in1k-uni16ti-803k` | `$EXP22/exp22-uniform16-teacherinit-lrdrop2-803k/checkpoints/step-16384-hf` |
| `in1k-uni16-1516k` | `$EXP22/exp22-uniform16-lrdrop-1516k/checkpoints/step-319488-hf` |
| `in1k-fovi-ti-1196k` | `$EXP22/exp22-fovi-teacherinit-lrdrop-1196k/checkpoints/step-155648-hf` |
| `in1k-fovi-1901k` | `$EXP22/exp22-fovi/checkpoints/step-1900544-hf` |

All four also take `--cfg.probe-repo canvit/dinov3-vitb16-lvd1689m-in1k-512x512-linear-clf-probe`
from the Hub, which is fused into the classification head (TPU parity — see the loss check
below). The two foveated arms add `--cfg.foveated-scale.fixed-scale 2.0`.

Four finetunes, one per pretrained backbone; the TPU recipe batch-adapted for one A100.
49 array jobs × 8192 = 401,408 steps each (~20 epochs at batch 64). `n_timesteps=4`, not the
task default of 10. Foveated arms evaluate under `random` with
`--cfg.foveated-scale.fixed-scale 2.0` — coarse-to-fine is uniform-only and out of
distribution for a fixed-scale foveated model.

**Only one number is reported and it is measured at the final timestep** — the classifier
reads the CLS token of the last glimpse only, so there is no per-timestep top-1 series.

**Watch the first logged `train/full/loss`:** it must sit well below `ln(1000) ≈ 6.9`. All
four arms start at 1.74–1.88, confirming the pretrained probe head was fused. A value near
6.9 would mean the finetune began from a random classifier, which is what `CFG_PROBE_REPO`
guards against.

An in1k checkpoint records its own architecture, so a finetuned model — whose backbone
exists nowhere else — loads straight from its `.pt` via `load_classifier`.

### Results

All four arms complete. Best `eval/top1`:

| arm | best `eval/top1` | run length |
|---|---|---|
| `in1k-uni16ti-803k` | **0.84962** | 393,216 of 401,408 |
| `in1k-fovi-ti-1196k` | **0.83716** | 401,408 |
| `in1k-uni16-1516k` | **0.83504** | 401,408 |
| `in1k-fovi-1901k` | **0.82718** | 401,408 |

Ordering `uni16ti > fovi-ti > uni16 > fovi`. Teacher init is worth ~1.4 pp on the uniform
patcher and ~1.0 pp on the foveated one.

`uni16ti-803k` is one 8192-step chunk short: its final array task hit the 2 h wall, and the
arrays were sized with no slack. Its curve had already flattened, so the number stands.

---

## exp34 — ADE20K frozen probes

### Setup

```bash
bash slurm/runs/exp34_ade20k_probe/ade20k-uni16ti-803k.sh
bash slurm/runs/exp34_ade20k_probe/ade20k-uni16-1516k.sh
bash slurm/runs/exp34_ade20k_probe/ade20k-fovi-ti-1196k.sh
bash slurm/runs/exp34_ade20k_probe/ade20k-fovi-1901k.sh
```

**Sources.** The same four exp22 pretrains as exp33, one per arm as `--cfg.model-repo`
(`ade20k-uni16ti-803k` ← `exp22-uniform16-teacherinit-lrdrop2-803k/checkpoints/step-16384-hf`,
and so on down the exp33 table). The segmentation head is trained here, so there is no
probe to supply. The two foveated arms add `--cfg.foveated-scale.fixed-scale 2.0`.

Four probe runs on the same four backbones. Frozen backbone — which is the ade20k default
(`cfg.mode = frozen`), so the launchers pass no `--preset`; `--preset probe` is the same
spec. 40,000 steps, random-view training, `n_timesteps 10`, scene 512, `canvas_grid 32`.
Single GPU — the ADE20K task does not support DDP.

**`resize_mode=squish` for every arm including foveated.** It distorts aspect ratio, so it
is not the right choice for a human-viewing comparison; whichever mode is used must be
reported alongside the number, because it moves the metric materially.

**The per-timestep `eval/miou_t*` are measured under *random* viewpoints, not
coarse-to-fine** — `eval_policy` is `auto`, and ADE20K's resolution is IID random from a
full-scene anchor, matching how the probe trains. ADE20K train-mIoU is deliberately not
logged.

### Results

All four completed. Best `miou_final` under the training eval (random viewpoints,
10 glimpses):

| arm | best `miou_final` |
|---|---|
| `ade20k-uni16ti-803k` | **0.45221** |
| `ade20k-fovi-ti-1196k` | **0.44344** |
| `ade20k-uni16-1516k` | **0.42363** |
| `ade20k-fovi-1901k` | **0.41917** |

#### Re-evaluated under coarse-to-fine, 21 glimpses

The training eval answers "how good is this probe under the viewpoints it trained on". This
answers the separate question "what does it score under the C2F deploy convention", whose
`EpisodeConfig` default is 21 glimpses.

```bash
bash scripts/eval_ade20k_c2f.sh jon_exp34_ade20k_probe        # -> logs/<group>/_c2f_eval/*.json
```

One GPU, ~10 min for all four (measured on a MIG `3g.40gb` A100 slice with 12 CPUs). The
script loops the four arms through `python -m canvit.harness.evaluate ade20k`, which rebuilds each
model through the ADE20K task and loads the probe's `best.pt` into it — no HF export step.
Full ADE20K val (2000 images), `squish-512`, `canvas_grid 32`, batch 16, each arm against
the backbone its probe was trained on. The two FOVEATED arms add
`--cfg.eval-override-scale 2.0 --cfg.foveated-scale.fixed-scale 2.0`.

mIoU in %, measured 2026-08-02 on each run's `checkpoints/best.pt`:

| arm | t0 | t1 | t2 | t4 | t9 | t14 | t20 | gain t0→t20 | scale pin |
|---|---|---|---|---|---|---|---|---|---|
| `uni16ti-803k` | 41.23 | 42.69 | 43.80 | 45.62 | 46.13 | 46.39 | **46.61** | +5.38 | — |
| `fovi-ti-1196k` | 37.68 | 40.55 | 41.92 | 44.25 | 45.18 | 45.56 | **45.91** | +8.23 | 2.0 |
| `uni16-1516k` | 38.23 | 39.90 | 41.17 | 43.16 | 43.65 | 43.89 | **44.12** | +5.88 | — |
| `fovi-1901k` | 35.26 | 38.49 | 39.87 | 41.84 | 42.28 | 42.89 | **43.03** | +7.77 | 2.0 |

Same ordering as the training eval, and every arm ends 1.1–1.8 pp above its own 10-glimpse
random-view `miou_final`.

**The foveated arms' larger gain is not a larger capability.** Each starts ~3 pp below its
uniform counterpart at t0 and spends the curve catching up, because for a fixed-scale
foveated backbone t0 is a scale-2.0 foveation rather than a full-image view.

**"Coarse-to-fine" is only literally true for the uniform arms.** `--cfg.eval-override-scale`
overwrites every generated scale and keeps the generated CENTERS, so the pinned foveated
arms run all 21 glimpses at the identical scale-2.0 foveation pattern, moved along the
quadtree's centre schedule (centre → 4 quadrant centres → 16 sub-quadrant centres). The
{1.0, 0.5, 0.25} scale ladder that makes C2F coarse-to-fine is exactly what the pin deletes.
Unpinned it would be worse, not better: a fixed-scale foveated model derives
`fix_size = scale * H`, so unseen scales put every glimpse out of distribution and mIoU
decays as glimpses accumulate. The four arms are therefore not compared policy-for-policy,
and should be reported that way.

**These numbers carry run-to-run noise.** `policies._shuffle_levels` calls `torch.randperm`
with no generator, so C2F draws a fresh within-level permutation per sample from the global
RNG. Re-evaluating identical weights moved `t20` by up to 0.5 pp on `fovi-1901k` and by
≤0.07 pp on the other three, while `miou_t0` matched to five decimals every time — level 0
has n=1 and skips the shuffle. Seeding the permutation would remove the question; until
then, do not read a sub-pp difference between two c2f runs as real.

---

## exp35 — ADE20K viewpoint policy, 10 seeds

### Setup

```bash
for s in 0 1 2 3 4 5 6 7 8 9; do
  SEED=$s bash slurm/runs/exp35_policy_qreg_10seed/policy-qreg-s0.sh
done
```

**Sources: none local.** Both halves come from the Hub, which makes this the one group with
no dependency on exp22 or on anything under `logs/` — the backbone is
`Ade20kConfig.model_repo`'s default (`canvit/canvitb16-add-vpe-pretrain-g128px-s512px-in21k-dv3b16-2026-02-02`,
so `CFG_MODEL_REPO` is deliberately unset) and the probe is
`--cfg.probe-repo canvit/probe-ade20k-40k-s512-c64-in21k`.

A `ViewpointScorer` trained by Q-regression against a **frozen** backbone and probe, so that
segmentation improves as fast as possible per glimpse; at deployment it takes the argmax over
its candidate grid. `--preset policy_only`, 9000 steps, 5 timesteps, batch 16, canvas grid
64, `resize_mode=squish`. Ten seeds.

**9000 steps rather than 8000**, because the loop evaluates when `step % val_every == 0` and
never reaches `max_steps`; at 8000 the last eval would be at 7000. The extra 1000 steps buy a
ninth eval. They do not reshape the LR schedule: `policy_only` freezes backbone and head, so
ADE20K's `warmup_onecycle` is never built. The scorer uses `JointPolicyConfig`'s own recipe —
`warmup_constant`, warmup = `int(0.125 * max_steps)`, then flat at 2e-4 — so the only effect
is a ramp to step 1125 instead of 1000.

`CFG_RESIZE_MODE=squish` is the measurement contract for this group. Changing it shifts CE by
roughly 0.016 — an order of magnitude more than the seed spread below — so a policy result
that looks dramatically better is far more likely to be a changed protocol than a better
policy.

**Free consistency check:** `ce_t0` / `miou_t0` are the glimpse taken *before* any policy
action, so they depend only on the frozen backbone, probe, resize and eval path. Matching
them against a probe-only evaluation isolates any difference to the policy itself.

### Results

All ten seeds complete. Each at its early-stop step: the eval with the lowest mean t1–t4 CE,
which is what `Ade20kRunTask.best_metric` selects `best.pt` on.

| seed | early-stop step | best `ce_mean` | `miou_final` (%) |
|---|---|---|---|
| s0 | 7000 | 0.68634 | 44.99 |
| s1 | 6000 | 0.68561 | 44.87 |
| s2 | 3000 | 0.68584 | 44.88 |
| s3 | 3000 | 0.68586 | 44.69 |
| s4 | 7000 | 0.68712 | 44.84 |
| s5 | 6000 | 0.68553 | 44.77 |
| s6 | 7000 | 0.68579 | 44.85 |
| s7 | 6000 | 0.68703 | 44.65 |
| s8 | 6000 | 0.68602 | 44.84 |
| s9 | 8000 | 0.68580 | 44.74 |
| **mean ± sd** | | **0.68609 ± 0.00056** | **44.81 ± 0.10** |

Ten seeds span 0.00159 in CE and 0.34 pp in mIoU — that spread is the resolution of this
recipe, and any two policy configurations closer together than it are not distinguishable
without more seeds. One seed (s9) early-stopped at step 8000, on the ninth eval that the
9000-step length exists to provide.

#### Policy comparison figure

![Viewpoint policies on ADE20K](assets/ComparisonPoliciesADE20K.png)

```bash
# measure the baselines on a GPU, read the trained-Q seeds from the run logs, then draw
python scripts/plot_policy_comparison.py --trained-dir logs/jon_exp35_policy_qreg_10seed

# baselines depend only on the frozen model + data + metric code, so a new policy group
# can copy them instead of re-measuring (no GPU, seconds)
python scripts/plot_policy_comparison.py --trained-dir logs/jon_exp35_policy_qreg_10seed \
    --reuse-baselines docs/assets/_policy_comparison_data.json

python scripts/plot_policy_comparison.py --from-cache    # re-style only, no measurement
```

Writes `docs/assets/ComparisonPoliciesADE20K.png` and caches every curve to
`docs/assets/_policy_comparison_data.json`, so restyling costs no GPU.

Every curve shares one frozen model — the published c64 pretrain plus
`canvit/probe-ade20k-40k-s512-c64-in21k`, canvas 64, squish-512, 5 glimpses, full val. Only
the viewpoint policy differs, and it is the same pair exp35 trains against, so the learned
curve and the baselines are strictly comparable.

mIoU in %, t = 0..4:

| policy | t0 | t1 | t2 | t3 | t4 |
|---|---|---|---|---|---|
| **Viewpoint-Q (trained), n=10** | 39.58 | **42.67** | **43.87** | **44.44** | **44.81** |
| 95% CI over seeds | — | ±0.08 | ±0.07 | ±0.07 | ±0.06 |
| EG-C2F | 39.57 | 42.19 | 43.28 | 44.04 | 44.66 |
| C2F | 39.57 | 41.28 | 42.52 | 43.89 | 44.68 |
| random (safe-box) | 39.57 | 41.10 | 42.04 | 42.67 | 42.91 |
| Viewpoint-Q (untrained), n=3 | 39.57 | 41.10 | 41.67 | 42.12 | 42.49 |

The learned policy leads at every t, and its advantage is largest early — +1.4 pp over C2F at
t1, narrowing to +0.1 pp by t4. That is the claim being tested: it should reach a given mIoU
in fewer glimpses, not necessarily end higher. The untrained-scorer row is the control that
separates "the learned policy works" from "any argmax trajectory works".

**Two things a reader gets wrong about this figure.** The dashed F-IID row is drawn from the
paper only: our `random` draws its scale from the safe-box area law rather than F-IID's fixed
fovea-sized scale, so it is a different policy plotted under its own name, not a
reproduction. And the trained curve's t0 (39.58) sits 0.01 pp above the baselines' (39.57)
even though t0 is pre-policy and the model is identical — the trained row is read from each
run's own logged eval, in a different process from the one-shot baseline measurement. That is
cross-process numerical noise, not a different model.

---

## Using the results afterwards

Checkpoints from exp33 and exp34 are **self-describing**: they record the architecture, not
just a path to the model they started from, so they load without their source repo:

```python
from canvit.core.model_source import load_classifier, load_segmentation
clf = load_classifier("logs/jon_exp33_in1k_finetune/<run>/checkpoints/best.pt")
seg = load_segmentation("logs/jon_exp34_ade20k_probe/<run>/checkpoints/best.pt")
```

An exp34 checkpoint also works directly as `--cfg.probe-repo` for a policy run, with no
conversion. Publishing goes through `canvit.checkpoint.to_hf` (whole models) or
`canvit.checkpoint.probe_to_hf` (the segmentation head alone).

`to_hf` auto-detects which layout to write from the payload: a distill checkpoint becomes
the `CanViTForPretrainingHFHub` layout, an in1k one the `CanViTForImageClassification`
layout (`metadata.task == "in1k"`). An ADE20K probe is the one that does not go through it —
its head is published with `probe_to_hf`.
