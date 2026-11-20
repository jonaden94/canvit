# exp27 — does the UNIFIED HARNESS reproduce the CanViT-PyTorch-RL policy recipe?

## FINAL VERDICT (2026-07-31): the harness reproduces the recipe. PASSED.

Deploy (best-mean-CE) checkpoints — the selection the published band and the 8 HF policies use.
Reference row = the 8 PUBLISHED policies through OUR eval, which removes every
accounting/batch-size confound (their doc: 0.6853 +- 0.0007 / 44.97 +- 0.10, so our eval
reproduces them to +0.0003 CE / -0.01 mIoU).

| arm | n | mean(t1-t4) CE | mIoU t4 | dCE (p) | dt4 (p) |
|---|---|---|---|---|---|
| **published, our eval** | 8 | 0.6856 +- 0.0006 | 44.96 +- 0.11 | — | — |
| `rl_train` (ported ref) | 5 | 0.6855 +- 0.0010 | 44.86 +- 0.09 | -0.0000 (0.55) | -0.106 (0.051) |
| **harness + scale fix** | 5 | **0.6859 +- 0.0009** | **44.87 +- 0.12** | +0.0003 (0.22) | -0.096 (0.082) |
| `rl_train` POOLED (arm E) | 5 | 0.6853 +- 0.0005 | 44.88 +- 0.10 | -0.0002 (0.76) | -0.084 (0.099) |

**The harness is indistinguishable from the ported reference and from the published policies on
CE** — the metric the band is defined by and the one the reward optimizes. Goal met.

**Arm E (the original's pooled rollout) is a NULL:** t4 +0.022 (p=0.377), CE -0.0002 (p=0.393).
So the ~0.1 mIoU t4 shortfall shared by ALL THREE arms is not the rollout architecture; being
common to every trainer we have, it sits upstream of all of them. Only marginally significant
(p=0.05..0.10) and CE matches exactly — since the reward IS CE, two policies can tie on CE and
differ slightly in mIoU. Remaining unaudited surface: the core-library revisions on the
authors' machine (doc 15 §A5.9/§A5.10).

**Beware two selection traps** found while doing this:
- Averaging each run's PEAK `eval/miou_t4` over its evals gives 44.970 — which coincidentally
  equals the published 44.97, but is a max-over-8-noisy-evals statistic compared against a
  best-CE-selected one. The honest same-rule number is 44.834 for those seeds.
- `best.pt` exists from the FIRST eval, so an in-flight run scores as if finished. It did once
  here; `compare_to_published_band.py` now requires the terminal checkpoint as proof.

---

## Earlier verdict (2026-07-30), kept for the reasoning

15 checkpoints, all re-scored at the LAST step through ONE eval in ONE process at eval batch
32 (every arm logs `eval_batch_size=32`, so no batch-size confound).
`canvit_claude/canvit/unification/compare_arms.py` produces this table and the tests.

| arm | n | mean(t1-t4) CE | mIoU t4 | jobs |
|---|---|---|---|---|
| band, last step (published) | 8 | 0.6863 | 44.91 | — |
| **C** `rl_train` (ported reference) | 5 | **0.6866 +- 0.0008** | **44.854 +- 0.057** | 15098292/93, 15103016-18 |
| **B** harness (0.8x policy gradient) | 5 | 0.6880 +- 0.0010 | 44.770 +- 0.133 | 15100922-26 |
| **D** harness + scale fix | 5 | 0.6871 +- 0.0015 | 44.755 +- 0.136 | 15103388-92 |

Exact one-sided permutation tests (252 splits; attainable floor 0.0040):

| comparison | dCE | p | dt4 | p |
|---|---|---|---|---|
| B vs C | +0.0014 | **0.0278** | -0.084 | 0.123 |
| D vs C | +0.0005 | 0.262 | -0.100 | 0.103 |
| D vs B (improvement) | -0.0009 | 0.151 | -0.016 | 0.583 |

**The harness/reference difference is REAL** (B vs C, p=0.0278 on the band's defining metric).
**A concrete cause was found and fixed**: `run_rollout` divided the policy loss by
`n_glimpses` when only `n_glimpses-1` glimpses carry one, making the harness's scorer gradient
exactly **0.8x** the reference's — a 20% smaller effective policy LR (commit bc0b16b, doc 15
§A5.1). **But the fix does not fully explain the gap:** it cuts CE +0.0014 -> +0.0005 (D vs C
no longer detectable, p=0.26) yet D vs B is only p=0.151 and **t4 did not move at all**.

**Do not read an arm's level off two seeds.** Arm C's first two seeds were on the lucky side
of its own distribution (0.6870/0.6859, then 0.6857/0.6870/0.6876); an earlier version of this
table quoted arm C as 0.6864 +- 0.0008 / 44.91 +- 0.03 from n=2 and called the harness gap
larger than it is. Also: compare at the LAST step, not best-ckpt — best-ckpt is a max over 8
noisy evals and flatters the noisier arm.

**NO CODE DIVERGENCE REMAINS** (doc 15 §A5.4). With `prime_on_policy=1.0` (no RNG), the same
batch and the same scorer, the two training rollouts are **bit-identical at every depth** —
ΔCE +0.000000 and 16/16 chosen glimpses agreeing at t1..t4, `reward_frac` equal to all printed
digits. With the loss composition already pinned bit-identical by
`harness/tests/test_policy_loss_scale.py`, the gradient is identical too. So arm D vs arm C
(dCE +0.0005, p=0.26) is **seed/stream noise**: independent shuffling, scorer init and
eps-greedy streams make identical code give different runs.

An earlier version of this README blamed a "fp32 vs bf16 logits" divergence for the residual.
That was wrong twice over — the recomputation is bit-identical, and the trace that appeared to
show a divergence was my own script omitting `rollout_and_loss`'s train-mode scorer forward
(the one that updates `frontend.bn` before the eval-mode selection reads it). Retracted in
doc 15 §A5.4. Reproduce: `canvit_claude/canvit/unification/diff_training_trace.py`.

Still unexplained, and possibly nothing: arm C's t4 sd is 0.057 vs the harness's 0.133/0.136.
At n=5 that ratio is weak evidence (F(4,4) is very wide). It is not a rollout code difference.

Measurement is not the issue: the harness eval is bit-identical (0.0000 on t0..t4 + ce_mean)
to the validated eval on **all three model sources** — published HF qband, `rl_train`, and
harness (`canvit_claude/canvit/unification/eval_equivalence.py`).

**FOUR defects had to be fixed, and every one of them was silent:**

1. **resize protocol** — center_crop vs the band's squish; made arm A look 0.016 CE *better*
   than a band quoted to +-0.0007.
2. **mIoU argmax/upsample order** — this repo was the only one in the stack argmaxing before
   upsampling; worth 0.19 mIoU (commit 68b635f, re-bases all earlier ADE20K mIoU).
3. **frozen head left in train mode** — `requires_grad_(False)` does not freeze BatchNorm, so
   the probe drifted under the reward (commit bc5e00e).
4. **probe BN polluted at startup** — `StateEncoder` construction ran the probe in train mode
   on a batch of ONE blank canvas, shifting `head.bn.running_mean` by 1.074 and the init
   template by 1.621288. This is the one that made the harness eval disagree with the
   reference at all (canvit_pytorch 1f5121b).

Plus BN mode (b) for glimpse SELECTION, which is what actually closed the remaining
0.19 mIoU (doc 15 §A3).

---

## Original plan (kept for the reasoning)

wandb project `exp27`. Two arms, seed 0, same GPU class, same pinned commits.

| arm | script | entry point | status before this |
|---|---|---|---|
| A (control) | `policy-oldloop-s0.sh` | `python -m canvit_pretrain.ade20k.rl_train` | gate-validated 2026-07-23 |
| B | `policy-harness-s0.sh` | `harness.run ade20k --preset policy_only` | **never run at scale** |

## Why a control arm at all

The published reference is the RL repo's 8-seed qband band, **0.6853 ± 0.0007** mean
t1–t4 val CE (`CanViT-PyTorch-RL/docs/qband_results.md`, per-seed spread 0.6845…0.6865),
with EG-C2F-c64 at 0.6949. The P3 gate already reproduced it here (jobs 15025279 /
15025337 → 0.6855 / 0.6867).

Arm A is re-run anyway for two reasons:

1. The band was measured on the RL repo's machine. A local reference removes the
   hardware/stack question from the comparison entirely.
2. Those gate runs predate `845e401` (per-timestep mIoU in the deploy eval), so they
   logged **CE only**. Arm B reports CE *and* mIoU; without arm A, half the comparison
   has no counterpart.

**Rule (learned in exp23/exp26): never gate a production A/B on one baseline run.** If A
and B disagree, the next step is a second seed of A — not a verdict. See
[[../../../../canvit_claude/canvit/unification/14-parity-coverage.md]].

## Verdict rule

Judge on **mean(t1–t4) val CE**, the metric the qband band is defined by and the one
arm B selects `best.pt` on (`neg_ce_mean`).

- Arm B inside 0.6845…0.6865 → the harness reproduces the recipe. The policy path is
  production-gated and CanViT-RL work can proceed on it.
- Arm B outside, arm A inside → a harness-specific problem. *(Historical: `score_res` was
  the named suspect here and it was the WRONG one — the actual causes were BN mode (a) for
  glimpse selection and the probe-BN pollution. score_res was closed anyway, doc 15 gap #5.)*
- Both outside → not a harness question; look at the stack (pins, probe repo, data).

Also worth watching, independent of the verdict: **mIoU at t1–t4** (band: 42.65 → 44.97)
and `reward_frac` trending positive. A policy that is learning nothing shows a flat
`reward_frac` near 0 while CE sits at its t0 value.

## What had to be fixed before B was even meaningful

All 2026-07-28/29, committed in `cea4dee`, detailed in
[[../../../../canvit_claude/canvit/unification/15-rl-recipe-parity-and-open-items.md]] §A:

1. **Validation deployed no policy.** `--preset policy_only` validated on RANDOM
   glimpses and selected `best.pt` on that mIoU. Fixed by the shared `eval_policy` knob
   (`--cfg.eval-policy policy`) + `best_metric` following to `neg_ce_mean`.
2. **Adam betas** — every group silently got torch's `(0.9, 0.999)`; RL uses `(0.9, 0.95)`.
3. **No LR ramp** — the policy group fell through to `warmup_steps=0`.
4. **Augmentation** — rl_train trains on unaugmented images; the harness augmented
   unconditionally. Needed its own flag: neutralising the aug knobs does NOT disable
   augmentation (`RandomCropWithLabel` + `PhotoMetricDistortion` have no knob).
5. **The probe head (blocking).** `probe_repo` was gated on `mode=="finetune"`, but
   `policy_only` runs FROZEN — so it built a **fresh random head**. The reward *is* the
   probe's CE reduction, so the scorer would have trained on pure noise with nothing
   failing and every log looking healthy.

## The first attempt was voided by the resize protocol (2026-07-29)

Jobs 15093707 / 15093712 / 15093767 / 15093768 ran **center_crop on both arms** and were
cancelled. Arm A s0 completed at mean t1–t4 CE **0.6693** — 0.016 *better* than the band,
~20× its own 0.0007 seed spread. That was not a better policy, it was a different
measurement.

CanViT-PyTorch-RL squish-resizes image **and** mask to `scene_size` everywhere — its
`config.py` docstring calls this "the measurement contract every entry point builds on",
and its dataset class is literally `Ade20kSquish`. The qband band and the EG-C2F
baselines exist only under squish. At the P3 gate commit `7e5afac`, `rl_train.py:329`
hardcoded it. Commit **`1a0b452`** ("Lift val resize_mode into ADE/policy configs, default
center_crop") turned it into a knob and defaulted it to `center_crop`, silently
decoupling the reference from its band; neither launcher set it.

Fixed three ways: `PolicyTrainConfig.resize_mode` is pinned back to `squish` (it is the
frozen reference — `Ade20kConfig` deliberately keeps `center_crop` for new work), both
launchers now set it explicitly, and a policy run under any other mode logs a loud
not-band-comparable warning. Pinned by `test_rl_train_defaults_to_the_bands_squish_protocol`
and `test_harness_policy_run_warns_when_not_band_comparable`.

**Everything else audited clean** against the original repo (2026-07-29): same
`model_repo` and probe rule, same `make_val_transforms` function (equivalence-tested for
both modes in `canvit_claude/canvit/unification/specialize_equivalence.py`), full val split with no
limit/stride, eval CE at full 512², objective mean over t1–t4, and every recipe
hyperparameter matching `TrainConfig` (lr 2e-4, wd 1e-2, betas .9/.95, clip 1.0,
score_res 128, 640k forwards, batch 16, horizon 4, warmup 0.125, target_momentum 0.997,
scales (0.5,0.25), centers 16, width 128, blocks 3, prime_on_policy 0.5, dueling). The
only remaining diffs are immaterial to the metric: eval batch 32 vs 16 (workers are 4 on BOTH — the old "4 vs 8" claim was wrong, and worker count provably does not change the data; doc 15 §A5.7).

## Gotcha for anyone editing arm B

`--cfg.no-augment` lives in `EXTRA_ARGS`, **not** as `CFG_AUGMENT=False`. tyro renders
bools as paired flags (`--cfg.augment` / `--cfg.no-augment`) while the launcher's
`CFG_FOO_BAR` mapping emits `--cfg.foo-bar VALUE`, which cannot express a flag. The same
applies to every other bool knob. Caught by a local smoke run before submitting; a
launcher-only test would have passed the wrong thing silently.

`ADE20K_ROOT` is **not** in `env.sh` — both scripts export it, as every exp24 script does.

## Submit

```bash
bash slurm/runs/exp27/policy-oldloop-s0.sh              # ~65 min on one A100
for s in 0 1 2 3 4; do SEED=$s bash slurm/runs/exp27/policy-harness-s0.sh; done  # ~75 min each
```

`RUN_NAME` includes the seed, so seeds get distinct run dirs. That matters: an earlier
re-run with a fixed `RUN_NAME` silently OVERWROTE the previous configuration's checkpoints.

Curves for the Figure-4B comparison: `canvit_claude/canvit/unification/plot_policy_curves.py
--policy-ckpts <each seed's step-8000.policy.pt>`.

Pins: `PRETRAIN=cea4dee`, `PYTORCH=017ce9b`, `FOVI=c399d3b` on both arms. `rl_train.py`
is untouched by `cea4dee` (the session changed the harness path), so arm A pins "today's
code" without depending on any of this session's work.
