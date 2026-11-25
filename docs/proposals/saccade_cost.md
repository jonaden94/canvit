# Proposal: optional per-saccade cost in the RL reward

> Written by Claude (AI, Claude Opus 5.5), 2026-11-25, for Valentin, as a proposal for Jonathan (and his Claude).
> Stage 1 by helper_1; stage 2 implemented and this file updated by helper_2 the same day.
> `saccade_cost.patch` is a `git diff` against `main` 40c1927 with BOTH stages. It applies cleanly
> (`git apply --check`) and is the local branch `saccade-cost` (not pushed). Default off.
> Tests: 30 new CPU tests (24 harness, 6 search) and the full suite pass (`.venv-cu126`, one torch
> thread, TMPDIR on lustre; `in1k/tests/test_in1k.py` excluded: it cannot be collected without read
> access to Jonathan's ImageNet path).

## TL;DR

- **Problem:** the search policy makes saccades that are too long. Scored on human COCO-Search18 and
  COCO-FreeView scanpaths, a cost on saccade length that we fit afterwards adds +1.18 bits/fix (log cost,
  mean of CS18 TP val, TA val and FreeView val; `canvit/scanpath/posthoc/saccade_cost.py`).
- **Proposal:** let RL pay for each saccade. `c(d) = log(1 + d/r)`, where `d` is the distance between
  consecutive glimpse centres and `r` is half a policy-grid cell. Option `linear` is `d/W`. The cost enters
  after standardisation: `target_t = z_batch(R) − β · z_batch(c_t)`. Each action carries its own cost.
- **Stage 1 (fixed β):** `--rl.saccade-cost log --rl.saccade-cost-weight 0.3`, plus the control β = 0.
- **Stage 2 (β-conditioned, recommended):** `--rl.saccade-cost-beta-mode conditioned
  --rl.saccade-cost-weight 1.0`. Each training trial draws β ~ U[0, 1]; the policy sees β next to the
  search cue (FiLM). One run per cost shape covers every β; at evaluation `--rl.saccade-cost-beta-eval`
  picks the β, also on an existing checkpoint.
- **Both are off by default** and then reproduce the current loss, gradients and the task digests bit
  for bit.

## 1. Why a cost, why this form

- **What the fits show:** at τ = 1 the policy's spatial entropy is already human-level. What it lacks
  is the distance bias: humans make short saccades and the policy does not. The post-hoc fits multiply the
  policy's own map by exp(−w · c(d)), with c exactly the training's cost below, and fit only w = λ/α (+ τ):
  - log fits best on all three datasets: w = 1.80 ± 0.01 for the main checkpoint, +1.28 / +0.84 / +1.41
    bits/fix on CS18 TP / TA / FreeView val;
  - linear: w = 10.46 ± 0.09, +1.25 / +0.79 / +1.33 (0.03–0.08 below log; the same ranking on the centre
    bias alone and on all 9 retinas);
  - quadratic and linear + log were also compared post hoc (2026-11-24) and dropped: quadratic was worst,
    linear + log added ≤ 0.04 over log. Only the two shapes the training implements are kept (Valentin,
    2026-11-25).
- **One definition:** on `scanpath-eval`, `canvit/scanpath/posthoc/cost_forms.py` holds the same two
  formulas as `SaccadeCost.cost` here (same d in scene units, r = 1/grid, linear = d/2); its test compares
  both on a grid of d bit for bit when both are importable (checked in a trial merge of the two branches).
  After the merge, `posthoc` imports the harness function and `cost_forms.py` goes (importers:
  `posthoc/saccade_cost.py`, `posthoc/priors.py` and its test).
- **Two forms, both physiologically motivated (Valentin, 2026-11-24):**
  - `log`: distance on the log-polar collicular/cortical map. r ≈ 0.84° is close to Schwartz's
    a ≈ 0.75°, and half a 32×32 grid cell is the same order.
  - `linear`: motor and time cost along the main sequence.
- **Why train with it rather than add it afterwards:** a cost in the reward changes what the policy
  learns to look at (it plans short hops), which a post-hoc reweighting cannot. Training with both forms
  tests which one helps search and humans more.

## 2. Weighting: combine after standardisation

These numbers come from the manager session's analysis of the W&B run `su3han7x`. The terminal reward
`R = −per_image_loss` shrinks over training:

| quantity | early | late | change |
|---|---|---|---|
| mean reward (`reward_frac`) | −2.2 | −0.24 | ~10× smaller |
| std of the batch mean (batch 16) | 0.70 | 0.235 | ~3× smaller |

A fixed λ on the raw reward is therefore weak early and dominant late. Standardising both terms
separately avoids this:

```
target_t = z_batch(R) − β · z_batch(c_t),   z_batch(x) = (x − mean_B x) / max(std_B x, eps)
```

- **Unit-free:** β is "cost std per reward std" and keeps its meaning as R shrinks.
- **Per action:** action t's target is lowered by its own `c_t`, not by the trajectory sum.
- **The eps floor** (`std_eps = 1e-2` in cost units; the log cost's batch std is about 1, the linear
  cost's about 0.2) stops a batch of nearly equal saccades from turning rounding noise into unit-scale
  targets.
- **Comparison option `combine='raw'`** (stage 1 only): `target = z(R − λ · Σ_t c_t)`, the cost in
  reward units.
- **Reward mask:** with the reward mask (search `condition='both'`), both z-scores use only the
  target-present rows, as the reward does. Target-absent trials keep only the entropy term.
- **Stage 2:** `z_batch(c_t)` stays a batch statistic; each trial's own β_b scales it:
  `target_{b,t} = z_batch(R)_b − β_b · z_batch(c_t)_b`. A constant β_b = β is stage 1 exactly.

## 3. Where it plugs in (all in the patch)

| file | stage | change |
|---|---|---|
| `harness/config.py` | 1 | `JointPolicyConfig.saccade_cost ∈ {none, log, linear}` (default `none`), `saccade_cost_weight` (β, λ, or β_max in stage 2; default 0), `saccade_cost_combine ∈ {standardized, raw}` |
| | 2 | `saccade_cost_beta_mode ∈ {fixed, conditioned}` (default `fixed`), `saccade_cost_beta_eval` (default 0) |
| `harness/policy/build.py` | 1 | builds `SaccadeCost(r = 1/centers_per_axis)`; refuses an active cost off the terminal-PG path, and a weight without a form |
| | 2 | `conditioned`: widens `film_cond_dim` by `BETA_FEATURES = 2`; refuses it without a FiLM scorer (every task but search), with weight ≤ 0 or with `raw`; refuses a `beta_eval` in `fixed` mode |
| `harness/rollout/engine.py` | 1 | the deferred-credit `traj` also stores `saccade_len = ‖centre_t − centre_{t−1}‖` (t0 FULL view = scene centre) and passes `[B, T]` to `trajectory_loss`; measured only when the cost is on |
| | 2 | `conditioned`: passes the task's `saccade_beta` [B] to `trajectory_loss` (an optional task hook, like `reward_mask`) |
| `harness/policy/joint.py` | 1 | `SaccadeCost` (forms, r, eps), `_batch_z`; `trajectory_loss(..., saccade_len=None)` lowers each step's target, or the reward for `raw`; logs `policy/saccade_len` and `policy/saccade_cost` |
| | 2 | `SaccadeCost.conditioned / beta_eval / sample_beta / features`; `trajectory_loss(..., saccade_beta=None)` takes a per-trial β |
| `search/task.py` | 2 | `bind` draws β per trial (`U[0, weight]`, held for the trajectory) and gives `BoundSearchTask` a cue with β appended plus the `saccade_beta` hook; `_policy_cond` = cue (+ β features; the eval β when none is given) used by the deploy rollout, the policy panel of the figures and the training-time figure (the trial's own β); `checkpoint_metadata` gains a `saccade_cost` entry when the cost is on |
| `search/config.py` | 1, 2 | states `saccade_cost="none"` explicitly in `search_policy_config()`, with both stages' flags in a comment |
| `harness/tests/test_saccade_cost.py`, `search/tests/test_saccade_cost_search.py` | 1, 2 | new; CPU only |

- **Units:** centres are in [−1, 1], so the scene width is 2, r = 1/32 on the 32×32 grid, and
  `linear` = d/2. The search scenes are square, so the same normalisation on both axes is a Euclidean
  distance in the scene.
- **Gating:** the cost exists only for objective `pg` with `reward='terminal'` (the search preset).
  Every other task and objective keeps its path untouched, and the build refuses an active cost there.
  Within search, the cost is gated per trial by the same `reward_mask`.
- **β = 0 with a form set:** saccade lengths are measured and logged, but the target is unchanged.
  This is the sweep's control arm.
- **β features:** `[β, log1p(β / 0.1)]`, appended to the cue embedding. The FiLM generators'
  last layer is zero-initialised (as for the cue), so a fresh conditioned policy starts as the
  unconditioned one and learns how much β should steer it.

## 4. Tests (in the patch)

Stage 1:
- **Off or zero weight is the historical loss, bit for bit:** `torch.equal` on the loss and on
  `dL/dscores`, for log and linear, with and without the reward mask, and for both combinations.
- **A positive β lowers the target of long saccades:** within each step, the shift is monotone in the
  saccade length, has mean 0 (no net push) and population std β.
- **`raw` combination:** equals the historical path fed with `R − λ Σ c`.
- **Cost forms:** d = r gives log 2; linear gives d/2.
- **Through `run_rollout` + `build_policy`** (the tiny foveated model of `test_reward_options`):
  β = 0 trains exactly as without the cost and logs `saccade_len` in (0, 2√2]; β = 0.3 changes the
  scorer gradient; inline PG, QReg, and a weight without a form are refused.

Stage 2:
- **A constant per-trial β is stage 1 at that β, bit for bit:** loss and `dL/dscores` (log, linear,
  with and without the mask), and through `run_rollout` the scorer gradient of a task with
  `saccade_beta ≡ 0.3` equals the fixed-β = 0.3 run.
- **Each trial is shifted by its own β:** shift = β_b × the unit-β shift; β_b = 0 leaves the target.
- **Refusals:** conditioned without a FiLM scorer, with weight 0, with `raw`; `beta_eval` in fixed mode;
  a conditioned loss without the per-trial β.
- **Search, off:** cue, scorer width (`film_cond_dim` = cond_dim), bound task and metadata as before.
- **Search, conditioned:** the scorer's FiLM input is cond_dim + 2; the bound cue is
  `[embedding, features(β_b)]` with β_b ∈ [0, weight]; validation/deploy use the eval β; with the FiLM
  generators off their zero init, eval β 0 and 1 give different policy maps; it trains end to end.
- **Checkpoint round trip:** a conditioned run started from a command line writes `best.pt` with
  `metadata["saccade_cost"]` (form, weight, combine, beta_mode, beta_eval, r) and the run config;
  `harness.evaluate search --opts.ckpt best.pt` rebuilds the widened scorer without drift and
  reproduces the training-time validation exactly; `--rl.saccade-cost-beta-eval 0.8` overrides the
  eval β of that checkpoint.

## 5. Runs

**Stage 1 (sanity):** the exp39 recipe, one seed:

- `--rl.saccade-cost log --rl.saccade-cost-weight 0.3`
- the control `--rl.saccade-cost log --rl.saccade-cost-weight 0`: same code path, only logging

**Stage 2 (one run per cost shape):**

- `--rl.saccade-cost log --rl.saccade-cost-beta-mode conditioned --rl.saccade-cost-weight 1.0
  --rl.saccade-cost-beta-eval 0.3` (β ~ U[0, 1] in training; validation, and so the best checkpoint,
  at β = 0.3)
- the same with `--rl.saccade-cost linear`
- evaluation sweep on one checkpoint: `python -m canvit.harness.evaluate search --opts.ckpt best.pt
  --rl.saccade-cost-beta-eval {0,0.1,0.3,1.0}`

**Log:**
- `policy/saccade_len` and `policy/saccade_cost`;
- `found_t3` for the policy, greedy and random selection (already logged);
- the reward and entropy series as now.

**Afterwards (our side):** `scripts/scanpath/new_checkpoint.sh <best.pt> <name>` scores each checkpoint
on human scanpaths (CS18 TP/TA val + test, FreeView val, the post-hoc cost fit, the summary page).
The question: does the trained cost close the +1.18 bits/fix the post-hoc fit finds, and at what cost
in `found_t3`? For a stage-2 checkpoint the scanpath evaluation must build the scorer's cue through
`SearchRunTask._policy_cond` (it calls `_policy_embed` today, on branch `scanpath-eval`) and take a β
per scoring run; that change is ours, after this lands.

## 6. Stage 2 design notes

- **Why β on the FiLM cue:** the cue is already a per-trial input to every FiLM-modulated stage of
  the scorer (`search/task.py` `policy_cond` → `engine` `ctx.cond` → `selector` `net(feats,
  cond=ctx.cond())`). Appending two β features widens one input instead of adding a scorer input;
  `film_cond_dim` is in the checkpoint's run config, so stage-1 checkpoints and old ones load as before.
- **Sampling:** U[0, weight] from the global torch RNG at `bind`, per rank, only when conditioned (off
  and stage 1 draw nothing, so their RNG streams are unchanged). A mixture with P(β = 0) > 0 or a
  log-uniform draw is a one-line change in `SaccadeCost.sample_beta` if the low-β end needs more mass.
- **Validation selects the best checkpoint at one β** (`saccade_cost_beta_eval`, default 0 = cost-free
  behaviour). Set it to the β of interest for training; evaluation can override it afterwards.

**Alternatives, briefly:**
- **A β schedule** (0 → β over training): cheaper, but confounded with training time. The early,
  cost-free phase shapes the policy differently, and one run still gives one point.
- **A length target** tuned like `entropy_floor_step`: a dual variable drives the mean saccade length
  to a target, for example the human mean. This is descriptive: it makes the length match by
  construction and uses human data in training, so the comparison with humans is no longer a test.

## 7. Overlap with Caglar's branch `caglar` (d7a207f, 2026-11-25)

`caglar` forked from main at 1646f6c3, before `canvit/search/` existed, so it has no search task;
`git apply --3way` of this patch on it conflicts in all four harness files. The overlaps are textual
(the same dataclasses and calls), not semantic: his Q-Prop critic works on the inline PG path
(`glimpse_loss`), the cost on the deferred one (`trajectory_loss`).

| file | `caglar` | this patch | resolution |
|---|---|---|---|
| `harness/config.py` | `JointPolicyConfig.qprop` (after the reward fields) | five `saccade_cost*` fields in the same dataclass | keep both |
| `harness/policy/build.py` | `PG(..., qprop=)`, a second `ViewpointScorer` (critic), `JointPolicy(critic=)` | import, `SaccadeCost` block after `resolve_reward`, `film_cond_dim` widening, `JointPolicy(saccade_cost=)` | keep both; the critic is built without `film_cond_dim`, so a conditioned search run with `qprop` would need the cue there too (not combined today) |
| `harness/policy/joint.py` | `critic` field, DDP sync/state dict of the critic, `crit_all` in `glimpse_loss` | `SaccadeCost`, `_batch_z`, `saccade_cost` field, `trajectory_loss` arguments | keep both; adjacent fields |
| `harness/rollout/engine.py` | threads `local_patches` / `write_snapshots` through the glimpse loop and `select()` | `measure_saccades`/`beta_kw` set-up before the loop, `prev_centre` at t0, `saccade_len` on `traj`, kwargs of the `trajectory_loss` call | keep both; adjacent lines in the same loop |
| `search/task.py`, `search/config.py` | absent on `caglar` | stage 2 | his branch takes them with main |

## 8. Open points for Jonathan

- **Combination and default:** is the standardised combination acceptable, with β = 0.3 as the stage-1
  default?
- **Stage 2 range:** β ~ U[0, 1] and eval/selection at β = 0.3?
- **The first saccade:** it is measured from t0's FULL view at the scene centre. Humans also start at
  the centre in COCO-Search18. Should t = 1 be exempt instead?
- **Gating:** gate the cost by `reward_mask` (as now), or also apply it to target-absent trials (which
  then train on the cost alone)?
