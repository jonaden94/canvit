# Proposal: optional per-saccade cost in the RL reward

> Written by Claude (AI, Claude Opus 5.5), 2026-11-25, for Valentin, as a proposal for Jonathan (and his Claude).
> Nothing here is applied on any branch. `saccade_cost.patch` is a `git diff` against `main` 40c1927 and
> applies cleanly (`git apply --check`). It is default-off. Its 12 CPU tests pass, as do the existing
> `test_task_digests.py`, `test_policy_loss_scale.py`, `test_reward_options.py`, `test_vpg.py`,
> `test_rollout_engine.py` and `canvit/search/tests`.

## TL;DR

- **Problem:** the search policy makes saccades that are too long. Scored on human COCO-Search18 and
  COCO-FreeView scanpaths, a cost on saccade length that we fit afterwards adds +1.18 bits/fix (log cost,
  mean of CS18 TP val, TA val and FreeView val; `canvit/scanpath/saccade_cost.py`).
- **Proposal:** let RL pay for each saccade. `c(d) = log(1 + d/r)`, where `d` is the distance between
  consecutive glimpse centres and `r` is half a policy-grid cell. Option `linear` is `d/W`. The cost enters
  after standardisation: `target_t = z_batch(R) − β · z_batch(c_t)`. Each action carries its own cost.
- **Stage 1 (the patch):** two runs, log cost with β = 0.3 and the control β = 0.
- **Stage 2 (recommended):** a β-conditioned policy. β is drawn per trial and given to the scorer through
  the same FiLM route as the search cue. One training run per cost shape then covers every β at
  evaluation. It is described in pseudo-code below, because it also touches the search deploy path.

## 1. Why a cost, why this form

- **What the fits show:** at τ = 1 the policy's spatial entropy is already human-level. What it lacks
  is the distance bias: humans make short saccades and the policy does not. All post-hoc fits use the
  policy's own predictions times a distance term:
  - the log cost fits best on all three datasets (log a = −1.80 ± 0.009 for the main checkpoint);
  - the linear cost adds about 0.03 on top of the log one;
  - a quadratic cost is worst.
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
- **Comparison option `combine='raw'`:** `target = z(R − λ · Σ_t c_t)`, with the cost in reward units.
- **Reward mask:** with the reward mask (search `condition='both'`), both z-scores use only the
  target-present rows, as the reward does. Target-absent trials keep only the entropy term.

## 3. Where it plugs in (all in the patch)

| file | change |
|---|---|
| `harness/config.py` | `JointPolicyConfig.saccade_cost ∈ {none, log, linear}` (default `none`), `saccade_cost_weight` (β or λ, default 0), `saccade_cost_combine ∈ {standardized, raw}` |
| `harness/policy/build.py` | builds `SaccadeCost(r = 1/centers_per_axis)`; refuses an active cost off the terminal-PG path, and a weight without a form |
| `harness/rollout/engine.py` | the deferred-credit `traj` also stores `saccade_len = ‖centre_t − centre_{t−1}‖` (t0 FULL view = scene centre) and passes `[B, T]` to `trajectory_loss`; measured only when the cost is on |
| `harness/policy/joint.py` | `SaccadeCost` (forms, r, eps), `_batch_z`; `trajectory_loss(..., saccade_len=None)` lowers each step's target, or the reward for `raw`; logs `policy/saccade_len` and `policy/saccade_cost` |
| `search/config.py` | states `saccade_cost="none"` explicitly in `search_policy_config()`, with the sweep flags in a comment |
| `harness/tests/test_saccade_cost.py` | new; CPU only |

- **Units:** centres are in [−1, 1], so the scene width is 2, r = 1/32 on the 32×32 grid, and
  `linear` = d/2. The search scenes are square, so the same normalisation on both axes is a Euclidean
  distance in the scene.
- **Gating:** the cost exists only for objective `pg` with `reward='terminal'` (the search preset).
  Every other task and objective keeps its path untouched, and the build refuses an active cost there.
  Within search, the cost is gated per trial by the same `reward_mask`.
- **β = 0 with a form set:** saccade lengths are measured and logged, but the target is unchanged.
  This is the sweep's control arm.

## 4. Tests (in the patch)

- **Off or zero weight is the historical loss, bit for bit:** `torch.equal` on the loss and on
  `dL/dscores`, for log and linear, with and without the reward mask, and for both combinations.
- **A positive β lowers the target of long saccades:** within each step, the shift is monotone in the
  saccade length, has mean 0 (no net push) and population std β.
- **`raw` combination:** equals the historical path fed with `R − λ Σ c`.
- **Cost forms:** d = r gives log 2; linear gives d/2.
- **Through `run_rollout` + `build_policy`** (the tiny foveated model of `test_reward_options`):
  - β = 0 trains exactly as without the cost and logs `saccade_len` in (0, 2√2];
  - β = 0.3 changes the scorer gradient;
  - inline PG, QReg, and a weight without a form are all refused.

## 5. Runs

**Stage 1 (sanity, the patch as is):** the exp39 recipe, one seed:

- `--rl.saccade-cost log --rl.saccade-cost-weight 0.3`
- the control `--rl.saccade-cost log --rl.saccade-cost-weight 0`: same code path, only logging

If budget allows, the full grid is β ∈ {0, 0.1, 0.3, 1.0} × {log, linear}, ideally on the multi-retina
backbone with the retinas varying (as exp40).

**Log:**
- `policy/saccade_len` and `policy/saccade_cost`;
- `found_t3` for the policy, greedy and random selection (already logged);
- the reward and entropy series as now.

**Afterwards (our side):** `scripts/scanpath/new_checkpoint.sh <best.pt> <name>` scores each checkpoint
on human scanpaths (CS18 TP/TA val + test, FreeView val, the post-hoc cost fit, the summary page).
The question: does the trained cost close the +1.18 bits/fix the post-hoc fit finds, and at what cost
in `found_t3`?

## 6. Stage 2 (recommended): one β-conditioned policy per cost shape

**Idea:** sample β per trial during training, give it to the policy as an input, and train once per
cost shape (log, linear). At evaluation, sweep β on the same checkpoint. This gives the whole
search/human-likeness trade-off curve from one run instead of four.

**How the search cue enters the policy today** (main 40c1927):

- `search/task.py:240`: `build_policy(..., film_cond_dim=cfg.cond_dim)` builds a FiLM-conditioned
  scorer (`core/policy/net.py:304`: `film_in` + `build_unet_film`).
- `search/task.py:78`: the bound task sets `policy_cond = lambda: policy_embed(targets)` ([B, cond_dim]).
- `engine.py`: `ctx.cond = task.policy_cond`.
- `selector.py:122-137`: the selector calls `net(feats, cond=ctx.cond())` on every glimpse.
- `search/task.py:325, 445`: the eval and deploy paths call `embed(targets)` directly.

**Minimal hook:** append β features to the cue. This needs no new scorer input, only a wider FiLM
conditioning vector.

```python
# config:     saccade_cost_beta: Literal["fixed", "sampled"] = "fixed"; saccade_cost_beta_max = 1.0
# search build_policy:  film_cond_dim = cfg.cond_dim + K_BETA          # K_BETA = 2
def beta_features(beta):            # [B] -> [B, 2]; beta = 0 is a real point
    return torch.stack([beta, torch.log1p(beta / 0.1)], dim=-1)

# BoundSearchTask.__init__ (training): one beta per trial, held for the whole trajectory
self.saccade_beta = beta_max * torch.rand(B)        # U[0, beta_max]; or a log-uniform mix with
                                                    # P(beta = 0) = 0.2
self.policy_cond = lambda: torch.cat([policy_embed(targets), beta_features(self.saccade_beta)], -1)

# engine: also pass the per-trial weight      (getattr(task, "saccade_beta", None) -> [B])
# joint.trajectory_loss: weight may be a [B] tensor instead of the scalar sc.weight
target = target - beta_b * _batch_z(cost[:, t], mask, eps=sc.std_eps)   # beta_b: [B]

# eval / deploy (search/task.py:325, 445; the scanpath GridScanpathModel): a fixed beta for all trials
cond = torch.cat([embed(targets), beta_features(torch.full((B,), beta_eval))], -1)
```

**Notes:**
- `z_batch(c_t)` stays a batch statistic. Each trial's own β then decides how much its cost counts.
- The scanpath evaluation gets `--cfg.saccade-beta` and scores a grid of β per checkpoint.
- Checkpoints trained with fixed β (stage 1) stay loadable: `film_cond_dim` is in the checkpoint's
  model config.

**Alternatives, briefly:**
- **A β schedule** (0 → β over training): cheaper, but confounded with training time. The early,
  cost-free phase shapes the policy differently, and one run still gives one point.
- **A length target** tuned like `entropy_floor_step`: a dual variable drives the mean saccade length
  to a target, for example the human mean. This is descriptive: it makes the length match by
  construction and uses human data in training, so the comparison with humans is no longer a test.

## 7. Open points for Jonathan

- **Combination and default:** is the standardised combination acceptable, with β = 0.3 as the stage-1
  default?
- **The first saccade:** it is measured from t0's FULL view at the scene centre. Humans also start at
  the centre in COCO-Search18. Should t = 1 be exempt instead?
- **Gating:** gate the cost by `reward_mask` (as now), or also apply it to target-absent trials (which
  then train on the cost alone)?
