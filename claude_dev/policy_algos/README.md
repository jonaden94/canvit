# policy-algos: new policy-learning algorithms on the exp36_1 setup (started 2026-11-13)

**Bottom line (2026-11-19, all runs done).** Nothing closes the gap to the GT oracle (held-out t4
~0.54). Six learning rules end at t4 0.450-0.454 best-over-evals, 5 seeds each: PG, QReg, dense
QReg, A2C, PPO and SAC-Discrete (multi-step Q is at 0.450). Four scorer changes are all within noise
(3 or 4 U-Net levels, no input BN, local readout). So are 25-50 % of the training data and
augmentation. The only gain is dense QReg (extra glimpsed candidates regressed per state): clearly
better at t1-t3, ~+0.001-0.002 at t4. The policy does not overfit and is not data-limited. The
limit is what can be inferred from the canvas, so ~0.454 at t4 is the practical ceiling for this
frozen backbone + probe. Details: Final results, Follow-up results, SAC results below.

**Why.** exp36_1 found QReg, on-policy PG, Q-Prop and canvas-only Q-Prop tied (t4 ≈ 0.452 mIoU),
while a greedy ground-truth oracle on the same frozen model reaches 0.544, and its held-out A/B
variant suggests most of that gap is real (`claude_dev/policy_oracle/`). So: do other ways of
*learning* the policy close part of it?

**Branch.** `policy-algos`, from **f3d6e6b** (Caglar's branch head = the code of every exp36_1 run),
worktree `repos/canvit-policy-algos`. Not from main: main lacks Caglar's branch, and its fovi port
(27b1bde geometry) differs from the fovi c399d3b geometry the exp22 backbone was trained on.

**Fixed across arms** (= exp36_1): exp22 backbone + exp34 probe, policy_only, 9000 steps, batch 16,
5 glimpses, view scale 2.0, eval every 1000, canvit pin = the branch commit, fovi c399d3b, W&B
`jon_exp36_1_policies_fovi`. **Owner's choices:** canvas-only scorer input (`--rl.feature-groups
ln_feat`), 5 seeds (0-4) per arm, same W&B project.

**Capacity.** Every actor / Q-net is the existing `ViewpointScorer` (5.54M). The A2C/PPO value
function is a second `ViewpointScorer` of the same size (V = mean of its candidate map; the 64-unit
dueling `vhead` would be far smaller than Q-Prop's critic). QTD's target net is a lagged copy (no
extra capacity).

| arm | objective | what is new |
|---|---|---|
| `qreg-lnfeat` | qreg | none -- the baseline the QReg-family arms need (exp36_1 had no canvas-only QReg) |
| `a2c-lnfeat` | `ac` | credit over the episode: GAE(gamma 0.7, lambda 1.0), learned V(s), 1 update per rollout |
| `ppo-lnfeat` | `ac --rl.ppo-epochs 2` | + PPO-clip 0.2, 2 passes per rollout (AdaptiveNN `engine_ppo.py` defaults) |
| `qtd-lnfeat` | `qtd` | Q target = r_t + max Q(s_{t+1}) (double-Q, Polyak target net tau 0.01) instead of r_t |
| `qreg-dense4-lnfeat` | `qreg --rl.qreg-extra-candidates 4` | 4 extra random cells glimpsed (no_grad) and regressed per state: 5 labels/state instead of 1 |

Rewards everywhere: the per-glimpse fractional probe-CE reduction of exp36_1. AC/QTD divide it by a
global running std (no per-depth mean shift, which would not cancel through bootstrapping).
Existing objectives (qreg/pg/vpg/qprop) take their old code paths unchanged.

Comparable exp36_1 arms: `pg-lnfeat` (5 seeds, running 2026-11-13), `qprop-lnfeat` (8), and
default-feature qreg/pg/qprop (7-8).

## Status log

- 2026-11-13 ~04:00: implemented (`rl.py` AC/QTD/gae/ac_loss, `joint.py` episode_loss,
  `engine.py` deferred per-glimpse rewards + dense extra glimpses), 13 new tests
  (`harness/tests/test_policy_algos.py`), launcher `slurm/runs/exp36_1_policies_fovi/algos_in_job.sh`.
- 04:00-04:35: GPU smoke tests of all 5 arms inside job 17284453 (60 steps, short eval): all exit
  0. Speeds there (GPU shared with the oracle, so ~1.4x slower than alone): qreg 1.07 s/step,
  a2c 1.4, ppo 2.3 (after faf7ac4's batched critic; 1.6/2.7 before), qtd 1.17, dense4 2.15.
  Peak reserved GPU memory: ppo 15.2 GB, dense4 12.1 GB. Two jobs on one V100 gave ~1.5x total
  throughput (the oracle slowed 15-30% while a training ran at ~72% of its solo speed).
- 04:38: **runs launched**, pinned **59e76f1** (code = faf7ac4). Work queue
  `logs/jon_exp36_1_policies_fovi/algos_queue/` (25 runs = 5 arms x seeds 0-4, seed-major,
  rotating arm order; `todo/ claimed/ done/ failed/`, worker logs `worker-*.log`), drained by:
  job **17288122** (jupyter V100, 48 h, `algos_queue_job.sbatch`: 2 workers, 2nd starts 15 min
  later) and a 3rd worker in the owner's job **17284453** that starts when the pg-lnfeat sweep
  there logs "sweep finished" (~18:00). That 3rd worker was started from a Claude session INSIDE
  17284453: if that session's step ends, the worker and its current run die -- requeue a stale
  `claimed/` entry by moving it back to `todo/`.
- 08:40: job 17288122 started 05:41. qreg-lnfeat-s0 done (159 min, sharing the GPU). **a2c-s0 and
  ppo-s0 OOMed at their first training step**: the co-running qreg process had grown its PyTorch
  cache to 17.3 GB (a process's cache is never released to the other process) and AC needs ~14.4 GB.
  Non-AC pairs fit (qtd 17.4 + dense4 11.4 GB). Fix, no code change: the 10 a2c/ppo runs moved to a
  separate queue `algos_queue_ac/` (failed s0 retries reset as fresh entries; their dead run dirs are
  `*.failed-<ts>`, their W&B runs show as crashed), drained by ONE solo worker in 17284453 (starts
  after pg-lnfeat, ~15:30; log `algos_queue_ac/worker-17284453-ac.log`); 17288122's two workers keep
  the qreg/qtd/dense4 queue. Estimate (pg-lnfeat solo runs at 0.44 s/step): AC queue ~22 h, ends ~Nov 14
  midday, 17284453 ends Nov 15 01:52.

## Scorer architecture ablation on pg-lnfeat (owner's request 2026-11-13 afternoon)

pg-lnfeat ties every other arm (t4 0.452), so the owner asked whether the scorer needs its
architecture. Two findings from reading `core/policy/net.py`:
- The U-Net pools 32 -> 16 -> 8 -> 4 -> 2 -> 1. With 7x7 depthwise kernels one 8x8-level kernel
  already spans 28 of the 32 canvas cells, so global context exists by the 8x8 level; at 2x2 and
  1x1 the block is mostly zero padding (at 1x1 a per-channel MLP). Those two levels + their
  decoder stages are ~1.7M of the 5.37M parameters.
- BatchNorm is NOT gone: the Frontend's first layer is a per-channel `BatchNorm2d` over the
  input (the body is LayerNorm). So even with `--rl.no-select-bn-eval` (which made PG
  on-policy in TRAINING) the scorer computes a different function at eval (running stats)
  than in training (batch stats). With `ln_feat` input -- already a per-token channel
  LayerNorm, no affine -- the BN has nothing to equalize, and the per-channel affine it adds
  is absorbed by the next 1x1 conv, so removing it (not replacing it by a LayerNorm, which
  would only repeat the input LayerNorm) costs no expressiveness.

New scorer options (defaults = the historical net, bit-identical: same state_dict and same
forward as the pre-change code, train and eval mode): `--rl.policy-levels N` (U-Net bottoms
out at 32 >> (N-1); default 6) and `--rl.policy-input-norm none` (default `bn`). Tests:
`core/tests/test_policy_arch_options.py` + `test_policy_levels_and_input_norm_reach_the_scorer_and_train`.

| arm | change vs pg-lnfeat | params |
|---|---|---|
| `pg-lnfeat-lev4` | U-Net down to 4x4 | 3.63M |
| `pg-lnfeat-lev3` | U-Net down to 8x8 | 2.76M |
| `pg-lnfeat-local` | no U-Net (`--rl.policy-readout local`), score = own cell only | 0.18M |
| `pg-lnfeat-nobn` | no input BatchNorm -> no BatchNorm anywhere, no train/eval gap | 5.37M |

Launcher `slurm/runs/exp36_1_policies_fovi/pg_arch_ablation.sh`, 5 seeds, one run per V100,
on a SELF-REQUEUEING queue job (`chain_queue.sh`, owner's request: no manual resubmission).
The jupyter QOS allows 2 submitted jobs per user and the owner keeps one, so a job cannot
pre-submit a successor; instead it requeues itself (same id, fresh 2-day limit) when its
workers stop for lack of walltime with work left. Verified on a CPU test job: `scontrol
requeue` from inside the batch script works (rc 0); the job restarted ~10 min later with
SLURM_RESTART_COUNT=1 and a fresh time limit (job 17298388).

## Final results (all runs done 2026-11-15; 0 failed)

ADE20K val mIoU, mean ± SE over seeds. "best" = the best of each run's evals (steps 1000-8000)
per glimpse count; "@8000" = the last eval. t0 (no policy) = 0.3767 for every arm; random-grid t4
0.430, held-out GT oracle t4 ~0.54 ([[policy-gt-oracle]] in memory).

| arm | n | best t1 | best t2 | best t3 | best t4 | t4 @8000 |
|---|---|---|---|---|---|---|
| **qreg-dense4-lnfeat** | 5 | **0.4212** | **0.4403** | **0.4480** | **0.4532±0.0006** | **0.4514±0.0008** |
| pg-lnfeat-lev3 | 5 | 0.4192 | 0.4373 | 0.4474 | 0.4526±0.0006 | 0.4508±0.0014 |
| pg-lnfeat | 5 | 0.4198 | 0.4369 | 0.4468 | 0.4523±0.0004 | 0.4510±0.0010 |
| qprop | 8 | 0.4203 | 0.4380 | 0.4468 | 0.4523±0.0004 | 0.4496±0.0006 |
| qprop-lnfeat | 8 | 0.4200 | 0.4381 | 0.4463 | 0.4521±0.0006 | 0.4506±0.0005 |
| pg-lnfeat-nobn | 5 | 0.4188 | 0.4376 | 0.4465 | 0.4520±0.0012 | 0.4507±0.0014 |
| pg-lnfeat-lev4 | 5 | 0.4193 | 0.4361 | 0.4460 | 0.4519±0.0003 | 0.4506±0.0009 |
| pg-lnfeat-local | 5 | 0.4176 | 0.4355 | 0.4449 | 0.4519±0.0003 | 0.4488±0.0014 |
| qreg | 7 | 0.4209 | 0.4382 | 0.4464 | 0.4519±0.0003 | 0.4504±0.0002 |
| pg | 8 | 0.4199 | 0.4377 | 0.4459 | 0.4509±0.0003 | 0.4501±0.0006 |
| a2c-lnfeat | 5 | 0.4189 | 0.4364 | 0.4451 | 0.4508±0.0005 | 0.4487±0.0010 |
| qreg-lnfeat | 5 | 0.4195 | 0.4387 | 0.4463 | 0.4508±0.0007 | 0.4482±0.0004 |
| qtd-lnfeat | 5 | 0.4182 | 0.4356 | 0.4444 | 0.4500±0.0007 | 0.4458±0.0010 |
| ppo-lnfeat | 5 | 0.4172 | 0.4345 | 0.4435 | 0.4497±0.0008 | 0.4451±0.0018 |

Reading:
- Algorithms: only dense QReg (4 extra scored candidates per step) beats the pg-lnfeat baseline,
  clearly at t1-t3 (+0.003 at t2, ~5 SE), within noise at t4. A2C, PPO and multi-step Q are at or
  below the simple baselines; PPO/qtd also lose most by the last eval (less stable).
- Architecture: none of the 4 changes moves t4 beyond noise (all within ±0.0007 of 0.4523). The
  U-Net bottleneck can stop at 8x8 (lev3, 2.8M params) or 4x4 at no cost; dropping the input BN
  costs nothing (and removes the train/eval difference); a purely LOCAL 0.18M-param readout
  reaches the same t4 but is ~0.002 lower at t1-t3 -- global context helps the early glimpses only.
- So the ~0.09 gap to the oracle is not closed by the RL algorithm or by scorer capacity/shape.
- The chain job 17313163 drained all 20 ablation runs in 23 h, so its self-requeue was not
  needed (the mechanism was tested separately with job 17298388).

## Follow-up (owner's go 2026-11-16): data limit or supervision limit?

The algorithm and architecture arms all plateau at t4 ~0.452 vs the oracle's ~0.54. Is that plateau a
limit of the 20k training images, or of how sparse the supervision is?

**1. Train vs val** (no training). The final (step-9000) checkpoints of pg-lnfeat and qreg-dense4-lnfeat
s0-4 are evaluated on 2000 TRAINING images and on val. The greedy GT oracle and the random grid are run
on the same 2000 training images. Compare the share of the random->oracle gap each policy captures on
train vs on val. A much larger share on train means the policy is data-limited.
Caveat: the probe was trained on these training images too, so absolute train numbers are higher. The
gap share is the comparable quantity.
- `policy_eval.py` runs a checkpoint through the training-time validation (`task.evaluate(joint=...)`)
  on any ADE20K root. The standalone `harness.evaluate` cannot deploy a policy (it passes no joint).
  Gate: best.pt of pg-lnfeat-s0 on val reproduces its logged step-7000 eval exactly (V100).
- Training subsets are symlink roots, `logs/jon_exp36_1_policies_fovi/ade20k_subsets/` (README there).
  `trainsub2000` has 2000 training images as its `validation` split.
- Outputs: `logs/jon_exp36_1_policies_fovi/trainval/*.json` (policies) and `logs/jon_exp36_1_oracle_train/`
  (oracle, `merge.txt`).

**2. Training arms** (`slurm/runs/exp36_1_policies_fovi/followup_queue.sh`, 5 seeds each):

| arm | change vs anchor | anchor |
|---|---|---|
| pg-lnfeat-nobn-aug | `--cfg.augment` (dinov3 ADE20K recipe: rescale 0.5-2, 512 crop, flip, photometric) | pg-lnfeat-nobn |
| pg-lnfeat-nobn-frac50 / -frac25 | trains on a fixed 50 % / 25 % of the training images | pg-lnfeat-nobn |
| qreg-dense16-lnfeat | 16 extra candidates per state | qreg-dense4-lnfeat |

- dense32 was dropped. It runs out of memory on a 32 GB V100 (512 glimpses in one forward) and would
  take ~14 h/run even if chunked. Measured: dense4 1.17 s/step, dense16 2.6 s/step (~7 h/run, 22 GB).
- Augmented masks are uint8. The policy reward rejected them, and aug-s0 died at step 0. Fixed in
  6927612; the ADE20K digests are unchanged. The queue is pinned to 6927612. Chain job 17336157 was
  requeued once to pick up the new pin, cutting frac25-s0 after 20 min; it reruns.
- Compute: chain `logs/jon_exp36_1_policies_fovi/chain_followup/` (job 17336157, self-renewing), plus a
  detached worker inside the owner's jupyter job 17334601 (`chain_followup/local/`). That worker runs
  step 1 first, then drains the same queue. It never requeues.

### Follow-up results (2026-11-17; dense16 s4 added 2026-11-19)

**Train vs val.** Final checkpoints, mean of 5 seeds. "Share" = the fraction of the random->oracle gap
closed. The oracle and random grid on the 2000 training images: t1-t4 0.5246/0.5578/0.5784/0.5901 and
0.4469/0.4577/0.4675/0.4721.

| | share t1 | t2 | t3 | t4 |
|---|---|---|---|---|
| pg-lnfeat, training images | 0.17 | 0.20 | 0.17 | 0.16 |
| pg-lnfeat, val | 0.21 | 0.23 | 0.21 | 0.18 |
| dense4, training images | 0.19 | 0.22 | 0.18 | 0.18 |
| dense4, val | 0.25 | 0.25 | 0.21 | 0.18 |

There is no overfitting: the share is not larger on the images the policy trained on.

**Training arms** (best over evals; ± = SE):

| arm | n | t1 | t2 | t3 | t4 | t4 @8000 |
|---|---|---|---|---|---|---|
| pg-lnfeat-nobn (anchor) | 5 | 0.4188 | 0.4376 | 0.4465 | 0.4520±0.0012 | 0.4507±0.0014 |
| + augmentation | 5 | 0.4171 | 0.4361 | 0.4455 | 0.4513±0.0004 | 0.4494±0.0010 |
| 50 % of training images | 5 | 0.4188 | 0.4363 | 0.4457 | 0.4514±0.0006 | 0.4498±0.0007 |
| 25 % of training images | 5 | 0.4189 | 0.4358 | 0.4455 | 0.4513±0.0006 | 0.4500±0.0008 |
| qreg-dense4-lnfeat (anchor) | 5 | 0.4212 | 0.4403 | 0.4480 | 0.4532±0.0006 | 0.4514±0.0008 |
| qreg-dense16-lnfeat | 5 | 0.4218 | 0.4413 | 0.4488 | 0.4540±0.0003 | 0.4517±0.0005 |

**Reading.** The policy is not data-limited: a quarter of the images gives the same result, and
augmentation does not help. Denser supervision helps only marginally: 4x more labels (dense4 ->
dense16) adds ~+0.001 per t, about 1-2 SE. So algorithm, architecture, data and supervision density are
all ruled out as the lever. What remains is information: the oracle knows from the labels where the
probe is wrong, and a policy can only infer that from the canvas. For this frozen backbone and probe,
~0.454 at t4 looks like the practical ceiling.

## SAC-Discrete (owner's request 2026-11-17)

`--rl.objective sac` (`rl.SAC`, `rl.sac_loss`, `JointPolicy._sac_loss`) implements SAC-Discrete
(Christodoulou 2019, arXiv 1910.07207), Algorithm 1:
- critics: twin soft-Q `ViewpointScorer`s of the actor's size (`joint.TwinCritic`) with Polyak targets;
- soft value: the exact expectation over pi (eq. 10);
- actor: the expected objective over all cells (eq. 12);
- temperature: a dual step toward the target entropy (eq. 11).

Departures from the paper, all chosen for comparability with the other arms:
- No replay buffer: one update per rollout on that rollout's own transitions, the same data and
  update budget as PG.
- Rewards are divided by a running std, as for AC/QTD.
- Target critics are Polyak-averaged with tau 0.005 (Haarnoja et al.), not hard-copied every 8000
  steps (the paper's Atari setting is nearly our whole run).

The second paper (SD-SAC, Zhou et al. TMLR 2024, arXiv 2209.10081) fixes long-horizon Atari failure
modes: entropy collapse under deceptive rewards and twin-min underestimation. It is not implemented.

The actor is pg-lnfeat's scorer (5.37M parameters, ln_feat input, no input BN), so the anchor is
pg-lnfeat-nobn. Two arms, 5 seeds each (`slurm/runs/exp36_1_policies_fovi/sac_queue.sh`, queue
`chain_sac`):

| arm | target entropy |
|---|---|
| sac-lnfeat-nobn | the paper's 0.98 * log 256 = 5.43 nats (max 5.55: near-uniform while training) |
| sac-ent1-lnfeat-nobn | 1.0 nat, PG's own entropy target |

Checks: 4 new tests in `test_policy_algos.py`, including a hand-computed loss and the actor
gradient = the expected-objective gradient. All 9 digests (8 task + rollout parity) are identical
to 7f3d16c on the same node. Smoke: 1.3 s/step vs pg's 0.43 (~3.5 h/run), 21.5 GB peak.

### SAC results (all 10 runs done 2026-11-18, 0 failed)

Best over evals; ± = SE; n = 5 each.

| arm | t1 | t2 | t3 | t4 | t4 @8000 |
|---|---|---|---|---|---|
| pg-lnfeat-nobn (anchor) | 0.4188 | 0.4376 | 0.4465 | 0.4520±0.0012 | 0.4507±0.0014 |
| sac-ent1-lnfeat-nobn (1.0 nat) | 0.4185 | 0.4366 | 0.4464 | 0.4523±0.0003 | 0.4504±0.0003 |
| sac-lnfeat-nobn (5.43 nats) | 0.4134 | 0.4317 | 0.4422 | 0.4482±0.0006 | 0.4470±0.0012 |
| qreg-dense16-lnfeat (best arm) | 0.4218 | 0.4413 | 0.4488 | 0.4540±0.0003 | 0.4517±0.0005 |

Mean t4 per eval step (0, 1000, ..., 8000):

| arm | 0 | 1k | 2k | 3k | 4k | 5k | 6k | 7k | 8k |
|---|---|---|---|---|---|---|---|---|---|
| pg-lnfeat-nobn | .4068 | .4466 | .4481 | .4488 | .4481 | .4490 | .4497 | .4487 | .4507 |
| sac-ent1 | .4068 | .4505 | .4506 | .4506 | .4514 | .4502 | .4500 | .4493 | .4504 |
| sac (paper target) | .4068 | .4109 | .4284 | .4394 | .4390 | .4418 | .4446 | .4460 | .4470 |

End-of-training diagnostics (last logged step, all seeds): policy entropy 5.38-5.48 vs target 5.43,
alpha 0.30-0.34 (paper target); 0.78-1.30 vs 1.0, alpha 0.016-0.021 (1.0 nat). Q(s,a) tracks the
TD target in both arms. The method runs as intended.

**Reading.** At PG's entropy target, SAC equals PG at every t. It only gets there sooner (plateau by
step 1000), which does not change the endpoint. At the paper's target it samples near-uniformly while
training. Its critic then learns about near-random sequences, not the greedy ones that are deployed,
so it is 0.004 lower at t4 and still rising at 8000. SAC is the sixth learning rule to plateau at
~0.452-0.454 (PG, QReg, dense QReg, A2C, PPO, SAC). This supports the follow-up's conclusion: the
ceiling comes from the information on the canvas, not from the learning rule.

