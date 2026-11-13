# policy-algos: new policy-learning algorithms on the exp36_1 setup (started 2026-11-13)

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
