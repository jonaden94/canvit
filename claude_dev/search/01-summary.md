# Visual search in canvit: what is implemented (short version)

Branch `search-v1` (pushed). Code: `canvit/search/`. The long design notes, with the
reasoning behind each choice, are in `00-design.md`; this file only says what exists.

## Run it

```bash
python -m canvit.harness.run search --cfg.model-repo <foveated CanViT .pt / HF dir> --cfg.run-group <group>
```

SLURM, both on a V100 (`jupyter` partition), in fp32 because V100 has no bfloat16:

- `slurm/runs/exp38_search_smoke/search-smoke.sh`: 300-step smoke run.
- `slurm/runs/exp39_search/search-v1-s0.sh`: the first training run, the defaults plus Haoze's
  augmentations, 5 h. Needs `TRAIN_COMMIT=`.

## What the task does

- **Input:** a COCO-Search18 image, letterboxed to 512×512, plus a cue (one of the 18 class
  labels). Target-present and target-absent trials are mixed, about 50/50.
- **Backbone:** a pretrained **foveated** CanViT, **frozen**. Uniform-patcher backbones are refused.
- **Episode:** 4 glimpses. t0 is the image centre; t1–t3 are chosen by the policy.
  Each glimpse updates the canvas (32×32 tokens).
- **Two trainable modules**, both conditioned on the cue by FiLM, trained **jointly**:

| module | input | output | trained by |
|---|---|---|---|
| task head | canvas + cue | 32×32 heatmap "where is the target" + 1 presence logit | supervised loss |
| policy (`ViewpointScorer`) | canvas (LayerNorm'd tokens only) + cue | 32×32 distribution over the next fixation | REINFORCE |

## Data (`data.py`)

- **Target-present:** `images/<cat>/`, boxes from `coco_search18_fixations_TP_*_split{1,2}.json`,
  deduplicated over the 10 subjects: 2163 train / 326 val pairs.
- **Target-absent:** `coco_search18_images_TA/<cat>/` + `coco_search18_fixations_TA_trainval.json`:
  2163 / 326 pairs. Only for split 1.
- **Batches:** a batch dict holds `image`, `target` (cue), `boxes` (xyxy in [-1, 1], NaN padded)
  and `present`.
- **Augmentation:** default is h-flip plus a random crop that always keeps the whole target
  box. The rest is optional.

## Task head (`head.py`)

LayerNorm → 1×1 conv → FiLM → ConvNeXt U-Net (FiLM after every stage, the same blocks as the
policy) → 1×1 conv to one logit per cell. Presence: mean-pool the last feature map → Linear →
1 logit. About 1.3M parameters. `--cfg.head linear` gives a linear head instead.

## Losses (`loss.py`, `task.py`)

- **Heatmap:** KL(target map ‖ softmax(heatmap)), averaged over **target-present trials only**.
  The target map is a Gaussian at the box centre, σ = 2.5/64 of the scene (Haoze).
- **Presence:** focal loss (γ = 2) on the presence logit, over all trials, × 0.75.
- **Total:** heatmap + 0.75 · presence, on the **last glimpse only** by default.

## Policy

- PG (REINFORCE), fixations sampled during training.
- **Reward:** −KL of the heatmap at the last glimpse, given to every action of the episode
  (terminal reward).
  - Standardised per step over the batch (FoAViS).
  - No learned baseline.
  - **Target-absent trials get no reward**: they are masked out of the policy term and out of
    the standardisation, as Haoze does.
- **Entropy:** the coefficient adapts so the entropy stays near 2.11 nats. That is FoAViS's 3.5
  nats on 64×64, scaled to 32×32.
- **At eval:** the policy takes the argmax.

## Validation (every 500 steps)

- **Fixation rules,** on the same val set:
  - `random`: seeded random fixations.
  - `greedy`: the heatmap argmax, never revisiting a cell.
  - `policy`: the learned policy's argmax.
- **Metrics,** keyed `<metric>_t<t>_<rule>` (X = `overlap_box_frac` × 100, default 10):

| metric | over | meaning |
|---|---|---|
| `found` | target-present | the task head's argmax after glimpse t lies inside the target box |
| `found_X_percent` | target-present | a square of side X % of the scene around that argmax overlaps the box (GVSM's fixation box; Haoze's old rule) |
| `hit` | target-present | some fixation 0..t (t0, the centre, included) lies inside the box |
| `hit_X_percent` | target-present | some fixation 0..t has its X % square overlapping the box |
| `correct_<one of the four>` | all | present: located by that criterion and judged "present"; absent: judged "absent" |
| `loss` | target-present | heatmap KL |
| `presence_acc`, `recall`, `specificity`, `precision`, `f1` | all / present / absent / … | presence at p > 0.5 |

- **Where they go:**
  - `{run_dir}/eval.jsonl`: every metric at every glimpse, one line per validation.
  - W&B `eval/`: `loss`, `found`, `hit`, `correct_found`, `presence_acc` at every glimpse;
    the rest at the last glimpse only.
  - W&B `overview/` (20 panels, last glimpse, same names as in `eval/`): `found` and
    `hit` and their X % versions for all three rules, `correct_found` (+ X %) for random
    and policy, `presence_acc` and `loss` for the policy, and `found_gain_…_policy_vs_random`
    (+ X %).
- **Best checkpoint:** `found` at the last glimpse under the policy (`found_t3_policy`).

## Figures (every 500 steps, `viz.py`)

- **Where:** `{run_dir}/visualization/search_train/` (a training rollout) and
  `search_val_{random,greedy,policy}/` (the first val batch).
- **Layout:** 6 samples as blocks in two rows: 3 target-present on top (green tint), 3
  target-absent below (orange tint), each titled with its cue. A block has one column per
  glimpse and these rows, each with the scanpath (white; current fixation = teal dot):
  1. fixations: the task head's argmax as a cross (green = `found`, yellow = only
     `found_X_percent`, red = neither, orange = target absent) and p(present);
  2. the policy distribution that chose this fixation, with its entropy;
  3. the task heatmap after this glimpse, with its KL;
  4. a PCA of the canvas.
- Heatmaps are probabilities (softmax over the 32×32 cells), each panel scaled to its own
  maximum.

## Options

Task config, `--cfg.<name>` (dashes, e.g. `--cfg.presence-loss ce`):

| option | default | choices / meaning |
|---|---|---|
| `model_repo` | required | the foveated backbone |
| `condition` | `both` | `tp` = target-present only (no presence head) |
| `split` | 1 | 2 = the dataset's split2 (needs `condition tp`) |
| `scene_size` | 512 | scene side in px |
| `n_timesteps` | 4 | glimpses per episode, t0 included |
| `foveated_scale` | fixed 2.0 | must match the backbone's pretraining |
| `head` | `unet` | `linear` |
| `head_width` / `head_block_layers` | 64 / 2 | head size |
| `cond_dim` | 64 | cue-embedding width |
| `embedding` | `separate` | `shared` = one cue embedding for head and policy (separate FiLM layers) |
| `detach_policy_embedding` | False | `shared` only: the policy does not train the embedding |
| `dropout` | 0 | before the head's output conv |
| `loss` | `kl_gaussian` | `nll_point` = all mass on the box-centre cell (Tom's comparison) |
| `target_sigma` | 2.5/64 | Gaussian σ as a fraction of the scene |
| `supervise` | `last` | `every` = loss on every glimpse (heatmap and presence) |
| `presence_loss` | `focal` | `ce` |
| `presence_focal_gamma` | 2.0 | |
| `presence_weight` | 0.75 | |
| `overlap_box_frac` | 0.1 | square side (fraction of the scene) of `found_X_percent` / `hit_X_percent`; X follows it |
| `eval_policies` | random, greedy, policy | |
| `batch_size` / `eval_batch_size` | 16 / 32 | |
| `peak_lr` / `weight_decay` / `warmup_steps` | 1e-4 / 0.01 / 500 | task head, warmup + cosine (Haoze: one AdamW for everything, lr 1e-4, wd 0.01, cosine over 200 epochs, no warmup) |
| `max_steps` / `val_every` / `viz_every` / `viz_samples` | 20000 / 500 / 500 / 6 | |

Augmentation, `--cfg.aug.<name>`:

| option | default |
|---|---|
| `hflip_p` | 0.5 |
| `crop` / `crop_scale` | True / (0.6, 1.0); the target box always stays fully inside |
| `color_jitter`, `rotate`, `affine`, `autocontrast` | all False (Haoze's settings when on) |

Policy, `--rl.<name>` (search defaults, not the generic ones):

| option | default | choices |
|---|---|---|
| `objective` | `pg` | `qreg` (no target-absent mask support), `vpg` is refused |
| `reward` | `terminal` | `difference` (KL drop per glimpse), `fractional` (relative drop); both need `supervise every` |
| `reward_center` | `batch_mean` | `none`, `running_mean` |
| `reward_scale` | `batch_std` | `none`, `running_std`, `ema_abs` (Haoze's EMA of \|loss\|, clip ±10) |
| `entropy_target` | 2.11 | nats |
| `entropy_update` | `per_rollout` | `per_glimpse` |
| `alpha_lr`, `entropy_alpha_init/min/max` | 0.01, 1e-3 / 1e-6 / 0.1 | the adaptive entropy coefficient |
| `width` / `block_layers` | 64 / 2 | policy size (≈ the head's) |
| `policy_lr`, `policy_weight_decay` | 2e-4, 1e-2 | |

What trains: `--preset default` = frozen backbone, head + policy jointly. Only `default` is tested
for search.

## Changes to shared code (all other tasks unchanged; the existing fingerprint tests are unchanged)

- **Policy network** (`core/policy/net.py`): optional FiLM conditioning, optional no-BatchNorm input.
- **Policy training** (`harness/policy/`): the reward kind, centring and scaling options; the
  entropy-target knobs; an optional per-trial reward mask.
- **Engine** (`harness/rollout/`): optional task hooks for per-glimpse loss weights, the policy's
  cue input and the reward mask.
- **Fingerprint tests:** three new ones, for the PG and VPG paths and for the scorer weights.
- **Launchers:** `harness_train.sbatch` accepts `TASK=search`.

## Refused combinations (error with the reason)

- `vpg` for search; a non-foveated backbone; more than 1 GPU.
- `reward fractional/difference` with `supervise last`.
- `condition both` with `split 2`.
- PG with a BatchNorm policy network plus eval-mode selection.

## Status

- **Tests:** CPU only (tiny model, synthetic data in the real file format). The last full-suite
  run had 586 passed and 2 failures in a new test; those were fixed and re-run on their own.
  The real backbone loads and evaluates on CPU.
- **GPU smoke run:** job 17122634 (A100) crashed at the first validation (the figure sliced the
  batch but not the policy's stored canvas); fixed. Re-run on a V100 (job 17131243) passed: 300 steps
  in 8 min, hit_final_policy (now `found_10_percent_t3_policy`) 0.10 -> 0.64 by step 200,
  figures written.
- **Not built:** `null_cell` (one softmax over cells + "absent"); a policy reward on
  target-absent trials; a sampled-policy eval rule; the standalone `SearchEval`; DDP.
