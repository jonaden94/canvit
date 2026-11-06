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
  augmentations. Needs `TIME=` and `TRAIN_COMMIT=`.

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
- **Metrics,** per glimpse t (`…_t{t}_{rule}`) and at the last glimpse (`…_final_{rule}`):

| metric | over | meaning |
|---|---|---|
| `loss` | target-present | heatmap KL |
| `hit` | target-present | square of 10% of the scene side around the heatmap argmax overlaps the box (Haoze) |
| `presence_acc`, `recall`, `specificity`, `precision`, `f1` | all / present / absent | presence at p > 0.5 |
| `correct` | all | present: hit and "present"; absent: "absent" |

- **Best checkpoint:** `hit_final_policy`.

## Figures (every 500 steps, `viz.py`)

- **Where:** `{run_dir}/visualization/search_train/` (a training rollout) and
  `search_val_{random,greedy,policy}/` (the first val batch).
- **Layout:** for 4 samples, one column per glimpse and these rows:
  1. fixations so far + the hit square + p(present);
  2. the policy distribution that chose this fixation;
  3. the task heatmap;
  4. a PCA of the canvas.

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
| `hit_box_frac` | 0.1 | hit square side as a fraction of the scene |
| `eval_policies` | random, greedy, policy | |
| `batch_size` / `eval_batch_size` | 16 / 32 | |
| `peak_lr` / `weight_decay` / `warmup_steps` | 1e-4 / 0.01 / 500 | task head, warmup + cosine (Haoze: one AdamW for everything, lr 1e-4, wd 0.01, cosine over 200 epochs, no warmup) |
| `max_steps` / `val_every` / `viz_every` / `viz_samples` | 20000 / 500 / 500 / 4 | |

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
  batch but not the policy's stored canvas); fixed and re-submitted on a V100.
- **Not built:** `null_cell` (one softmax over cells + "absent"); a policy reward on
  target-absent trials; a sampled-policy eval rule; the standalone `SearchEval`; DDP.
