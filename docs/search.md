# Visual search (task `search`)

Class-cued visual search on COCO-Search18 with a frozen foveated CanViT: a target-conditioned task head
(heatmap + target present/absent) and a fixation policy, trained jointly. Code: `canvit/search/`. This page
says what exists and how to run it; the design reasoning behind each choice is kept with the project notes
outside the repository.

## Run it

```bash
python -m canvit.harness.run search --cfg.model-repo <foveated CanViT .pt / HF dir> --cfg.run-group <group>
```

SLURM, both on a V100 (`jupyter` partition), in fp32 because V100 has no bfloat16:

- `slurm/runs/exp38_search_smoke/search-smoke.sh`: 300-step smoke run.
- `slurm/runs/exp39_search/search-v1-s0.sh`: the first training run, the defaults plus Haoze's
  augmentations, 5 h. Needs `TRAIN_COMMIT=`.

Evaluate a finished run from its checkpoint alone (the run's config, architecture and policy are
read from `best.pt`; see the README, "Evaluating a checkpoint"):

```bash
python -m canvit.harness.evaluate search --opts.ckpt <run>/checkpoints/best.pt --opts.out <out>.json
```

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

- **Dataset:** the padded release (Haoze's setting) at `datasets/coco_search18`, in its
  restructured layout of 2026-11-06 (the only one supported): all images in a flat `images/`,
  the JSONs in `fixations/`. The 18 classes are a fixed list (`COCO_SEARCH18_CLASSES`).
- **Target-present:** boxes from `fixations/coco_search18_fixations_TP_*_split{1,2}.json`,
  deduplicated over the 10 subjects: 2163 train / 326 val pairs.
- **Target-absent:** `fixations/coco_search18_fixations_TA_trainval.json`: 2163 / 326 pairs.
  Only for split 1.
- **Batches:** a batch dict holds `image`, `target` (cue), `boxes` (xyxy in [-1, 1], NaN padded),
  `present` and `text` (the category name, for the text-cued path).
- **Augmentation:** default is h-flip plus a random crop that always keeps the whole target
  box. The rest is optional.
- **Unpadded COCO-Search18** (`datasets/visual_search/coco_search18_nopadding`, the COCO originals
  with mapped boxes) has the same layout and loads through the same adapter with
  `--cfg.data-root <that folder>`; it is letterboxed like every other picture.

### Curated release (`--cfg.dataset release`, `release.py`): loader only, no model yet

The visual-search training set built by get_datasets, read from a frozen release folder
(`--cfg.release-dir`, default `datasets/visual_search/search_release/basic-v1`; the spec is the
folder's `README.md`). The cue of a trial is its **text** (`batch["text"]`, a list of str); there
is no class index. The search model has no text encoder yet, so building a model on the release
is refused; only the loaders exist.

- **Pictures:** one `os.pread` per picture from the pack shards, PIL `draft` (decoded no smaller
  than the scene needs), the release's orientation fix. Same scene transform as COCO-Search18.
- **Training order:** the release's reference epoch sampler, vendored as `release_sampler.py`
  (reproduces the reference epochs row for row; `test_release.py`). An epoch (~349k trials, half
  target-absent) is built in the main process (~9 GB of RAM, ~16 s); the epochs follow each other
  as one endless stream, fixed by `--cfg.seed`. A resume at step s continues at trial
  s x batch_size of that stream. Each rank takes `trials[rank::world]` (DDP itself is still off
  for search).
- **Workers** are spawned, not forked (a forked worker would copy the 9 GB sampler). Throughput
  with the default augmentation at 512 px: ~30 samples/s per worker, 390 with 16 workers.
- **Validation:** all of `val.parquet` (7,364 present + 7,364 absent trials); the last glimpse's
  `found`, `correct_found` and `presence_acc` also per class-size band
  (`..._band{lt10,10-99,100-299,300-999,1k-10k,ge10k}`, n = training examples of the class).

### Visual prompts (`--cfg.visual-prompts`): loader only

On a release with visual prompts (from `basic-v2` on; spec section 7 of its README), every sample also carries
the trial's visual prompt: `vp_feat` [14, 768] (DINOv3 ViT-B cell features of another instance of the prompt, from
another picture: 1x1, 2x2 and 3x3 grids on its mask's box), `vp_cov` [14] (mask coverage per cell) and `vp_valid`
(about 77 % of trials have one; otherwise zeros). Training trials take the sampler's `vp_idx`, val the fixed one of
`val.parquet`. `vp_features.npy` is read by mmap in each worker. On COCO-Search18 the flag is for evaluation only:
each trial gets a fixed prompt of its category from the release's `vp_test` (the box with the smallest
`blake2b(f"{image}|{category}|{box_id}", digest_size=8)`); training on COCO-Search18 with it is refused. Off (the
default), samples are exactly as before.

### Text encoder (`text_encoder.py`): built, not wired into the model yet

`TextEncoder()(batch["text"]) -> [B, 768]`: Talk2DINO for DINOv3 ViT-B, frozen. CLIP ViT-B/16's text tower
(OpenAI's weights via the transformers port `openai/clip-vit-base-patch16`, HF cache) and Talk2DINO's projection
(`dino_stuff/Talk2DINO/weights/dinov3_vitb_mlp_infonce.pth`, sha256-checked) to the DINOv3 ViT-B patch-feature
space (Talk2DINO was trained against the same DINOv3 weights as our teacher). Computed every step in fp32 without
gradients, ~7 ms per batch of 16-64 texts on an A100; `cache=True` keeps each distinct text's vector. Not
normalised. COCO-Search18 batches carry `text` = the category name, so the zero-shot test uses the same path.
How the vector conditions the head and the policy is not decided yet.

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
  - W&B `overview/` (29 panels, same names as in `eval/`). At the last glimpse: `found`
    and `hit` and their X % versions for all three rules, `correct_found` (+ X %) for random
    and policy, `presence_acc` and `loss` for the policy, and `found_gain_…_policy_vs_random`
    (+ X %). The development over the earlier glimpses: `found` (policy, random) and `hit`
    (policy).
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
| `dataset` | `coco_search18` | `release` = the curated release (`release_dir`); loader only so far |
| `release_dir` | `.../search_release/basic-v1` | the release folder |
| `visual_prompts` | False | samples carry `vp_feat` / `vp_cov` / `vp_valid` (release with visual prompts; COCO-Search18: eval only) |
| `retina` | None | a multi-retina backbone is used through this one retina (fixed for training and eval), by name, e.g. `cmf0.75_res65_cps6`; required for such a backbone, refused otherwise. The same option exists for `ade20k` and `in1k` |
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

## Refused combinations (error with the reason)

- `vpg` for search; a non-foveated backbone; more than 1 GPU.
- A model on `dataset release` (text cue, no text encoder yet).
- `reward fractional/difference` with `supervise last`.
- `condition both` with `split 2`.
- PG with a BatchNorm policy network plus eval-mode selection.

## Not built yet

- `null_cell` (one softmax over cells + "absent").
- A policy reward on target-absent trials.
- A sampled-policy eval rule.
- DDP.
