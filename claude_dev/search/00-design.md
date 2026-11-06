# Visual search in canvit — v1 design (2026-11-05)

Status: **design, not implemented.** Agreed with the owner in conversation on 2026-11-05.
Reference implementation: `autoregressive_detection`, branch `haoze-wip` (Haoze Zhu,
2026-11-02). Porting rule from the owner: **take Haoze's defaults, but adapt sizes and
resolutions to what is natural for canvit.** Every formulation choice the owner named (loss,
supervision, reward, normalisation, embeddings, augmentations) is a config option, not a
hard-coded decision.

## 1. What v1 is

Class-cued visual search on COCO-Search18, target-present (TP) trials only. A frozen CanViT
ViT-B backbone (foveated patcher only) builds the canvas from a sequence of fixations. Two small
trainable modules sit on top of it:

- a **task head**, which maps the canvas and the target class to a heatmap over the 32×32
  canvas grid ("where is the target");
- a **policy** (the existing `ViewpointScorer`), which maps the canvas and the target class to a
  distribution over the next fixation on the same 32×32 grid.

Both are conditioned on the target class by FiLM. The default trains the task head and the
policy **jointly**, with the backbone frozen.

Out of scope for v1: image cues, the uniform patcher, DDP, human-scanpath metrics, learned
value baselines. Target-absent (TA) trials and a presence head were added on 2026-11-06 once
the TA images were downloaded: see §12.

## 2. Data

### 2.1 What is on disk

`/user/henrich1/u25995/jonathan/datasets/coco_search18`:

- `images/<category>/<name>.jpg`: 3101 TP images, 1680×1050, already zero-padded by the
  dataset authors.
- `coco_search18_fixations_TP_{train,valid}_split1.json`: one record per (subject, trial), with
  `name`, `task` (category), `bbox` `[x, y, w, h]` in 1680×1050 pixels.
- Split 1 gives **2163 unique (image, target) train pairs and 326 val pairs**. The 21.6k
  records are 10 subjects on those pairs, and the bbox is identical across subjects
  (checked), so we deduplicate on `(name, task)`.
- The TA fixation JSON (`coco_search18_fixations_TA_trainval.json`) has no bbox and a `split`
  field with the same 2163 / 326 pairs as TP split1. Its images were NOT in the original
  download; `COCOSearch18-images-TA.zip` (same server) was fetched on 2026-11-06 and extracted
  to `coco_search18_images_TA/<category>/` (3101 images, 1680×1050; all 2489 train/val TA pairs
  present). The test split is withheld.

### 2.2 Sample contract (dataset-independent)

Everything downstream of the adapter sees only:

```
image:  float [3, S, S]   ImageNet-normalized, letterboxed to the square scene (S = 512)
target: long  []          class index into the dataset's class list
boxes:  float [N, 4]      target boxes in scene-normalized [-1, 1] xyxy (N = 1 TP, N = 0 TA)
present: bool []          target present (§12)
```

A new search dataset needs only a new adapter that produces this. The class vocabulary comes
from the adapter (18 classes here) and sizes the embedding tables. This one coupling to the
dataset is inherent to label cues.

### 2.3 Augmentation (train only)

Implemented with `torchvision.transforms.v2`, which transforms `tv_tensors.BoundingBoxes`
together with the image. Albumentations is not installed in our venvs, and v2 needs no new
dependency. The transforms apply to the 1680×1050 image, **before** letterboxing. Every one
is a flag.

| aug | default | from |
|---|---|---|
| horizontal flip | on, p 0.5 | Haoze |
| random scaled crop | on, scale (0.6, 1.0), aspect kept | owner asked for it; the range is our choice |
| colour jitter (b/c/s 0.4, hue 0.1, p 0.8) + greyscale p 0.1 | off | Haoze |
| rotation ±60°, affine (rot ±45°, translate ±10 %, scale 0.8–1.2, shear ±15°) | off | Haoze |
| autocontrast p 0.5 | off | Haoze |

**Target retention.** The full geometric chain is checked, not only the crop. A sample is
kept only if at least `min_box_fraction` (default 1.0) of the original box area survives inside
the frame. Otherwise the chain is redrawn, up to `max_aug_tries` (default 10), and then the
un-augmented sample is used. Rotation turns the box into the axis-aligned box of the rotated
box, which inflates it. That is accepted.

## 3. Model

### 3.1 Wrapper

`canvit/search/model.py::CanViTForVisualSearch` holds `canvit` (bare CanViT, loaded like the
ade20k wrapper), `head` (§3.3) and the embeddings (§3.2). The scorer belongs to the
`JointPolicy` as on every other task.

### 3.2 Target embeddings

`nn.Embedding(n_classes, cond_dim)`, `cond_dim = 64` (Haoze). Flag
`embedding: separate | shared`:

- `separate` (default): one embedding for the task head, one for the policy.
- `shared`: one embedding read by both, each with its own FiLM projection MLPs. The task loss
  and the policy loss both train it. With `detach_policy_embedding: true` the policy reads a
  detached copy, so only the task loss trains it.

*As implemented:* the task embedding is `SearchHead.embed`. Under `separate` the policy's
embedding is attached to the scorer module as `scorer.target_embed`, so the harness treats it
as policy parameters everywhere (requires_grad, gradient clipping, DDP sync, the policy
checkpoint) with no extra plumbing. Under `shared` the policy reads `head.embed` (detached if
`detach_policy_embedding`), and the head's optimizer group steps it.

### 3.3 Task head

`canvit/search/head.py::SearchHead`, selected by `head: unet | linear`:

- `unet` (default): LN → 1×1 conv 768→w → **the scorer's ConvNeXt-V2 U-Net** (same blocks,
  6 encoder stages 32→1 with average pooling, 5 decoder stages with skip concatenation, one
  head stage) → 1×1 conv → logits `[B, 32, 32]`. FiLM is applied after the input projection
  and after every encoder and decoder stage, as Haoze conditions every backbone level.
- `linear`: LN → 1×1 conv 768→w → FiLM → 1×1 conv. The cheap sanity baseline.

Width and blocks per stage are config options. The default is `w = 64` with 2 blocks per
stage. At that size the scorer has 1.05M parameters (5.4M at its default of 128 wide with 3
blocks per stage), and the task head should come out similar. Haoze's 64×64 heatmap over 640 px
becomes canvit's native **32×32, one logit per canvas token**.

**FiLM**, as in Haoze (`FiLMGenMLP`, `apply_generic_film`): per stage, an MLP
(Linear → ReLU → Linear, hidden width max(128, 2·cond_dim)) maps the embedding to γ and β, and
the stage output becomes `(1 + γ)·GroupNorm(x) + β`, with GroupNorm over min(8, channels)
groups. The MLP's last layer is zero-initialised, so at init each FiLM stage outputs
`GroupNorm(x)` whatever the target: the conditioning starts out independent of the target
(it is not the identity, because the GroupNorm is always applied).

**Shared code, not shared weights.** The U-Net forward is factored into a helper in
`core/policy/net.py`, used by both `ViewpointScorer` and `SearchHead`. The scorer keeps its
attribute names (`enc`, `dec`, `head`), so published scorer checkpoints still load.

### 3.4 Policy

The existing `ViewpointScorer` with:

- `feature_groups = ("ln_feat",)`: the plain canvas, with a per-token LayerNorm only.
- `action_space = "fixation"` and `centers_per_axis = 32`. Verified 2026-11-05: at 32 the
  candidate `grid_sample` reproduces the 32×32 score map exactly (max abs diff 0.0), so one
  action is one canvas cell.
- **New:** optional FiLM conditioning, `ViewpointScorer(..., film_cond_dim: int | None = None)`
  and `forward(feats, cond=None)`. FiLM sits at the same places as in the task head. With
  `film_cond_dim=None` the module, its state_dict, its config.json and its output are
  unchanged, and old config.json files without the key load as `None`.
- **No input BatchNorm.** The scorer's only BatchNorm (`Frontend.bn`) exists to equalise the
  scales of the handcrafted feature groups that the ade20k and distill policies read: the
  change and entropy maps, a few channels at scales like 1e-3, next to the 768 canvas
  channels. Search reads none of those, so it does not need it. The class gets a constructor
  argument `input_norm: batch | none`, set by the task and not exposed as a search config
  option. ade20k and distill keep `batch`, which leaves their behaviour and the published
  policy checkpoints (which contain BatchNorm weights) unchanged. Search sets `none`. Its only input, `ln_feat`, is already LayerNorm'd
  per token (`features._lnc`), and the frontend applies another LayerNorm after the input
  projection. With `none` the scorer has no train/eval difference at all: an action is
  sampled from the same forward pass its log-probability is read from, as in FoAViS, and the
  choice for one image does not depend on the rest of the batch.
- Same defaults as the task head (64 / 2).

t0 is the existing foveated FULL viewpoint: a centred fixation at the fixed view scale, like
the human trials and Haoze's start location.

### 3.5 Glimpse geometry

Foveated patcher only. The search task **refuses** a uniform-patcher `model_repo`. View scale
is fixed at `foveated_scale.fixed_scale` (default 2.0), with the same "must match pretraining
scale" warning as ade20k. Default `n_glimpses = 4`: t0 plus 3 chosen, as in Haoze.

## 4. Task loss

`per_image_loss(readout) -> [B]` is the per-sample loss, and also the raw material of the
policy reward. `step_loss` is its batch mean.

- `loss: kl_gaussian | nll_point`.
  - `kl_gaussian` (default, Haoze): KL(target ‖ softmax over the 32×32 logits). The target is
    a normalised Gaussian per box centre; boxes are summed and renormalised. σ is a fraction of
    the scene side: `target_sigma = 2.5/64 ≈ 0.039` (Haoze's σ of 2.5 cells on his 64×64
    heatmap), i.e. 1.25 cells on our grid.
  - `nll_point`: cross-entropy on the cell containing the box centre. This is the σ → 0 limit.
    It is the option from Tom's comparison (`external_info_provided_by_jonathan/toms_loss_vergleich.md`).
    His result was measured on next-human-fixation prediction and scored at the exact pixel,
    which favours sharp targets. It may not carry over to box localisation, and that is why
    it is a flag.
- `supervise: last | every`. `last` (default, Haoze): only the last glimpse's loss trains the
  head. `every`: the mean over all glimpses.

Engine support for `last`: an optional task hook `step_weight(t, n_glimpses) -> float`
scales each glimpse's task loss. If the hook is absent the weight is 1.0, guarded by `== 1.0`,
so the existing tasks hit the identical expression and the digests cannot move.

## 5. Policy training

### 5.1 Objective

Default: canvit's existing `PG`, i.e. REINFORCE with softmax sampling and **no learned
baseline**. QReg stays available. VPG (learned baseline) is refused for search in v1: its
`vhead` reads pooled, unconditioned features, so it cannot be target-aware, and the owner does
not want learned baselines for now.

### 5.2 Reward, centring and scaling: engine-level flags for every task

New `JointPolicyConfig` fields. `default` resolves to each objective's current behaviour, so
no existing run changes.

| flag | values | `default` resolves to |
|---|---|---|
| `reward` | `fractional` = (L_{t-1}−L_t)/L_{t-1}; `difference` = L_{t-1}−L_t (FoAViS); `terminal` = −L_T for every action (Haoze, VPG) | qreg/pg → `fractional`, vpg → `terminal` |
| `reward_center` | `none`, `running_mean` (EMA, per depth), `batch_mean` (per step over the batch, FoAViS) | qreg/pg → `running_mean` |
| `reward_scale` | `none`, `running_std` (EMA, per depth), `batch_std` (per step over the batch, FoAViS), `ema_abs` (÷ EMA of mean abs reward, clip ±10, Haoze) | qreg/pg → `running_std` (pg `z_subtract_only` → `none`) |

`running_mean` + `running_std` is exactly today's `RunningNorm`. FoAViS's `per_step` is
`batch_mean` + `batch_std`. Haoze's normalisation without his learned baseline is `none` (or
`running_mean`) + `ema_abs`.

**Search defaults:** `reward = terminal`, `reward_center = batch_mean`, `reward_scale = batch_std`.

Engine mechanics:

- `fractional` and `difference` are credited inline per glimpse (today's path). `difference`
  is a new expression next to the existing one.
- `terminal` with PG uses the deferred path VPG already has: buffer `(scores, flat_idx)` per
  step and build the loss at the last glimpse. `JointPolicy.defers_credit` becomes "VPG or
  reward == terminal".
- The policy loss stays the mean over the T−1 decision steps, as the harness does today. The
  `n_glimpses/(n_glimpses−1)` rescale is unchanged.

### 5.3 Entropy target

This is the existing PG dual ascent, `log α += η(H* − H̄)` with α clipped. It is the same
update as FoAViS Appendix J, written in log space. New knobs, all `None` = today's behaviour:

| knob | today | search default (FoAViS, adapted) |
|---|---|---|
| `entropy_target` H* | 1.0 nats | **2.11 nats** = 3.5 − ln 4 |
| `alpha_lr` η | 0.05 | 0.01 |
| `entropy_alpha_init` | = `entropy_bonus` (0.01) | 1e-3 |
| `entropy_alpha_min` / `_max` | 0.01 / 10 | 1e-6 / 0.1 |
| `entropy_update` | `per_glimpse` | `per_rollout` |

- **Why 2.11 nats.** FoAViS used H* = 3.5 nats on a 64×64 grid. On 32×32, the same spatial
  spread is H* − ln(4096/1024) = 2.11 nats: e^H ≈ 8.3 effective cells, about 0.8 % of the image
  area, as in FoAViS.
- **Why `per_rollout`.** Today's PG takes one dual step per glimpse, T−1 steps per training
  step, using the pre-update α. FoAViS takes one step per training step on the mean entropy
  over the decision steps. `per_rollout` reproduces FoAViS. `per_glimpse` stays the default
  elsewhere.
- This is a floor in both implementations: the bonus only raises entropy. The owner reports
  that in FoAViS it held entropy at the target in practice, because REINFORCE sharpens the
  policy and the shrinking weight balances that.
- `log_alpha` is already checkpointed. It is per rank under DDP, which is irrelevant here,
  since search is single-GPU.

### 5.4 Conditioning plumbing

The bound search task exposes `policy_cond() -> [B, cond_dim]`, the policy embedding of this
batch's targets. It is recomputed on each call, so no graph is reused across a chunk boundary.
*As implemented:* `RolloutCtx` has a field `cond` (default None). The engine sets
`ctx.cond = task.policy_cond` after `start_rollout` when the task has the hook, and
`PolicySelector.select` calls `self.net(feats, cond=ctx.cond())` when it is set. No selector's
`start_rollout` signature changed. `deploy_rollout_viewpoints` takes `cond=` and sets it the
same way. With no hook the scorer calls are the historical ones.

### 5.5 BatchNorm and sampling

With a BatchNorm in the scorer, `select_bn_eval=True` (the harness default) picks the action
from an eval-mode forward pass but differentiates the log-probability of a train-mode pass. For
an objective that samples its action (PG, VPG), that makes the REINFORCE gradient off-policy,
because the two passes rank actions very differently. Search avoids the question: with
`input_norm = none` the scorer has no BatchNorm, so both passes are the same computation and
`select_bn_eval` is ignored. See §11 for the existing PG path.

## 6. Spec and run defaults

Default spec (search's own; the generic `joint` preset trains the backbone):
`train_backbone=False`, `train_head=True`, `train_policy=True`, `task_weight=1`,
`policy_weight=1`, `bptt=none`, fixed horizon `n_glimpses`. Presets `probe` (head only, random
fixations) and `policy_only` (frozen head) work through the existing spec machinery.

| setting | default | note |
|---|---|---|
| scene / canvas | 512 px / 32×32 | canvit native |
| batch size | 16 | ade20k's; 2163 samples ≈ 135 steps/epoch |
| head optimiser | AdamW lr 1e-4, wd 0.01, warmup_cosine | Haoze's lr/wd (his search config; 0.05 was his COCO-pretraining config, corrected 2026-11-06) |
| policy optimiser | `JointPolicyConfig` defaults (lr 2e-4, wd 1e-2, betas 0.9/0.95) | existing recipe |
| max steps / val every | 20000 / 500 | about 150 epochs; Haoze's search config sets 200 |
| dropout | 0.0 (flag) | owner: not by default |
| DDP | refused (`supports_ddp=False`) | map-style dataset, as ade20k |

## 7. Evaluation (minimal)

`evaluate()` runs the val set at `n_glimpses` under an eval policy:

- `random`: fixed seed per eval.
- `greedy`: fixate the argmax of the current task heatmap, never revisiting a cell. A
  zero-training baseline the learned policy must beat.
- `policy`: argmax of the scorer.

Per timestep it reports `loss_t` and `hit_t`. `hit` means Haoze's criterion: a square box of
side 0.1·scene (64 px at 640) around the heatmap argmax intersects a target box.
`best_metric = hit` at the last glimpse. With TA trials (§12) `loss` and `hit` average over the
TP trials only, and the presence metrics are added.

Differences from Haoze's validation (checked 2026-11-05 in `haoze-wip`'s `losses.py`): he
reports only the last glimpse, on TP rows, plus L1 to the box centre, the RL loss terms and the
presence metrics; and his validation SAMPLES fixations from the policy (no greedy switch), where
our `policy` rule takes the argmax. So `hit_final_policy` is not directly comparable to his
accuracy. The owner accepted this (2026-11-06); a sampled rule can be added if needed.

**Figures** (`canvit/search/viz.py`, `viz_every`, default 500): per glimpse and sample, the
fixations with the hit square, the policy distribution that chose the fixation (recomputed from
the previous canvas; a test checks its argmax is the fixated cell), the task heatmap, and a
PCA of the canvas. Written for the training rollout (`search_train/`) and for the first val
batch under each eval rule (`search_val_<rule>/`).

## 8. Invalid combinations (refused with the reason)

- `reward ∈ {fractional, difference}` with `supervise = last`: early glimpses' losses come from
  a head never trained on them, so the rewards are noise.
- `reward = terminal` with `qreg`: QReg regresses the immediate reward at the taken cell.
- `vpg` with any non-default `reward_center` / `reward_scale`: VPG owns its baseline and
  advantage normalisation.
- `vpg` on the search task in v1 (see §5.1).
- `n_glimpses < 2` with `difference` / `fractional`.
- `entropy_*` knobs with an objective that has no dual (`qreg`, `vpg`).
- `pg` or `vpg` with `input_norm = batch` and `select_bn_eval = true` (§5.5, §11), on any task.
- Uniform patcher; `world_size > 1`.

## 9. Files

New: `canvit/search/{__init__,config,data,head,model,task,loss}.py` and
`canvit/search/tests/`; a `SearchCmd` in `harness/cli.py`; a `SearchEval` in
`harness/evaluate.py` (not yet written, see §10b).

Changed (shared code, all behind `None`/`default` values that reproduce today):
`core/policy/net.py` (FiLM option, `input_norm` option, shared U-Net helper), `harness/config.py`
(`JointPolicyConfig` fields), `harness/policy/{rl,joint,build}.py` (centring, scaling, entropy
knobs, refusals), `harness/rollout/{engine,selector,eval_viewpoints}.py` (reward kinds,
`step_weight` hook, `cond_fn`).

## 10. Test plan

1. **Pin before touching.** The digests cover qreg policies only (`test_policy_qreg_{uniform,
   foveated}_digest`), and no PG digest exists. First add a PG digest (inline fractional) and a
   VPG digest, recorded on the unmodified tree, and run the whole suite (`.venv-cu126`,
   `--basetemp` in the scratchpad).
2. **No-op proof.** After each shared-code change, all digests (the parity digest, the 8 task
   digests, the 2 qreg digests, the new PG/VPG ones) must be unchanged at defaults.
3. **Units.**
   - The reward kinds, and centring/scaling against hand-computed tensors.
   - `per_rollout` vs `per_glimpse` dual steps.
   - FiLM: `cond=None` is bit-identical to the old scorer, and zero-init FiLM is the identity
     at init.
   - `input_norm = none`: the scorer's train-mode and eval-mode outputs are identical.
   - KL → NLL as σ → 0.
   - Box transforms: flip, crop and letterbox keep boxes consistent, and retention never emits
     a sample without its target.
   - Each refusal in §8.
4. **End to end.** A 2-step CPU run on a synthetic search dataset built by the real adapter
   path. Then a short 1-GPU smoke run on COCO-Search18, checking that train loss falls and
   that `hit_t` is logged for all three eval policies.

## 10b. Implementation status (2026-11-05, branch `search-v1`)

Implemented and tested on CPU: everything in §2–§8 except the standalone eval command
(`SearchEval` in `harness/evaluate.py`, not written; training-time validation works). Files:
`canvit/search/{config,data,loss,head,model,task}.py`, `SearchCmd` in `harness/cli.py`.

- **Checks.** Full suite green. All digests unchanged at defaults, including three new
  policy digests recorded on the unmodified tree (`f35585a`): PG, VPG, and QReg with the
  scorer hashed. The old foveated QReg digest did not move when the policy loss was scaled
  by 0.8. New tests: `harness/tests/test_reward_options.py` and `search/tests/`.
- **End to end.** Joint training runs through `harness.run` with both embedding modes, and
  evaluation runs under random, greedy and policy fixations. Both use a tiny foveated ViT-S on
  a synthetic dataset in the real file format.
- **Real data.** 2163 / 326 pairs and no missing image files. Loading takes about 21 ms per
  sample with the default augmentation, 52 ms with all of them.
- **Bug found while testing.** The Gaussian target underflowed to all zeros at small σ, which
  gave a silent zero loss. It is now normalised in log space (a softmax over the grid).
- **Not yet run:** anything on a GPU or with a real pretrained backbone.

## 11. Findings and open points

- **Existing PG + `select_bn_eval=True` is off-policy.** `build_policy` refuses this for VPG,
  and its comment says the True default exists for "QReg/PG, which do not sample". But PG does
  sample (`mode="sample"`), so the same bias applies to any PG run with a BatchNorm scorer left
  at the default. Agreed fix (owner, 2026-11-05): extend the refusal to PG whenever
  `input_norm = batch`. No launcher under `slurm/runs/` passes `objective pg` (grep,
  2026-11-05), so no existing run changes.
- With 2163 training pairs, overfitting is the main risk (owner: try first, react after).
  A COCO-detection pretraining set would need COCO 2017 images, which are not on disk.
- `deploy_rollout_viewpoints` gets its selector from `deploy_selector(joint)` and drives it
  through the same `start_rollout` / `select` calls (read 2026-11-05), so passing `cond_fn` to
  `start_rollout` (§5.4) covers the deploy eval too. It needs a `cond_fn` parameter of its own.

## 12. Target-absent trials (2026-11-06, owner decisions)

**Baseline = Haoze's scheme** (`haoze-wip` `config_search.yaml` over `config_default.yaml`,
`losses.py`, `modules_rl.py`, read 2026-11-06):

- **Data** `condition: tp | both` (default `both`, Haoze's): all TP and all TA pairs, about
  50/50. A TA sample keeps its cue class, has no box, and any crop is valid. TA comes with split1
  only, so `split 2` + `both` is refused.
- **Presence readout.** It must read the head's target-conditioned features, not the canvas:
  the backbone is frozen and never sees the cue, so the canvas and its CLS are target-blind.
  The head's last FiLM'd feature map (the one the output conv reads) is mean-pooled, then a
  Linear gives one logit, log p(present) − log p(absent) (his 2-way softmax as a sigmoid). No
  LSTM: the canvas carries memory across glimpses, and his search config turns it off anyway.
- **Presence loss** `focal` (γ = 2) or `ce`, weight `presence_weight` = 0.75. His focal α = 0.75
  multiplies every sample alike, so it is a loss weight, not class balancing. Supervised at the
  glimpses `supervise` selects (default last, his `use_final_only`): one flag for heatmap and
  presence, because the engine's `step_weight` hook weights the whole task loss; separate flags
  would need a per-term hook.
- **Heatmap loss** on TP trials only (his `heatmap_loss_ta_mode: disabled`); a TA row's target
  map is all zero, so its KL is 0 and it is masked out of the mean.
- **Policy.** The reward (−KL at the last glimpse) exists for TP trials only. TA trials are left
  out of the score-function term and out of the batch standardisation (his `valid_mask`; the
  mean is over valid rows). The entropy term covers all trials. Engine change: an optional
  `reward_mask` attribute on the bound task, passed to `JointPolicy.{trajectory,glimpse}_loss`
  and `pg_loss(mask=)`; PG only (others raise). No mask = the old calls, unchanged.
- **Eval.** `loss`/`hit` over TP trials; `presence_acc`, `recall` (TP said present),
  `specificity` (TA said absent), `precision`, `f1` at p > 0.5; and `correct` (TP: hit and
  "present"; TA: "absent"), per glimpse and rule. `best_metric` stays `hit_final_policy`.

**Discussed and deferred** (owner preferred the baseline first):

- **`null_cell`:** append the pooled readout's logit `a` to the 1024 heatmap logits and take one
  softmax over 1025 outcomes. TP target = the Gaussian on the cells; TA target = all mass on the
  absent cell. One KL for all trials; it equals the cell-conditional heatmap KL − log p(present)
  on TP and −log p(absent) on TA (an identity, not two losses). p(absent) = σ(a − logsumexp h),
  so the overall level of the heatmap logits carries presence evidence. It would give the policy
  one reward for every trial (TA: log p(absent), i.e. fixations that let the model rule the
  target out); the TP and TA reward scales differ, so it would need rewards standardised within
  each condition (unbiased, since the condition is fixed before the action), and groups of ~8 at
  batch 16 are noisy. Not built; it reuses the same pooled readout.
- **A presence reward on TA trials** (−presence loss), for the same reason: Haoze's policy learns
  nothing from TA trials. Owner: revisit after the baseline.
- Haoze's `uniform` / `center_complement` TA heatmap modes and DDM: not ported.

