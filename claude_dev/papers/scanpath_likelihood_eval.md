> Written by Claude (AI, Claude Opus 5.5), 2026-11-18, from https://arxiv.org/abs/2102.12239 (v2, full text read), https://jov.arvojournals.org/article.aspx?articleid=2778776 (full text read via the Semantic Scholar PDF copy https://pdfs.semanticscholar.org/b85f/9944d64c393e5dc09d4ccc8ee858cc40f27b.pdf), https://arxiv.org/abs/2505.10169 (v2, full text read), https://github.com/matthias-k/pysaliency (source read: models.py, utils.py, datasets.py).

# Likelihood evaluation of scanpath models: the Kümmerer/Bethge conventions

## 1. Kümmerer & Bethge, "State-of-the-Art in Human Scanpath Prediction", arXiv 2102.12239 (v2, Oct 2021)
Verified in text.

**Method:**
- Score each fixation f_i by the model's **conditional** prediction p(f_i | f_0..f_{i-1}, I). This is teacher forcing on the human history.
- The joint likelihood factorises by the chain rule. Scores are averaged.
- For non-probabilistic models, AUC/NSS are computed on the "conditional priority map".
- Models that output only a point are turned into a Gaussian map.

**Why not ScanMatch/MultiMatch:**
- In a toy example (Fig. 3), deliberately *wrong*, more deterministic models score 7–8% *higher* than the true generating model.
- Likelihood cannot rank a wrong model above the generating one: LL = −(entropy + KL).

**Initial fixation (verified):**
- "In all datasets, each trial of a subject begins with a initial forced central fixation. We include this initial forced fixation in the scanpath history that the models are conditioned on, but we don't evaluate how well models predict it, since it is not voluntary."
- MIT1003: the initial fixation is kept as history. Where it is invalid, a central fixation is substituted.
- CAT2000: a central fixation is *added*, because the provided scanpaths lack it. Duplicate consecutive fixations are removed.

**Averaging (verified):** "we first average the performances of each fixation per image and then over all images."
- Note: pysaliency's default `average='fixation'` is a plain mean over all fixations (see §4). State which one you use.

**Units:** log base 2, bit/fixation.
- "LL" = IG relative to a **uniform** model.
- "IG" = IG relative to the **centre-bias** model.

**Baselines (verified):**
- *Centre bias:* "for each image Gaussian kernel density estimate over all fixations on other images in the same dataset" (leave-one-image-out).
- *Fixation-number-dependent centre bias:* fitted separately per fixation-index bin. Bins: MIT 1, 2, 3–5, 6+; CAT2000 1, 2, 3, 4, 5–6, 7–10, 11–15, 16+. It gains only +0.044 bit (MIT1003) and +0.13 (CAT2000) over the plain centre bias.
- *Spatial gold standard:* a mixture of (Gaussian KDE over **other subjects'** fixations on the same image) + uniform + centre bias. Bandwidth and weights are fitted for maximum likelihood under leave-one-subject-out cross-validation. Both "subject LOO" and "joint" (all subjects in the KDE, an upper bound) are reported. **It ignores history, so it is not a ceiling for scanpath models.** CCN-DeepGaze III beats the LOO gold standard.

**Metrics excluded:**
- sAUC, because it penalises the centre bias that the initial fixation and short saccades legitimately produce.
- CC/SIM/KLDiv, because a conditional distribution has only one fixation.

**Numbers (bit/fix):**

| dataset | uniform | centre bias | DeepGaze II | spatial GS (LOO) | spatial GS (joint) | CCN-DeepGaze III |
|---|---|---|---|---|---|---|
| MIT1003 | 0 | 0.906 | 1.870 | 2.120 | 2.762 | 2.257 |
| MIT300 | 0 | 0.782 | 1.724 | 2.117 | 2.555 | 2.171 |
| CAT2000 train | 0 | 1.439 | – | 1.882 | 2.299 | 2.291 |

The MIT/Tübingen Saliency Benchmark adopted this protocol for MIT300 and CAT2000 (test fixations not public).

## 2. DeepGaze III, Kümmerer, Bethge, Wallis, JOV 22(5):7, 2022
Verified in text.

**Model:** p(f_1..f_N | f_0, I) = ∏ p(f_i | f_0..f_{i-1}, I). It conditions on the forced initial fixation f_0 but never predicts it.

**Architecture:**
- The image is downscaled ×2 and passed through DenseNet201 (3 layers, 2048 channels).
- Spatial priority readout: 1×1 convolutions 8→16→1.
- Scanpath network: each of the **last 4 fixations** is encoded as 3 maps (Euclidean distance, dx, dy) → 1×1 convolutions.
- Fixation-selection network: 1×1 convolutions 128→8→1 → Gaussian blur → + centre-bias log density → softmax.
- 28,601 trainable parameters.

**Training:**
- Pretrained on SALICON mouse data (spatial only), then MIT1003 spatial with the initial fixation dropped.
- Then the scanpath network is added and everything fine-tuned.
- **10-fold cross-validation over images** (8 train / 1 validation / 1 test). Each MIT1003 image is evaluated by the fold that held it out. MIT300 uses the average over folds. 8 seeds.

**Evaluation:**
- The initial forced fixation is history only, never scored.
- Log-likelihood is averaged per image, then over images.
- Centre bias = Gaussian KDE over all other images' fixations plus a uniform component, fitted for maximum likelihood on MIT1003.
- Gold standard as above.

**Numbers:**
- MIT1003: LL 2.442 bit/fix, IG 1.536, AUC 0.916, NSS 3.257. Centre bias 0.906; DeepGaze III without scanpath 1.945; without image 1.695; spatial GS (LOO) 2.120.
- MIT300: LL 2.248, IG 1.466.
- CAT2000: LL 3.064.

**History ablation:**
- The current fixation adds 0.35 bit. Second-to-last 0.12, then 0.02, then 0.01.
- **Most of the history information is in the last 1–2 fixations.**

**Per fixation index (Fig. 10, LL vs fixation index):**
- Performance is highest for the first free fixation and decays over the next 2–3 fixations.
- The centre-bias model and the scanpath-independent model decay similarly. The image-independent model stays flat (and rises slightly from fixation 3).
- Interpretation: the first 2–3 fixations go to high-salience regions, first near the centre, then anywhere. Later fixations go to less salient regions and are steered more by scanpath dynamics.
- **For an entropy target:** early free-viewing fixations are more concentrated / predictable; later ones are more dispersed in the image-driven part.

**Discussion:** the authors expect visual search to have "much more complicated interactions between scene content and scanpath history" and name COCO-Search18 as the next step. DeepGaze III itself is free-viewing only.

## 3. Kümmerer, Khanuja, Bethge, "Modeling Saliency Dataset Bias", ICCV 2025 (arXiv 2505.10169)
Verified in text. This is a **spatial** (static) saliency paper, not a scanpath paper. It is relevant for baselines, dataset differences and the COCO-FreeView benchmark.

**Datasets:**
- MIT1003, CAT2000, COCO-FreeView, DAEMONS, FIGRIM. "We discard the initial forced fixation."
- Own 10-fold image splits via `pysaliency.filter_datasets.train_validation_fold` where no official split exists (CAT2000 stratified by category).
- Benchmark: **MIT/Tübingen Saliency Benchmark now includes MIT300, CAT2000 and COCO-Freeview** test sets.

**Finding:**
- Training on one dataset and testing on another loses about 40% IG. About 60% of that gap is dataset-specific bias.
- Fewer than 20 interpretable per-dataset parameters close about 75% of the gap: multiscale weights, centre bias, priority scaling (a nonlinearity), centre-bias weight, and **blur ("fixation scatter")**.

**Fitted blur σ per dataset (Fig. 18a):** about **0.4–0.9 degrees of visual angle**, with DAEMONS needing the smallest blur.
- This is the effective spatial precision of fixation maps in those datasets, i.e. eye-tracker error plus fixation scatter.

**Baselines (verified, App. E/K):**
- Centre bias = KDE over all other images' fixations plus a uniform component. Bandwidth and uniform weight are fitted by **leave-one-image-out** cross-validated maximum likelihood.
- In low-data settings a KDE + centred Gaussian + uniform mixture works best.
- Gold standard = uniform + centre bias + KDE over other observers on the same image. Parameters are fitted per image; reported as leave-one-subject-out, plus an "upper bound" that mixes all observers.

**Spatial gold standards (IG bit/fix vs centre bias, Table 1):**

| | MIT1003 | CAT2000 | COCO-FreeView | DAEMONS | FIGRIM |
|---|---|---|---|---|---|
| subject-LOO | 1.213 | 0.494 | 0.869 | 1.347 | 1.054 |
| upper bound | 1.829 | 0.873 | 1.511 | 1.722 | 1.642 |

Best model: 1.240 / 0.522 / 1.031 / 1.258 / 1.117. Benchmark (Table 2): MIT300 1.246, CAT2000 0.493, COCO-FreeView 1.073 (full joint training).

**Resolution handling:** the readout is computed at 1/8 resolution, upscaled bilinearly to 1/2, blurred in degrees, then evaluated per pixel.

## 4. pysaliency conventions (verified from source, master branch)
- **Coordinates:** pixel units, x = column, y = row, origin top-left.
  - `x_int = np.asarray(x, dtype=int)`, i.e. truncation toward zero, not rounding.
  - A model's `conditional_log_density(stimulus, x_hist, y_hist, t_hist, ...)` must return an **H×W per-pixel log density that sums (in exp) to 1 over pixels**.
  - LL of a fixation = `log_density[y_int, x_int]`, natural log.
- `information_gains(...) = (LL_model − LL_baseline)/ln 2`. The default baseline is `UniformModel()`, which gives what the papers call "LL".
- `average='fixation'` (default) = mean over all fixations. `average='image'` = mean per image, then over images, which is what the papers report.
- `pysaliency.ScanpathModel`: history is passed as `x_hist`/`y_hist` (all previous fixations including the initial central one). The fixation itself is the scored point.
- **Implication for a 32×32 policy:** convert cell probabilities to a per-pixel density. Either piecewise-constant, log p_pix = log P_cell − log(n_pixels_in_cell), or bilinearly upsampled and renormalised as the SceneWalk-X wrapper does. Otherwise LL is not comparable with DeepGaze-family numbers. IG vs a centre bias is invariant to this constant only if the baseline is evaluated on the same pixel grid.

## 5. Does the grid resolution matter? (my own back-of-envelope calculation, not from a paper)
Script: `/tmp/claude-844252/-mnt-lustre-grete-usr-u13879-normative-modelling/739b6471-dad3-45b2-bf66-42680ce8a9eb/scratchpad/scripts/gridloss.py` (session scratchpad, not in the repo; may be cleared).

Setup:
- The "true" next-fixation density is an isotropic Gaussian with standard deviation σ, placed at a random position on the grid.
- A predictor knows it exactly but can output only per-cell mass, either piecewise-constant or bilinearly upsampled.
- Information lost = KL(true ‖ coarse):

| σ (in cell widths) | loss, piecewise-constant | loss, bilinear |
|---|---|---|
| 0.25 | 1.23 bit | 1.45 bit |
| 0.5 | 0.41 | 0.41 |
| 1 | 0.12 | 0.08 |
| 2 | 0.03 | 0.02 |
| 4 | 0.006 | 0.005 |

Applied to COCO-Search18 (1680×1050 px; check the visual angle in `coco_search18_eval.md`):
- A 32×32 grid has cells of 52.5×32.8 px.
- Human conditional densities are multimodal and much wider than the gold-standard blur (0.4–0.9°). The loss is roughly bounded by the row whose σ matches the narrowest mode.
- **Expect order 0.1 bit/fix lost to a 32×32 grid when modes are about 1 cell wide, and over 0.4 bit if they are half a cell wide.**
  - Search target-present fixations land on the target, so they can be very concentrated.
  - DG3.5-VL reaches 4.23 bit/fix IG on target-present trials, so the losses there could matter.
- DeepGaze3.5-VL (100×100) beats pixel-level DeepGaze III. At 1% resolution, discretisation is not the bottleneck; at about 3% (32×32) it might start to be.
- **Recommended control:** score the human gold standard (or DeepGaze III) after the same 32×32 coarsening and report the loss.
