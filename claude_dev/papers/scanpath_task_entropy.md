> Written by Claude (AI, Claude Opus 5.5), 2026-11-18, from https://doi.org/10.1167/jov.20.5.3 (Backhaus et al. 2020, PMC7409614) , https://arxiv.org/abs/1803.07352 (Schütt et al. 2019 JOV) , https://arxiv.org/abs/1409.7686 (Kümmerer, Wallis, Bethge PNAS 2015 preprint) , https://arxiv.org/abs/1704.08615 (Kümmerer et al. ECCV 2018) , https://arxiv.org/abs/1610.05982 (Rothkegel et al. 2017 JOV) , https://arxiv.org/abs/2112.11067 (Schwetlick, Backhaus, Engbert 2023) , https://jov.arvojournals.org/article.aspx?articleid=2191846 (Mills et al. 2011) , https://jov.arvojournals.org/article.aspx?articleid=2121535 (Koehler et al. 2014) , https://arxiv.org/abs/1802.04069 (Rothkegel et al. 2019), plus the papers in the sibling notes. Most of this reading was done by a research sub-agent (also Claude), which tagged each item [V] full text read / [A] abstract only. I re-checked the Backhaus entropy table and the Schütt "0.5–0.6 bit/fix, much broader" sentence in the extracted text myself.

# How spread and entropy of human fixations depend on task, fixation index and target: material for a policy entropy target

## Read this first: which entropy?
Most published numbers are about the **marginal** fixation density: all fixations of all observers on an image, pooled over the trial.

A policy's per-step distribution is a **conditional** next-fixation distribution, p(f_t | f_0..f_{t-1}, I). Its entropy is much lower than the marginal entropy.

- **Upper bound from the best model (from the chain rule; my inference):** the true conditional entropy H(f_t | history, I) ≤ the best model's cross-entropy = log2(#cells) − IG_vs_uniform.
- So **the best published scanpath likelihoods give upper bounds on the human conditional entropy:**

| dataset | IG vs uniform | per-step entropy ceiling, 32×32 (10-bit max) |
|---|---|---|
| MIT1003, DeepGaze III | 2.44 bit/fix | ≤ ~7.6 bits |
| MIT1003, DeepGaze3.5-VL (2.18 over centre bias, plus ~0.9 for the centre bias) | ~3.1 | ≤ ~7 bits |
| COCO-Search18, DG3.5-VL search model | > 4 (TP IG 4.23 is already over the centre bias) | well below 6 bits on TP trials |

These are rough figures. The 32×32 resolution loss (see `scanpath_likelihood_eval.md` §5) adds back a little. **The true human value is lower still, by however much better models will get.**

- **Marginal entropies** (below) are a looser upper bound. They mix the between-subject and within-trial spread.
- **Practical consequence:** matching a policy's entropy to the human *marginal* entropy would make it far too diffuse per step. If an entropy target is used at all, calibrate it on the conditional log-likelihood. The cleanest calibration is to choose the temperature or entropy weight that maximises held-out human log-likelihood (teacher-forced). Likelihood-optimal calibration is the principled "human-like" setting.

## Task
**Backhaus, Engbert, Rothkegel, Trukenbrod 2020, JOV 20(5):3** (head-free, mobile tracking, images 40.6°×20.1°, 8 s) [V, I re-checked the table].
- Entropy of the per-image fixation density on a 128×128 grid (Scott-rule KDE; maximum 14 bits):

| | Count People | Count Animals | Guess Country | Guess Time |
|---|---|---|---|---|
| entropy (bits) | 13.051 | 13.476 | 13.327 | 13.394 |
| LOSO predictability vs uniform (bit/fix) | 1.187 | 0.745 | 0.936 | 0.830 |
| mean saccade amplitude | 6.27° | 6.45° | 6.76° | 6.83° |

- The entropies are 0.5–1 bit below the maximum. The **task changes marginal entropy by only about 0.4 bit**, and the direction depends on the target: counting people is most concentrated, counting animals most spread.
- Distance to centre does not depend on task for 0–400 ms. It becomes task-dependent after 400 ms and saturates after 1–2 s.
- Converted to 32×32, assuming the density is smooth within 4×4 cells (my conversion): about 9.05–9.48 bits out of 10.

**Schütt et al. 2019, JOV 19(3):1** (memorisation vs search, 31°×25°) [V, re-checked the key sentence]:
- Search fixations are "distributed much broader over the images than in the standard scene viewing task". The empirical (gold-standard) density carries only **0.5–0.6 bit/fix** for search (depending on the target), against **about 1.4 bit/fix** for memorisation.
- Cross-validated KDE bandwidths: memorisation 0.5–0.7° (centre bias 1.1–1.8°); search 0.7–1.1° (centre bias 1.3–2.0°). Later fixation indices prefer larger bandwidths.
- In these experiments the targets were small, low-salience patches. That is very different from COCO-Search18 object categories, where target-present fixations are *more* concentrated than free viewing (next item).

**COCO-Search18 / COCO-FreeView (Chen et al. CVPRW 2022; HAT CVPR 2024)** [V by the sub-agent]:
- Human half-vs-half NSS: TP 5.31 ≫ FV 2.58 ≈ TA 2.44.
- "TA search fixations were more spatially dispersed than free-viewing fixations."
- Scanpath length: TP 2.6, TA 5.0, FV 14.5 fixations.
- Centre bias: strong for FV, weaker for TA, absent for TP.
- DeepGaze3.5-VL: IG TP 4.23, TA 2.29, FV 2.15 bit/fix.
- **For COCO-Search18, the ordering of per-step concentration is TP ≫ TA ≈ FV.**

**Koehler et al. 2014, JOV 14(3):14** [V-web; numbers only in figures]: between-observer fixation variability is lowest for cued object search, highest for free viewing and saliency search, and grows with fixation number.

**Mills et al. 2011, JOV 11(8):17** [V-web]: model-estimated saccade amplitude at 1 s was free viewing 5.0°, memory 7.5°, pleasantness 7.6°, search (for a small embedded letter) 8.6°. No dispersion or entropy measure.

**Schwetlick, Backhaus, Engbert 2023, Psych Rev** [V]: fitted SceneWalk attentional span σ_A is narrower for the counting (search-like) tasks (7.8, 9.6) than for the guessing tasks (11.1, 10.6).

## Fixation index (time in trial)
- **Free viewing: the first free fixation is the most concentrated and predictable, and spread grows with index.**
  - DeepGaze III, JOV 2022, Fig. 10 [V]: LL is highest for fixation 1 and decays over 2–3 fixations. The image-driven component decays; the dynamics component stays flat.
  - What Moves the Eyes, NeurIPS 2025 (project page) [V]: "DeepGaze shows higher confidence (less entropy) early on … in agreement with the empirical data". The fix added to SceneWalk is a fixation-index-dependent temperature that decays exponentially. **This is direct support for an index-dependent entropy schedule** (low early, higher later).
  - Schütt 2019 [V]: (i) a strong centre bias on the first saccade (centre bias alone 1.6 bit/fix at fixation 1); (ii) broadening; (iii) a steady state after about 10 fixations (centre bias about 0.25 bit/fix).
  - Rothkegel 2017 [V]: distance to centre stabilises at about 1–2 s.
  - DG3.5-VL, Fig. 5 [V]: the spatial-only model drops from about 2.7 to about 0.25 bit/fix (vs uniform) by fixation 12. Late fixations are predictable only through each observer's own history.
- **Search (COCO-Search18):** guided from the first saccade. About 50% of targets are fixated at saccade 1, 82% by 2 and 94% by 3. DG3.5-VL finds search IG "elevated from the first saccade onward, with no initial ramp-up". In FFM, prediction gets worse for later fixations, especially on TA.
- **Not found:** any paper reporting conditional or marginal fixation entropy *per fixation index* as numbers.

## Target category
- COCO-Search18 (Sci Rep 2021): search difficulty varies by category (bottle and bowl hardest; stop sign and toilet easiest). Most per-category numbers are in the supplement, not extracted.
- DG3.5-VL: TP IG far exceeds TA IG for every category. Unseen categories reach IG similar to seen ones (about 3.0–3.3).
- Schütt 2019: search predictability varies only slightly by target (0.5–0.6 bit/fix).
- Rothkegel 2019 Sci Rep [V, figures only]: low-spatial-frequency targets produce larger saccades than high-frequency targets.

## Spatial precision
- Eye-tracker error is about 0.5° (lab) to 1° (mobile).
- Cross-validated gold-standard and KDE bandwidths:
  - 27 px ≈ 0.75° on MIT1003 (Kümmerer et al. 2015 preprint)
  - 0.5–1.1° (Schütt 2019)
  - fitted blur 0.4–0.9° across five datasets (Kümmerer et al. ICCV 2025)
  - HAT's baseline kernel is "one degree" in the paper, but its code uses a 24 px σ at 320×512 (about 2.5°)
- A 32×32 grid on COCO-Search18 (54°×35°) has cells of about 1.7°×1.1°: coarser than these bandwidths. See `scanpath_likelihood_eval.md` §5 and `coco_search18_eval.md` (HAT 80×128 → 20×32 costs 0.15 bit/fix on TA).
