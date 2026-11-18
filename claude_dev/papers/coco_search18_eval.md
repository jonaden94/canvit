> Written by Claude (AI, Claude Opus 5.5), 2026-11-18, from https://arxiv.org/abs/2207.01166 , https://arxiv.org/abs/2303.09383 , https://arxiv.org/abs/2303.15274 , https://arxiv.org/abs/2005.14310 , https://www.nature.com/articles/s41598-021-87715-9 (Chen et al. 2021 Sci Rep, with supplement) , https://openaccess.thecvf.com/content/CVPR2022W/VSS/html/Chen_Characterizing_Target-Absent_Human_Attention_CVPRW_2022_paper.html (COCO-FreeView) , https://sites.google.com/view/cocosearch/home , https://github.com/cvlab-stonybrook/HAT and https://github.com/cvlab-stonybrook/Scanpath_Prediction (code). Most of the reading was done by a research sub-agent (also Claude). I spot-checked three quotes myself in the extracted paper text: the HAT §11.5 baseline, the Sci Rep "test fixations withheld" sentence, and FFM's px/deg. Everything else is the sub-agent's reading, which it marked as verified against paper text or code.

# Goal-directed search scanpaths on COCO-Search18: datasets, metrics, likelihood evaluation

## Dataset facts
**COCO-Search18** (Chen, Yang, Ahn, Samaras, Hoai, Zelinsky, Sci Rep 11:8776, 2021):
- 10 subjects, 18 target categories, 6,202 images (3,101 target-present + 3,101 target-absent). 299,037 fixations, of which 268,760 are on correct trials.
- Display: 1680×1050 px at 47 cm, about **54°×35°**, i.e. about 31 px/deg at full resolution.
- Models conventionally work at **320×512** (about 9.1–9.5 px/deg; FFM states p = 9.14) with a 20×32 action grid (IRL and FFM).
- Trials start at a central fixation.
- Split: 70/10/20 train/val/test, random within each category. The paper gives no exact counts.

**Test fixations** (paper text spot-checked by me; site checked by the sub-agent):
- Sci Rep 2021: "the fixations on the test images are withheld", for a managed benchmark.
- The Scanpath_Prediction repo history shows a commit of 2022-02-15, "Adding testing set download link to the website". The README still says the release includes the test set.
- The site now says "We are taking down the testing data in order to set up an online benchmark", and the test link is gone.
- **So test fixations were public from about 2022 and have since been withdrawn. Check whether the group has a local copy. Otherwise evaluate on val, or on a held-out part of train.** The takedown date is unknown.

**COCO-FreeView** (Chen et al., "Characterizing Target-absent Human Attention", CVPRW 2022):
- Same 6,202 images, 10 subjects, 5 s free viewing, 822,602 fixations.
- Test fixations are likewise taken down ("to set up an online evaluation service").
- COCO-FreeView is now a test set of the MIT/Tübingen Saliency Benchmark (Kümmerer et al. ICCV 2025), which strongly suggests the online benchmark is that one.

## Benchmark metrics (definitions from papers plus code)
- **TFP:** cumulative probability that the target has been fixated after each saccade.
- **TFP-AUC:** area under that curve. In code: `cdf[0]+cdf[-1]+sum(cdf[1:-1])`.
- **Probability Mismatch:** sum over fixations of |TFP_human − TFP_model|. From IRL, Yang et al. CVPR 2020.
- **Sequence Score (SS):** scanpaths become strings of fixation-cluster IDs, which are aligned with Needleman–Wunsch.
  - In code, the clusters are MeanShift clusters fitted per image on human fixations, with 0/1 symbol similarity, gap 0, normalised by the longer string. The papers do not name the clustering method.
  - **SemSS:** the same procedure with COCO-segmentation category labels instead of cluster IDs. The implementation changed between FFM and HAT (human TA SemSS 0.542 vs 0.372), so SemSS is not comparable across those papers.
- IRL protocol: 10 sampled scanpaths of 6 fixations each, first fixation excluded. Human SS 0.490 and TFP-AUC 5.200 (TP).
- Kümmerer & Bethge 2021 show that alignment-based scanpath metrics can rank wrong, overly deterministic models above the true one. DG3.5-VL shows SED and FED disagree on COCO-Search18.

## Likelihood ("conditional") metrics on COCO-Search18
Only two families report them:
- **FFM, ECCV 2022 ("Target-absent Human Attention")**
- **HAT, CVPR 2024 ("Unifying Top-down and Bottom-up Scanpath Prediction")**

Gazeformer (CVPR 2023), ChenLSTM (CVPR 2021) and IRL (CVPR 2020) report no likelihood metrics. DeepGaze3.5-VL (2026) is the third source; see `deepgaze35.md`.

**HAT definitions (Supp. §11.5, spot-checked):**
- "the baseline is a fixation density map constructed by averaging the smoothed density (with a Gaussian kernel of one degree of visual angle) maps of all training fixations. For target-present and target-absent visual search settings, we use a (target) category-wise fixation density map."
- **The IG baseline is therefore a per-category, training-set fixation prior. It is not a cross-validated centre-bias KDE as in the Kümmerer protocol.**
- Predicted maps are resized to 320×512 before scoring. HAT's native output is 80×128.

**HAT code facts (sub-agent read the code):**
- log2, so bits.
- Pooled average over all scored fixations, not per image.
- The first (central) fixation is never scored. The termination step is excluded.
- On target-present trials, human scanpaths are truncated at the first target fixation.
- The prior uses training fixations without the first fixation. The code's default kernel σ is 24 px at 320×512, about 2.5°, which **contradicts the paper's "one degree"**. Check before reproducing.

**Numbers:**

| model | condition | cIG | cNSS | cAUC | SS | SemSS |
|---|---|---|---|---|---|---|
| HAT | TP | 2.399 | 5.086 | 0.977 | 0.470 | 0.543 |
| HAT | TA | 1.686 | 3.103 | 0.961 | 0.402 | 0.382 |
| HAT | FV (COCO-FreeView test) | 1.485 | 3.382 | 0.965 | 0.369 | – |
| FFM | TP | 1.548 | 2.376 | – | 0.451 | 0.562 |
| FFM | TA | 0.729 | 1.524 | – | 0.372 | 0.516 |

- In HAT's FV table, DeepGaze III gets cIG 0.140 and FFM 0.329. These are low, presumably because DeepGaze III was not trained on COCO-FreeView and the baseline is the dataset's own prior.
- **Grid-resolution ablation (HAT, TA):** a 20×32 output gives cIG 1.534 vs 1.686 at 80×128. Coarsening by 4× per axis cost 0.15 bit/fix. This is the only direct published measurement of grid resolution vs likelihood on COCO-Search18 that we found.
- DeepGaze3.5-VL (`deepgaze35.md`): TP 4.23, TA 2.29, overall 3.27 bit/fix, with HAT re-evaluated by them at TP 3.03 and TA 1.68. These do not match HAT's own 2.399 for TP. **The baseline and split differ (dataset centre bias vs category prior), so IG values are not comparable across papers unless the baseline is the same.**

## Search vs free viewing: spread, length, fixation index (no likelihood-by-index numbers found)
**Scanpath length (CVPRW 2022):** mean number of fixations is TP 2.61, TA 5.02, FV 14.5.
- Search is short. Most TP likelihood comes from 1–3 fixations per trial.

**Dispersion:**
- "TA search fixations were more spatially dispersed than free-viewing fixations" (CVPRW 2022).
- All tasks show a strong centre bias on the first new fixation.
- HAT supplement: centre bias strong for FV, weaker for TA, not obvious for TP.

**Inter-subject consistency (CVPRW 2022, NSS half-vs-half):**

| | TP | TA | FV |
|---|---|---|---|
| NSS | 5.306 | 2.437 | 2.578 |
| AUC | 0.921 | 0.863 | 0.859 |

TP fixations are far more concentrated across subjects. TA is similar to FV, matching DG3.5-VL's "TA ≈ FV predictability".

**TFP (Sci Rep 2021):** about 0.5 of targets are fixated on the 1st saccade, 0.82 by the 2nd and 0.94 by the 3rd.
- Categories differ: bottle and bowl are hardest; stop sign and toilet easiest.

**By fixation index:**
- Target guidance dominates TP from the 1st new fixation and grows over fixations 1–3.
- In FFM, SS(first 2) > SS(first 4) > SS(all): later fixations are harder to predict, most strongly for TA.
- DG3.5-VL: search IG is "elevated from the first saccade onward".

## Implications for CanViT (my inference)
- **Grid size:** a 32×32 grid on 54°×35° gives cells of about 1.7°×1.1°. That is larger than the 1° kernel HAT uses and the 0.4–0.9° blur Kümmerer et al. fit.
  - Expect a measurable resolution cost on TP, where target fixations are tight. HAT's own 80×128→20×32 ablation lost 0.15 bit on TA.
  - Report the gold standard or a strong model coarsened to 32×32 as a control.
- **Score:** fixations 1..n given 0..n−1, with fixation 0 = centre, never scored. Truncate TP at the first target fixation (HAT convention) or say that you do not.
- **Report both baselines** if you want comparability:
  - the HAT/FFM per-category training prior, for comparison with HAT
  - a cross-validated centre-bias KDE (Kümmerer), for comparison with DeepGaze-family numbers
- State fixation-pooled vs per-image averaging.
