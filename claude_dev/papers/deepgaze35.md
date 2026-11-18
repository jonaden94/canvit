> Written by Claude (AI, Claude Opus 5.5), 2026-11-18, from https://arxiv.org/abs/2607.02083 (full PDF text read: main paper and appendix), https://link.springer.com/chapter/10.1007/978-3-032-37271-0_14 (only the search listing; the page needs a login).

# DeepGaze3.5-VL: Modeling Scanpaths via Autoregressive Token Prediction

Agrawal, Bethge, Kümmerer (Univ. Tübingen / Tübingen AI Center). arXiv 2607.02083v1, 2 Jul 2026.
A Springer chapter with the same title exists (DOI 10.1007/978-3-032-37271-0_14). I could not open it, so the venue is **not verified**.
The owner's "DeepGaze 3.5" is this paper. The 2.18 bits on MIT1003 is confirmed in the abstract and Table 1.

## What it is (verified)
- A large VLM fine-tuned with LoRA (InternVL3.5-8B with an InternViT-300M vision encoder; LoRA on all LLM MLP layers). It replaces DeepGaze III's hand-built spatial priority, scanpath and fixation-selection networks.
- **Fixations are text tokens.** x and y are each written as a 2-digit integer from 00 to 99 (per cent of image width/height, origin top-left, y downwards). That gives a **100×100 grid**: 4 tokens per fixation (x1, x2, y1, y2).
- When scoring, the logits are restricted to the 10 digit tokens and renormalised at each digit position, so all probability mass stays on the grid.
- The full 10,000-cell map takes a prefix tree of 1+10+100+1000 = 1,111 forward passes. "Rapid IG" needs only 4 passes per fixation, conditioned on the ground-truth history. The authors say the two give the same IG.
- "We upscale the 10,000-cell probability map to original image resolution and renormalize before metric computations." The paper does not say whether the map is divided by cell area or how it is interpolated.
- Conditioning is written into the prompt: subject ID, task ("predict a scanpath searching for a clock"), and fixation duration (1 ms bins) as an extra predicted attribute.
- The comparison model **DeepGaze III-6B** is DeepGaze III retrained with an InternViT-6B backbone on the same splits. DeepGaze III with InternViT-300M gets 1.45 bits on MIT1003, so better visual features alone do not explain the gain.

## Datasets (verified, Appendix 6.6)
All splits are the authors' own image-level splits. They are **not** the official benchmark test sets.

| dataset | content | train / held-out scanpaths |
|---|---|---|
| MIT1003 | 1003 images, 15 obs, 3 s | 14,775 / **270** |
| CAT2000 (train part) | 2000 images, 5 s | 33,470 / 678 |
| COCO-FreeView | 4317 images, 10 obs, 5 s | 42,188 / 860 |
| DAEMONS | 2200 images used, 8 s | 27,792 / 2,917 |
| FiGrIm | 2787 images, 2 s | 42,864 / 867 |

- COCO-Search18 is used for a separate search-conditioned model (Sec. 3.6). **The split used is not stated.**
- The MIT1003 held-out set is only 270 scanpaths. The numbers are therefore not comparable with DeepGaze III's 10-fold cross-validated 2.442 bit/fix LL-vs-uniform.

## Evaluation protocol
- **Verified:** IG = (LL_model − LL_centerbias)/ln 2, in bits per fixation. "Each dataset has its own independent centerbias baseline, which is held fixed and applied uniformly across all models." It cites Tatler 2007 and Kümmerer et al. 2015.
- **Not stated in the paper:**
  - how the centre bias is fitted (presumably the pysaliency cross-validated KDE used in the group's other papers; this is inference)
  - whether the forced initial central fixation is scored
  - per-fixation vs per-image averaging
  - maximum history length
  - any gold standard (none is reported)
- **Inferred from the training-prompt examples (Fig. 16/17):** the target sequence starts near the centre, e.g. "(50, 48), (21, 48), …" and "(52, 48), …". This suggests the first, central fixation is part of the generated (and possibly scored) sequence. Fig. 8's caption speaks of "the first post-central fixation". **This is ambiguous; ask the authors or check the code if it is released.**
- **Verified:** the per-fixation-index plots (Fig. 5) are in "IG over Uniform (bits/fix)". Table 1 is IG over the centre bias.
- **Verified:** AUC uses the full prefix-tree map.

## Headline numbers (verified, Table 1; IG bit/fix vs dataset centre bias / AUC)
| model | MIT1003 | COCO-FV | CAT2000 | DAEMONS | FiGrIm |
|---|---|---|---|---|---|
| TPPGaze | 1.18 | 1.41 | – | 1.52* | – |
| HAT | 1.25* | 1.39 | 1.07* | 1.53* | 0.99* |
| DeepGaze III-6B | 1.49 / .91 | 1.69 / .92 | 1.60 / .93 | 2.32 / .92 | 1.59 / .92 |
| DG3.5-VL leave-one-dataset-out | 1.94* | 1.79* | 1.66* | 2.30* | 1.73* |
| DG3.5-VL IID-only | 2.03 | 2.15 | 2.01 | 2.67 | 1.88 |
| DG3.5-VL combined | **2.18 / .94** | **2.19** | **2.07** | **2.75** | **1.93** |

\* = not trained on that dataset.

**Other numbers (verified):**
- Duration conditioning raises IG: MIT 2.03→2.12, COCO-FV 2.15→2.30, DAEMONS 2.67→2.84.
- Subject tokens add about +0.09 bit/fix on MIT1003 (5-fold leave-subject-out, which needs a different split).
- IG correlates with MMMU score across VLMs: Spearman ρ = 0.93.

## Search vs free viewing (verified, Sec. 3.6 and Fig. 9)
- Search model on COCO-Search18: aggregate **3.27 bit/fix**, against **2.15** for the free-viewing model "on the same COCO images".
- **Target-present 4.23, target-absent 2.29** ("close to free viewing predictability").
- HAT on the same data: TP 3.03, TA 1.68.
- Per-fixation IG for search is "elevated from the first saccade onward, with no initial ramp-up". Free-viewing IG is "lower and flatter" (Fig. 9, fixation indices 1–14).
- Unseen target categories reach IG similar to seen ones (about 3.0–3.3, appendix).
- **Alignment metrics on COCO-Search18 (Appendix Table 2), for comparison only:**
  - TP: SS 0.65 (GazeXplain 0.54, Gazeformer 0.47); SemSS 0.47
  - TA: SS 0.58
  - The authors argue these metrics favour "mean scanpath" regression, and that SED and FED give contradictory rankings.

## Fixation-index and entropy findings relevant to an entropy target (verified)
- Fig. 5 (MIT1003): the spatial-only model (history = fixations pooled over all subjects) falls from about 2.7 to about 0.25 bit/fix (vs uniform) by fixation 12.
  - Adding fixation index (history-invariant ordinal model) gives a stable +0.3–0.6 bit.
  - The full model's advantage grows with fixation index: late fixations are predictable only through each observer's own history.
- Duration counterfactual: the predicted spatial entropy rises from 9.56 bits (50 ms) to 10.68 bits (600 ms) on the 100×100 grid. For scale, log2(10^4) = 13.29 bits is uniform. Short fixations give sharper next-fixation maps.
- **No entropy-vs-fixation-index numbers are given.**

## Resolution (verified)
The authors list as a limitation: "Our coordinate discretization to a 100×100 grid caps spatial resolution at 1% of image extent." Even so, the 100×100 model beats pixel-level DeepGaze III by a wide margin. The grid is therefore not the binding constraint at 1% resolution.

## Relevance to CanViT
- The protocol is the same family as ours: teacher forcing, a discretised grid upsampled to pixels, IG vs the dataset centre bias.
- Our 32×32 grid is about 3× coarser per axis than their 100×100.
- The search numbers (3.27 overall, 4.23 TP, 2.29 TA) are the only published likelihood numbers for COCO-Search18 that I found in this paper. The split is not stated, so treat them as indicative.
