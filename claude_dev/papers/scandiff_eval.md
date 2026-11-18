> Written by Claude (AI, Claude Opus 5.5), 2026-11-18, from https://arxiv.org/abs/2507.23021 (v1 LaTeX source, ICCV 2025), https://github.com/aimagelab/ScanDiff (commit 9eff26c, 2026-05-13), its data zips at https://ailb-web.ing.unimore.it/publicfiles/ScanDiff_ICCV2025/ (only the test JSONs and one SemSS map were fetched, by HTTP range), issues #2 and #8 of that repo, and the PyPI packages multimatch-gaze 0.1.3 and GazeParser 0.11.1.

# ScanDiff (Cartella et al., ICCV 2025): how it evaluates scanpaths

Tags: **[V]** verified in paper text, code or data, or recomputed by me. **[I]** inferred.
Paths are relative to the ScanDiff repo (`src/...`). "Paper" means the arXiv LaTeX source.

## 0. Headline: the tables report KL divergences, not similarity scores

- **[V]** Every MM/SM/SS/SemSS cell in Tables 1 and 2, supp. Tables 6 (OSIE) and 7 (search MM breakdown) (and in the ablations) is a **KL divergence (lower is better)**. Paper §4.1: "the distribution of human vs. generated metrics is compared against the distribution of the human consistency metrics, using the Kullback-Leibler divergence". The protocol comes from TPP-Gaze (D'Amelio et al. 2025).
- **[V]** `gazetools/metrics/kld.py:8-19`: `perform_kld(human, model)` builds `np.linspace(min, max, 100)` edges over both samples pooled, so 99 bins. It normalises both histograms and returns `scipy.stats.entropy(h_human, h_model + 1e-10)`, which is KL(human ‖ model) in nats.
- **[V]** Both samples are **all pairwise scores pooled over every test image**, not per-image means:
  - Model sample: every (human, generated) pair of an image, via `itertools.product`.
  - Human sample: every unordered human pair, via `itertools.combinations`. For SS it is ordered pairs with self excluded, so each pair is counted twice. That does not change a normalised histogram.
- Consequences:
  - A raw SS/MM value is **not** comparable to these tables. A CanViT number is comparable only as the same KLD against the same human sample (same test JSONs, same coordinate frame).
  - The KLD rewards matching the **spread** of human-human similarity. A deterministic policy that emits 10 identical scanpaths gets a narrow model distribution and is penalised. So CanViT must **sample** its scanpaths.
- **[V]** Variability metrics (paper Table 5, `diversity1`) are raw scores, higher is better:
  - **RSS** = Recall SS. A human scanpath counts as covered if some generated scanpath has SS > threshold. The threshold is the dataset's human-consistency SS, a config value (§3).
  - **DSS** = SS(g,h) / (1 + |SS(g,g) − SS(h,h)|), computed per image and then averaged (`sequence_score.py:381-501`).

## 1. Datasets, splits, conditions

| set | test size (from the released JSON) [V] | split | frame used for metrics |
|---|---|---|---|
| COCO-Search18 TP | 612 (image, task), 6118 scanpaths, 10 subj (2 pairs have 9) | Gazeformer/HAT test split. Issue #2: "recover the test split ... from gazeformer" | JSON px 512×320 → /512, /320 → ×512, ×384 |
| COCO-Search18 TA | 612 (image, task), 6120 scanpaths, 10 subj | same | JSON px 1680×1050 → ×512/1680, ×384/1050 |
| COCO-FreeView | 648 images, 6462 scanpaths (630 images with 10 subj, 18 with 9) | **ScanDiff's own split**. Paper §4.1: FV + MIT1003 pooled, 70/15/15. Issue #2: the official FV test is not available | 1680×1050 → 512×384 |
| MIT1003 | 151 images × 15 subj = 2265 scanpaths | ScanDiff's own split (same 70/15/15) | per-image W×H → 512×384 |
| OSIE | zero-shot only (supp. Table 6) | `processed_fixations/osie_fixations_test.json` | → 512×384 |

- **[V]** All coordinates are scaled to [0,1] per axis and then multiplied by **512×384** (`utils/test_utils.py:14,23`). The `images_width=1680, images_height=1050` defaults in `run_metrics.py:143` are overridden by that `size`.
  - The mapping is **anisotropic**: COCO 512×320 is stretched vertically to 384, and MIT1003 aspect ratios are distorted.
  - Normalisation is applied in `cocosearch18_tp.py:81-85`, `cocosearch18_ta.py:79-82`, `coco_freeview.py` and `mit1003.py`. It is active because `use_abs_coords: false` in `configs/data/unified_datamodule.yaml:11`, which overrides the per-dataset `true` (`utils/instantiators.py:73,94`).
- **[V] First fixation is included.** It is fixation 0 in every released JSON:
  - TP mean (249,155) on 512×320
  - TA (817,509) and FV (837,520) on 1680×1050
  - MIT1003 at (0.498, 0.496) of the image, std 0.02–0.03
  - No code drops it. Note: ScanDiff's MIT1003 JSON already *contains* a near-centre first fixation, whereas our pysaliency MIT1003 adapter prepends the centre (design §2.2). Use their JSON for comparability.
- **[V] Max length 16.** Human scanpaths are truncated to 16 in the collator (`data/components/collator.py:29`, `max_len: 16`). Every SS/SemSS call uses `truncate=16`. The model emits up to 16 fixations plus a validity head.
- **[V] Number of samples.** Per image (or (image, task)) the model generates `num_viewers` scanpaths = the number of human scanpaths for that key (10, or 15 on MIT1003; `evaluator.py:106-118`). Each is compared with every human scanpath: 10×10 pairs, never best-of.
- **[V]** Visual search conditions are evaluated separately (TP, TA). There is one model for search and one for free viewing (`checkpoints/scandiff_{visualsearch,freeview}.pth`).

## 2. Metrics, implementation, parameters

Driver: `gazetools/metrics/run_metrics.py:137-621`. Metric list: `configs/evaluation/default.yaml` and `jobs/eval.sh`.

| metric (table col.) | implementation [V] | parameters [V] |
|---|---|---|
| **MultiMatch** Sh/Len/Dir/Pos/Dur (MM, Avg) | pip **`multimatch-gaze==0.1.3`** (MIT), `docomparison(sp1, sp2, screensize=(512, 384))`, wrapper `multimatch.py:66` | No grouping/simplification (package defaults `grouping=False`, thresholds 0). Durations in ms. A scanpath with <3 fixations is padded to 3 with fixation (1,1,1) (`:21,40`). No `T` → duration 100 ms for every fixation (`:14,33`). Sh = vector similarity. The **Avg KLD is the mean of the per-sub-score KLDs** (`run_metrics.py:216`), not the KLD of the averaged MM. |
| **ScanMatch** w/ Dur, w/o Dur (SM) | pip **`GazeParser==0.11.1`** (`GazeParser.ScanMatch`, **GPL-3**), `scanmatch.py` | `ScanMatch(Xres=512, Yres=384, Xbin=16, Ybin=12, TempBin=50 \| 0)`. The `Xbin=14/8` computed at `:85-90` is dead code. Package defaults apply: `Threshold=3.5`, `GapValue=0`. w/o Dur = `TempBin=0` (`run_metrics.py:232`), w/ Dur = 50 ms (`:259`). Pads <3 fixations with (1,1[,1]) **in place**, mutating the shared dicts (`scanmatch.py:19-29`), so every later metric sees padded model *and* human scanpaths. |
| **Sequence Score** w/o Dur, w/ Dur (SS) | own code after IRL/HAT: `nw_matching` (`gazetools/utils.py:16-32`, 0/1 similarity, gap 0, ÷ longer length) and `compute_SS`, `compute_self_SS`, `compute_SS_Time`, `compute_self_SS_Time` (`sequence_score.py`) | Clusters are **precomputed and shipped**: `data/<ds>/clusters_<ds>_512_384.npy`, a pickled dict key → {`strings`, `cluster` (sklearn MeanShift), `times`}. Built by `data/components/*/preprocess_*_clusters.py`: MeanShift per (split, condition, task, image), fitted on that split's human fixations of that condition only (TP and TA separately), in 512×384 px. Bandwidth = `estimate_bandwidth` × factor from (0.2…1.8) maximising the improved interaction rate (recipe of Scanpath_Prediction issue #24). **Human scanpaths with <3 fixations are padded with fixations at (1,1) px and duration 1 ms before clustering.** Truncation 16 for both. w/ Dur: each cluster symbol is repeated `int(T_ms / 50)` times before NW. Model strings come from `MeanShift.predict`. |
| **SemSS** w/o Dur, w/ Dur | own code (`compute_SSS*`, `sequence_score.py:83-328`, `scanpath2categories` in `gazetools/utils.py:35`) | Panoptic label maps from `data/cocosearch18/SemSS/segmentation_maps/<img>.npy.gz`, shipped in their cocosearch18.zip (1128 maps, **320×512 uint8**, labels ≥ 10; I checked one). These are the HAT/Gazeformer SemSS maps [I: same file layout as HAT's `SemSS/`; not diffed]. Coordinates are mapped back 512×384 → 512×320 and the label is read at `int(x), int(y)`. No clustering. Truncation 16; w/ Dur uses tempbin 50. Search only. |
| SED | own copy of Zanca's `string_edit_distance` (`visual_attention_metrics.py:320`), n=10 grid | Computed for the model only, mean per image. **Not in any paper table.** |
| NSS / CC / AUC-Judd / KLdiv | Zanca's code, `saliency.py` | Implemented and **not reported**. There is **no** cIG, cNSS, cAUC or likelihood. |
| TFP / TFP-AUC / prob. mismatch | — | **Not used anywhere** (grep for `cdf`/`fixOnTarget`/`bbox` finds nothing in the metrics). |
| RSS, DSS (Table 5) | `compute_SS(_Time)` with threshold; `compute_DSS_Time`, `compute_DSS_no_dur` | RSS threshold = the human-consistency SS in `configs/data/test_datasets/*.yaml` (§3). |

### Code defects that change numbers [V in code; the effect on published cells is I]

1. **SemSS without durations truncates labels to their first character.**
   - Mechanism: `scanpath2categories` returns `(label, t)` tuples when `T` is present and bare label strings otherwise. `compute_SSS` then takes `i[0]` (`sequence_score.py:104,114`), so for a model without `T` label "27" becomes "2", while the humans keep full labels.
   - HAT, which has no durations, shows SemSS w/o Dur 3.837 (TP) and 3.472 (TA) in Table 2, where every other model is ≤ 1.6. [I] This defect probably explains those values.
   - **Give CanViT a dummy `T` so this path is not taken.**
2. **`compute_DSS_no_dur` compares duration lists, not cluster strings**, in its model-vs-model term (`sequence_score.py:562`, `nw_matching(p_string1, p_string2)` on `T`).
3. Order dependence: the ScanMatch padding (point 1 of the SM row) silently feeds into SS/SemSS. It applies whenever `scan_match*` is in the metric list, which it is in both configs.
4. `run_metrics.py:147` admits that the "Model"/"Human" means are a global mean rather than a per-image mean. The KLD is unaffected.

### Models without durations [V from the table dashes]

HAT and DeepGaze III rows show "–" for MM-Dur, SM w/ Dur, SS w/ Dur, SemSS w/ Dur and DSS. Their MM Avg is the mean of the 4 spatial KLDs. Check: HAT FV (1.099+0.434+0.042+0.444)/4 = 0.505, as printed. ScanDiff's Avg is over all 5 (0.388/5 = 0.078). Without `T` the code crashes for SM/SS with durations; for MM it substitutes 100 ms.

**[I]** Table 5 still gives HAT an RSS (0.861 on TP). So the RSS column is presumably the no-duration variant (`threshold_no_dur`), at least for HAT.

## 3. Human consistency and baseline numbers

- **[V]** The paper reports **no human row**. By construction the human KLD is 0.
- **[V]** Raw human-consistency SS values live in the configs. They serve as the RSS thresholds, and the comment calls them "human consistency score":

| dataset | SS w/ Dur (`threshold`) | SS w/o Dur (`threshold_no_dur`) |
|---|---|---|
| COCO-Search18 TP | 0.506 | 0.664 |
| COCO-Search18 TA | 0.353 | 0.377 |
| COCO-FreeView | 0.314 | 0.348 |
| MIT1003 | 0.406 | 0.408 |
| OSIE | 0.332 | 0.345 |

- **[V, recomputed]** I reran their clustering recipe on the TP test JSON (scratch script, sklearn 1.8):
  - With their <3-fixation (1,1) padding, human SS w/o Dur is **0.664**, matching the config exactly.
  - Without the padding it is 0.644.
  - In isotropic 512×320 instead of 512×384, without padding, it is 0.643. So the anisotropic frame barely matters here.
  - 45% of TP test scanpaths have <3 fixations, so the padding matters there. On TA it is 10%.
  - SS w/ Dur is **0.506**, also matching the config, but only when their time quirk is copied. In about half the TP records `T` has one more entry than `X`, and `preprocess_*_clusters.py` takes `T[:len(X)]` *after* padding `X`. The padded fixations therefore get real trailing durations instead of 1 ms. Done naively (`T[:n]`, then pad), the value is 0.556.
  - **These values are not comparable to HAT's 0.476 (test) or our 0.491 (split1 valid).** Even without padding ScanDiff gets 0.643–0.644. Frame and padding account for only 0.02 of that gap. The rest must come from split (test vs valid), HAT/IRL pooling TA fixations into the clusters, or how HAT aggregates. [I] I did not resolve which.

**KLD cells, lower is better** (paper Table 1 for FV/MIT; Table 2 plus supp. Table 7 for TP/TA). "g" = retrained by ScanDiff on its splits (gray rows). Other rows are original checkpoints.

COCO-FreeView, then MIT1003. Columns: MM Sh/Len/Dir/Pos/Dur/Avg | SM w/, w/o | SS w/, w/o:

| model | COCO-FreeView | MIT1003 |
|---|---|---|
| DeepGazeIII (orig) | .037 .016 .019 .028 – .025 \| – .368 \| – .393 | – |
| HAT (orig) | 1.099 .434 .042 .444 – .505 \| – 1.025 \| – .331 | 1.196 .522 .381 2.386 – 1.121 \| – 2.112 \| – 1.305 |
| ChenLSTM (orig) | .034 .128 .105 .045 .189 .100 \| .574 .373 \| .344 .442 | .028 .073 .149 .107 .110 .094 \| .168 .161 \| .192 .316 |
| GazeXplain (orig) | .151 .195 .874 .164 .382 .353 \| 3.915 3.423 \| 2.278 5.616 | .018 .065 .079 .058 .188 .082 \| .035 .094 \| .072 1.419 |
| DeepGazeIII g | .009 .017 .059 .038 – .031 \| – .348 \| – .417 | .025 .020 .210 .074 – .082 \| – .210 \| – 3.878 |
| GazeXplain g | .346 .226 .032 .033 .068 .141 \| .049 .038 \| .017 .007 | .060 .046 .065 .025 .044 .048 \| .158 .051 \| .128 .043 |
| TPP-Gaze g | .063 .017 .061 .038 .010 .038 \| .125 .226 \| .033 .130 | .039 .036 .139 .068 .027 .062 \| .244 .257 \| .144 .280 |
| **ScanDiff** | .131 .048 .021 .037 .151 .078 \| .015 .027 \| .013 .038 | .050 .015 .042 .019 .072 .040 \| .041 .065 \| .026 .047 |

COCO-Search18. Columns: MM Avg | SM w/, w/o | SS w/, w/o | SemSS w/, w/o:

| model | TP | TA |
|---|---|---|
| ChenLSTM | .197 \| .011 .236 \| .040 .262 \| .084 .162 | .075 \| .010 .012 \| .036 .775 \| .044 .677 |
| Gazeformer (orig) | .281 \| .027 .340 \| .119 .268 \| .131 .208 | .089 \| .061 .075 \| .102 .245 \| .085 .139 |
| HAT (orig) | .118 \| – .263 \| – .148 \| – 3.837 | .052 \| – .063 \| – .097 \| – 3.472 |
| GazeXplain (orig) | .166 \| .023 .237 \| .070 .232 \| .140 .188 | .046 \| .062 .048 \| .038 .191 \| .043 .206 |
| Gazeformer g | .251 \| .045 .508 \| .048 .349 \| .095 .262 | .526 \| 1.184 1.688 \| .319 .520 \| .671 1.043 |
| GazeXplain g | .167 \| .010 .238 \| .050 .217 \| .092 .224 | .037 \| .030 .026 \| .028 .146 \| .038 .143 |
| TPP-Gaze g | .507 \| 2.317 3.995 \| .893 2.381 \| .736 1.605 | .135 \| .775 .887 \| .427 .537 \| .231 .300 |
| **ScanDiff** | .048 \| .037 .079 \| .019 .074 \| .072 .078 | .020 \| .005 .008 \| .008 .031 \| .007 .024 |

MM sub-scores, Sh/Len/Dir/Pos/Dur (supp. Table 7, table numbers counted from the LaTeX source):

| model | TP | TA |
|---|---|---|
| HAT | .161 .089 .115 .108 – | .108 .040 .024 .034 – |
| ScanDiff | .077 .048 .067 .015 .033 | .008 .010 .007 .008 .067 |

Table 5, DSS / RSS (↑):

| model | COCO-FV | MIT1003 | TP | TA |
|---|---|---|---|---|
| HAT | – / .645 | – / .615 | – / .861 | – / .748 |
| GazeXplain | .099 / .032 | .302 / .674 | .406 / .689 | .283 / .716 |
| TPP-Gaze | .271 / .732 | .313 / .758 | .284 / .516 | .221 / .663 |
| ScanDiff | .277 / .736 | .354 / .815 | .425 / .747 | .312 / .800 |

## 4. Licence and packaging

- **[V]** The repo has **no LICENSE file**, and the GitHub API reports `license: null`. Legally that means all rights reserved: **do not vendor or copy their code into canvit** (which is MIT).
  - Reading it, and running an unmodified clone outside our tree, is the safe use.
  - Licensing could be requested from the authors (issue tracker; open issue #1 is about an HF release).
- **[V]** It is not a package: no `setup.py`/`pyproject.toml`. It is a Hydra/Lightning project with pinned `requirements.txt`, including detectron2, torch 2.3.1 and sklearn 1.5.2.
- **[V]** The two metric packages it relies on are installable:
  - `multimatch-gaze` 0.1.3: MIT.
  - `GazeParser` 0.11.1: GPL-3 per its `LICENSE.txt`, sdist only. Fine as an optional runtime dependency for our own evaluation; do not copy it into the MIT tree.
- **[V]** The data zips are public, with no licence stated. The cluster `.npy` files are **pickles of sklearn MeanShift objects**: unpickling executes code, and they need a sklearn close to 1.5.2.

## 5. Our `canvit/scanpath/generative.py` vs ScanDiff

Matches:
- `nw_score` equals their `nw_matching`.
- The MeanShift bandwidth recipe is the same, with the same factors and the improved interaction rate.
- A model's SS is the mean over the image's humans; human SS leaves the scanpath itself out.
- Clusters are fitted on TP humans only. This is a deviation from IRL but the *same* as ScanDiff, which clusters TP and TA separately.
- Fixation 0 is included.

Deviations, each of which breaks comparability with ScanDiff's tables:

1. **Statistic.** Ours returns raw per-scanpath SS. ScanDiff reports KL(human-pair distribution ‖ model-human-pair distribution) with 99 equal bins over the pooled range. We need the pooled pair lists and `perform_kld`.
2. **Coordinate frame.** Ours uses isotropic letterboxed [-1,1], which is equivalent to 512×320 px. ScanDiff uses an anisotropic 512×384 (or W×H → 512×384). MeanShift is equivariant only under *isotropic* scaling (generative.py's own docstring), so in principle the clusters differ. Empirically on TP test human SS the difference is 0.643 vs 0.644 (§3). Convert to 512×384 anyway for exactness, or load their cluster pickles.
3. **Short-scanpath padding.** ScanDiff pads every scanpath with <3 fixations with (1,1) px fixations: humans at cluster time, models via the ScanMatch mutation. Ours does not. On TP this moves human SS from 0.644 to 0.664.
4. **Truncation.** ScanDiff truncates both model and human to 16 fixations. Ours leaves truncation to the caller. ScanDiff does **not** cut model scanpaths at the first target hit (HAT `cutFixOnTarget`), so `cut_at_target` must not be applied for a ScanDiff comparison.
5. **Split.** ScanDiff uses the Gazeformer TP/TA *test* split (612 + 612). Ours was checked on split1 *valid*. FV and MIT1003 use ScanDiff's own 70/15/15 split, available only from their JSONs.
6. **Number of model scanpaths.** ScanDiff samples as many as there are humans and compares all pairs. Our TFP path assumes one rollout per trial.
7. **TFP / AUC / probability mismatch** have no ScanDiff counterpart. They remain HAT/IRL-comparable only.
8. MultiMatch, ScanMatch and SemSS are absent from generative.py; ScanDiff's versions are in §2.

## 6. Recommendation for CanViT (no durations)

1. Write CanViT samples into ScanDiff's generation pickle format and run their metric code **from an unmodified clone outside the repo**. This is the only way to get numerically identical KLDs, given the padding and mutation quirks.
   - Pickle format: dict key → `{'size', 'scanpaths': [N,16,3] in [0,1], PAD = -3, 'question_id': '', 'dataset': 'cocosearch18_tp'|...}`, with key = `(img, task)` for search and `img` for free viewing.
   - Entry point: `Evaluator.compute_metrics` (`evaluator.py:288`), which reloads `generations_*.pkl` / `original_*.pkl`.
   - It needs a small stub datamodule exposing `.test_collators[0].PAD`, `.time_in_ms=False`, and `.test_datasets[0].{name, root_path, threshold, threshold_no_dur}`.
   - Use their test JSONs, cluster files and SemSS maps from the data zips.
   - Sample N = number of humans per key, ≤16 fixations each, fixation 0 = centre start. Map coordinates from scene [-1,1] to image pixels and then to [0,1] per axis.
2. **Give every fixation a constant dummy duration** (e.g. 0.3 s) only to avoid SemSS defect 1. Then report only the duration-free columns: MM Sh/Len/Dir/Pos with Avg over 4, SM w/o Dur, SS w/o Dur, SemSS w/o Dur, plus RSS w/o Dur. Put "–" in the Dur columns, as ScanDiff does for HAT and DeepGaze III.
3. If we reimplement instead, which is licence-clean:
   - MM: `multimatch-gaze` `docomparison(screensize=[512,384])`.
   - SM: `GazeParser.ScanMatch(Xres=512, Yres=384, Xbin=16, Ybin=12, TempBin=0)`.
   - SS: generative.py plus a 512×384 frame, (1,1) padding to length 3 and truncation 16, or their cluster pickles.
   - SemSS: their 320×512 maps.
   - KLD: 99 bins over the pooled min-max, `entropy(h_human, h_model + 1e-10)`.
   - Validate on the TP test human sample first. Targets, reproduced by my scratch script: SS w/o Dur 0.664 and w/ Dur 0.506.
