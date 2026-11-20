# Human scanpath evaluation in canvit — design (2026-11-18)

> Written by Claude (AI, Claude Opus 5.5) for Valentin, 2026-11-18, branch `scanpath-eval`. Open decisions for the
> owner are marked **[OPEN]**; decisions already taken are dated.

## 1. Goal

Measure how human-like CanViT's fixations are. First on **COCO-Search18** (goal-directed search),
then on **free-viewing** data (MIT1003, then the DeepGaze set: CAT2000, COCO-FreeView, DAEMONS,
FIGRIM). One pipeline for both: a free-viewing trial is a search trial without a cue.

Owner decisions (2026-11-18):
- Both metric families, **likelihood first** (§4.1), then the generative COCO-Search18 benchmark (§4.2).
- Any viewpoint policy must be evaluable, plus zero-training baselines — not only the search runs,
  whose checkpoints are not readable without group `HPC_nib00021` (see §8).
- COCO-Search18 comes from the owner's own copy, symlinked into the layout canvit reads (§2.1).
- Work on branch `scanpath-eval`.

Non-goals for v1: fixation durations (CanViT has no notion of time), training on human data.

## 2. Data

### 2.1 COCO-Search18

`/mnt/lustre-grete/usr/u13879/datasets/coco_search18_canvit/`, built by
`make_coco_search18_layout.py` (this dir) as symlinks into the original release:
2779 images, the four TP fixation JSONs. **No target-absent data** (not in this copy), so
`condition=tp` only until TA is downloaded.

A human trial = one record: `name, subject, task, bbox, X, Y, T, length, correct, RT, split`.
X/Y are pixels on the 1680×1050 display canvas (the image incl. the release's zero padding)
and **include the initial centre fixation** as fixation 0. Split1 valid: 3258 trials,
326 (image, task) pairs, 10 subjects, mean length 3.72, max 34.

### 2.2 Free viewing (later)

MIT1003 is on disk in pysaliency format (`datasets/MIT1003/{fixations,stimuli}.hdf5`); its stored
scanpaths **do not contain the central start fixation** (checked in autoregressive_detection's
`human_scanpaths.py`), so the adapter prepends the image centre as fixation 0. CAT2000's
`FIXATIONLOCS` carry no order (fixation maps only, so likelihood of the spatial map only).
COCO-FreeView: same images as COCO-Search18, free viewing — the cleanest search-vs-free contrast.

### 2.3 Sample contract (dataset-independent)

    image:     float [3, S, S]   letterboxed into the square scene, exactly as canvit/search/data.py
    cue:       long  []          class index, or -1 for free viewing
    fix:       float [L, 2]      human fixations (x, y) in scene coordinates [-1, 1], NaN-padded
    n_fix:     long  []          valid length (fixation 0 = the start fixation)
    boxes:     float [N, 4]      target boxes xyxy [-1, 1] (search only; empty for free viewing)
    trial_id / subject / image_id  for grouping (per-subject, leave-one-subject-out)

Pixel → scene: the same letterbox as `SearchDataset` (scale `S / max(H, W)`, centred,
zero padding), applied to the display canvas. One function, shared with the boxes, so a fixation
and a box can never disagree about where the image is.

## 3. Models: one interface, many adapters

```python
class ScanpathModel(Protocol):
    def next_fixation_logits(self, batch, history: Tensor, t: int) -> Tensor:  # [B, G, G]
        """Teacher forcing: log-scores for fixation t given the HUMAN's fixations 0..t-1."""
    def rollout(self, batch, n: int, mode: str) -> Tensor:                    # [B, n, 2]
        """The model's own scanpath (argmax or sampled)."""
```

| adapter | what | needs |
|---|---|---|
| `canvit_policy` | a trained `ViewpointScorer` + its CanViT, any task (search, ade20k, in1k, distill) | a harness checkpoint |
| `canvit_search_heatmap` | softmax of the search task head, visited cells masked ("greedy" as a distribution) | a search checkpoint |
| `uniform` | flat over the displayed image | nothing |
| `centre_bias` | the training split's fixation density (KDE), no image | training fixations |
| `gold_standard` (later) | leave-one-subject-out KDE of the other subjects' fixations: the spatial ceiling | nothing |

**Teacher forcing reuses the deploy code path.** `PolicySelector.select()` runs the encoder (which
keeps a rolling reference across glimpses, `enc.prev`) and the scorer, and stores the scores on
`last_aux`. The adapter calls it unchanged, reads `last_aux["scores"]` as the next-fixation
distribution, and then **replaces the chosen centre by the human's fixation** before `advance`.
So the encoder state, the cue embedding and the view scale follow exactly the deploy rollout;
only the centres differ. t0 is the FULL anchor, which for a foveated model is the centre
fixation at the training scale — the same as the human's fixation 0 on COCO-Search18.

**Uniform-patcher policies** (the public `canvit/*policy-ade20k*` checkpoints) have a scale
dimension in their action space. **[OPEN] D1:** marginalise over scales for the likelihood and
force the human fixation at the scale the policy's argmax chose — or support foveated policies only.

## 4. Metrics

### 4.1 Likelihood (teacher-forced; DeepGaze III / Kümmerer & Bethge 2021)

Per human fixation t ≥ 1, the model's distribution over the G×G candidate cells is turned into
a **density over the displayed image** (mass uniform within a cell, renormalised over the cells
that overlap the image, not the letterbox padding), and evaluated at the human fixation.

| metric | definition |
|---|---|
| `ll` | mean log2-density per fixation, relative to `uniform` (bits/fix) |
| `ig` | `ll(model) − ll(centre_bias)` (bits/fix) — the headline number |
| `cnss` | NSS of the conditional map at the fixation |
| `cauc` | AUC of the conditional map: human fixation vs all cells |

Reported overall and per fixation index t (1, 2, …), averaged over fixations, then over trials
**[OPEN] D2** (pysaliency averages over fixations; per-trial averaging weights short trials up).

**[OPEN] D3:** how many fixations to score. The search policies are trained for 4 glimpses
(t0..t3); humans use up to 34. Proposal: all fixations up to `max_fix` (default 6), with the
per-index breakdown showing where the model leaves its training horizon.

### 4.2 Generative (COCO-Search18 benchmark; Yang 2020, Chen 2021, Gazeformer, HAT)

On the model's own scanpaths (argmax or sampled) vs the 10 human scanpaths per (image, task):

- **TFP curve**: fraction of trials with the target fixated within k fixations; `tfp_auc`;
  `prob_mismatch` = Σ_k |TFP_model(k) − TFP_human(k)|.
- **Sequence Score**: string-edit similarity after clustering fixations (Yang 2020).
- **MultiMatch** (shape, direction, length, position; duration dropped).
- **SemSS**: needs COCO panoptic labels — later.

The existing search `hit` metric is the k-step TFP of the model alone; §4.2 adds the human side.

## 5. Entry point and layout

```bash
python -m canvit.harness.evaluate scanpath --cfg.dataset coco_search18 --cfg.model uniform
python -m canvit.harness.evaluate scanpath --cfg.dataset coco_search18 --cfg.model canvit_policy \
    --opts.ckpt <search best.pt> --cfg.foveated-scale.fixed-scale 2.0
```

`ScanpathEval` has its own `run()` (like `Ade20kDinoV3Eval`, which also bypasses
`evaluate()`), because there is no task head to restore for the baselines. One JSON record out,
with the protocol (dataset, split, max_fix, model, scale) for provenance.

```
canvit/scanpath/
├── data.py      human-scanpath adapters (coco_search18, later mit1003, …) -> the §2.3 contract
├── models.py    the ScanpathModel adapters (§3)
├── metrics.py   §4, pure functions on tensors
├── evaluate.py  the run() behind `harness.evaluate scanpath`
└── tests/       synthetic data in the real file formats (the search conftest is the template)
```

`core/` stays untouched; `search/` is read (data letterbox, model, policy) but not changed.

## 6. Refused combinations (planned)

- A fixed-scale foveated policy without `--cfg.foveated-scale.fixed-scale` (no silent default).
- `condition both` on a copy without target-absent data.
- A search adapter on a free-viewing dataset (no cue).

## 7. Test plan and success criteria

- Coordinates: a fixation at a box corner lands on that corner in scene coordinates; the centre
  pixel maps to (0, 0); round trip pixel → scene → pixel.
- Metrics on hand-built maps: a delta map at the fixation gives the maximal `ll`; `uniform` gives
  `ll = 0` exactly; `centre_bias` against itself gives `ig = 0`.
- Teacher forcing: forcing the policy's OWN argmax reproduces `deploy_rollout_viewpoints` bit-for-bit
  (the check that the forced path did not change anything but the centres).
- Real data: `uniform` and `centre_bias` on COCO-Search18 split1 valid run end to end on CPU; the
  centre bias must beat uniform (`ll > 0`).

## 8. Blockers and dependencies

- **Group access.** The foveated backbone (exp22), the search runs (exp39) and the shared logs are
  under `/mnt/vast-nhr/projects/nib00021`, unreadable without `HPC_nib00021`. Until then only
  the baselines and the public uniform-patcher policies can run.
- **Free-viewing policy.** "Reconstruction" = a policy trained on the `distill` objective
  (`--preset policy_only`). Whether a foveated one exists: ask Jonathan.
- **Test fixations.** Later COCO-Search18 releases include the test split's fixations, which
  Gazeformer and HAT report on; not in this copy. **[OPEN] D4:** valid split1 (what the search runs
  validate on) vs downloading the test split for comparability with the literature.

## 9. Status (2026-11-18)

**Rebuilt on pysaliency (2026-11-18, owner: "use pysaliency conventions and the package; use
established packages, delete own code they replace").** pysaliency (GitHub 4728226, 0.2.23;
installed into `.venv-cu126` with `uv pip install --constraint <freeze>` because `uv lock` needs the
private `../fovi` clone) now loads the data (`get_COCO_Search18(merge_tasks=False,
unique_images=True)`, pointed at the local release files by a md5-checking mirror; its
`_modify_image` patched for numpy 2) and computes every likelihood number (LL / IG vs `UniformModel`
and `CrossvalidatedBaselineModel`, NSS, AUC; fixations with history >= 1; `average='image'` and
`'fixation'`). Deleted: own `metrics.py`, own centre-bias KDE, own JSON trial loader. Own code left:
`data.py` (letterbox pysaliency stimuli/scanpaths into tensors), `models.py` (`GridScanpathModel`:
CanViT's precomputed grid logits as a pysaliency `ScanpathModel`; density wrapper for NSS/AUC;
centre-bias bandwidth/eps fit), `canvit_models.py` (teacher-forced episode + `precompute`),
`generative.py` (no package exists; kept for comparison with generative models such as ScanDiff),
`evaluate.py`. Off-stimulus fixations are clipped (rebuilding `ScanpathFixations`, which copies
coordinates at construction). The numbers below were made with the deleted own metrics (v0) and
are superseded once the pysaliency run is in.

v0 (own code, now deleted): `data.py` (COCO-Search18 TP trials -> §2.3), `metrics.py` (§4.1),
`models.py` (uniform, centre bias), `canvit_models.py` (search `canvit_policy` / `canvit_heatmap`,
§3), `evaluate.py` + `harness.evaluate scanpath`. Interface change from §3: one teacher-forced
episode per batch, `forced_logits(batch, n, grid) -> [B, n, G*G]` (a per-step call would rerun
the backbone over the whole history: n(n+1)/2 glimpses instead of n).

Checks passed: the §7 coordinate and metric tests; boxes and pixels identical to `SearchDataset`;
**forcing the policy's own argmax reproduces the deploy rollout's scorer outputs bit for bit**
(`test_canvit_models.py`), and its argmax is the cell deploy fixated.

First numbers, COCO-Search18 split1 valid, 32x32 grid, 512 scene, fixations 1..6 (3227 trials
with >= 1 fixation after the start; 8357 fixations):

| model | ll (bits/fix, vs uniform) | ig vs centre bias | nss | auc |
|---|---|---|---|---|
| uniform | 0.000 | -0.570 | 0.000 | 0.500 |
| centre bias (sigma 0.02, train-fitted) | 0.570 | 0 | 0.836 | 0.735 |

Per index the centre bias falls from 0.82 (t1) to ~0.2-0.3 bits (t3..t6): later fixations are less
central. Reference ceiling at t1: a leave-one-subject-out KDE of the other subjects' fixation 1 on
the same grid scores 2.63 bits/fix (scratch analysis, not yet a model in the code).

Generative metrics (§4.2) as pure functions in `generative.py` (TFP curve / AUC / probability
mismatch, Sequence Score), following HAT `common/metrics.py` and IRL `irl_dcb/metrics.py`; the SS
clustering recipe (MeanShift, bandwidth factor by improved interaction rate) is from the authors in
Scanpath_Prediction issue #24, since that code was never released. Check on the real split1 valid
humans: the TFP curve matches the reference code run on the raw JSON exactly (AUC 4.624 at K=6;
the published 5.200 is the test split), human-vs-other-humans SS 0.491 (authors' corrected test
value 0.476). NOT yet wired into `evaluate.py`: needs the models' own rollouts (`rollout()`, §3).
sklearn comes in only transitively (via `dinov3`); declare it directly when this is committed.

Not built: generative eval wiring, gold-standard model, free-viewing datasets, uniform-patcher
policies (D1). The CanViT models have not run on real checkpoints (group access).

## 10. Findings outside this project (for Jonathan)

- `pandas` reaches the venv only through `fovi` (`uv.lock`), but `dinov3.eval.segmentation`
  imports it, so `canvit.ade20k` / `harness.evaluate` fail to import in a venv synced without the
  `fovi-legacy` group (which needs a sibling `../fovi` clone). Stopgap here: `uv pip install
  pandas==3.0.3` (the locked version). Fix: declare pandas as a direct dependency.
- `in1k/tests/test_in1k.py`: its `skipif(not _IN1K_SHARDS.is_dir())` RAISED `PermissionError` at
  collection for a user outside `HPC_nib00021`, failing the whole suite. Fixed on this branch
  (`_readable_dir`).
- A venv built on a conda/miniforge base interpreter breaks `sqlite3` (hence `comet_ml`) on Grete
  (`CXXABI_1.3.15` missing). Build it with `uv sync --python-preference only-managed`.

## 11. Target entropy for the search policy (owner question, 2026-11-18)

- FoAViS: 3.5 nats on its 64x64 grid, but only in the joint-without-stop-grad ablation; the
  default recipe has NO entropy term, and 3.5 was never swept -- it was copied from where the
  best-transferring policy happened to sit (`autoreg/core/losses.py:302-307`; paper
  Appendix.tex:992, 1344-1349). Same dual-ascent rule as canvit's (eta 0.01, clamp [1e-6, 0.1]).
- canvit's 2.11 = 3.5 - ln 4: right for keeping the same SPATIAL spread (~0.8% of the image,
  ~8 cells of 32x32), not the same normalised entropy (0.30 vs FoAViS's 0.42).
- Humans on COCO-Search18, same 32x32 grid (scratch analysis, LOSO-bandwidth KDE per (image,
  task), sigma 1.5 cells): fixation 1 ~4.6 nats (~100 cells), fixations 2-4 ~4.2-4.3 nats; the
  estimator cannot go below ~3.6 nats (the kernel's own entropy). So a human-like policy is far
  broader than 2.11.
- Whether that matters depends on the goal: the entropy target shapes TRAINING (exploration,
  task performance); for the human-likelihood eval a policy's sharpness can be calibrated after
  training (one softmax temperature fitted on the train split, as DeepGaze-style comparisons do).
  **[OPEN] D5:** keep 2.11 for search performance and calibrate for the eval, or train a
  human-matched variant (~4.2-4.6 nats).

## 12. Human biases in the policies (owner decisions, 2026-11-20)

Owner: biases shape the policy in RL training AND at inference, so both routes are wanted; the evaluation
must work for any task's policy (reconstruction/distill, ADE20K, IN1k, search, future tasks), not only search.

Biases to model (sources: `claude_dev/papers/schwetlick_kuemmerer.md` = D'Agostino, Schwetlick, Bethge &
Kümmerer, "What Moves the Eyes", NeurIPS 2025, which extends SceneWalk-X using DeepGaze III):
- temperature over the trial (fixation-index dependent);
- saccadic momentum (continue the previous direction; stronger after long saccades and later in the trial);
- cardinal attention bias with a leftward asymmetry early on, adapting over time;
- saccade-amplitude distribution, inhibition / facilitation of return (SceneWalk);
- the centre bias (already a pysaliency baseline).

Plan (proposed 2026-11-20, owner: "sounds good"):
1. Each bias is written ONCE, as a small module in `canvit/core/policy/` (no harness imports, as `core/`
   requires): a log-prior over the G x G grid given the history (previous fixations, fixation index), with
   a few parameters.
2. (a) Inference / evaluation first: `scores / T(t) + sum_k w_k log prior_k(history)`, parameters fitted on
   the human TRAINING split by pysaliency log-likelihood. Measures how many bits each bias adds over CanViT.
   Lives in `canvit/scanpath/` (the teacher-forced `_episode` already passes every step's scores through
   one place).
3. (b) Training: one optional hook in the shared `PolicySelector` that adds the prior to the scores before
   choosing, behind a config flag that defaults off (so the parity and pinning digests stay unchanged), as
   the existing cue-conditioning hook does. Shared code: agree the hook with Jonathan first.
4. Prerequisite: a task-agnostic episode (any task's policy + optional cue), see the review of FoAViS/autoreg
   and canvit's per-task model/policy construction (in progress).
