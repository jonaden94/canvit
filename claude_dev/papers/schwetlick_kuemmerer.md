> Written by Claude (AI, Claude Opus 5.5), 2026-11-18, from https://bethgelab.github.io/what-moves-the-eyes/ , https://neurips.cc/virtual/2025/loc/san-diego/poster/117056 , https://mlanthology.org/neurips/2025/dagostino2025neurips-moves/ , https://github.com/bethgelab/what-moves-the-eyes (README and source code), https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2024.1389609/full , https://arxiv.org/abs/2112.11067 (abstract only).

# Lisa Schwetlick and Matthias Kümmerer

The owner wrote "Schwittlick". The author is **Lisa Schwetlick** (Univ. Potsdam, Engbert lab; now EPFL).

The owner confirmed (2026-11-18) that the paper meant is the NeurIPS 2025 paper below.

**Caveat on access:** OpenReview serves its PDF only behind a browser challenge, and no arXiv version exists yet ("coming soon" on the project page). I could **not** read the full paper text. The content below comes from:
- the project page (abstract and figure captions)
- the NeurIPS poster page
- the released code

Treat the method and protocol details as **unverified unless marked as coming from the code**.

## 1. What Moves the Eyes: Doubling Mechanistic Model Performance Using Deep Networks to Discover and Test Cognitive Hypotheses
D'Agostino, Schwetlick, Bethge*, Kümmerer* (*shared senior authors). NeurIPS 2025. OpenReview id `cnrwqlr8dg`.

**Idea (verified on project page and abstract):**
- DeepGaze III serves as a "scientific instrument".
- The authors look for "controversial fixations": fixations where DeepGaze III's conditional log-likelihood far exceeds SceneWalk's, ranked by LLD or WLLD (log-likelihood difference, weighted).
- Inspecting these suggests mechanisms, which are added to SceneWalk. The extended model is "SceneWalk-X" in the code.

**The three mechanisms (verified on project page, qualitative only):**
1. **Time-dependent temperature scaling.** "DeepGaze shows higher confidence (less entropy) early on and predicts fixations in more salient locations, in agreement with the empirical data. SceneWalk is not showing this effect."
   - Fix: a fixation-index-dependent temperature with exponential decay.
   - **This is the most directly relevant result for an entropy target:** in free viewing, the next-fixation distribution is sharper (lower entropy) for early fixations and broadens with fixation index.
   - Numbers are not available to me.
2. **Saccadic momentum and return.** There is a bias to continue in the previous saccade's direction. It is stronger after long saccades and later in the scanpath, and it persists after controlling for where salient objects are. The mechanism is modulated by previous saccade length and by fixation index.
3. **Cardinal attention bias with a leftward asymmetry.** Early saccades go left, in both the data and DeepGaze III. The bias adapts over time.

**Headline (verified, abstract):**
- SceneWalk's "explained variance" on MIT1003 doubles, 35% → 70%. "Closed over 56% of the gap to deep networks."
- A per-mechanism contribution figure covers MIT1003 and DAEMONS.
- **The exact definition of "explained variance" is not verified.** Most likely it is the share of the log-likelihood gap between a baseline and a reference closed (e.g. (LL_model − LL_cb)/(LL_ref − LL_cb)). This is inference; check the paper.

**Datasets:** MIT1003 and DAEMONS (verified from the figure caption and the code configs `example_MIT_config.yaml` and `example_DAEMONS_config.yaml`).

**Protocol details visible in the code** (`deepgaze_vs_scenewalk/scenewalk_jax/pysaliency_wrapper.py`, `configs/example_MIT_config.yaml`), verified by reading the source:
- SceneWalk runs on a **128×128 grid** (`MAP_SIZE: 128`), in degrees of visual angle. `pixel_per_degree: 36` for MIT1003.
- The saliency input is DeepGaze's density, bilinearly zoomed down to 128×128 and renormalised.
- For pysaliency evaluation, the 128×128 conditional density is **bilinearly upsampled (`scipy.ndimage.zoom`, order=1) to image resolution and renormalised to sum to 1 over pixels**, then logged. A faster path takes SceneWalk's own log-likelihoods and shifts them "to account for the stimulus size", i.e. the per-pixel log density = log cell mass − log(pixels per cell).
  - This is exactly the conversion a 32×32 CanViT policy needs.
- Scanpaths are scored from the 2nd fixation on: `len(scanpath) == len(xs) - 1`. The first (initial central) fixation is history only, consistent with Kümmerer & Bethge 2021.
- There is a "pseudo-crossvalidated gold standard" module (`gold_standard/pseudo_crossvalidated_gold_standard.py`). It builds a per-subject, per-image interpolated gold standard. What it is used for in the paper (probably the ceiling for "explained variance") is not verified.

## 2. Potsdam data set of eye movement on natural scenes (DAEMONS)
Schwetlick, Kümmerer, Bethge, Engbert. Frontiers in Psychology 15:1389609, 2024. Data report. Verified from the full HTML text.

- **Images:** 2,400 landscape photos at 1920×1080, shown at 32° visual angle. Viewing distance 95 cm, EyeLink 1000 at 1000 Hz.
- **Trials:** 8 s each, starting at a central fixation marker (175 ms after image onset). The task is free viewing with an image-recognition check every 20 images.
- **Splits:**
  - train: 2,000 images × 10 observers
  - validation: 200 images × 50 observers
  - test: 200 images × 50 observers
  - 250 participants in total
- **Hosting:** data on OSF (osf.io/cn5yp images, osf.io/ewyg7 fixations). **The article does not say whether test fixations are public.**
- **Processing:** saccade detection with λ = 4 and at least 5 samples; fixations under 25 ms are merged. Saccade amplitude peaks around 2°, fixation duration around 150 ms.
- **No model evaluation and no dispersion or entropy analysis** in the data report.
- DAEMONS is used by DeepGaze3.5-VL and by Kümmerer et al. ICCV 2025.
  - It has the highest IG of the free-viewing sets in DG3.5-VL: 2.75 bit/fix.
  - Its gold standard in ICCV 2025 is 1.347 bit/fix (subject-LOO, spatial).
  - So DAEMONS fixations are comparatively predictable. The long 8 s trials and many observers help.

## 3. Earlier related items (not read in full)
- **Schwetlick, Kümmerer, Engbert, Bethge (2022), "DeepGaze vs SceneWalk: what can DNNs and biological scan path models teach each other?"** Listed on the Bethge-lab site as Journal of Vision (very likely a VSS/ECVP conference abstract). It is the precursor of the NeurIPS paper. Text not read.
- **Schwetlick, Backhaus, Engbert (2023), "A dynamical scan-path model for task-dependence during scene viewing"** (arXiv 2112.11067; Psychological Review). Kümmerer is not an author.
  - It extends SceneWalk to task-dependence across four tasks, with fixation durations.
  - Relevant to "task changes scanpath statistics", but I read only the abstract. The fixation-spread-by-task question is passed to `scanpath_task_entropy.md`, if it exists.
