# Human scanpath evaluation (`evaluate scanpath`)

> Written by Claude (AI, Claude Opus 5.5), 2026-11-25, for Valentin Hassler's scanpath evaluation.

How human-like are a checkpoint's fixations? The evaluation runs the checkpoint on human scanpaths with
teacher forcing: at every human fixation it asks the model for its distribution over the next fixation, given
the human's earlier fixations, and scores that distribution against where the human looked next. The scores are
pysaliency's. Code: `canvit/scanpath/`. This page says what exists and how to run it. The design reasoning is
kept with the project notes outside the repository.

## Install

The evaluation needs [pysaliency](https://github.com/matthias-k/pysaliency) (MIT licence), an optional
dependency. Nothing else in canvit imports it.

```bash
uv sync --extra scanpath          # or: pip install -e '.[scanpath]'
```

Without it, `python -m canvit.harness.evaluate scanpath ...` stops with "install the scanpath extra". Every
other subcommand works unchanged.

## Run it

```bash
# a checkpoint's policy on COCO-Search18 validation, target-present trials
python -m canvit.harness.evaluate scanpath --cfg.model canvit_policy --cfg.ckpt <run>/checkpoints/best.pt \
    --cfg.dataset coco_search18 --cfg.condition tp --cfg.data-location <data dir> --out <out>.json

# baselines (no checkpoint): the centre bias and the uniform model
python -m canvit.harness.evaluate scanpath --cfg.model centre_bias --cfg.dataset coco_freeview \
    --cfg.data-location <data dir> --out <out>.json
```

- **Checkpoint:** the checkpoint alone defines the model, as for the other `evaluate` subcommands. The run
  config, the architecture and the policy are read from `best.pt`, for any task (search, distill, ade20k, in1k).
- **Models** (`--cfg.model`):
  - `canvit_policy`: the learned policy;
  - `canvit_heatmap`: a search run's task head;
  - `centre_bias` and `uniform`: baselines.
- **Search checkpoints on free-viewing data:** they are read as the uniform mixture over their 18 cues.
- **GPU and CPU stages:** running the model needs a GPU (minutes); scoring in pysaliency is CPU work.
  - The GPU stage alone: `--cfg.shard i N --cfg.save-predictions <i>.npz`.
  - Scoring the saved predictions later on CPU: `--cfg.predictions <0>.npz <1>.npz ... --cfg.device cpu`.
  - Use `--cfg.processes` for parallel scoring.
- **Smoke test:** `--cfg.limit-scanpaths 50`.

## Data

Every dataset is pysaliency's, loaded by its own loaders and cached under `--cfg.data-location`. This flag is
required. Its default is `$SCANPATH_DATA_DIR` when that is set. The first use downloads and builds the cache
(COCO-Search18: about 20 GB unpacked in `$TMPDIR`; point `TMPDIR` at a big filesystem for that run).

| `--cfg.dataset` | parts | notes |
|---|---|---|
| `coco_search18` | train, validation (split1), test | target present and absent (`--cfg.condition tp/ta/both`); one stimulus per (image, cue) |
| `coco_freeview` | train, validation | free viewing, same images |
| `daemons` | train, validation | |
| `mit1003`, `cat2000`, `figrim` | all | no split: the centre bias is fitted leave-one-image-out on the same set |

- **COCO-Search18 split1:** pysaliency downloads the release. To use local copies instead (checked against
  pysaliency's md5s), pass `--cfg.release-dirs <dir> [<dir> ...]`. The files are
  `COCOSearch18-images-{TP,TA}.zip`, `COCOSearch18-fixations-TP.zip` and
  `coco_search18_fixations_TA_trainval.json`.
- **COCO-Search18 official test split** (`--cfg.part test`): not in the public release. Its fixations were
  redistributed by Gazeformer and ScanDiff. Place them under the data location as
  `scandiff_data/cocosearch18/coco_search18_fixations_{TP,TA}_test.json`. Its images come from the release
  image zips, so `--cfg.release-dirs` is required here.
- **Target-present test scanpaths end at the first fixation on the target.** Score validation the same way with
  `--cfg.condition tp --cfg.cut-at-target`, so that validation and test are comparable.

## What is scored

The conventions are pysaliency's and Kümmerer & Bethge's:

- **Which fixations:** every fixation that has a history. Fixation 0 is the start fixation and serves only as
  history.
- **Units:** bits per fixation.
- **`ll`:** log-likelihood over pysaliency's uniform model.
- **`ig`:** information gain over the centre bias. The centre bias is a leave-one-image-out KDE on the
  evaluated split; its bandwidth is fitted on the training split (`--cfg.centre-bias-fit` caches the fit).
- **`nss`, `auc`:** of the conditional density, on a seeded subset of at most 20 000 fixations.
- **Averaging:** each metric comes averaged over images (`*_image`, the papers' headline) and over fixations
  (`*_fix`), overall and per fixation index (`*_t<t>`).
- **Grid models:** a CanViT model predicts on its policy grid (32×32 for search). Each cell's probability is
  spread uniformly over the stimulus pixels in that cell.
- **Display frame:** each stimulus is letterboxed into the scene exactly as the checkpoint's own training data
  was. This is the run task's `display_frame` attribute: search uses COCO-Search18's 1680×1050 displays, the
  other tasks the square scene. `--cfg.display-frame` overrides it, for deliberate comparisons only.
- **Temperature (diagnostic, never fed back into training):** with `--cfg.train-scanpaths N`, a softmax
  temperature is fitted on N training scanpaths, globally and per fixation index. The tempered `ll` and `ig`
  are reported under `diagnostics`.

## Outputs

One JSON record per call (`--out`). It has these fields:

| field | contents |
|---|---|
| `task`, `dataset`, `split`, `model` | what was scored |
| `protocol` | the full config, the evaluated part, the display frame and where it came from, the pysaliency version, the centre-bias fit, and the checkpoint's provenance (path, step, run config) |
| `metrics` | `n_fixations`, `n_scanpaths`; `ll`, `ig`, `nss`, `auc` as `*_image`, `*_fix`, `*_t<t>` |
| `diagnostics` | grid models: the predictive entropy, and the temperature fits with their tempered scores |

Optional files:

- `--cfg.save-predictions`: the predicted maps, `.npz`, readable with `GridScanpathModel.load`.
- `--cfg.save-train-predictions`: the same for the training subset.
- `--cfg.save-samples`: the policy's own sampled scanpaths, in stimulus pixels.

## Tests

The tests are in `canvit/scanpath/tests/` and use synthetic pysaliency data. They cover:

- the scene geometry against `SearchDataset`;
- the metrics against pysaliency;
- checkpoints of every task loading into the same policy;
- feeding a policy its own choices reproducing its deploy rollout bit for bit.

Without pysaliency they are not collected. The one test that needs the real COCO-Search18 test cache skips
when the data is absent.
