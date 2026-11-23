"""CanViT's own scanpaths on the COCO-Search18 TEST trials, in the input format of the ScanDiff evaluator driver
(normative_modelling/external/scandiff_driver/run_scandiff_eval.py; it scores them with ScanDiff's own code).

    source .envrc && .venv-cu126/bin/python scripts/scanpath/sample_scandiff.py <best.pt> tp <out.json> \
        [--temperature 1.0] [--seed 0]

Per (image, task) key of ScanDiff's test JSON, one sampled scanpath per human scanpath (as ScanDiff requires),
as long as that human's, at most 16 fixations (ScanDiff's max_len), fixation 0 the centre anchor (the humans'
include their start fixation too). A human scanpath with ONE fixation (the start only; ScanDiff keeps them, its
loaders filter no length) gets the model's one-fixation scanpath, the anchor at the image centre, without a model
call; their number is in ``meta``. Sampled by ``canvit_models.sample_scanpaths`` from the density the likelihood
scores, with the trial's cue. Output: {"frame": [1680, 1050], "scanpaths": [{name, task, X, Y}], "meta": ...},
X/Y in display pixels of the 1680x1050 release images. The test fixations are not in pysaliency's COCO-Search18
(train/valid only): the keys and lengths come from ScanDiff's test JSONs (the Gazeformer test split).

Written by Claude (AI, Claude Opus 5.5), 2026-11-23.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pysaliency
import torch
from pysaliency.external_datasets.coco_search18 import TASKS
from pysaliency.utils.variable_length_array import VariableLengthArray

from canvit.scanpath.canvit_models import PolicyEpisode, sample_scanpaths
from canvit.scanpath.data import make_scanpath_loader

SCANDIFF_TEST = "{root}/scandiff_data/cocosearch18/coco_search18_fixations_{cond}_test.json"
FRAMES = {"TP": (512, 320), "TA": (1680, 1050)}  # the frame of ScanDiff's X/Y per file (driver docs)
W, H = 1680, 1050
MAX_LEN = 16


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("condition", choices=["tp", "ta"])
    ap.add_argument("out", type=Path)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    a = ap.parse_args()
    cond = a.condition.upper()
    humans = json.load(open(SCANDIFF_TEST.format(root=os.environ["SCANPATH_DATA_DIR"], cond=cond)))
    fw, fh = FRAMES[cond]

    # one stimulus per (image, task) key, the cue in pysaliency's `task` attribute (as its COCO-Search18 loader)
    keys = sorted({(s["name"], s["task"]) for s in humans})
    index = {k: i for i, k in enumerate(keys)}
    images = Path(os.environ["COCO_SEARCH18_ROOT"]) / "images"
    stimuli = pysaliency.FileStimuli([str(images / name) for name, _ in keys],
                                     attributes={"task": np.array([TASKS.index(t) for _, t in keys])})
    # the humans' scanpaths only give the sampled lengths (and the keys); positions rescaled to display pixels
    # one-fixation humans: the model's scanpath is its anchor alone (no glimpse to choose); the rest are sampled
    single = [s for s in humans if len(s["X"]) < 2]
    multi = [s for s in humans if len(s["X"]) >= 2]
    xs = [list(np.asarray(s["X"], float)[:MAX_LEN] * W / fw) for s in multi]
    ys = [list(np.asarray(s["Y"], float)[:MAX_LEN] * H / fh) for s in multi]
    n = np.array([index[(s["name"], s["task"])] for s in multi])
    fix = pysaliency.ScanpathFixations(pysaliency.Scanpaths(  # ts: pysaliency requires one; the index (unused)
        xs=VariableLengthArray(xs), ys=VariableLengthArray(ys), n=n,
        ts=VariableLengthArray([list(range(len(x))) for x in xs])))

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ep = PolicyEpisode.load(a.ckpt, device=dev, need_policy=True)
    loader = make_scanpath_loader(stimuli, fix, scene_size=ep.task.cfg.scene_size, batch_size=a.batch_size,
                                  num_workers=2)
    smp = sample_scanpaths(ep, loader, temperature=a.temperature, seed=a.seed, max_len=MAX_LEN)

    per_key = defaultdict(int)
    records = []
    for x, y, sp, L in zip(smp["xs"], smp["ys"], smp["scanpath"], smp["n_fix"]):
        name, task = keys[n[sp]]
        per_key[(name, task)] += 1
        records.append({"name": name, "task": task, "X": x[:L].tolist(), "Y": y[:L].tolist()})
    for s in single:  # the anchor = the scene centre = the image centre (the letterbox is centred)
        per_key[(s["name"], s["task"])] += 1
        records.append({"name": s["name"], "task": s["task"], "X": [W / 2], "Y": [H / 2]})
    human_per_key = defaultdict(int)
    for s in humans:
        human_per_key[(s["name"], s["task"])] += 1
    assert per_key == human_per_key, "one sample per human scanpath and key (ScanDiff's rule)"
    a.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"frame": [W, H], "scanpaths": records,
               "meta": {"ckpt": str(a.ckpt), "condition": cond, "split": "test (ScanDiff/Gazeformer)",
                        "temperature": a.temperature, "seed": a.seed, "max_len": MAX_LEN,
                        "lengths": "matched to each human scanpath",
                        "single_fixation_humans": {
                            "n": len(single), "model_scanpath": "the anchor alone, [image centre], no model call",
                            "why": "the human made no saccade after the start fixation; ScanDiff keeps these "
                                   "scanpaths (no length filter), so the per-key counts must include them"},
                        "checkpoint": ep.provenance}},
              open(a.out, "w"), default=str)
    print(f"{a.out}: {len(records)} scanpaths, {len(keys)} keys")


if __name__ == "__main__":
    main()
