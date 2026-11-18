"""Synthetic human scanpath data as pysaliency objects -- the data contract of ``canvit.scanpath``
(pysaliency's own loaders produce the same types for the real datasets). Small stimuli (105x168,
COCO-Search18's 1050x1680 aspect) keep pysaliency's per-pixel metrics fast; fixations cluster near
the centre and start at the centre, as in COCO-Search18."""

import numpy as np
import pysaliency
import pytest
import torch
from pysaliency.utils.variable_length_array import VariableLengthArray
from torchvision.io import write_jpeg

from canvit.search.tests.conftest import fake_cs18  # noqa: F401 -- the tiny search model reads its classes there

H, W = 105, 168


def scanpaths(n_stimuli: int, *, per_stimulus: int = 3, seed: int = 0, present=True, h=H, w=W):
    rng = np.random.default_rng(seed)
    xs, ys, ns, tp, bbox = [], [], [], [], []
    for n in range(n_stimuli):
        for s in range(per_stimulus):
            k = 1 + (n + s) % 5
            xs.append([w / 2] + list(np.clip(w / 2 + 0.15 * w * rng.standard_normal(k), 0, w - 1)))
            ys.append([h / 2] + list(np.clip(h / 2 + 0.15 * h * rng.standard_normal(k), 0, h - 1)))
            ns.append(n)
            p = present if isinstance(present, bool) else bool(present[n])
            tp.append(int(p))
            bbox.append([0.1 * w, 0.2 * h, 0.2 * w, 0.2 * h] if p else [np.nan] * 4)
    return pysaliency.ScanpathFixations(pysaliency.Scanpaths(
        xs=VariableLengthArray(xs), ys=VariableLengthArray(ys), n=np.array(ns),
        ts=VariableLengthArray([list(range(len(x))) for x in xs]),
        scanpath_attributes={"target_present": np.array(tp), "target_bbox": np.array(bbox, dtype=float)}))


@pytest.fixture(scope="session")
def fake_search(tmp_path_factory):
    """(stimuli_train, fixations_train, stimuli_val, fixations_val): JPEG file stimuli with a ``task``
    attribute, as pysaliency's COCO-Search18 loader returns; val mixes target present and absent."""
    root = tmp_path_factory.mktemp("scanpath")
    g = torch.Generator().manual_seed(0)
    out = []
    for part, n in (("train", 8), ("val", 4)):
        paths = []
        for i in range(n):
            p = root / f"{part}_{i}.jpg"
            write_jpeg(torch.randint(0, 255, (3, H, W), generator=g, dtype=torch.uint8), str(p), quality=80)
            paths.append(str(p))
        stimuli = pysaliency.FileStimuli(paths, attributes={"task": np.arange(n) % 18})
        out += [stimuli, scanpaths(n, seed=len(out), present=[i % 2 == 0 for i in range(n)])]
    return tuple(out)
