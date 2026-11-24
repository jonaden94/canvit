"""Human-bias priors on top of a grid scanpath model, fitted post hoc and measured in bits (design §12, step 2a).

Per scored fixation t (history f_0..f_{t-1} in scene coordinates, f_{t-1} the current fixation; cells j of the
G x G scene grid with centres c_j; d_j = |c_j - f_{t-1}|, theta_j its direction, 0 = right, pi/2 = up):

    log p(j) = log_softmax_j( logits_j / tau + log pixels_j
                              + amp_a log(d_j + r) - amp_b d_j                     (1) saccade amplitude (gamma)
                              + ret_w sum_{k<t} exp(-|c_j - f_k|^2 / 2 ret_s^2)    (2) return (+ facilitation)
                              + dir_card cos(4 theta_j) + dir_lr cos(theta_j)       (3) cardinal and right(+)/left(-)
                              + cb_w log centre_bias_j )                            (4) the dataset's centre bias

r = half a cell, so the current cell has a finite amplitude term. Exact at cell level, like
``models.fit_temperature``: pysaliency's LL is log p(cell) - log(pixels in cell), and the second term does not
depend on any parameter. Fitted jointly by maximum likelihood on the TRAINING subset's predictions (scipy L-BFGS-B,
gradients by torch autograd); scored on validation as IG over the centre bias with the same exact shift
``evaluate`` uses for the temperature: per fixation, IG_new = IG_record + (log p_new - log p_canvit(tau = 1)) / ln 2,
averaged per image, then over images (pysaliency ``average='image'``).

pysaliency 0.2.23 has no SceneWalk-style priors (no amplitude, return or direction terms; checked 2026-11-23);
it provides the centre bias (``CrossvalidatedBaselineModel``, used for prior (4), built exactly like the scores'
reference in ``evaluate.centre_bias_model``). The priors follow SceneWalk / D'Agostino, Schwetlick, Bethge &
Kümmerer (NeurIPS 2025) in spirit, with one simple parametric form each. Written by Claude (AI, Claude Opus 5.5),
2026-11-23.

    source .envrc && .venv-cu126/bin/python -m canvit.scanpath.priors \
        $LOGS_DIR/scanpath_eval/<ckpt>/coco_search18_tp_validation.json   # -> <ckpt>/priors/<stem>.json
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize

log = logging.getLogger(__name__)

PRIORS: dict[str, tuple[str, ...]] = {  # prior -> its parameters
    "amplitude": ("amp_a", "amp_b"),
    "return": ("ret_w", "ret_log_s"),
    "direction": ("dir_card", "dir_lr"),
    "centre": ("cb_w",),
}
INIT = {"log_tau": 0.0, "amp_a": 0.0, "amp_b": 0.0, "ret_w": 0.0, "ret_log_s": float(np.log(0.1)), "dir_card": 0.0,
        "dir_lr": 0.0, "cb_w": 0.0}
BOUNDS = {"log_tau": (-4.0, 4.0), "ret_log_s": (np.log(0.01), np.log(1.0))}  # others free


@dataclass
class CellBatch:
    """Everything the priors need per scored fixation, on the scene grid (C = G*G cells)."""
    logits: np.ndarray  # [N, C] the model's logits (zeros: no model)
    logcount: np.ndarray  # [N, C] log(stimulus pixels in the cell), -inf off the image
    cell: np.ndarray  # [N] the cell the fixation landed in
    t: np.ndarray  # [N] history length = fixation index
    n: np.ndarray  # [N] stimulus index (for the per-image average)
    hist: list[np.ndarray]  # N x [t, 2] the history f_0..f_{t-1}, scene coordinates, last = current fixation
    log_cb: np.ndarray  # [N, C] log centre-bias density per cell (mean over the cell's pixels)

    def __len__(self) -> int:
        return len(self.cell)


def cell_centres(grid: int) -> np.ndarray:
    """[G*G, 2] (x, y) cell centres in scene coordinates [-1, 1], row-major (index = row * G + col)."""
    c = (np.arange(grid) + 0.5) / grid * 2 - 1
    x, y = np.meshgrid(c, c)
    return np.stack([x.ravel(), y.ravel()], -1)


def features(b: CellBatch) -> dict[str, torch.Tensor]:
    """The parameter-free pieces of the priors, once per batch."""
    grid = int(round(b.logits.shape[1] ** 0.5))
    cc = cell_centres(grid)
    cur = np.stack([h[-1] for h in b.hist])  # [N, 2]
    dx, dy = cc[None, :, 0] - cur[:, None, 0], cc[None, :, 1] - cur[:, None, 1]  # [N, C]; scene y points down
    d = np.hypot(dx, dy)
    theta = np.arctan2(-dy, dx)
    owner = np.repeat(np.arange(len(b)), [len(h) for h in b.hist])  # history rows -> their fixation
    hx = np.concatenate(b.hist)  # [M, 2]
    d2 = ((cc[None] - hx[:, None]) ** 2).sum(-1)  # [M, C]
    f32 = {"log_d": np.log(d + 1.0 / grid), "d": d, "cos4": np.cos(4 * theta), "cos1": np.cos(theta), "d2": d2,
           "log_cb": b.log_cb}
    out = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in f32.items()}
    out |= {"logits": torch.as_tensor(b.logits, dtype=torch.float64),
            "logcount": torch.as_tensor(b.logcount, dtype=torch.float64),
            "owner": torch.as_tensor(owner), "cell": torch.as_tensor(b.cell)}
    return out


def log_probs(p: dict[str, torch.Tensor], f: dict[str, torch.Tensor]) -> torch.Tensor:
    """[N, C] log p(cell) for parameters ``p`` (missing ones = INIT, i.e. that prior is off)."""
    def g(k):
        return p[k] if k in p else torch.tensor(INIT[k], dtype=torch.float64)
    z = f["logits"] / torch.exp(g("log_tau")) + f["logcount"]
    z = z + g("amp_a") * f["log_d"] - g("amp_b") * f["d"] + g("dir_card") * f["cos4"] + g("dir_lr") * f["cos1"]
    z = z + g("cb_w") * f["log_cb"]
    if "ret_w" in p:
        bumps = torch.exp(-f["d2"].double() / (2 * torch.exp(2 * g("ret_log_s"))))
        z = z + p["ret_w"] * torch.zeros_like(z).index_add_(0, f["owner"], bumps)
    return torch.log_softmax(z, dim=1)


def free_params(priors: tuple[str, ...], with_model: bool) -> list[str]:
    return (["log_tau"] if with_model else []) + [k for pr in priors for k in PRIORS[pr]]


def fit(b: CellBatch, priors: tuple[str, ...], *, with_model: bool = True) -> dict[str, float]:
    """Maximum-likelihood parameters (mean log p of the human cell) on ``b``; ``with_model=False``: the logits
    are ignored (uniform), only the priors are fitted."""
    names = free_params(priors, with_model)
    f = features(b if with_model else _no_model(b))
    rows = torch.arange(len(b))

    def nll(x: np.ndarray) -> tuple[float, np.ndarray]:
        xt = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        loss = -log_probs(dict(zip(names, xt)), f)[rows, f["cell"]].mean()
        loss.backward()
        return float(loss.detach()), xt.grad.numpy()

    res = minimize(nll, np.array([INIT[k] for k in names]), jac=True, method="L-BFGS-B",
                   bounds=[BOUNDS.get(k, (None, None)) for k in names])
    if not res.success:
        log.warning("prior fit %s: %s", names, res.message)
    return dict(zip(names, map(float, res.x)))


def _no_model(b: CellBatch) -> CellBatch:
    return CellBatch(**{**b.__dict__, "logits": np.zeros_like(b.logits)})


def cell_log_probs(b: CellBatch, params: dict[str, float], *, with_model: bool = True) -> np.ndarray:
    """[N] log p(the human's cell) under ``params``."""
    f = features(b if with_model else _no_model(b))
    with torch.no_grad():
        lp = log_probs({k: torch.tensor(v, dtype=torch.float64) for k, v in params.items()}, f)
    return lp[torch.arange(len(b)), f["cell"]].numpy()


def image_average(v: np.ndarray, n: np.ndarray) -> float:
    """pysaliency's ``average='image'``: mean per stimulus, then over stimuli."""
    from types import SimpleNamespace

    from pysaliency.utils import average_values
    return float(average_values(v, SimpleNamespace(n=n), average="image"))


# --------------------------------------------------------------------------------------- real data
def cell_log_mean(log_density: np.ndarray, cell: np.ndarray, count: np.ndarray) -> np.ndarray:
    """[G*G] log of the mean of exp(``log_density``) over each cell's pixels (``cell``: the pixels' cell index, as
    models._pixel_cells; ``count``: pixels per cell). One bincount (helper_2's fast_cell_batch, 2026-11-24; the
    per-cell masks it replaces were O(cells x pixels), 74 min on COCO-FreeView). Cells without pixels: 0, they are
    masked by logcount = -inf (0 * -inf = nan)."""
    ld = np.asarray(log_density, dtype=np.float64).ravel()
    mx = ld.max()
    s = np.bincount(np.asarray(cell).ravel(), weights=np.exp(ld - mx), minlength=len(count))
    out, on = np.zeros(len(count)), np.asarray(count) > 0
    out[on] = np.log(s[on]) + mx - np.log(np.asarray(count, dtype=np.float64)[on])
    return out


def cell_batch(model, stimuli, fixations, centre_bias) -> CellBatch:
    """A ``CellBatch`` from a ``GridScanpathModel``'s predictions on ``fixations`` (the scored ones: every
    fixation with a history, in ``models._scored_cells`` order) and the centre bias (a pysaliency model) the
    scores are relative to."""
    from .data import pixel_to_scene
    from .models import _pixel_cells, _scored_cells, without_start_fixations
    logits, cells, logcount, t = _scored_cells(model, stimuli, fixations)
    fx = without_start_fixations(fixations)
    hist, log_cb, cache = [], [], {}
    grid = int(round(logits.shape[1] ** 0.5))
    for i in range(len(fx)):
        n = int(fx.n[i])
        hw = tuple(int(v) for v in stimuli.sizes[n])
        k = int(t[i])
        xy = torch.tensor(np.stack([fx.x_hist[i][:k], fx.y_hist[i][:k]], -1), dtype=torch.float64)
        # the display frame as the predictions (None: the plain letterbox; framed predictions were misplaced before)
        hist.append(pixel_to_scene(xy, hw=hw, scene=model.scene, frame=model.frame).numpy())
        if n not in cache:  # log of the mean centre-bias density over each cell's pixels
            cache[n] = cell_log_mean(centre_bias.log_density(stimuli.stimuli[n]),
                                     *_pixel_cells(hw[0], hw[1], model.scene, grid, model.frame))
        log_cb.append(cache[n])
    return CellBatch(logits=logits, logcount=logcount, cell=cells, t=np.asarray(t), n=np.asarray(fx.n), hist=hist,
                     log_cb=np.stack(log_cb))


VARIANTS: dict[str, tuple[tuple[str, ...], bool]] = {  # name -> (priors, with the CanViT logits)
    "canvit_tau": ((), True),
    **{f"canvit_tau+{p}": ((p,), True) for p in PRIORS},
    "canvit_tau+all": (tuple(PRIORS), True),
    "priors_only": (tuple(PRIORS), False),
    "centre_only": (("centre",), False),  # check: ~0 bits over the centre bias by construction
}


def evaluate_priors(train: CellBatch, val: CellBatch, record: dict, variants=VARIANTS) -> dict:
    """Fit every variant on ``train``; IG over the centre bias on ``val`` (overall per image, and per fixation
    index), as the record's IG plus the exact per-fixation shift from CanViT at tau = 1."""
    m = record["metrics"]
    base = cell_log_probs(val, {"log_tau": 0.0})  # CanViT at tau = 1: what the record's IG is of
    out = {}
    for name, (priors, with_model) in variants.items():
        p = fit(train, priors, with_model=with_model)
        delta = (cell_log_probs(val, p, with_model=with_model) - base) / np.log(2)
        tr = cell_log_probs(train, p, with_model=with_model)
        out[name] = {"priors": list(priors), "with_canvit": with_model, "params": p,
                     "tau": float(np.exp(p["log_tau"])) if "log_tau" in p else None,
                     "train_ll_cell_bits": float(tr.mean() / np.log(2)),
                     "ig_image": m["ig_image"] + image_average(delta, val.n),
                     "ig_t": {int(i): m[f"ig_t{i}"] + float(delta[val.t == i].mean())
                              for i in np.unique(val.t) if f"ig_t{i}" in m}}
        log.info("%s: IG %.3f bits/fix %s", name, out[name]["ig_image"], p)
    return out


def annotate(res: dict) -> dict:
    """Unambiguous labels: every variant's ``ig_over_centre_bias`` is ABSOLUTE (bits/fix over the dataset's centre
    bias), and ``gain`` is its difference to its base row: CanViT@tau over CanViT at tau = 1, CanViT@tau + priors
    over CanViT@tau, a no-image variant over the centre bias; CanViT@tau + all also over priors only."""
    who = "heatmap" if "heatmap" in res["model"] else "CanViT"
    v = res["variants"]
    for name, x in v.items():
        ig = x["ig_image"]
        if name == "canvit_tau":
            label, base, base_ig = f"{who}@tau", f"{who} at tau = 1", res["record_ig_image"]
        elif x["with_canvit"]:
            what = "all four priors" if len(x["priors"]) == len(PRIORS) else " + ".join(x["priors"])
            label, base, base_ig = f"{who}@tau + {what}", f"{who}@tau", v["canvit_tau"]["ig_image"]
        else:
            what = "all four priors" if len(x["priors"]) == len(PRIORS) else " + ".join(x["priors"])
            label, base, base_ig = f"no image: {what}", "the centre bias", 0.0
        x |= {"label": label, "ig_over_centre_bias": ig, "gain": {"over": base, "bits": ig - base_ig}}
    if "canvit_tau+all" in v and "priors_only" in v:
        v["canvit_tau+all"]["gain_over_priors_only"] = v["canvit_tau+all"]["ig_image"] - v["priors_only"]["ig_image"]
    return res


def main() -> None:
    from .data import load_dataset
    from .evaluate import ScanpathConfig, centre_bias_model, select, train_subset
    from .models import GridScanpathModel
    ap = argparse.ArgumentParser()
    ap.add_argument("records", type=Path, nargs="+", help="CanViT records (<stem>.json with <stem>.npz and "
                                                          "<stem>_train<N>.npz beside them)")
    ap.add_argument("--annotate-only", action="store_true",
                    help="only (re)write the labels and gains of existing priors/<stem>.json (no fit)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    for path in a.records:
        out = path.parent / "priors" / f"{path.stem}.json"
        if a.annotate_only:
            out.write_text(json.dumps(annotate(json.loads(out.read_text())), indent=1))
            log.info("annotated %s", out)
            continue
        rec = json.loads(path.read_text())
        pr = rec["protocol"]
        train_npz = Path(pr.get("save_train_predictions")
                         or path.with_name(f"{path.stem}_train{pr['train_scanpaths']}.npz"))
        if not train_npz.exists():
            raise SystemExit(f"{path}: no training predictions {train_npz} (datasets without a split: not yet)")
        cfg = ScanpathConfig(dataset=rec["dataset"], condition=pr["condition"], train_scanpaths=pr["train_scanpaths"],
                             train_seed=pr["train_seed"], data_location=Path(pr["data_location"]))
        sv, fv = load_dataset(rec["dataset"], pr["part"], location=Path(pr["data_location"]))
        fv = select(fv, condition=pr["condition"], limit=pr.get("limit_scanpaths"))
        st, ft = train_subset(cfg)
        fit_cb = pr["centre_bias"]
        val = cell_batch(GridScanpathModel.load(path.with_suffix(".npz")), sv, fv, centre_bias_model(sv, fv, fit_cb))
        train = cell_batch(GridScanpathModel.load(train_npz), st, ft, centre_bias_model(st, ft, fit_cb))
        res = {"dataset": rec["dataset"], "split": rec["split"], "model": rec["model"], "record": str(path),
               "train_predictions": str(train_npz), "n_train_fixations": len(train), "n_val_fixations": len(val),
               "centre_bias": fit_cb, "record_ig_image": rec["metrics"]["ig_image"],
               "variants": evaluate_priors(train, val, rec), "evaluated_at": datetime.now(UTC).isoformat()}
        annotate(res)
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(res, indent=1))
        log.info("wrote %s", out)


if __name__ == "__main__":
    main()
