"""The saccade-length (distance) bias a checkpoint still misses, as a post-hoc log-prior on its map; measured in bits.

Per scored fixation (current fixation f, cells j of the G x G scene grid, d_j = |c_j - f| in DEGREES):

    log p(j) = log_softmax_j( logits_j / tau_dataset + log pixels_j + prior(d_j) )
    prior:  none  |  log  a log(d + r)  |  linear+log  a log(d + r) - b d          (r = half a grid cell)

One shared fit of the prior over the focus datasets the checkpoint has (COCO-Search18 TP + TA validation,
COCO-FreeView validation; their 2000-scanpath TRAINING predictions ``<stem>_train2000.npz``), tau per dataset (a
per-dataset calibration). Maximum likelihood on cells (scipy L-BFGS-B, torch autograd), SEs from the
image-clustered sandwich covariance. Scored on validation as IG over the centre bias with the exact per-fixation
shift ``priors`` and ``evaluate`` use. Under entropy-regularised RL with temperature alpha a reward cost c(d)
multiplies the policy by exp(-c(d)/alpha), so the fitted log prior is the cost per unit alpha:
c(d)/alpha = -a log(1 + d/r) (+ b d). "strength" of the log form = -a (nats per unit log(1 + d/r)).

Analysis and the choice of the form: normative_modelling/external/saccade_cost/ (2026-11-24; same numbers for exp39 s0,
the log form wins on both the policy and the centre bias). Written by Claude (AI, Claude Opus 5.5), 2026-11-24.

    source .envrc && .venv-cu126/bin/python -m canvit.scanpath.saccade_cost $LOGS_DIR/scanpath_eval/<entry>
        # -> <entry>/saccade_cost.json (read by scripts/scanpath/figures.py --summary)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize

log = logging.getLogger(__name__)

STEMS = {"tp": "coco_search18_tp_validation", "ta": "coco_search18_ta_validation", "fv": "coco_freeview_validation"}
# pixels per degree of the eye-tracking displays (as protocol.deepgaze.ppd of the DeepGaze III records)
PPD = {"coco_search18": 31.1, "coco_freeview": 31.1}
FORMS = {"none": (), "log": ("a",), "linear+log": ("a", "b")}
TAUS = np.unique(np.concatenate([np.geomspace(0.25, 128.0, 37), [1.0]]))  # the best-tau sweep (val)
CHUNK = 6000  # fixations per autograd pass


# --------------------------------------------------------------------------------------------------- data
def _batch(path: Path, part: str) -> dict:
    """One record's CellBatch (``priors.cell_batch``, val or its training subset) reduced to what the fit needs:
    logits, log pixels per cell, the human's cell, image index, and d (deg) from the current fixation to every cell."""
    from .data import letterbox_geometry, load_dataset
    from .evaluate import ScanpathConfig, centre_bias_model, select, train_subset
    from .models import GridScanpathModel
    from .priors import cell_batch, cell_centres
    torch.set_num_threads(1)
    rec = json.loads(path.read_text())
    pr = rec["protocol"]
    if part == "val":
        st, fx = load_dataset(rec["dataset"], pr["part"], location=Path(pr["data_location"]))
        fx = select(fx, condition=pr["condition"], limit=pr.get("limit_scanpaths"))
        npz = path.with_suffix(".npz")
    else:
        cfg = ScanpathConfig(dataset=rec["dataset"], condition=pr["condition"], train_scanpaths=pr["train_scanpaths"],
                             train_seed=pr["train_seed"], data_location=Path(pr["data_location"]))
        st, fx = train_subset(cfg)
        npz = train_npz(path, rec)
    model = GridScanpathModel.load(npz)
    b = cell_batch(model, st, fx, centre_bias_model(st, fx, pr["centre_bias"]))
    grid = int(round(b.logits.shape[1] ** 0.5))
    cur = np.stack([h[-1] for h in b.hist])
    cc = cell_centres(grid)
    d_scene = np.hypot(cc[None, :, 0] - cur[:, None, 0], cc[None, :, 1] - cur[:, None, 1])
    # scene units -> stimulus pixels -> degrees: x_scene = (x_px cw / w + left) / scene * 2 - 1
    deg = np.empty(len(b))
    for n in np.unique(b.n):
        h, w = (int(v) for v in st.sizes[int(n)])
        cw = letterbox_geometry(h, w, model.scene, model.frame)[1]
        deg[b.n == n] = w * model.scene / (2 * cw) / PPD[rec["dataset"]]
    return {"logits": b.logits.astype(np.float32), "logcount": b.logcount.astype(np.float32), "cell": b.cell,
            "n": b.n, "d": (d_scene * deg[:, None]).astype(np.float32), "r": (deg / grid).astype(np.float32),
            "grid": grid}


def train_npz(path: Path, rec: dict) -> Path:
    pr = rec["protocol"]
    p = pr.get("save_train_predictions")
    return Path(p) if p else path.with_name(f"{path.stem}_train{pr.get('train_scanpaths', 2000)}.npz")


def available(entry: Path) -> dict[str, Path]:
    """The focus records of ``entry`` with validation and training predictions."""
    out = {}
    for k, stem in STEMS.items():
        p = entry / f"{stem}.json"
        if p.exists() and p.with_suffix(".npz").exists() and train_npz(p, json.loads(p.read_text())).exists():
            out[k] = p
    return out


def _tensors(key: str, D: dict) -> dict:
    return {"key": key, "logits": torch.as_tensor(D["logits"], dtype=torch.float64),
            "logcount": torch.as_tensor(D["logcount"], dtype=torch.float64),
            "d": torch.as_tensor(D["d"], dtype=torch.float64),
            "r": torch.as_tensor(D["r"], dtype=torch.float64)[:, None],
            "cell": torch.as_tensor(D["cell"]).long(), "n": np.asarray(D["n"])}


def _chunk(D: dict, i: int, j: int) -> dict:
    return {k: (v[i:j] if isinstance(v, (torch.Tensor, np.ndarray)) else v) for k, v in D.items()}


# ---------------------------------------------------------------------------------------------------- fit
def log_p(form: str, p: dict, D: dict, tau: torch.Tensor | float | None = None) -> torch.Tensor:
    """[N, C] log p(cell); ``tau`` None: exp(p['log_tau_<key>'])."""
    if tau is None:
        tau = torch.exp(p[f"log_tau_{D['key']}"])
    z = D["logits"] / tau + D["logcount"]
    if "a" in FORMS[form]:
        z = z + p["a"] * torch.log(D["d"] + D["r"])
    if "b" in FORMS[form]:
        z = z - p["b"] * D["d"]
    return torch.log_softmax(z, 1)


def _human(lp: torch.Tensor, D: dict) -> torch.Tensor:
    return lp[torch.arange(len(D["cell"])), D["cell"]]


def fit(form: str, train: list[dict]) -> tuple[dict[str, float], dict[str, float]]:
    """Maximum-likelihood parameters (tau per dataset, the prior shared) and their image-clustered SEs."""
    names = [f"log_tau_{D['key']}" for D in train] + list(FORMS[form])
    N = sum(len(D["cell"]) for D in train)

    def pieces():
        for D in train:
            for i in range(0, len(D["cell"]), CHUNK):
                yield _chunk(D, i, i + CHUNK)

    def nll(x):
        val, grad = 0.0, np.zeros(len(x))
        for C in pieces():
            xt = torch.tensor(x, dtype=torch.float64, requires_grad=True)
            loss = -_human(log_p(form, dict(zip(names, xt)), C), C).sum() / N
            loss.backward()
            val, grad = val + float(loss.detach()), grad + xt.grad.numpy()
        return val, grad

    res = minimize(nll, np.zeros(len(names)), jac=True, method="L-BFGS-B",
                   bounds=[(-4, 4) if k.startswith("log_tau") else (None, None) for k in names])
    if not res.success:
        log.warning("saccade cost fit %s: %s", form, res.message)
    xh = torch.tensor(res.x, dtype=torch.float64)
    H, meat = np.zeros((len(names),) * 2), np.zeros((len(names),) * 2)
    for D in train:
        _, inv = np.unique(D["n"], return_inverse=True)
        G = np.zeros((inv.max() + 1, len(names)))  # per-image sums of the per-fixation scores
        for i in range(0, len(D["cell"]), CHUNK):
            C = _chunk(D, i, i + CHUNK)
            H += torch.autograd.functional.hessian(lambda x: -_human(log_p(form, dict(zip(names, x)), C), C).sum(),
                                                   xh).numpy()
            np.add.at(G, inv[i:i + CHUNK], torch.func.jacfwd(lambda x: _human(log_p(form, dict(zip(names, x)), C),
                                                                              C))(xh).numpy())
        meat += G.T @ G
    Hi = np.linalg.inv(H)
    se = np.sqrt(np.clip(np.diag(Hi @ meat @ Hi), 0, None))
    return dict(zip(names, map(float, res.x))), dict(zip(names, map(float, se)))


def image_average(v: np.ndarray, n: np.ndarray) -> float:
    from .priors import image_average as ia
    return ia(v, n)


def val_ig(form: str, p: dict, V: dict, record_ig: float, taus=None) -> np.ndarray:
    """IG over the centre bias (bits/fix, per image) on validation: the record's IG (policy at tau = 1, no prior)
    plus the exact per-fixation shift. ``taus`` None: the fitted tau; else one value per tau (prior fixed)."""
    pt = {k: torch.tensor(v, dtype=torch.float64) for k, v in p.items()}
    grid = [None] if taus is None else list(taus)
    shift = np.zeros((len(grid), len(V["cell"])))
    with torch.no_grad():
        for i in range(0, len(V["cell"]), CHUNK):
            C = _chunk(V, i, i + CHUNK)
            ref = _human(log_p("none", {}, C, tau=1.0), C)
            for j, tau in enumerate(grid):
                shift[j, i:i + CHUNK] = ((_human(log_p(form, pt, C, tau=tau), C) - ref) / np.log(2)).numpy()
    return np.array([record_ig + image_average(s, V["n"]) for s in shift])


# ------------------------------------------------------------------------------------------------- driver
def evaluate_entry(entry: Path, processes: int = 6) -> dict:
    recs = available(entry)
    if not recs:
        raise SystemExit(f"{entry}: no focus record with validation and training predictions")
    jobs = [(k, part) for k in recs for part in ("train", "val")]
    with ProcessPoolExecutor(min(processes, len(jobs))) as ex:
        built = dict(zip(jobs, ex.map(_batch, [recs[k] for k, _ in jobs], [p for _, p in jobs])))
    log.info("cell batches: %s", {f"{k}/{p}": len(v["cell"]) for (k, p), v in built.items()})
    deg_unit = float(np.median(np.concatenate([v["r"] for v in built.values()]))) * built[jobs[0]]["grid"]
    grid = built[jobs[0]]["grid"]
    train = [_tensors(k, built.pop((k, "train"))) for k in recs]
    fits = {form: fit(form, train) for form in FORMS}
    for form, (p, se) in fits.items():
        log.info("fit %s: %s", form, {k: f"{v:.4g} +- {se[k]:.2g}" for k, v in p.items()})
    n_train = {D["key"]: len(D["cell"]) for D in train}
    del train
    per = {}
    for k, path in recs.items():
        V = _tensors(k, built.pop((k, "val")))
        rig = json.loads(path.read_text())["metrics"]["ig_image"]
        p0 = fits["none"][0]
        row = {"ig_tau": float(val_ig("none", p0, V, rig)[0]), "tau": float(np.exp(p0[f"log_tau_{k}"]))}
        for form, tag in (("log", "log"), ("linear+log", "linlog")):
            p = fits[form][0]
            row[f"ig_{tag}"] = float(val_ig(form, p, V, rig)[0])
            row[f"gain_{tag}"] = row[f"ig_{tag}"] - row["ig_tau"]
            row[f"tau_{tag}"] = float(np.exp(p[f"log_tau_{k}"]))
        sweep = val_ig("log", {"a": fits["log"][0]["a"]}, V, rig, TAUS)
        row["best_tau_log_val"], row["ig_best_tau_log_val"] = float(TAUS[int(np.argmax(sweep))]), float(sweep.max())
        per[k] = row
        log.info("%s: %s", k, {kk: round(v, 3) for kk, v in row.items()})
    (pl, sl), (pll, sll) = fits["log"], fits["linear+log"]
    gl = float(np.mean([v["gain_log"] for v in per.values()]))
    names = {"tp": "CS18 TP", "ta": "TA", "fv": "FreeView"}
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                cwd=Path(__file__).parent).stdout.strip()
    except OSError:
        commit = ""
    return {
        "key_line": f"distance bias still missing: +{gl:.2f} bits/fix (log cost, mean of "
                    f"{', '.join(names[k] for k in per)} val)",
        "gain_log_mean": gl, "gain_linlog_mean": float(np.mean([v["gain_linlog"] for v in per.values()])),
        "fit": {"log": {"a": pl["a"], "a_se": sl["a"], "lambda_over_alpha": -pl["a"]},
                "linlog": {"a": pll["a"], "a_se": sll["a"], "b_per_deg": pll["b"], "b_se": sll["b"]}},
        "per_dataset": per, "n_train_fixations": n_train, "deg_per_scene_unit": deg_unit, "r_deg": deg_unit / grid,
        "records": {k: str(v) for k, v in recs.items()},
        "notes": "IG: bits/fix over the centre bias, validation, per image then over images. One fit of the prior "
                 "shared over the datasets listed, tau per dataset (tau, tau_log, tau_linlog: fitted on the 2000 "
                 "training scanpaths). best_tau_log_val: argmax of validation IG over tau with the log prior fixed "
                 "(optimistic). lambda_over_alpha = -a: RL cost c(d)/alpha = lambda/alpha * log(1 + d/r), r = half a "
                 "grid cell.",
        "written_by": "canvit.scanpath.saccade_cost (Claude, Claude Opus 5.5)", "source": f"canvit {commit}",
        "computed_at": datetime.now(UTC).isoformat()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("entries", type=Path, nargs="+", help="records dirs (checkpoints.json entries)")
    ap.add_argument("--processes", type=int, default=int(os.environ.get("PROCESSES", 6)))
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    for entry in a.entries:
        res = evaluate_entry(entry, a.processes)
        (entry / "saccade_cost.json").write_text(json.dumps(res, indent=1))
        log.info("%s -> %s", res["key_line"], entry / "saccade_cost.json")


if __name__ == "__main__":
    main()
