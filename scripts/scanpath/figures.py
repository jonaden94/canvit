"""Figures and tables for the scanpath working notes (ShareLaTeX `Getting_Info_scanpath_prediction`).

    source .envrc && .venv-cu126/bin/python scripts/scanpath/figures.py --out <ShareLaTeX project dir> \
        --fonts <dir with LiberationSans-*.ttf>

Writes figures/*.pdf and tables/*.tex there; every number in them comes from this script. Data: the human
scanpaths from pysaliency (`canvit.scanpath.data.load_dataset`), the evaluation records and saved predictions in
`$LOGS_DIR/scanpath_eval/` (`baselines/`, `deepgaze3/`, one directory per CanViT checkpoint), and the ScanDiff
driver's sanity runs (`--scandiff-runs`). A figure whose inputs do not exist yet is skipped with a message.
Style: Valentin's paper-figures guidelines (FoAViS `notebooks/paper_common.py`, copied below because it cannot be
imported here): Liberation Sans 8 pt, 0.5 pt black lines, Okabe-Ito colours paired with markers, figures built at
the page width they are included at. Written by Claude (AI, Claude Opus 5.5), 2026-11-20; restyled and extended
(records, saccades) 2026-11-23.
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from scipy.stats import norm  # noqa: E402

from canvit.scanpath.data import eval_part, load_dataset  # noqa: E402
from canvit.scanpath.evaluate import select  # noqa: E402
from canvit.scanpath.generative import target_hit_index, tfp_auc, tfp_curve  # noqa: E402
from canvit.scanpath.models import GridScanpathModel, grid_log_density, policy_entropy  # noqa: E402

# ---------------------------------------------------------------------------------------------------- style
# From FoAViS notebooks/paper_common.py (OKABE_ITO, BASELINE_COLOR, set_paper_style, despine, mark_fixation).
OKABE_ITO = {"black": "#000000", "orange": "#E69F00", "skyblue": "#56B4E9", "green": "#009E73",
             "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7"}
BASELINE_COLOR = "0.35"  # every baseline / reference line
TEXTWIDTH_IN = 6.53  # this document: US letter, 2.5 cm margins (preamble.tex)
FS = 8
# One meaning per colour in every figure: (colour, marker, label). Humans are the data, drawn in black.
STYLE = {
    "human": (OKABE_ITO["black"], "o", "humans"),
    "canvit": (OKABE_ITO["blue"], "s", "CanViT search policy"),
    "deepgaze3": (OKABE_ITO["orange"], "^", "DeepGaze III"),
    "centre_bias": (BASELINE_COLOR, "D", "centre bias"),
    "uniform": (BASELINE_COLOR, "v", "uniform"),
}
PDF_META = {"CreationDate": None}  # no timestamp: a rerun with the same data writes the same PDF
PNG_PREVIEW = False  # --png: also write a .png beside every .pdf (for looking at renders; the notes use the .pdf)


def set_style(fonts: Path | None) -> None:
    """paper_common.set_paper_style(8), with Liberation Sans registered from ``fonts`` (FoAViS notebooks/fonts)."""
    for f in sorted(fonts.glob("*.ttf")) if fonts else ():
        font_manager.fontManager.addfont(str(f))
    family = "Liberation Sans"
    if not any(f.name == family for f in font_manager.fontManager.ttflist):
        warnings.warn("Liberation Sans not found (pass --fonts or set SCANPATH_FONTS_DIR): falling back to DejaVu Sans")
        family = "DejaVu Sans"
    plt.rcParams.update({
        "font.family": family, "font.size": FS, "axes.titlesize": FS, "axes.labelsize": FS,
        "xtick.labelsize": FS, "ytick.labelsize": FS, "legend.fontsize": FS - 1, "figure.titlesize": FS,
        "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5, "xtick.major.size": 2.0, "ytick.major.size": 2.0, "lines.linewidth": 0.5,
        "patch.linewidth": 0.5, "grid.linewidth": 0.5, "hatch.linewidth": 0.5, "axes.edgecolor": "black",
        "axes.labelcolor": "black", "text.color": "black", "xtick.color": "black", "ytick.color": "black",
        "axes.prop_cycle": matplotlib.cycler(color=[OKABE_ITO[k] for k in
                                                    ("orange", "skyblue", "green", "purple", "blue", "vermillion")]),
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": None, "savefig.pad_inches": 0.01,
        "pdf.fonttype": 42, "ps.fonttype": 42, "legend.frameon": False,
        "mathtext.fontset": "custom", "mathtext.rm": family, "mathtext.it": family + ":italic",
        "mathtext.bf": family + ":bold",
    })


def despine(ax, offset: float = 4) -> None:
    """Separated axes for line/bar/scatter charts: top and right removed, left and bottom offset outwards."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_position(("outward", offset))


def mark_fixation(ax, x, y, current: bool = True, s: float = 30, lw: float = 0.9) -> None:
    """A fixation is a '+': cyan = the current/next fixation, white = earlier ones, black outline."""
    import matplotlib.patheffects as pe
    ax.scatter(x, y, s=s, marker="+", c="cyan" if current else "white", lw=lw, zorder=5,
               path_effects=[pe.Stroke(linewidth=lw + 1.0, foreground="k"), pe.Normal()])


def save(fig, path: Path) -> None:
    fig.savefig(path, metadata=PDF_META)
    if PNG_PREVIEW:
        fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------------------------------- human reference
G = 32  # the search policy's fixation grid
POLICY_TARGET = 2.11  # nats, the search policy's entropy target (search/config.py)
CONDITIONS = {"tp": "target present", "ta": "target absent"}


def present_mask(fix) -> np.ndarray:
    return np.asarray(fix.scanpaths.scanpath_attributes["target_present"]).astype(bool)


def example_trials(stimuli, fix, out: Path) -> None:
    """One target-present and one target-absent stimulus with every subject's scanpath."""
    from pysaliency.external_datasets.coco_search18 import TASKS
    sp, tp = fix.scanpaths, present_mask(fix)
    n_sp = np.bincount(sp.n, minlength=len(stimuli))

    def pick(mask):  # first stimulus of the condition with >= 8 scanpaths of length >= 3
        for i in range(len(stimuli)):
            rows = np.flatnonzero((sp.n == i) & mask)
            if n_sp[i] >= 8 and len(rows) and (np.asarray(sp.length)[rows] >= 3).mean() > 0.5:
                return i, rows
        raise RuntimeError("no stimulus found")

    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, 2.25), layout="constrained")
    for ax, mask, name in ((axes[0], tp, "target present"), (axes[1], ~tp, "target absent")):
        i, rows = pick(mask)
        ax.imshow(stimuli.stimuli[i])
        for r in rows:
            L = int(sp.length[r])
            ax.plot(sp.xs[r][:L], sp.ys[r][:L], "-", color="white", lw=0.6, alpha=0.85)
            mark_fixation(ax, sp.xs[r][1:L], sp.ys[r][1:L], current=False, s=14, lw=0.6)
        b = sp.scanpath_attributes["target_bbox"][rows[0]]
        if np.isfinite(b).all():  # the target: an area, so a rectangle
            ax.add_patch(plt.Rectangle(b[:2], b[2], b[3], fill=False, ec=OKABE_ITO["yellow"], lw=1.0))
        ax.set_title(f"{name}: cue '{TASKS[int(stimuli.attributes['task'][i])]}', {len(rows)} subjects")
        ax.axis("off")
    save(fig, out / "example_trials.pdf")


K_SUBJECTS = 4  # subjects per entropy estimate, the same at every t (see human_entropy)


def kde_entropy(pts: np.ndarray, hw: tuple[int, int], sigma_cells: float = 1.5) -> float:
    """Entropy (nats) of a Gaussian KDE of ``pts`` [n, 2] (pixels), integrated over the policy's 32x32 grid on
    the displayed image, with a 0.1% uniform floor."""
    h, w = hw
    xe, ye = np.linspace(0, w, G + 1), np.linspace(0, h, int(round(G * h / w)) + 1)
    s = sigma_cells * w / G
    mx = np.diff(norm.cdf((xe[None] - pts[:, :1]) / s), axis=1)
    my = np.diff(norm.cdf((ye[None] - pts[:, 1:]) / s), axis=1)
    m = (my[:, :, None] * mx[:, None, :]).sum(0)
    m = 0.999 * m / m.sum() + 0.001 / m.size
    return float(-(m * np.log(m)).sum())


def kde_range(hw: tuple[int, int], k: int = K_SUBJECTS) -> tuple[float, float]:
    """What this estimator can return for k points: all at one place (floor) and spread far apart (ceiling)."""
    h, w = hw
    floor = kde_entropy(np.full((k, 2), [w / 2, h / 2]), hw)
    spread = np.array([[w * (i + 1) / (k + 1), h / 2 + (-1) ** i * h / 4] for i in range(k)])
    return floor, kde_entropy(spread, hw)


def human_entropy(stimuli, fix, mask, t_max=6, k=K_SUBJECTS, draws=5, seed=0) -> dict[int, tuple]:
    """Per fixation index t: (median, q25, q75, n_stimuli) over stimuli of the entropy (nats) of the subjects'
    fixation t (``kde_entropy``). EVERY estimate uses exactly k subjects: only stimuli with >= k subjects still
    searching at t enter, k of them drawn ``draws`` times and the entropies averaged. Using all subjects instead
    made the curve look flat (other session, 2026-11-20): searchers drop from 10 per stimulus at t=1 to a median
    of 4 at t=6, and a KDE of n points has a ceiling that grows with n
    (claude_journal/scripts_2026-11-20/entropy_n.py)."""
    rng = np.random.default_rng(seed)
    sp = fix.scanpaths
    lengths = np.asarray(sp.length)
    out = {}
    for t in range(1, t_max + 1):
        Hs = []
        for i in np.unique(sp.n[mask]):
            rows = np.flatnonzero((sp.n == i) & mask & (lengths > t))
            if len(rows) < k:
                continue
            pts = np.array([[sp.xs[r][t], sp.ys[r][t]] for r in rows])
            hw = tuple(int(v) for v in stimuli.sizes[int(i)])
            Hs.append(np.mean([kde_entropy(pts[rng.choice(len(rows), k, replace=False)], hw)
                               for _ in range(draws)]))
        if len(Hs) >= 30:  # fewer stimuli: too noisy to plot (target present has 27 at t=5)
            out[t] = (float(np.median(Hs)), float(np.percentile(Hs, 25)), float(np.percentile(Hs, 75)), len(Hs))
    return out


def entropy_figure(stimuli, fix, out: Path) -> dict:
    """Humans only, both conditions in one panel (present: solid circles, absent: dashed squares)."""
    tp = present_mask(fix)
    res = {"target present": human_entropy(stimuli, fix, tp), "target absent": human_entropy(stimuli, fix, ~tp)}
    floor, ceiling = kde_range(tuple(int(v) for v in stimuli.sizes[0]))
    fig, (ax, axn) = plt.subplots(2, 1, figsize=(0.54 * TEXTWIDTH_IN, 2.9), sharex=True, height_ratios=(3, 1),
                                  layout="constrained")
    ax.axhspan(floor, ceiling, color="0.92", lw=0)
    ax.text(0.9, ceiling + 0.08, f"estimator range, {K_SUBJECTS} subjects", color=BASELINE_COLOR, va="bottom")
    for (name, e), ls, mk in zip(res.items(), ("-", "--"), ("o", "s")):
        t = np.array(list(e))
        med, q25, q75, n = (np.array([v[k] for v in e.values()]) for k in range(4))
        ax.fill_between(t, q25, q75, color="k", alpha=0.08, lw=0)
        ax.plot(t, med, ls=ls, marker=mk, ms=3, color="k", lw=0.8, label=name)
        axn.plot(t, n, ls=ls, marker=mk, ms=2.5, color="k", lw=0.8)
    ax.axhline(POLICY_TARGET, color=BASELINE_COLOR, ls="--")
    ax.text(0.9, POLICY_TARGET + 0.1, f"policy entropy target {POLICY_TARGET} nats", color=BASELINE_COLOR)
    ax.axhline(np.log(G * 20), color=BASELINE_COLOR, ls=":")
    ax.text(0.9, np.log(G * 20) + 0.1, "uniform on the image", color=BASELINE_COLOR)
    ax.set(ylabel="entropy (nats), median, IQR", ylim=(0, 7.4))
    ax.legend(loc="lower right")
    axn.set(xlabel="fixation index t", ylabel="stimuli", xlim=(0.7, max(max(e) for e in res.values()) + 0.3),
            ylim=(0, None))
    for a in (ax, axn):
        despine(a)
    save(fig, out / "human_entropy.pdf")
    return res | {"kde_range": (floor, ceiling)}


def tfp_figure(stimuli, fix, out: Path, max_fix=6) -> tuple[np.ndarray, float]:
    """Target-fixation probability of the humans (target present): curve averaged within task, then over tasks."""
    sp, tp = fix.scanpaths, present_mask(fix)
    rows = np.flatnonzero(tp)
    L = torch.as_tensor(np.asarray(sp.length)[rows])
    xy = np.full((len(rows), int(L.max()), 2), np.nan)
    for j, r in enumerate(rows):
        xy[j, : L[j]] = np.stack([sp.xs[r][: L[j]], sp.ys[r][: L[j]]], -1)
    b = np.asarray(sp.scanpath_attributes["target_bbox"])[rows]
    boxes = torch.tensor(np.concatenate([b[:, :2], b[:, :2] + b[:, 2:]], -1)[:, None], dtype=torch.float32)
    tasks = torch.as_tensor(np.asarray(stimuli.attributes["task"])[sp.n[rows]])
    hit = target_hit_index(torch.tensor(xy, dtype=torch.float32), L, boxes)  # pixels; inside-box is frame-free
    curve = tfp_curve(hit, tasks, max_fix=max_fix).numpy()
    auc = float(tfp_auc(torch.as_tensor(curve)))
    fig, ax = plt.subplots(figsize=(0.42 * TEXTWIDTH_IN, 2.0), layout="constrained")  # minipage 0.42
    color, marker, _ = STYLE["human"]
    ax.plot(range(len(curve)), curve, marker=marker, ms=3, color=color, lw=0.8)
    ax.set(xlabel="fixations after the start", ylabel="P(target fixated)", ylim=(0, 1))
    despine(ax)
    save(fig, out / "human_tfp.pdf")
    return curve, auc


# ------------------------------------------------------------------------------------------- human saccades
# The datasets in the order of the figures: (slug, condition, short title). Free viewing has no condition.
PANELS = [("coco_search18", "tp", "COCO-Search18\ntarget present"),
          ("coco_search18", "ta", "COCO-Search18\ntarget absent"),
          ("coco_freeview", "both", "COCO-FreeView"), ("mit1003", "both", "MIT1003"), ("cat2000", "both", "CAT2000"),
          ("daemons", "both", "DAEMONS"), ("figrim", "both", "FIGRIM")]
LEN_BINS = np.linspace(0, 1, 41)  # saccade length, fraction of the stimulus' longer side
DIR_BINS = np.deg2rad(np.arange(-7.5, 360, 15))  # 24 directions, bins centred on the cardinal directions


def saccades(xs: list[np.ndarray], ys: list[np.ndarray], n: np.ndarray, sizes) -> dict[str, np.ndarray]:
    """Every saccade between consecutive fixations of each scanpath (pixel coordinates; fixation 0 is the start
    fixation, which every dataset here has at the image centre). ``length``: fraction of the stimulus' longer
    side, which is the side of CanViT's square scene, so humans and model share the unit; ``angle``: radians,
    0 = rightwards, pi/2 = upwards (image y points down); ``first``: the scanpath's first saccade.
    (pysaliency 0.2.23 has no saccade statistics, only scanpath plotting; checked 2026-11-23.)"""
    length, angle, first = [], [], []
    for x, y, i in zip(xs, ys, n):
        ok = np.isfinite(x) & np.isfinite(y)
        dx, dy = np.diff(np.asarray(x)[ok]), np.diff(np.asarray(y)[ok])
        if not len(dx):
            continue
        length.append(np.hypot(dx, dy) / max(sizes[int(i)]))
        angle.append(np.mod(np.arctan2(-dy, dx), 2 * np.pi))
        first.append(np.arange(len(dx)) == 0)
    return {k: np.concatenate(v) for k, v in (("length", length), ("angle", angle), ("first", first))}


def human_saccades(fixations, stimuli) -> dict[str, np.ndarray]:
    sp = fixations.scanpaths
    L = np.asarray(sp.length)
    return saccades([sp.xs[r][: L[r]] for r in range(len(L))], [sp.ys[r][: L[r]] for r in range(len(L))],
                    np.asarray(sp.n), stimuli.sizes)


def saccade_summary(s: dict[str, np.ndarray]) -> dict[str, float]:
    """Saccades, median length, share horizontal / vertical (within 22.5 deg of the axis), share of first saccades
    going left."""
    a, f = s["angle"], s["first"]
    off_h = np.minimum(np.abs(np.angle(np.exp(1j * a))), np.abs(np.angle(np.exp(1j * (a - np.pi)))))
    off_v = np.abs(np.angle(np.exp(1j * (a - np.pi / 2))) % np.pi)
    off_v = np.minimum(off_v, np.pi - off_v)
    left = (a[f] > np.pi / 2) & (a[f] < 3 * np.pi / 2)
    return {"n": int(len(a)), "median_length": float(np.median(s["length"])),
            "horizontal": float((off_h < np.pi / 8).mean()), "vertical": float((off_v < np.pi / 8).mean()),
            "first_left": float(left.mean()), "first_median_length": float(np.median(s["length"][f]))}


def dir_hist(angle: np.ndarray) -> np.ndarray:
    """Share of saccades per direction bin (angles in [0, 2 pi); the bin around 0 deg wraps)."""
    h, _ = np.histogram(np.where(angle >= DIR_BINS[-1], angle - 2 * np.pi, angle), bins=DIR_BINS)
    return h / max(h.sum(), 1)


def saccade_figure(sacc: dict[str, dict], out: Path, model: dict[str, dict] | None = None) -> None:
    """Columns = datasets. Top: saccade length histogram (density over the fraction of the longer side). Bottom:
    direction histogram (share of saccades per 15 deg bin), all saccades filled, first saccades as outline.
    ``model``: the same statistics of a model's sampled scanpaths, drawn as a blue line over the humans."""
    keys = [k for k in sacc]
    every = list(sacc.values()) + list((model or {}).values())
    ymax = max(np.histogram(s["length"], LEN_BINS, density=True)[0].max() for s in every)  # one scale for all
    rmax = max(max(dir_hist(s["angle"]).max(), dir_hist(s["angle"][s["first"]]).max()) for s in every)
    fig = plt.figure(figsize=(TEXTWIDTH_IN, 2.6), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.02, hspace=0.02)
    gs = fig.add_gridspec(2, len(keys), height_ratios=(1, 1.3))
    hc = STYLE["human"][0]
    for j, key in enumerate(keys):
        s = sacc[key]
        ax = fig.add_subplot(gs[0, j])
        ax.hist(s["length"], bins=LEN_BINS, density=True, color="0.8", ec="none")
        ax.hist(s["length"], bins=LEN_BINS, density=True, histtype="step", color=hc, lw=0.5)
        if model and key in model:
            ax.hist(model[key]["length"], bins=LEN_BINS, density=True, histtype="step", color=STYLE["canvit"][0],
                    lw=0.8)
        ax.set_title(key, fontsize=FS - 1)
        ax.set(xlim=(0, 0.8), xticks=(0, 0.4, 0.8), xticklabels=("0", ".4", ".8"), ylim=(0, ymax * 1.02),
               yticks=(0, 4, 8))
        if j:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("density")
        if j == len(keys) // 2:
            ax.set_xlabel("saccade length (fraction of the image's longer side)")
        despine(ax, offset=2)
        pa = fig.add_subplot(gs[1, j], projection="polar")
        th = DIR_BINS[:-1] + np.pi / 24  # bin centres
        pa.bar(th, dir_hist(s["angle"]), width=np.pi / 12, color="0.8", ec="none")
        pa.bar(th, dir_hist(s["angle"][s["first"]]), width=np.pi / 12, fill=False, ec=hc, lw=0.5)
        if model and key in model:
            h = dir_hist(model[key]["angle"])
            pa.plot(np.r_[th, th[:1]], np.r_[h, h[:1]], color=STYLE["canvit"][0], lw=0.8)
        pa.set(xticks=np.deg2rad([0, 90, 180, 270]), yticks=[], ylim=(0, rmax))
        pa.set_xticklabels(["R", "U", "L", "D"] if j == 0 else [], fontsize=FS - 1)
        pa.tick_params(pad=-1)
        pa.spines["polar"].set_linewidth(0.5)
        pa.grid(lw=0.3, color="0.7")
    handles = [plt.Rectangle((0, 0), 1, 1, color="0.8", label="humans, all saccades"),
               plt.Rectangle((0, 0), 1, 1, fill=False, ec=hc, lw=0.5, label="humans, first saccade")]
    if model:
        handles.append(plt.Line2D([], [], color=STYLE["canvit"][0], lw=0.8, label=STYLE["canvit"][2] + ", samples"))
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles))
    save(fig, out / "saccades.pdf")


def saccade_table(summ: dict[str, dict], out: Path) -> dict[str, str]:
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"dataset & saccades & median length & horizontal & vertical & 1st: median length & 1st: left \\",
             r"\midrule"]
    for key, s in summ.items():
        lines.append(f"{key.replace(chr(10), ' ')} & {s['n']:,} & {s['median_length']:.2f} & "
                     f"{100 * s['horizontal']:.0f}\\% & {100 * s['vertical']:.0f}\\% & "
                     f"{s['first_median_length']:.2f} & {100 * s['first_left']:.0f}\\% \\\\".replace(",", "{,}"))
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "saccades.tex").write_text("\n".join(lines) + "\n")
    m = list(summ.values())
    return {"saccLenMin": f"{min(s['median_length'] for s in m):.2f}",
            "saccLenMax": f"{max(s['median_length'] for s in m):.2f}",
            "saccHorMin": f"{100 * min(s['horizontal'] for s in m):.0f}",
            "saccHorMax": f"{100 * max(s['horizontal'] for s in m):.0f}",
            "saccLeftMin": f"{100 * min(s['first_left'] for s in m):.0f}",
            "saccLeftMax": f"{100 * max(s['first_left'] for s in m):.0f}"}


# ------------------------------------------------------------------------------------------------- records
def load_records(results: Path) -> list[dict]:
    """Every scanpath evaluation record under ``results``, with ``source`` = its directory (``baselines``,
    ``deepgaze3`` or a CanViT checkpoint name) and ``kind`` = the STYLE key. Records written before 2026-11-23
    sit at the top level, cover COCO-Search18 only and have no ``dataset``/``split``; they are used only while
    ``baselines/`` does not exist. Records of a partial run (``limit_scanpaths``) are skipped."""
    recs = []
    for p in sorted(results.rglob("*.json")):
        try:
            r = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        if not isinstance(r, dict) or r.get("task") != "scanpath":
            continue
        top = p.parent == results
        if top and (results / "baselines").is_dir():
            continue
        pr = r["protocol"]
        if pr.get("limit_scanpaths") is not None:  # a smoke test on part of the data, not a result
            print(f"load_records: skipped {p} (limit_scanpaths={pr['limit_scanpaths']})")
            continue
        r.setdefault("dataset", "coco_search18")
        r.setdefault("split", f"{pr['condition']}_{pr['part']}")
        r["source"] = "baselines" if top else p.parent.name
        r["kind"] = r["model"] if r["source"] == "baselines" else ("deepgaze3" if r["source"] == "deepgaze3"
                                                                    else "canvit")
        r["path"] = p
        recs.append(r)
    return recs


def per_index(metrics: dict, key: str) -> tuple[np.ndarray, np.ndarray]:
    t = sorted(int(k[len(key) + 2:]) for k in metrics if k.startswith(f"{key}_t"))
    return np.array(t), np.array([metrics[f"{key}_t{i}"] for i in t])


def ig_figure(recs: list[dict], out: Path, t_max: int = 8) -> bool:
    """IG over the centre bias (bits/fixation) per fixation index, COCO-Search18 TP and TA validation."""
    panels = [(c, [r for r in recs if r["dataset"] == "coco_search18" and r["split"] == f"{c}_validation"])
              for c in CONDITIONS]
    if not any(rs for _, rs in panels):
        print("ig_figure: no COCO-Search18 validation records yet")
        return False
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, 2.1), sharey=True, layout="constrained")
    seen = {}
    for ax, (cond, rs) in zip(axes, panels):
        ax.axhline(0, color=STYLE["centre_bias"][0], lw=0.5)
        for r in sorted(rs, key=lambda r: list(STYLE).index(r["kind"])):
            if r["kind"] == "centre_bias":
                continue
            t, v = per_index(r["metrics"], "ig")
            keep = t <= t_max
            color, marker, label = STYLE[r["kind"]]
            if r["kind"] == "canvit" and len({x["source"] for x in recs if x["kind"] == "canvit"}) > 1:
                label = f"{label} ({r['source']})"
            ls = ":" if r["kind"] == "uniform" else "-"
            seen[label] = ax.plot(t[keep], v[keep], ls=ls, marker=marker, ms=3, color=color, lw=0.8, label=label)[0]
        ax.set(title=CONDITIONS[cond], xlabel="fixation index t", xticks=range(1, t_max + 1), xlim=(0.6, t_max + 0.4))
        despine(ax)
    axes[0].set_ylabel("IG over centre bias (bits/fix)")
    seen["centre bias"] = plt.Line2D([], [], color=STYLE["centre_bias"][0], lw=0.5)
    fig.legend(handles=list(seen.values()), labels=list(seen), loc="outside upper center", ncol=len(seen))
    save(fig, out / "ig_by_index.pdf")
    return True


# ------------------------------------------------------------------------------------ CanViT predictions
TRAINED_FIXATIONS = 3  # the exp39 search policy: 4 glimpses = the start fixation + 3 chosen ones
DATA_LOCATION: Path | None = None  # set by main (or the notebook): pysaliency's dataset location


@lru_cache(maxsize=8)
def dataset(slug: str, part: str):
    """``load_dataset`` once per (dataset, part) per process."""
    return load_dataset(slug, part, location=DATA_LOCATION)


def predictions(rec: dict):
    """(stimuli, the scored fixations, GridScanpathModel) of a CanViT record whose ``--cfg.save-predictions`` .npz
    sits beside it, else None. The .npz keys index the scanpaths after ``select`` with the record's protocol."""
    npz = rec["path"].with_suffix(".npz")
    if not npz.exists():
        return None
    pr = rec["protocol"]
    st, fx = dataset(rec["dataset"], pr.get("part", eval_part(rec["dataset"])))
    return st, select(fx, condition=pr.get("condition", "both"), limit=pr.get("limit_scanpaths")), \
        GridScanpathModel.load(npz)


def samples(rec: dict):
    """(stimuli, the sampled scanpaths' npz) of a CanViT record, from ``samples_<dataset>_<split>.npz`` beside it."""
    p = rec["path"].parent / f"samples_{rec['path'].stem}.npz"
    if not p.exists():
        return None
    pr = rec["protocol"]
    return dataset(rec["dataset"], pr.get("part", eval_part(rec["dataset"])))[0], np.load(p)


def canvit_records(recs: list[dict], ds: str, split: str) -> list[dict]:
    return [r for r in recs if r["kind"] == "canvit" and r["dataset"] == ds and r["split"] == split]


def by_index(H: np.ndarray, t: np.ndarray, t_max: int) -> dict[int, tuple[float, float, float]]:
    """Per fixation index (>= 20 values): median, q25, q75."""
    return {i: (float(np.median(H[t == i])), *(float(np.percentile(H[t == i], q)) for q in (25, 75)))
            for i in range(1, t_max + 1) if (t == i).sum() >= 20}


def policy_entropy_figure(recs: list[dict], human: dict, kde: tuple[float, float], out: Path, t_max: int = 6) -> bool:
    """DIAGNOSTIC. Top: entropy (nats) of the policy's next-fixation distribution per fixation index at tau = 1
    and at the global tau fitted on training scanpaths, next to the humans' (median and IQR over stimuli, k
    subjects). Bottom: the fitted temperatures, per index (secondary) and global (headline)."""
    if not any(canvit_records(recs, "coco_search18", f"{c}_validation") for c in CONDITIONS):
        print("policy_entropy_figure: no CanViT COCO-Search18 records yet")
        return False
    color, marker, label = STYLE["canvit"]
    fig, axes = plt.subplots(2, 2, figsize=(TEXTWIDTH_IN, 3.4), sharex=True, sharey="row", height_ratios=(2.3, 1),
                             layout="constrained")
    handles = {}
    for j, cond in enumerate(CONDITIONS):
        ax, axt = axes[:, j]
        ax.axhspan(*kde, color="0.92", lw=0)
        ax.axvspan(TRAINED_FIXATIONS + 0.5, t_max + 0.5, color="0.97", lw=0)
        ax.axhline(POLICY_TARGET, color=BASELINE_COLOR, ls="--")
        e = human[CONDITIONS[cond]]
        t = np.array(list(e))
        med, q25, q75 = (np.array([v[k] for v in e.values()]) for k in range(3))
        ax.fill_between(t, q25, q75, color="k", alpha=0.08, lw=0)
        handles["humans (KDE of 4 subjects)"] = ax.plot(t, med, marker="o", ms=3, color="k", lw=0.8)[0]
        for r in canvit_records(recs, "coco_search18", f"{cond}_validation")[:1]:
            d = r.get("diagnostics", {})
            glob = d.get("tempered_global", {}).get("temperature")
            p = predictions(r)
            if p is None:  # only the record: means at tau = 1
                ti, v = per_index({k: v for k, v in d.items() if k.startswith("entropy_t")}, "entropy")
                handles[rf"{label}, $\tau$ = 1 (mean)"] = ax.plot(ti[ti <= t_max], v[ti <= t_max], marker=marker, ms=3,
                                                              color=color, lw=0.8)[0]
            else:
                st, fx, m = p
                for temp, ls, name in ((1.0, "-", r"$\tau$ = 1"),
                                       (glob, "--", rf"global $\tau$ = {glob:.2f}" if glob else "")):
                    if temp is None:
                        continue
                    H, ti = policy_entropy(m.with_temperature(float(temp)), st, fx)
                    b = by_index(H, ti, t_max)
                    tt = np.array(list(b))
                    if ls == "-":
                        ax.fill_between(tt, [b[i][1] for i in tt], [b[i][2] for i in tt], color=color, alpha=0.15,
                                        lw=0)
                    handles[f"{label}, {name}"] = ax.plot(tt, [b[i][0] for i in tt], ls=ls, marker=marker, ms=3,
                                                          color=color, lw=0.8, mfc="white" if ls == "--" else color)[0]
            per = d.get("tempered_per_index", {}).get("temperature")
            if isinstance(per, dict):
                tt = np.array(sorted(int(k) for k in per))
                axt.plot(tt[tt <= t_max], [per[str(i)] if str(i) in per else per[i] for i in tt[tt <= t_max]],
                         marker=marker, ms=3, color=color, lw=0.8, ls="none")
            if glob:
                axt.axhline(glob, color=color, ls="--")
            axt.axhline(1, color=BASELINE_COLOR, lw=0.5)
        ax.set(title=CONDITIONS[cond], ylim=(0, 7.4))
        axt.set(xlabel="fixation index t", xticks=range(1, t_max + 1), xlim=(0.6, t_max + 0.4))
        for a in (ax, axt):
            despine(a)
    axes[0, 0].set_ylabel("entropy (nats)")
    axes[1, 0].set_ylabel(r"fitted $\tau$")
    handles[f"policy entropy target ({POLICY_TARGET} nats)"] = plt.Line2D([], [], color=BASELINE_COLOR, ls="--")
    handles["range of the human estimator"] = plt.Rectangle((0, 0), 1, 1, color="0.92")
    handles["beyond the trained horizon"] = plt.Rectangle((0, 0), 1, 1, color="0.97", ec="0.8", lw=0.3)
    fig.legend(handles=list(handles.values()), labels=list(handles), loc="outside upper center", ncol=3)
    save(fig, out / "policy_entropy.pdf")
    return True


def qualitative_figure(recs: list[dict], out: Path, steps=(1, 2, 3, 4), n_rows: int = 3, seed: int = 0) -> bool:
    """Rows: human scanpaths drawn at random (seed) among COCO-Search18 target-present scanpaths with more than
    max(steps) fixations. Columns: the policy's next-fixation density for fixation t after the human's
    fixations 0..t-1 (magma, per-panel scale; earlier fixations white +, the human's fixation t cyan +); last
    column (when samples exist): all humans' scanpaths on the stimulus (white) and CanViT's samples (blue). The
    target box in yellow; the cue at the left."""
    from pysaliency.external_datasets.coco_search18 import TASKS
    rs = [r for r in canvit_records(recs, "coco_search18", "tp_validation") if predictions(r) is not None]
    if not rs:
        print("qualitative_figure: no CanViT predictions (.npz) yet")
        return False
    r = rs[0]
    st, fx, m = predictions(r)
    smp = samples(r)
    sp = fx.scanpaths
    L = np.asarray(sp.length)
    ok = [i for i in range(len(L)) if L[i] > max(steps) and all((i, t) in m.logits for t in steps)]
    rows = np.sort(np.random.default_rng(seed).choice(ok, n_rows, replace=False))
    ncols = len(steps) + (smp is not None)
    h0, w0 = st.sizes[int(sp.n[rows[0]])]
    fig, axes = plt.subplots(n_rows, ncols, figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN / ncols * h0 / w0 * n_rows + 0.3),
                             squeeze=False, layout="constrained")
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.01, hspace=0.01)
    for a, i in zip(axes, rows):
        n = int(sp.n[i])
        img, hw = st.stimuli[n], tuple(int(v) for v in st.sizes[n])
        x, y = np.asarray(sp.xs[i]), np.asarray(sp.ys[i])
        box = np.asarray(sp.scanpath_attributes["target_bbox"][i])  # xywh pixels
        for ax, t in zip(a, steps):
            ax.imshow(img)
            ax.add_patch(plt.Rectangle(box[:2], box[2], box[3], fill=False, ec=OKABE_ITO["yellow"], lw=0.6))
            ax.imshow(np.exp(grid_log_density(m.logits[(int(i), t)], hw=hw, scene=m.scene)), cmap="magma", alpha=0.7)
            mark_fixation(ax, x[:t], y[:t], current=False, s=18, lw=0.7)
            mark_fixation(ax, x[t], y[t], current=True, s=24, lw=0.8)
            if i == rows[0]:
                ax.set_title(f"fixation {t}\n" + ("(beyond training)" if t > TRAINED_FIXATIONS else ""))
        if smp is not None:
            ax = a[-1]
            ax.imshow(img)
            for k in np.flatnonzero(sp.n == n):
                ax.plot(np.asarray(sp.xs[k])[: L[k]], np.asarray(sp.ys[k])[: L[k]], color="white", lw=0.5)
            st_, z = smp
            for k in np.flatnonzero(z["n"] == n):
                ax.plot(z["xs"][k], z["ys"][k], color=STYLE["canvit"][0], lw=0.6, marker=".", ms=2)
            if i == rows[0]:
                ax.set_title("humans (white)\nCanViT samples (blue)")
        for ax in a:
            ax.set(xticks=[], yticks=[])
            for sp_ in ax.spines.values():
                sp_.set_visible(False)
        a[0].set_ylabel(f"cue '{TASKS[int(st.attributes['task'][n])]}'")
    save(fig, out / "qualitative.pdf")
    return True


# The evaluated (dataset, split) pairs in the order of the cross-dataset figure and table.
EVALS = [("coco_search18", "tp_validation", "COCO-Search18 TP"), ("coco_search18", "ta_validation", "COCO-Search18 TA"),
         ("coco_freeview", "validation", "COCO-FreeView"), ("mit1003", "all", "MIT1003"), ("cat2000", "all", "CAT2000"),
         ("daemons", "validation", "DAEMONS"), ("figrim", "all", "FIGRIM")]
METRICS = [("ig_image", "IG over centre bias\n(bits/fix)"), ("ll_image", "LL over uniform\n(bits/fix)"),
           ("auc_image", "AUC"), ("nss_image", "NSS")]


def read_references(path: Path | None) -> list[dict]:
    """helper_2's reference_numbers.csv (model, dataset, split, condition, metric, value, unit, comparable,
    protocol_notes, source): the published rows with one of our metrics (ig, ll, auc, nss) whose numbers are at
    least defined like ours (``comparable`` yes / metric_only), mapped onto our EVALS rows."""
    import csv
    if path is None or not path.exists():
        return []
    norm_ = {e[0].replace("_", ""): e[0] for e in EVALS}
    out = []
    for row in csv.DictReader(path.open()):
        ds = norm_.get(row["dataset"].lower().replace("-", "").replace("_", "").replace(" ", ""))
        if ds is None or row["metric"] not in ("ig", "ll", "auc", "nss") or row["comparable"] not in ("yes",
                                                                                                    "metric_only"):
            continue
        cond = row["condition"].strip().lower()
        split = next((e[1] for e in EVALS if e[0] == ds and (ds != "coco_search18" or e[1].startswith(cond))), None)
        if split is None:
            continue
        try:
            value = float(row["value"])
        except ValueError:
            continue
        out.append({"model": row["model"], "dataset": ds, "split": split, "metric": f"{row['metric']}_image",
                    "value": value, "comparable": row["comparable"], "source": row["source"]})
    return out


REF_MARKERS = ["P", "X", "*", "h"]  # published models other than DeepGaze III, in order of appearance


def cross_dataset_figure(recs: list[dict], out: Path, refs: list[dict] = ()) -> bool:
    """Rows = evaluated (dataset, split), columns = metrics (per-image means, pysaliency); one marker per model.
    Published numbers (``read_references``) are hollow markers: DeepGaze III in its colour, other models purple.
    None of them is on exactly our split (comparable = metric_only), so they place our numbers, not rank them."""
    models = [k for k in ("canvit", "deepgaze3") if any(r["kind"] == k for r in recs)]
    if not models:
        print("cross_dataset_figure: no CanViT / DeepGaze III records yet")
        return False
    kinds = models + ["centre_bias"]
    fig, axes = plt.subplots(1, len(METRICS), figsize=(TEXTWIDTH_IN, 0.24 * len(EVALS) + 0.85), sharey=True,
                             layout="constrained")
    handles = {}
    for ax, (key, name) in zip(axes, METRICS):
        for k, kind in enumerate(kinds):
            color, marker, label = STYLE[kind]
            for row, (ds, split, _) in enumerate(EVALS):
                r = next((r for r in recs if r["kind"] == kind and r["dataset"] == ds and r["split"] == split), None)
                if r is None or key not in r["metrics"]:
                    continue
                handles[label] = ax.plot(r["metrics"][key], row + (k - (len(kinds) - 1) / 2) * 0.22, marker=marker,
                                         ms=4, color=color, ls="none")[0]
        others = sorted({f["model"] for f in refs if "deepgaze iii" not in f["model"].lower()})
        for f in refs:
            if f["metric"] != key:
                continue
            row = next(i for i, e in enumerate(EVALS) if e[:2] == (f["dataset"], f["split"]))
            dg3 = "deepgaze iii" in f["model"].lower()
            c, mk = (STYLE["deepgaze3"][0], STYLE["deepgaze3"][1]) if dg3 else \
                (OKABE_ITO["purple"], REF_MARKERS[others.index(f["model"]) % len(REF_MARKERS)])
            lab = f"{f['model']}, published" + ("" if f["comparable"] == "yes" else " (other split)")
            handles[lab] = ax.plot(f["value"], row + 0.33, marker=mk, ms=4, mfc="none", mec=c, ls="none")[0]
        if key == "ig_image":
            ax.axvline(0, color=BASELINE_COLOR, lw=0.5)
        ax.set(xlabel=name)
        despine(ax)
    axes[0].set(yticks=range(len(EVALS)), yticklabels=[e[2] for e in EVALS], ylim=(len(EVALS) - 0.5, -0.5))
    fig.legend(handles=list(handles.values()), labels=list(handles), loc="outside upper center",
               ncol=min(len(handles), 4))
    save(fig, out / "cross_dataset.pdf")
    return True


def results_table(recs: list[dict], out: Path) -> dict[str, str]:
    """tables/results.tex: per evaluated (dataset, split), IG / LL / AUC / NSS (per-image means) of every model
    with a record; returns macros for the headline numbers (CanViT IG on COCO-Search18)."""
    kinds = [k for k in ("canvit", "deepgaze3", "centre_bias", "uniform") if any(r["kind"] == k for r in recs)]
    lines = [r"\begin{tabular}{llrrrrr}", r"\toprule", r"data & model & fixations & IG & LL & AUC & NSS \\"]
    macros = {}
    for ds, split, title in EVALS:
        rows = []
        for kind in kinds:
            r = next((r for r in recs if r["kind"] == kind and r["dataset"] == ds and r["split"] == split), None)
            if r is None:
                continue
            m = r["metrics"]
            vals = " & ".join(f"{m[k]:.3f}".replace("-", "$-$") for k in ("ig_image", "ll_image", "auc_image",
                                                                            "nss_image"))
            rows.append(f" & {STYLE[kind][2]} & {int(m['n_fixations'])} & {vals} \\\\")
            if kind == "canvit" and ds == "coco_search18":
                tag = split[:2].upper()
                macros |= {f"cvIG{tag}": f"{m['ig_image']:.2f}", f"cvLL{tag}": f"{m['ll_image']:.2f}"}
                macros |= {f"cvIG{tag}t{i}": f"{m[f'ig_t{i}']:.2f}".replace("-", "$-$") for i in range(1, 4)
                           if f"ig_t{i}" in m}
        if rows:
            rows[0] = title + rows[0]
            lines += [r"\midrule", *rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "results.tex").write_text("\n".join(lines) + "\n")
    return macros


def canvit_saccades(recs: list[dict]) -> dict[str, dict]:
    """Saccade statistics of CanViT's sampled scanpaths, keyed like the human panels (PANELS titles)."""
    res = {}
    for slug, cond, title in PANELS:
        split = f"{cond}_validation" if slug == "coco_search18" else eval_part(slug)
        for r in [r for r in recs if r["kind"] == "canvit" and r["dataset"] == slug and r["split"] == split][:1]:
            if (sm := samples(r)) is not None:
                st, z = sm
                res[title] = saccades(list(z["xs"]), list(z["ys"]), z["n"], st.sizes)
    return res


def baselines_table(results: Path, out: Path) -> dict[str, str]:
    """tables/baselines.tex from the COCO-Search18 baseline records; returns number macros for the text."""
    rows, macros = [], {}
    recs = [r for r in load_records(results) if r["source"] == "baselines" and r["dataset"] == "coco_search18"]
    for cond in CONDITIONS:
        for model in ("uniform", "centre_bias"):
            r = next((r for r in recs if r["model"] == model and r["split"] == f"{cond}_validation"), None)
            if r is None:
                continue
            m = r["metrics"]
            if model == "centre_bias":
                tag = cond.upper()
                macros |= {f"cb{tag}": f"{m['ll_image']:.2f}", f"cb{tag}one": f"{m['ll_t1']:.2f}",
                           f"cb{tag}three": f"{m['ll_t3']:.2f}", f"cb{tag}nfix": f"{int(m['n_fixations'])}"}
            rows.append(f"{CONDITIONS[cond].split()[1]} & {model.replace('_', ' ')} & "
                        f"{int(m['n_fixations'])} & {m['ll_image']:.3f} & {m['ll_fix']:.3f} & "
                        f"{m['auc_image']:.3f} & {m['nss_image']:.3f} \\\\")
    lines = [r"\begin{tabular}{llrrrrr}", r"\toprule",
             r"condition & model & fixations & LL (image) & LL (fix.) & AUC & NSS \\", r"\midrule", *rows,
             r"\bottomrule", r"\end{tabular}"]
    (out / "baselines.tex").write_text("\n".join(lines) + "\n")
    return macros


def scandiff_table(runs: Path, out: Path) -> None:
    """ScanDiff's metrics (KL divergence to the human-human distribution; lower = more human-like) for the
    sanity 'models', validation split, 100 stimuli each."""
    # duration-free columns only (CanViT has no fixation durations; scandiff_eval.md)
    cols = ["MM_Avg_spatial", "ScanMatchNoDur", "SequenceScore"]
    lines = [r"\begin{tabular}{llrrr}", r"\toprule", r"condition & model & MultiMatch & ScanMatch & SeqScore \\",
             r"\midrule"]
    for cond in ("tp", "ta"):
        for model in ("human", "centre", "random"):
            p = runs / f"cocosearch18_{cond}_validation_{model}_k100.json"
            if not p.exists():
                continue
            kld = json.loads(p.read_text())["kld"]
            vals = [f"{kld[k]:.2f}" for k in cols]
            lines.append(f"{'present' if cond == 'tp' else 'absent'} & {model} & " + " & ".join(vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "scandiff_sanity.tex").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------------------------------- main
def env_dir(arg: Path | None, *names: str, sub: str = "") -> Path | None:
    if arg is not None:
        return arg
    for n in names:
        if v := os.environ.get(n):
            return Path(v) / sub if n == "LOGS_DIR" else Path(v)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fonts", type=Path, default=None, help="Liberation Sans .ttf dir; default $SCANPATH_FONTS_DIR")
    ap.add_argument("--scandiff-runs", type=Path, default=None,
                    help="the ScanDiff driver's sanity runs; default $SCANDIFF_RUNS_DIR")
    ap.add_argument("--results", type=Path, default=None,
                    help="evaluation records; default $SCANPATH_RESULTS_DIR, else $LOGS_DIR/scanpath_eval")
    ap.add_argument("--data", type=Path, default=None, help="pysaliency dataset location; default $SCANPATH_DATA_DIR")
    ap.add_argument("--references", type=Path, default=None,
                    help="published numbers (reference_numbers.csv); default $SCANPATH_REFERENCES")
    ap.add_argument("--png", action="store_true", help="also write .png previews beside the .pdf figures")
    a = ap.parse_args()
    global PNG_PREVIEW
    PNG_PREVIEW = a.png
    a.fonts = env_dir(a.fonts, "SCANPATH_FONTS_DIR")
    a.scandiff_runs = env_dir(a.scandiff_runs, "SCANDIFF_RUNS_DIR")
    a.results = env_dir(a.results, "SCANPATH_RESULTS_DIR", "LOGS_DIR", sub="scanpath_eval")
    a.data = env_dir(a.data, "SCANPATH_DATA_DIR")
    a.references = env_dir(a.references, "SCANPATH_REFERENCES")
    if a.scandiff_runs is None or a.results is None or a.data is None:
        raise SystemExit("set --scandiff-runs/--results/--data or their env variables (source .envrc)")
    set_style(a.fonts)
    figs, tabs = a.out / "figures", a.out / "tables"
    figs.mkdir(parents=True, exist_ok=True)
    tabs.mkdir(parents=True, exist_ok=True)

    global DATA_LOCATION
    DATA_LOCATION = a.data
    sv, fv = dataset("coco_search18", "validation")
    example_trials(sv, fv, figs)
    ent = entropy_figure(sv, fv, figs)
    curve, auc = tfp_figure(sv, fv, figs)
    scandiff_table(a.scandiff_runs, tabs)
    cb_macros = baselines_table(a.results, tabs)
    recs = load_records(a.results)
    ig_figure(recs, figs)
    policy_entropy_figure(recs, ent, ent["kde_range"], figs)
    qualitative_figure(recs, figs)
    cross_dataset_figure(recs, figs, read_references(a.references))
    res_macros = results_table(recs, tabs)

    sacc = {}
    for slug, cond, title in PANELS:
        st, fx = dataset(slug, eval_part(slug))
        sacc[title] = human_saccades(select(fx, condition=cond, limit=None), st)
    saccade_figure(sacc, figs, model=canvit_saccades(recs) or None)
    summ = {k: saccade_summary(s) for k, s in sacc.items()}
    sacc_macros = saccade_table(summ, tabs)

    tp = present_mask(fv)
    stats = {"validation_stimuli": len(sv), "scanpaths_tp": int(tp.sum()), "scanpaths_ta": int((~tp).sum()),
             "median_length_tp": float(np.median(np.asarray(fv.scanpaths.length)[tp])),
             "median_length_ta": float(np.median(np.asarray(fv.scanpaths.length)[~tp])),
             "entropy": {k: {t: v[0] for t, v in e.items()} for k, e in ent.items() if k != "kde_range"},
             "entropy_n_stimuli": {k: {t: v[3] for t, v in e.items()} for k, e in ent.items() if k != "kde_range"},
             "kde_range_nats": list(ent["kde_range"]),
             "tfp_curve": [round(float(c), 4) for c in curve], "tfp_auc": auc,
             "saccades": {k.replace("\n", " "): v for k, v in summ.items()}}
    (tabs / "numbers.json").write_text(json.dumps(stats, indent=1))
    ent_tp, ent_ta = (stats["entropy"][k] for k in ("target present", "target absent"))
    floor, ceiling = ent["kde_range"]
    macros = {  # the numbers the text quotes, so none is typed by hand
        "nStimVal": stats["validation_stimuli"], "nTP": stats["scanpaths_tp"], "nTA": stats["scanpaths_ta"],
        "medLenTP": f"{stats['median_length_tp']:.0f}", "medLenTA": f"{stats['median_length_ta']:.0f}",
        "tfpAUC": f"{auc:.2f}", "tfpOne": f"{curve[1]:.2f}", "tfpThree": f"{curve[3]:.2f}", "tfpSix": f"{curve[6]:.2f}",
        "entTPone": f"{ent_tp[1]:.1f}", "entTPmin": f"{min(ent_tp.values()):.1f}",
        "entTAmin": f"{min(ent_ta.values()):.1f}", "entTAmax": f"{max(ent_ta.values()):.1f}",
        "entTPlast": f"{ent_tp[max(ent_tp)]:.1f}", "entTAone": f"{ent_ta[1]:.1f}",
        "entTAlast": f"{ent_ta[max(ent_ta)]:.1f}", "entTlast": max(ent_tp),
        "kSubj": K_SUBJECTS, "entFloor": f"{floor:.1f}", "entCeil": f"{ceiling:.1f}",
    } | cb_macros | sacc_macros | res_macros
    (tabs / "numbers.tex").write_text(
        "% Generated by canvit scripts/scanpath/figures.py -- do not edit by hand.\n"
        + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items()))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
