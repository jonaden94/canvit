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

from canvit.scanpath.data import eval_part, letterbox_geometry, load_dataset  # noqa: E402
from canvit.scanpath.evaluate import select  # noqa: E402
from canvit.scanpath.generative import target_hit_index, tfp_auc, tfp_curve  # noqa: E402
from canvit.scanpath.models import GridScanpathModel, grid_log_density, policy_entropy  # noqa: E402

# ---------------------------------------------------------------------------------------------------- style
# From FoAViS notebooks/paper_common.py (OKABE_ITO, BASELINE_COLOR, set_paper_style, despine, mark_fixation).
# one preliminary checkpoint: exp39 s0, best validation step 15000 of a finished 20000-step run;
# every CanViT label names it
CKPT = "exp39 s0"
# The search task head's map read as a fixation distribution: off in every figure and table since 2026-11-23
# (Valentin: conceptually the wrong measurement); records stay on disk; --heatmap turns it back on.
SHOW_HEATMAP = False
OKABE_ITO = {"black": "#000000", "orange": "#E69F00", "skyblue": "#56B4E9", "green": "#009E73",
             "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7"}
BASELINE_COLOR = "0.35"  # every baseline / reference line
TEXTWIDTH_IN = 6.53  # this document: US letter, 2.5 cm margins (preamble.tex)
FS = 8
# One meaning per colour in every figure: (colour, marker, label). Humans are the data, drawn in black.
STYLE = {
    "human": (OKABE_ITO["black"], "o", "humans"),
    "canvit": (OKABE_ITO["blue"], "s", f"CanViT search policy ({CKPT})"),
    "canvit_heatmap": (OKABE_ITO["skyblue"], "o", f"CanViT search heatmap ({CKPT})"),  # the task head as a distribution
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


def saccades(xs: list[np.ndarray], ys: list[np.ndarray], n: np.ndarray, sizes) -> dict[str, np.ndarray]:
    """Every saccade between consecutive fixations of each scanpath (pixel coordinates; fixation 0 is the start
    fixation, which every dataset here has at the image centre). ``length``: fraction of the stimulus' longer
    side, which is the side of CanViT's square scene, so humans and model share the unit; ``angle``: radians,
    0 = rightwards, pi/2 = upwards (image y points down); ``first``: the scanpath's first saccade; ``length_px``:
    the length in pixels.
    (pysaliency 0.2.23 has no saccade statistics, only scanpath plotting; checked 2026-11-23.)
    Also per saccade: ``sp`` (scanpath number), ``index`` (1 = the saccade leaving the start fixation),
    ``turn`` (degrees, signed angle from the previous saccade, 0 = straight on, +-180 = straight back; NaN for
    the first), ``n`` (stimulus) and ``longer`` (the stimulus' longer side, pixels)."""
    keys = ("length", "length_px", "angle", "first", "sp", "index", "turn", "n", "longer")
    out: dict[str, list] = {k: [] for k in keys}
    for j, (x, y, i) in enumerate(zip(xs, ys, n)):
        ok = np.isfinite(x) & np.isfinite(y)
        dx, dy = np.diff(np.asarray(x)[ok]), np.diff(np.asarray(y)[ok])
        if not len(dx):
            continue
        longer = max(sizes[int(i)])
        a = np.arctan2(-dy, dx)
        out["length_px"].append(np.hypot(dx, dy))
        out["length"].append(out["length_px"][-1] / longer)
        out["angle"].append(np.mod(a, 2 * np.pi))
        out["first"].append(np.arange(len(dx)) == 0)
        out["sp"].append(np.full(len(dx), j))
        out["index"].append(np.arange(1, len(dx) + 1))
        out["turn"].append(np.concatenate([[np.nan], np.degrees(np.angle(np.exp(1j * np.diff(a))))]))
        out["n"].append(np.full(len(dx), int(i)))
        out["longer"].append(np.full(len(dx), float(longer)))
    return {k: np.concatenate(v) if v else np.zeros(0) for k, v in out.items()}


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
            "median_length_px": float(np.median(s["length_px"])),
            "horizontal": float((off_h < np.pi / 8).mean()), "vertical": float((off_v < np.pi / 8).mean()),
            "first_left": float(left.mean()), "first_median_length": float(np.median(s["length"][f]))}


def saccade_table(summ: dict[str, dict], out: Path, ppd: dict[str, float] | None = None,
                  model: dict[str, dict] | None = None) -> dict[str, str]:
    """Humans per dataset, then (``model``) CanViT's samples on the datasets it has them for. ``ppd``: pixels per
    degree of visual angle per panel title, where the viewing geometry is known (helper_2's DeepGaze III records,
    protocol.deepgaze.ppd); the degrees column shows a dash elsewhere. The macros describe the humans only."""
    ppd = ppd or {}
    lines = [r"\begin{tabular}{llrrrrrrr}", r"\toprule",
             r"dataset & from & saccades & median length & median length (deg) & horizontal & vertical & "
             r"1st: median length & 1st: left \\"]
    for who, rows in (("humans", summ), (f"CanViT samples ({CKPT})", model or {})):
        if rows:
            lines.append(r"\midrule")
        for key, s in rows.items():
            deg = f"{s['median_length_px'] / ppd[key]:.1f}" if key in ppd else "--"
            lines.append(f"{key.replace(chr(10), ' ')} & {who} & {s['n']:,} & {s['median_length']:.2f} & {deg} & "
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
HELD_OUT: dict[str, str] = {}  # dataset -> reason; FIGRIM released 2026-11-23 (helper_1: framed record = headline)


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
        if not isinstance(r, dict) or r.get("task") != "scanpath" or "metrics" not in r:
            continue  # also a shard of a sharded run (no metrics until merged)
        if p.stem.endswith("_scene"):  # the unframed comparison run (helper_1); the framed record is the headline
            continue
        if "retina" in p.relative_to(results).parts:  # the retina sweep (helper_1): out-of-distribution retinas,
            continue                                   # read only by its own figure
        top = p.parent == results
        if top and (results / "baselines").is_dir():
            continue
        pr = r["protocol"]
        if r.get("dataset") in HELD_OUT and p.parent != results and p.parent.name not in ("baselines", "deepgaze3"):
            print(f"load_records: held out {p} ({HELD_OUT[r['dataset']]})")
            continue
        if pr.get("limit_scanpaths") is not None:  # a smoke test on part of the data, not a result
            print(f"load_records: skipped {p} (limit_scanpaths={pr['limit_scanpaths']})")
            continue
        r.setdefault("dataset", "coco_search18")
        r.setdefault("split", f"{pr['condition']}_{pr['part']}")
        r["source"] = "baselines" if top else p.parent.name
        if r["source"] in ("baselines", "deepgaze3"):
            r["kind"] = r["model"] if r["source"] == "baselines" else "deepgaze3"
        else:
            r["kind"] = "canvit_heatmap" if "heatmap" in r["model"] else "canvit"  # canvit_search_heatmap
            if r["kind"] == "canvit_heatmap" and not SHOW_HEATMAP:
                continue
        r["path"] = p
        recs.append(r)
    for ds in {r["dataset"] for r in recs}:  # every IG and every reference row against ONE centre bias per dataset
        fits = {(r["protocol"]["centre_bias"]["bandwidth"], r["protocol"]["centre_bias"]["eps"])
                for r in recs if r["dataset"] == ds and isinstance(r["protocol"].get("centre_bias"), dict)}
        if len(fits) > 1:
            raise SystemExit(f"{ds}: records measured against different centre-bias fits {sorted(fits)}; "
                             "re-run the stale ones or move them out of the results directory")
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
            ls = {"uniform": ":", "canvit_heatmap": "--"}.get(r["kind"], "-")
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
    """(stimuli, the sampled scanpaths' npz) of a CanViT record, from ``samples_<dataset>_<split>.npz`` beside it
    (free viewing with a search policy: ``samples_<dataset>_<split>_cuemix.npz``, the 18-cue mixture per step)."""
    p = next((q for q in (rec["path"].parent / f"samples_{rec['path'].stem}{x}.npz" for x in ("", "_cuemix"))
              if q.exists()), None)
    if p is None:
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
                                       (glob, "--", r"global $\tau$ (fitted)")):
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
        glob_t = [r.get("diagnostics", {}).get("tempered_global", {}).get("temperature")
                  for r in canvit_records(recs, "coco_search18", f"{cond}_validation")[:1]]
        ax.set(title=CONDITIONS[cond] + (rf" (global $\tau$ = {glob_t[0]:.2f})" if glob_t and glob_t[0] else ""),
               ylim=(0, 7.4))
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
                ax.set_title(f"humans (white)\nCanViT {CKPT} (blue)")
        for ax in a:
            ax.set(xticks=[], yticks=[])
            for sp_ in ax.spines.values():
                sp_.set_visible(False)
        a[0].set_ylabel(f"cue '{TASKS[int(st.attributes['task'][n])]}'")
    save(fig, out / "qualitative.pdf")
    return True


def argmax_points(logits: np.ndarray, hw: tuple[int, int], scene: int) -> np.ndarray:
    """[N, G*G] grid logits of one stimulus -> [N, 2] (x, y) stimulus pixels of each argmax cell, the argmax taken
    over the cells that overlap the image (letterbox padding excluded); the point is the centre of the part of
    the cell that lies on the image. Also returns that centre for every image cell ([M, 2], for a random cell)."""
    h, w = hw
    g = int(round(logits.shape[1] ** 0.5))
    ch, cw, top, left = letterbox_geometry(h, w, scene)
    edges = np.arange(g + 1) * scene / g  # cell edges in scene pixels
    x0, x1 = np.clip(edges[:-1], left, left + cw), np.clip(edges[1:], left, left + cw)
    y0, y1 = np.clip(edges[:-1], top, top + ch), np.clip(edges[1:], top, top + ch)
    on = ((y1 - y0) > 0)[:, None] & ((x1 - x0) > 0)[None, :]  # [row, col]
    cx, cy = ((x0 + x1) / 2 - left) * w / cw, ((y0 + y1) / 2 - top) * h / ch  # stimulus pixels
    centres = np.stack(np.broadcast_arrays(cx[None, :], cy[:, None]), -1).reshape(-1, 2)
    best = np.where(on.ravel()[None], logits, -np.inf).argmax(1)
    return centres[best], centres[on.ravel()]


def policy_vs_human(recs: list[dict], out: Path, steps=(1, 2, 3, 4), near: float = 0.05) -> dict[str, str]:
    """tables/policy_vs_human.tex: for fixation t after the human's fixations 0..t-1 (teacher forced), the human's
    fixation t vs the policy's argmax cell: saccade amplitude from fixation t-1, distance to the nearest earlier
    fixation (both means, fraction of the longer side), share within ``near`` of an earlier fixation; and at t = 1
    (target present) the share inside the target box, with the expectation for a uniformly random image cell."""
    rows, macros = [], {}
    for cond in CONDITIONS:
        r = next((r for r in canvit_records(recs, "coco_search18", f"{cond}_validation") if predictions(r)), None)
        if r is None:
            continue
        st, fx, m = predictions(r)
        sp = fx.scanpaths
        L = np.asarray(sp.length)
        stat = {t: {k: [] for k in ("amp_h", "amp_p", "near_h", "near_p")} for t in steps}
        box = {"h": [], "p": [], "rand": []}
        for i in range(len(L)):
            ts = [t for t in steps if t < L[i] and (i, t) in m.logits]
            if not ts:
                continue
            n = int(sp.n[i])
            hw = tuple(int(v) for v in st.sizes[n])
            side = max(hw)
            xy = np.stack([np.asarray(sp.xs[i])[: L[i]], np.asarray(sp.ys[i])[: L[i]]], -1)
            pts, cells = argmax_points(np.stack([m.logits[(i, t)] for t in ts]), hw, m.scene)
            for t, pp in zip(ts, pts):
                for who, q in (("h", xy[t]), ("p", pp)):
                    stat[t][f"amp_{who}"].append(np.hypot(*(q - xy[t - 1])) / side)
                    stat[t][f"near_{who}"].append(np.hypot(*(xy[:t] - q).T).min() / side)
            if cond == "tp" and ts[0] == 1:
                b = np.asarray(sp.scanpath_attributes["target_bbox"][i])  # xywh pixels

                def inside(q):
                    return ((q[..., 0] >= b[0]) & (q[..., 0] <= b[0] + b[2])
                            & (q[..., 1] >= b[1]) & (q[..., 1] <= b[1] + b[3]))

                box["h"].append(inside(xy[1]))
                box["p"].append(inside(pts[0]))
                box["rand"].append(inside(cells).mean())
        for t in steps:
            v = {k: np.asarray(x) for k, x in stat[t].items()}
            if not len(v["amp_h"]):
                continue
            rows.append(f"{CONDITIONS[cond].split()[1]} & {t} & {len(v['amp_h']):,} & {v['amp_h'].mean():.2f} & "
                        f"{v['amp_p'].mean():.2f} & {v['near_h'].mean():.2f} & {v['near_p'].mean():.2f} & "
                        f"{(v['near_h'] < near).mean():.2f} & {(v['near_p'] < near).mean():.2f} \\\\"
                        .replace(",", "{,}"))
        if box["h"]:
            vals = {k: float(np.mean(x)) for k, x in box.items()}
            macros |= {"pvhBoxHumanTP": f"{vals['h']:.3f}", "pvhBoxPolicyTP": f"{vals['p']:.3f}",
                       "pvhBoxRandomTP": f"{vals['rand']:.3f}", "pvhBoxN": f"{len(box['h'])}"}
    if not rows:
        print("policy_vs_human: no CanViT COCO-Search18 predictions yet")
        return {}
    lines = [r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
             r" & & & \multicolumn{2}{c}{amplitude} & \multicolumn{2}{c}{nearest earlier fixation} & "
             rf"\multicolumn{{2}}{{c}}{{within {near:g} of an earlier one}} \\",
             r"condition & $t$ & fixations & humans & policy & humans & policy & humans & policy \\", r"\midrule",
             *rows]
    if macros:
        lines += [r"\midrule", rf"\multicolumn{{9}}{{l}}{{target present, $t=1$, inside the target box: humans "
                               rf"{macros['pvhBoxHumanTP']}, policy argmax {macros['pvhBoxPolicyTP']}, random image "
                               rf"cell {macros['pvhBoxRandomTP']} ({macros['pvhBoxN']} scanpaths)}} \\"]
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "policy_vs_human.tex").write_text("\n".join(lines) + "\n")
    return macros


# Post-hoc priors (canvit.scanpath.priors): variant key, macro tag, row label. Policy (squares) and heatmap
# (circles) share a row; macros prIG<tag><cond> / prGain<tag><cond> (policy), prHmIG<tag><cond> / prHmGain<tag><cond>
# (heatmap), cond = TP | TA | FV.
PRIOR_ROWS = [("canvit_tau", "Tau", r"CanViT@$\tau$"), ("canvit_tau+amplitude", "Amp", "+ amplitude"),
              ("canvit_tau+return", "Ret", "+ return"), ("canvit_tau+direction", "Dir", "+ direction"),
              ("canvit_tau+centre", "Cb", "+ centre bias"), ("canvit_tau+all", "All", "+ all four priors"),
              ("priors_only", "Only", "no image: all four priors"), ("centre_only", "CbOnly", "no image: centre bias")]
PRIOR_CONDS = {"tp": ("coco_search18", "tp_validation", "COCO-Search18 TP"),
               "ta": ("coco_search18", "ta_validation", "COCO-Search18 TA"),
               "fv": ("coco_freeview", "validation", "COCO-FreeView")}


def load_priors(results: Path) -> dict[tuple[str, str], dict]:
    """{(source, cond): priors JSON} for PRIOR_CONDS (canvit.scanpath.priors, annotated), source policy | heatmap."""
    out = {}
    sources = [("policy", "*/priors/*_validation.json")]
    if SHOW_HEATMAP:
        sources.append(("heatmap", "*/heatmap/priors/*_validation.json"))
    for src, pattern in sources:
        for p in sorted(results.glob(pattern)):
            r = json.loads(p.read_text())
            cond = next((c for c, (ds, sp, _) in PRIOR_CONDS.items() if ds == r["dataset"] and sp == r["split"]),
                        None)
            if cond is not None and all("label" in v for v in r["variants"].values()):
                out[(src, cond)] = r
    return out


def _signed(v: float) -> str:
    return f"{round(v, 2) + 0.0:+.2f}".replace("-", r"$-$")  # + 0.0: no "-0.00"


def _num(v: float) -> str:
    return f"{round(v, 2) + 0.0:.2f}".replace("-", r"\ensuremath{-}")


def _deepgaze(recs: list[dict], cond: str) -> dict | None:
    ds, sp, _ = PRIOR_CONDS[cond]
    return next((r for r in recs if r["kind"] == "deepgaze3" and r["dataset"] == ds and r["split"] == sp), None)


def priors_figure(recs: list[dict], results: Path, out: Path, tabs: Path) -> dict[str, str]:
    """ABSOLUTE IG over the centre bias (bits/fix, validation) of CanViT with the post-hoc priors fitted on training
    scanpaths: policy squares, heatmap circles, priors without any image grey; panels COCO-Search18 TP | TA |
    COCO-FreeView (policy = cue mixture); DeepGaze III dashed. Also tables/priors.tex and the pr* macros."""
    pri = load_priors(results)
    conds = [c for c in PRIOR_CONDS if ("policy", c) in pri]
    if not conds:
        print("priors_figure: no annotated priors results yet")
        return {}
    macros: dict[str, str] = {}
    cell = {(src, k, cond): v for (src, cond), r in pri.items() for k, v in r["variants"].items()}
    fig, axes = plt.subplots(1, len(conds), figsize=(TEXTWIDTH_IN, 0.2 * len(PRIOR_ROWS) + 0.95), sharey=True,
                             squeeze=False, layout="constrained")
    for ax, cond in zip(axes[0], conds):
        ax.axvline(0, color=BASELINE_COLOR, lw=0.5)
        dg = _deepgaze(recs, cond)
        if dg is not None:
            x = dg["metrics"]["ig_image"]
            ax.axvline(x, color=STYLE["deepgaze3"][0], ls="--", lw=0.8)
            ax.annotate(f"DGIII {x:+.2f}", (x, -0.5), xytext=(-2, 0), textcoords="offset points", ha="right",
                        color=STYLE["deepgaze3"][0], fontsize=FS - 1, va="center")
        for row, (key, tag, _) in enumerate(PRIOR_ROWS):
            for src in ("policy", "heatmap"):
                v = cell.get((src, key, cond))
                if v is None or (src == "heatmap" and not v["with_canvit"]):  # image-free rows are shared
                    continue
                ig, pre = v["ig_over_centre_bias"], "prHm" if src == "heatmap" else "pr"
                macros[f"{pre}IG{tag}{cond.upper()}"] = _num(ig)
                macros[f"{pre}Gain{tag}{cond.upper()}"] = _num(v["gain"]["bits"])
                if "gain_over_priors_only" in v:
                    macros[f"prGain{'Hm' if src == 'heatmap' else ''}AllOverOnly{cond.upper()}"] = _num(
                        v["gain_over_priors_only"])
                color, marker = (STYLE["canvit" if src == "policy" else "canvit_heatmap"][:2] if v["with_canvit"]
                                 else (BASELINE_COLOR, "D"))
                dy = ({"policy": -0.14, "heatmap": 0.14}[src] if v["with_canvit"] and SHOW_HEATMAP else 0.0)
                ax.plot(ig, row + dy, marker=marker, ms=4, color=color, ls="none")
        ax.set(title=PRIOR_CONDS[cond][2], xlabel="IG over centre bias (bits/fix)")
        despine(ax)
    axes[0, 0].set(yticks=range(len(PRIOR_ROWS)), yticklabels=[r[2] for r in PRIOR_ROWS],
                   ylim=(len(PRIOR_ROWS) - 0.5, -0.9))
    legend = {STYLE["canvit"][2] + "; free viewing: cue mixture":
              plt.Line2D([], [], color=STYLE["canvit"][0], marker="s", ls="none", ms=4)}
    if any(k[0] == "heatmap" for k in pri):
        legend[STYLE["canvit_heatmap"][2]] = plt.Line2D([], [], color=STYLE["canvit_heatmap"][0], marker="o",
                                                        ls="none", ms=4)
    legend["priors without image"] = plt.Line2D([], [], color=BASELINE_COLOR, marker="D", ls="none", ms=4)
    legend["DeepGaze III"] = plt.Line2D([], [], color=STYLE["deepgaze3"][0], ls="--", lw=0.8)
    fig.legend(list(legend.values()), list(legend), loc="outside upper center", ncol=2)
    save(fig, out / "priors.pdf")

    # one block per source; gain = over the block's first row (that row: over tau = 1), caption says so
    ncond = len(PRIOR_CONDS)
    lines = [r"\begin{tabular}{l" + "rr" * ncond + "}", r"\toprule",
             " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{t}}}" for *_, t in PRIOR_CONDS.values()) + r" \\",
             "".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(ncond)),
             "" + " & IG & gain" * ncond + r" \\"]
    for src, name in (("policy", "CanViT search policy (free viewing: cue mixture)"),
                      ("heatmap", "CanViT search heatmap")):
        block = [(k, lab) for k, _, lab in PRIOR_ROWS
                 if any((src, k, c) in cell and cell[(src, k, c)]["with_canvit"] for c in PRIOR_CONDS)]
        if not block:
            continue
        lines += [r"\midrule", rf"\multicolumn{{{1 + 2 * ncond}}}{{l}}{{\emph{{{name}}}}} \\"]
        for key, lab in block:
            vs = [cell.get((src, key, c)) for c in PRIOR_CONDS]
            vals = " & ".join(f"{_signed(v['ig_over_centre_bias'])} & {_signed(v['gain']['bits'])}" if v else "&"
                              for v in vs)
            lines.append(f"{lab.replace('CanViT@', 'at fitted ')} & {vals} \\\\")
    lines += [r"\midrule", rf"\multicolumn{{{1 + 2 * ncond}}}{{l}}{{\emph{{references}}}} \\"]
    for key, _, lab in PRIOR_ROWS:
        vs = [cell.get(("policy", key, c)) for c in PRIOR_CONDS]
        if any(vs) and not next(v for v in vs if v)["with_canvit"]:
            lines.append(f"{lab} & " + " & ".join(f"{_signed(v['ig_over_centre_bias'])} & " if v else "&"
                                                   for v in vs) + r" \\")
    dgs = [_deepgaze(recs, c) for c in PRIOR_CONDS]
    lines.append("DeepGaze III & " + " & ".join(f"{_signed(d['metrics']['ig_image'])} & " if d else "&"
                                                for d in dgs) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (tabs / "priors.tex").write_text("\n".join(lines) + "\n")
    return macros


def priors_index_figure(recs: list[dict], results: Path, out: Path, t_max: int = 8) -> bool:
    """IG over the centre bias per fixation index (validation) of CanViT at tau, CanViT (policy and heatmap) + all
    four post-hoc priors, and the priors without any image, next to DeepGaze III; panels as in priors_figure."""
    pri = load_priors(results)
    conds = [c for c in PRIOR_CONDS if ("policy", c) in pri]
    if not conds:
        print("priors_index_figure: no annotated priors results yet")
        return False
    lines_ = [("policy", "canvit_tau", STYLE["canvit"][0], "s", ":", r"CanViT policy@$\tau$"),
              ("policy", "canvit_tau+all", STYLE["canvit"][0], "s", "-", "policy + all four priors"),
              ("heatmap", "canvit_tau+all", STYLE["canvit_heatmap"][0], "o", "--", "heatmap + all four priors"),
              ("policy", "priors_only", BASELINE_COLOR, "D", "-", "no image: all four priors")]
    fig, axes = plt.subplots(1, len(conds), figsize=(TEXTWIDTH_IN, 2.2), sharey=True, squeeze=False,
                             layout="constrained")
    seen = {}
    for ax, cond in zip(axes[0], conds):
        ax.axhline(0, color=STYLE["centre_bias"][0], lw=0.5)
        for src, key, color, marker, ls, label in lines_:
            v = pri.get((src, cond), {}).get("variants", {}).get(key)
            if v is None:
                continue
            t = np.array(sorted(int(i) for i in v["ig_t"] if int(i) <= t_max))
            seen[label] = ax.plot(t, [v["ig_t"][str(i)] for i in t], ls=ls, marker=marker, ms=3, color=color,
                                  lw=0.8)[0]
        dg = _deepgaze(recs, cond)
        if dg is not None:
            t, y = per_index(dg["metrics"], "ig")
            keep = t <= t_max
            color, marker, label = STYLE["deepgaze3"]
            seen[label] = ax.plot(t[keep], y[keep], marker=marker, ms=3, color=color, lw=0.8)[0]
        ax.set(title=PRIOR_CONDS[cond][2], xlabel="fixation index t", xticks=range(1, t_max + 1),
               xlim=(0.6, t_max + 0.4))
        despine(ax)
    axes[0, 0].set_ylabel("IG over centre bias (bits/fix)")
    seen["centre bias"] = plt.Line2D([], [], color=STYLE["centre_bias"][0], lw=0.5)
    fig.legend(handles=list(seen.values()), labels=list(seen), loc="outside upper center", ncol=3)
    fig.suptitle(f"CanViT {CKPT}", x=0.01, ha="left", fontsize=FS)
    save(fig, out / "priors_by_index.pdf")
    return True


# --------------------------------------------------------- saccade statistics in the FoAViS style (Fig. 3)
# The unit switch (Valentin has not decided; manager's recommendation 2026-11-23): degrees for humans ACROSS
# datasets, fraction of the longer side (= CanViT's scene) for model vs human. Values: "deg" | "frac".
SACCADE_UNITS = {"effects": "deg", "compare": "frac"}
SHORT = {"deg": 1.0, "frac": 0.025}  # a "short" saccade: < 1 deg, or < 2.5% of the longer side
# Task regime per dataset (colour), one marker / line style per dataset inside a regime. Okabe-Ito hues not
# used for models elsewhere: humans are black in model figures, but here every line is human.
REGIME = {"search": OKABE_ITO["green"], "free viewing": OKABE_ITO["vermillion"], "memory": OKABE_ITO["purple"]}
DATASET_STYLE = {  # PANELS title -> (regime, marker, line style)
    "COCO-Search18\ntarget present": ("search", "o", "-"), "COCO-Search18\ntarget absent": ("search", "s", "--"),
    "COCO-FreeView": ("free viewing", "o", "-"), "MIT1003": ("free viewing", "s", "--"),
    "CAT2000": ("free viewing", "^", ":"), "DAEMONS": ("free viewing", "D", "-."),
    "FIGRIM": ("memory", "o", "-")}
RANDOM_REF = dict(color="0.8", lw=3.0, zorder=1, solid_capstyle="butt")  # FoAViS's uniform random policy line


def scanpath_xy(fixations) -> tuple[list, list, np.ndarray]:
    sp = fixations.scanpaths
    L = np.asarray(sp.length)
    return [sp.xs[r][: L[r]] for r in range(len(L))], [sp.ys[r][: L[r]] for r in range(len(L))], np.asarray(sp.n)


def random_xy(xs, ys, n, sizes, seed: int = 0) -> tuple[list, list, np.ndarray]:
    """The geometry-only reference: each scanpath keeps its start fixation and length, every later fixation is
    uniform on its image (FoAViS's uniform random policy)."""
    rng = np.random.default_rng(seed)
    rx, ry = [], []
    for x, y, i in zip(xs, ys, n):
        h, w = sizes[int(i)]
        k = int(np.isfinite(np.asarray(x, float)).sum())
        rx.append(np.concatenate([np.asarray(x[:1], float), rng.uniform(0, w, max(k - 1, 0))]))
        ry.append(np.concatenate([np.asarray(y[:1], float), rng.uniform(0, h, max(k - 1, 0))]))
    return rx, ry, n


def amplitude(s: dict, unit: str, ppd: float | None) -> np.ndarray:
    if unit == "deg":
        if ppd is None:
            raise ValueError("degrees need pixels per degree")
        return s["length_px"] / ppd
    return s["length"]


def _horizontal(a: np.ndarray) -> np.ndarray:  # within 22.5 deg of left/right (as saccade_summary)
    return np.minimum(np.abs(np.angle(np.exp(1j * a))), np.abs(np.angle(np.exp(1j * (a - np.pi))))) < np.pi / 8


def _vertical(a: np.ndarray) -> np.ndarray:
    return _horizontal(a - np.pi / 2)


def effect_curves(s: dict, unit: str, ppd: float | None, imax: int = 8, min_n: int = 100) -> dict[str, np.ndarray]:
    """Per saccade index 1..imax (NaN where fewer than ``min_n`` saccades): median amplitude, share horizontal,
    share vertical, share short (< SHORT[unit]), share returning (|turn| > 150 deg; index >= 2)."""
    amp, idx = amplitude(s, unit, ppd), s["index"]
    out = {k: np.full(imax, np.nan) for k in ("amplitude", "horizontal", "vertical", "short", "return", "n")}
    for i in range(1, imax + 1):
        m = idx == i
        out["n"][i - 1] = m.sum()
        if m.sum() < min_n:
            continue
        out["amplitude"][i - 1] = np.median(amp[m])
        out["horizontal"][i - 1] = _horizontal(s["angle"][m]).mean()
        out["vertical"][i - 1] = _vertical(s["angle"][m]).mean()
        out["short"][i - 1] = (amp[m] < SHORT[unit]).mean()
        if i >= 2:
            out["return"][i - 1] = (np.abs(s["turn"][m]) > 150).mean()
    return out


def effect_summary(s: dict, unit: str, ppd: float | None) -> dict[str, float]:
    """One number per effect: median amplitude; its change from saccade 2 to saccades >= 5 (ratio, < 1 = shorter
    later: coarse-to-fine); shares horizontal / vertical / short / returning over all saccades; share of first
    saccades going left."""
    amp, idx, a = amplitude(s, unit, ppd), s["index"], s["angle"]
    late = idx >= 5
    first = idx == 1
    left = (a[first] > np.pi / 2) & (a[first] < 3 * np.pi / 2)
    later = idx >= 2
    return {"amplitude": float(np.median(amp)),
            "decline": float(np.median(amp[late]) / np.median(amp[idx == 2])) if late.sum() >= 100 else np.nan,
            "horizontal": float(_horizontal(a).mean()), "vertical": float(_vertical(a).mean()),
            "short": float((amp < SHORT[unit]).mean()), "return": float((np.abs(s["turn"][later]) > 150).mean()),
            "first_left": float(left.mean())}


def saccade_effects(ppd: dict[str, float], out: Path, tabs: Path, imax: int = 8) -> dict[str, str]:
    """(a) Humans across all datasets: per saccade index, median amplitude, share horizontal / vertical, share
    returning, share short; and the first saccade's share leftward per dataset. Grey band: the uniform random
    reference (each scanpath's start and length, later fixations uniform on the image), range over datasets.
    Unit: SACCADE_UNITS["effects"]. Also tables/saccade_effects.tex (human value and random reference per
    dataset, mean over datasets) and macros sfx<Effect><Hum|Rnd>Mean."""
    unit = SACCADE_UNITS["effects"]
    hum, rnd, summ = {}, {}, {}
    for slug, cond, title in PANELS:
        st, fx = dataset(slug, eval_part(slug))
        xs, ys, n = scanpath_xy(select(fx, condition=cond, limit=None))
        p = ppd.get(title)
        if unit == "deg" and p is None:
            print(f"saccade_effects: no ppd for {title!r}, skipped")
            continue
        h, r = saccades(xs, ys, n, st.sizes), saccades(*random_xy(xs, ys, n, st.sizes), st.sizes)
        hum[title], rnd[title] = effect_curves(h, unit, p, imax), effect_curves(r, unit, p, imax)
        summ[title] = (effect_summary(h, unit, p), effect_summary(r, unit, p))
    ulab = {"deg": "deg", "frac": "fraction of longer side"}[unit]
    panels = [("amplitude", f"median amplitude ({ulab})", 1), ("horizontal", "share horizontal (±22.5°)", 1),
              ("vertical", "share vertical (±22.5°)", 1), ("return", "share returning (turn > 150°)", 2),
              ("short", f"share short (< {SHORT[unit]:g}{'°' if unit == 'deg' else ''})", 1)]
    fig, axs = plt.subplots(2, 3, figsize=(TEXTWIDTH_IN, 3.9), layout="constrained")
    t = np.arange(1, imax + 1)
    for ax, (key, ylab, i0) in zip(axs.flat, panels):
        lo = np.nanmin([r[key] for r in rnd.values()], axis=0)
        hi = np.nanmax([r[key] for r in rnd.values()], axis=0)
        ok = (t >= i0) & np.isfinite(lo)
        ax.fill_between(t[ok], lo[ok], hi[ok], color="0.88", lw=0, zorder=0)
        for title, c in hum.items():
            reg, mk, ls = DATASET_STYLE[title]
            ax.plot(t[t >= i0], c[key][t >= i0], color=REGIME[reg], marker=mk, ls=ls, ms=2.5, lw=0.8)
        ax.set(xticks=t, xlabel="saccade index", ylabel=ylab)
        if key != "amplitude":
            ax.set_ylim(0, None)
        despine(ax)
    ax = axs.flat[5]  # first saccade: share leftward, one row per dataset
    names = list(summ)
    for j, title in enumerate(names):
        reg, mk, _ = DATASET_STYLE[title]
        hv, rv = summ[title][0]["first_left"], summ[title][1]["first_left"]
        ax.plot([rv], [j], marker="|", color="0.5", ms=6, ls="none")
        ax.plot([hv], [j], marker=mk, color=REGIME[reg], ms=4, ls="none")
    ax.axvline(0.5, color=BASELINE_COLOR, lw=0.5)
    ax.set(yticks=range(len(names)), yticklabels=[k.replace("\n", " ").replace("COCO-Search18 target ", "CS18 ")
                                                  for k in names],
           ylim=(len(names) - 0.5, -0.5), xlabel="first saccade: share leftward", xlim=(0.25, 0.85))
    despine(ax)
    handles = [plt.Line2D([], [], color=REGIME[DATASET_STYLE[k][0]], marker=DATASET_STYLE[k][1],
                          ls=DATASET_STYLE[k][2], ms=3, lw=0.8) for k in hum]
    short = {"COCO-Search18\ntarget present": "CS18 TP", "COCO-Search18\ntarget absent": "CS18 TA"}
    labels = [short.get(k, k) + f" ({DATASET_STYLE[k][0]})" for k in hum]
    handles.append(plt.Rectangle((0, 0), 1, 1, color="0.88", lw=0))
    labels.append("uniform random (range; last panel |)")
    fig.legend(handles, labels, loc="outside upper center", ncol=4)
    save(fig, out / "saccade_effects.pdf")

    cols = [("amplitude", "{:.1f}" if unit == "deg" else "{:.2f}", "amplitude" + (r" ($^\circ$)" if unit == "deg"
                                                                                  else "")),
            ("decline", "{:.2f}", r"late / 2nd"), ("horizontal", "{:.0%}", r"horiz.\ \%"),
            ("vertical", "{:.0%}", r"vert.\ \%"), ("first_left", "{:.0%}", r"1st left \%"),
            ("return", "{:.0%}", r"return \%"), ("short", "{:.0%}", r"short \%")]
    lines = [r"\begin{tabular}{l" + "r" * len(cols) + "}", r"\toprule",
             "dataset & " + " & ".join(c[2] for c in cols) + r" \\", r"\midrule"]

    def cell(v, f, pct=False):  # in the table the % sits in the header
        return "--" if not np.isfinite(v) else f.format(v).replace("%", "" if pct else r"\%")

    for title, (h, r) in summ.items():
        name = {"COCO-Search18\ntarget present": "CS18 TP", "COCO-Search18\ntarget absent": "CS18 TA"}.get(title, title)
        lines.append(name + " & " + " & ".join(f"{cell(h[k], f, True)} ({cell(r[k], f, True)})" for k, f, _ in cols)
                     + r" \\")
    mean_h = {k: float(np.nanmean([v[0][k] for v in summ.values()])) for k, _, _ in cols}
    mean_r = {k: float(np.nanmean([v[1][k] for v in summ.values()])) for k, _, _ in cols}
    lines += [r"\midrule", "mean & " + " & ".join(
        f"{cell(mean_h[k], f, True)} ({cell(mean_r[k], f, True)})" for k, f, _ in cols) + r" \\", r"\bottomrule",
        r"\end{tabular}"]
    (tabs / "saccade_effects.tex").write_text("\n".join(lines) + "\n")
    tag = {"amplitude": "Amp", "decline": "Decline", "horizontal": "Hor", "vertical": "Ver", "first_left": "Left",
           "return": "Ret", "short": "Short"}
    macros = {}
    for k, f, _ in cols:
        macros[f"sfx{tag[k]}HumMean"] = cell(mean_h[k], f)
        macros[f"sfx{tag[k]}RndMean"] = cell(mean_r[k], f)
    return macros


def _kde_reflect(x: np.ndarray, grid: np.ndarray, bw: float) -> np.ndarray:
    """Gaussian KDE on ``grid``, reflected at 0 (amplitudes are >= 0). From FoAViS notebooks/paper_plots.py."""
    x = np.asarray(x, float).ravel()
    z, zr = (grid[:, None] - x[None]) / bw, (grid[:, None] + x[None]) / bw
    return (np.exp(-0.5 * z ** 2) + np.exp(-0.5 * zr ** 2)).mean(1) / (bw * np.sqrt(2 * np.pi))


def _kde_circ(a_deg: np.ndarray, grid: np.ndarray, bw: float) -> np.ndarray:
    """Gaussian KDE of angles in degrees on (-180, 180], wrapped. From FoAViS notebooks/paper_plots.py."""
    d = grid[:, None] - np.asarray(a_deg, float).ravel()[None]
    return sum(np.exp(-0.5 * ((d + o) / bw) ** 2) for o in (-360, 0, 360)).mean(1) / (bw * np.sqrt(2 * np.pi))


def _after(s: dict, amp: np.ndarray, grid: np.ndarray, bw: float = 0.35, min_w: float = 30.0) -> np.ndarray:
    """FoAViS's 'after a long saccade': next vs preceding amplitude, both z-scored within the scanpath (saccades
    from the start fixation left out), Nadaraya-Watson on ``grid``; NaN where the kernel weight is < ``min_w``."""
    keep = s["index"] >= 2
    sp, a = s["sp"][keep], amp[keep]
    xs, ys = [], []
    for j in np.unique(sp):
        v = a[sp == j]
        if len(v) >= 3 and v.std() > 0:
            z = (v - v.mean()) / v.std()
            xs.append(z[:-1])
            ys.append(z[1:])
    if not xs:
        return np.full(len(grid), np.nan)
    x, y = np.concatenate(xs), np.concatenate(ys)
    w = np.exp(-0.5 * ((grid[:, None] - x[None]) / bw) ** 2)
    sw = w.sum(1)
    return np.where(sw >= min_w, (w * y[None]).sum(1) / np.maximum(sw, 1e-12), np.nan)


def _min_fixations(xs, ys, n, k: int):
    keep = [j for j, x in enumerate(xs) if np.isfinite(np.asarray(x, float)).sum() >= k]
    return [xs[j] for j in keep], [ys[j] for j in keep], np.asarray(n)[keep]


def saccade_compare(rec: dict, ppd: float | None, out: Path, nsac: int = TRAINED_FIXATIONS) -> bool:
    """(b) CanViT's sampled scanpaths vs the humans', FoAViS Fig. 3 layout: row 1 amplitude of saccades f1-f2 ..
    (reflected KDE + 10/25/50/75/90 quantile strips), row 2 their direction (wrapped KDE), row 3 next vs
    preceding amplitude and the turn angle at f2, f3. Only scanpaths with >= nsac + 1 fixations (both sides).
    Humans black dashed, CanViT blue, the uniform random reference thick grey behind. Unit:
    SACCADE_UNITS["compare"] (frac: % of the longer side, with degrees on a secondary axis when ``ppd``)."""
    sm = samples(rec)
    if sm is None:
        return False
    st, z = sm
    ds, split = rec["dataset"], rec["split"]
    cond = split.split("_")[0] if ds == "coco_search18" else "both"
    _, fx = dataset(ds, rec["protocol"].get("part", eval_part(ds)))
    hxy = _min_fixations(*scanpath_xy(select(fx, condition=cond, limit=None)), nsac + 1)
    cxy = _min_fixations(list(z["xs"]), list(z["ys"]), z["n"], nsac + 1)
    src = {"humans": saccades(*hxy, st.sizes), "canvit": saccades(*cxy, st.sizes),
           "random": saccades(*random_xy(*hxy, st.sizes), st.sizes)}
    unit = SACCADE_UNITS["compare"]
    scale = 100.0 if unit == "frac" else 1.0
    amp = {k: amplitude(s, unit, ppd) * scale for k, s in src.items()}
    sty = {"humans": dict(color=STYLE["human"][0], ls="--", lw=0.9), "canvit": dict(color=STYLE["canvit"][0],
                                                                                   ls="-", lw=0.9),
           "random": RANDOM_REF}
    gx = np.linspace(0, 100 if unit == "frac" else float(np.percentile(amp["random"], 99)), 401)
    ga = np.linspace(-180, 180, 361)
    fig = plt.figure(figsize=(TEXTWIDTH_IN, 0.9 * TEXTWIDTH_IN), layout="constrained")
    gs = fig.add_gridspec(3, nsac)
    A = [fig.add_subplot(gs[0, t]) for t in range(nsac)]
    D = [fig.add_subplot(gs[1, t]) for t in range(nsac)]
    C = fig.add_subplot(gs[2, 0])
    T = [fig.add_subplot(gs[2, t]) for t in range(1, nsac)]
    peak = dmax = tmax = 0.0
    for k, s in src.items():
        for t in range(1, nsac + 1):
            m = s["index"] == t
            h = _kde_reflect(amp[k][m], gx, 2.0 if unit == "frac" else 0.5)
            A[t - 1].plot(gx, h, **sty[k])
            peak = max(peak, float(h.max()))
            dirs = np.degrees(np.angle(np.exp(1j * s["angle"][m & (s["length"] >= 0.01)])))
            d = _kde_circ(dirs, ga, 8.0)
            D[t - 1].plot(ga, d, **sty[k])
            dmax = max(dmax, float(d.max()))
            if t >= 2:
                mt = m & np.isfinite(s["turn"]) & (s["length"] >= 0.01)
                tt = _kde_circ(s["turn"][mt], ga, 8.0)
                T[t - 2].plot(ga, tt, **sty[k])
                tmax = max(tmax, float(tt.max()))
    grid = np.linspace(-1.5, 2.0, 71)
    for k, s in src.items():
        C.plot(grid, _after(s, amp[k], grid), **sty[k])
    dy = 0.075 * peak
    for t in range(1, nsac + 1):
        ax = A[t - 1]
        for i, k in enumerate(("humans", "canvit")):
            q = np.percentile(amp[k][src[k]["index"] == t], (10, 25, 50, 75, 90))
            y = -(i + 0.8) * dy
            c = sty[k]["color"]
            ax.plot([q[0], q[4]], [y, y], color=c, lw=0.6, solid_capstyle="butt")
            ax.plot([q[1], q[3]], [y, y], color=c, lw=3.0, alpha=0.45, solid_capstyle="butt")
            ax.plot([q[2]], [y], marker=STYLE["human" if k == "humans" else "canvit"][1], ms=3, color=c,
                    mec="white", mew=0.4, ls="none")
        ax.axhline(0, color="0.6", lw=0.5, zorder=1)
        ax.set_title(f"$f_{{{t}}}$–$f_{{{t + 1}}}$", loc="left")
        ax.set(ylim=(-2.4 * dy, peak * 1.05), xlim=(gx[0], gx[-1]),
               xlabel="amplitude (% of longer side)" if unit == "frac" else "amplitude (deg)")
        ax.spines["left"].set_bounds(0, peak * 1.05)
        if unit == "frac" and ppd:
            longer = float(np.median(src["humans"]["longer"]))
            sec = ax.secondary_xaxis("top", functions=(lambda v, f=longer / ppd / 100: v * f,
                                                       lambda v, f=longer / ppd / 100: v / f))
            sec.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}°")
            sec.tick_params(labelsize=FS - 1)
        D[t - 1].set(xlim=(-180, 180), xticks=[-180, -90, 0, 90, 180], ylim=(0, dmax * 1.08),
                     xticklabels=["left", "down", "right", "up", "left"], xlabel="direction",
                     title=f"$f_{{{t}}}$–$f_{{{t + 1}}}$")
        if t > 1:
            A[t - 1].set_yticklabels([])
            D[t - 1].set_yticklabels([])
    for j, ax in enumerate(T):
        ax.set(xlim=(-180, 180), xticks=[-180, -90, 0, 90, 180], ylim=(0, tmax * 1.08), xlabel="turn angle (deg)",
               title=f"at $f_{{{j + 2}}}$")
        if j:
            ax.set_yticklabels([])
    C.axhline(0, color="0.6", lw=0.5)
    C.axvline(0, color="0.85", lw=0.5)
    C.set(xlabel="preceding amplitude (z)", ylabel="next amplitude (z)", ylim=(-0.9, 0.9))
    A[0].set_ylabel("density")
    D[0].set_ylabel("density")
    T[0].set_ylabel("density") if T else None
    for ax in A + D + T + [C]:
        despine(ax)
    title = {"tp_validation": "COCO-Search18 target present", "ta_validation": "COCO-Search18 target absent"}.get(
        split, dict((s, t.replace("\n", " ")) for s, _, t in PANELS).get(ds, ds))
    handles = [plt.Line2D([], [], **{k: v for k, v in sty[k].items() if k in ("color", "ls", "lw")}) for k in sty]
    names = [f"humans (n = {len(hxy[0]):,})", f"CanViT samples, {CKPT} (n = {len(cxy[0]):,})",
             "uniform random fixations"]
    fig.legend(handles, names, loc="outside upper center", ncol=3, title=f"{title}, scanpaths with ≥ {nsac + 1}"
               " fixations")
    slug = {"tp_validation": "cs18_tp", "ta_validation": "cs18_ta"}.get(split, ds)
    save(fig, out / f"saccade_compare_{slug}.pdf")
    return True


# ------------------------------------------------------------------------------------------- retina sweep
TRAINED_RETINA = "cmf0.75_res65_cps6"  # the retina exp39 s0 was trained with (helper_1, 2026-11-23)
SEARCH_VAL_TRIALS = 326  # target-present val pairs behind found_t3 (helper_1, 2026-11-23; not in the record)
RES_MARKER = {65: "o", 80: "s", 92: "^"}  # retina resolution -> marker; cmf -> fill lightness
CMF_FILL = {0.75: OKABE_ITO["blue"], 2.2: "#6AA6D8", 7.7: "white"}


def load_retinas(results: Path) -> list[dict]:
    """helper_1's retina sweep, <ckpt>/retina/<cmf..._res..._cps...>/: search_validation.json (Jonathan's search
    harness) and coco_search18_{tp,ta}_validation.json (our records). One dict per retina with what exists."""
    import re
    out = []
    for d in sorted(results.glob("*/retina/cmf*_res*_cps*")):
        m = re.fullmatch(r"cmf([\d.]+)_res(\d+)_cps(\d+)", d.name)
        if m is None or not (d / "search_validation.json").exists():
            continue
        r = {"name": d.name, "cmf": float(m[1]), "res": int(m[2]), "cps": int(m[3]),
             "search": json.loads((d / "search_validation.json").read_text())["metrics"]}
        for c in CONDITIONS:
            p = d / f"coco_search18_{c}_validation.json"
            if p.exists():
                r[c] = json.loads(p.read_text())
        out.append(r)
    return out


def retina_figure(results: Path, out: Path) -> dict[str, str]:
    """H1 (search vs human alignment) across retinas: x = targets found within 3 fixations (policy, Jonathan's
    search harness, COCO-Search18 validation), y = IG over the centre bias (top: at the tau fitted on training
    scanpaths, bottom: at tau = 1), columns TP | TA. Marker = retina resolution, fill = cortical magnification;
    the trained retina ringed. Only the trained retina is in distribution: head and policy saw no other."""
    rets = [r for r in load_retinas(results) if any(c in r for c in CONDITIONS)]
    if len(rets) < 2:
        print(f"retina_figure: {len(rets)} retinas with scanpath records, need 2")
        return {}
    fig, axes = plt.subplots(2, 2, figsize=(TEXTWIDTH_IN, 3.8), sharex=True, sharey="row", layout="constrained")
    for j, cond in enumerate(CONDITIONS):
        for i, at in enumerate(("fitted", "one")):
            ax = axes[i, j]
            ax.axhline(0, color=STYLE["centre_bias"][0], lw=0.5)
            for r in rets:
                if cond not in r:
                    continue
                x = r["search"]["found_t3_policy"]
                y = (r[cond]["diagnostics"]["tempered_global"]["ig_image"] if at == "fitted"
                     else r[cond]["metrics"]["ig_image"])
                ax.errorbar(x, y, xerr=np.sqrt(x * (1 - x) / SEARCH_VAL_TRIALS), color=OKABE_ITO["blue"], lw=0.5,
                            capsize=0, zorder=1)  # binomial SE of found_t3
                ax.plot(x, y, marker=RES_MARKER.get(r["res"], "D"), ms=5, ls="none", mec=OKABE_ITO["blue"],
                        mfc=CMF_FILL.get(r["cmf"], "0.5"), mew=0.8)
                if r["name"] == TRAINED_RETINA:
                    ax.plot(x, y, marker="o", ms=11, mfc="none", mec="k", mew=0.6, ls="none")
                    ax.annotate("trained", (x, y), xytext=(7, 4), textcoords="offset points", fontsize=FS - 1)
            ax.set(title=CONDITIONS[cond] if i == 0 else None)
            if i == 1:
                ax.set_xlabel("targets found within 3 fixations (policy)")
            despine(ax)
    axes[0, 0].set_ylabel("IG over centre bias\n" + r"at fitted $\tau$ (bits/fix)")
    axes[1, 0].set_ylabel("IG over centre bias\n" + r"at $\tau$ = 1 (bits/fix)")
    handles = [plt.Line2D([], [], marker=m, ms=5, ls="none", mec="0.25", mfc="0.85")  # shape only, no fill code
               for m in RES_MARKER.values()]
    labels = [f"resolution {k}" for k in RES_MARKER]
    handles += [plt.Line2D([], [], marker="o", ms=5, ls="none", mec=OKABE_ITO["blue"], mfc=c) for c in
                CMF_FILL.values()]
    labels += [f"cortical magnification {k:g}" for k in CMF_FILL]
    fig.legend(handles, labels, loc="outside upper center", ncol=3,
               title=f"CanViT search policy {CKPT}, evaluated through {len(rets)} retinas (trained through one)")
    save(fig, out / "retina.pdf")
    # macros: means over the resolutions of each cortical magnification, tags A/B/C = cmf 0.75/2.2/7.7 (manager)
    macros = {}
    for tag, cmf in zip("ABC", sorted(CMF_FILL)):
        g = [r for r in rets if r["cmf"] == cmf and all(c in r for c in CONDITIONS)]
        if not g:
            continue
        tg = [r["tp"]["diagnostics"]["tempered_global"] for r in g]
        macros[f"retFound{tag}"] = f"{np.mean([r['search']['found_t3_policy'] for r in g]):.2f}"
        macros[f"retIGtauTP{tag}"] = _num(np.mean([t["ig_image"] for t in tg]))
        macros[f"retIGtauTA{tag}"] = _num(np.mean([r["ta"]["diagnostics"]["tempered_global"]["ig_image"] for r in g]))
        macros[f"retAUCTP{tag}"] = f"{np.mean([r['tp']['metrics']['auc_image'] for r in g]):.3f}"
        macros[f"retTauTP{tag}"] = f"{np.mean([t['temperature'] for t in tg]):.1f}"
    return macros


# --------------------------------------------------------------------------------------- temperature sweep
TAU_EVALS = [("coco_search18", "tp_validation", "COCO-Search18 TP"), ("coco_search18", "ta_validation",
             "COCO-Search18 TA"), ("coco_freeview", "validation", "COCO-FreeView"), ("mit1003", "all", "MIT1003")]
TAUS = np.geomspace(0.25, 128.0, 37)


def temperature_sweep(rec: dict, chunk: int = 20000) -> dict | None:
    """IG over the centre bias (bits/fix, per image as pysaliency) and mean map entropy (nats, over the image's
    cells) of a CanViT record's saved validation predictions at every tau in TAUS, plus tau = 1 and the fitted tau.
    IG(tau) = the record's IG + the exact per-fixation shift of log p(cell) from tau = 1 (as evaluate.py's tempered
    diagnostic). Cached in <record dir>/temperature/<stem>.json (recomputed when the .npz is newer)."""
    from scipy.special import log_softmax

    from canvit.scanpath.models import _scored_cells, without_start_fixations
    from canvit.scanpath.priors import image_average
    cache = rec["path"].parent / "temperature" / f"{rec['path'].stem}.json"
    npz = rec["path"].with_suffix(".npz")
    if cache.exists() and (not npz.exists() or cache.stat().st_mtime > npz.stat().st_mtime):
        return json.loads(cache.read_text())
    pred = predictions(rec)
    if pred is None:
        return None
    st, fx, model = pred
    fit = rec.get("diagnostics", {}).get("tempered_global", {})
    taus = np.unique(np.concatenate([TAUS, [1.0], [fit["temperature"]] if fit else []]))
    sx = without_start_fixations(fx)
    shift, H = np.zeros((len(taus), len(sx))), np.zeros((len(taus), len(sx)))
    for i in range(0, len(sx), chunk):
        logits, cells, logcount, _ = _scored_cells(model, st, sx[i:i + chunk])
        rows = np.arange(len(cells))
        base = log_softmax(logits + logcount, axis=1)[rows, cells]
        for j, tau in enumerate(taus):
            lp = log_softmax(logits / tau + logcount, axis=1)
            shift[j, i:i + len(cells)] = (lp[rows, cells] - base) / np.log(2)
            H[j, i:i + len(cells)] = -(np.exp(lp) * np.where(np.isfinite(lp), lp, 0.0)).sum(1)
    ig0 = rec["metrics"]["ig_image"]
    res = {"record": str(rec["path"]), "taus": taus.tolist(), "fitted_tau": fit.get("temperature"),
           "fitted_ig_record": fit.get("ig_image"), "ig_image": [ig0 + image_average(s, sx.n) for s in shift],
           "entropy_fix": H.mean(1).tolist(), "n_fixations": len(sx)}
    cache.parent.mkdir(exist_ok=True)
    cache.write_text(json.dumps(res, indent=1))
    return res


def temperature_figure(recs: list[dict], out: Path) -> dict[str, str]:
    """How much the temperature matters for the policy (validation): top, IG over the centre bias vs tau (log
    axis); bottom, the mean entropy of the policy's map vs tau with the humans' entropy (KDE of K_SUBJECTS
    subjects' fixation t, median over stimuli, mean over t = 1..TRAINED_FIXATIONS). tau = 1 (as trained) and the
    tau fitted on training scanpaths are marked. Macros tauIG<One|Fit><tag>, tauFit<tag>."""
    rows = []
    for ds, split, title in TAU_EVALS:
        rec = next(iter(canvit_records(recs, ds, split)), None)
        sw = temperature_sweep(rec) if rec is not None else None
        if sw is not None:
            rows.append((ds, split, title, rec, sw))
    if not rows:
        print("temperature_figure: no CanViT predictions yet")
        return {}
    color, marker, label = STYLE["canvit"]
    fig, axes = plt.subplots(2, len(rows), figsize=(TEXTWIDTH_IN, 3.2), sharex=True, sharey="row", squeeze=False,
                             layout="constrained")
    macros = {}
    for j, (ds, split, title, rec, sw) in enumerate(rows):
        taus, ig, H = np.array(sw["taus"]), np.array(sw["ig_image"]), np.array(sw["entropy_fix"])
        one, fit = int(np.argmin(abs(taus - 1.0))), sw["fitted_tau"]
        k = int(np.argmin(abs(taus - fit))) if fit else None
        ax, axh = axes[:, j]
        ax.axhline(0, color=STYLE["centre_bias"][0], lw=0.5)
        ax.plot(taus, ig, color=color, lw=0.8)
        axh.plot(taus, H, color=color, lw=0.8)
        st, fx = dataset(ds, rec["protocol"].get("part", eval_part(ds)))
        mask = np.ones(len(fx.scanpaths), dtype=bool)
        if ds == "coco_search18":
            mask = present_mask(fx) if split.startswith("tp") else ~present_mask(fx)
        hum = human_entropy(st, fx, mask, t_max=TRAINED_FIXATIONS)
        if hum:
            h_ref = float(np.mean([v[0] for v in hum.values()]))
            axh.axhline(h_ref, color=STYLE["human"][0], ls="--", lw=0.8)
            axh.annotate(f"humans {h_ref:.1f}", (taus[-1], h_ref), xytext=(0, -2), textcoords="offset points",
                         ha="right", va="top", fontsize=FS - 1)
        for a, y in ((ax, ig), (axh, H)):
            a.plot(taus[one], y[one], marker="o", mfc="white", mec=color, ms=4, ls="none")
            if k is not None:
                a.plot(taus[k], y[k], marker=marker, color=color, ms=4, ls="none")
        tag = {"tp_validation": "TP", "ta_validation": "TA"}.get(split, {"coco_freeview": "FV", "mit1003": "MIT"}
                                                                  .get(ds, ""))
        macros[f"tauIGOne{tag}"] = _num(ig[one])
        if k is not None:
            macros[f"tauIGFit{tag}"], macros[f"tauFit{tag}"] = _num(ig[k]), f"{fit:.1f}"
            if sw["fitted_ig_record"] is not None and abs(ig[k] - sw["fitted_ig_record"]) > 0.01:
                print(f"temperature_figure: {title}: sweep IG at fitted tau {ig[k]:.3f} != record "
                      f"{sw['fitted_ig_record']:.3f}")
        at_bound = fit is not None and fit >= 0.999 * np.exp(4.0)  # evaluate.py fits log tau in [-4, 4]
        note = (f"\nfitted $\\tau$ at the fit's bound ({fit:.0f})" if at_bound else f"\nfitted $\\tau$ = {fit:.1f}")
        ax.set(title=title + (note if fit else ""), xscale="log")
        axh.set(xlabel=r"temperature $\tau$")
        despine(ax), despine(axh)
    axes[0, 0].set(ylabel="IG over centre bias\n(bits/fix)", ylim=(-12, 1.5))  # tau < ~0.5 runs off the bottom
    axes[1, 0].set_ylabel("map entropy (nats)")
    handles = [plt.Line2D([], [], color=color, lw=0.8),
               plt.Line2D([], [], marker="o", mfc="white", mec=color, ms=4, ls="none"),
               plt.Line2D([], [], marker=marker, color=color, ms=4, ls="none"),
               plt.Line2D([], [], color=STYLE["human"][0], ls="--", lw=0.8)]
    fig.legend(handles, [label + "; free viewing: cue mixture", r"$\tau$ = 1 (as trained)",
                         r"$\tau$ fitted on training scanpaths",
                         f"humans (KDE of {K_SUBJECTS} subjects, t = 1-{TRAINED_FIXATIONS})"],
               loc="outside upper center", ncol=2)
    save(fig, out / "temperature.pdf")
    return macros


# The evaluated (dataset, split) pairs in the order of the cross-dataset figure and table.
EVALS = [("coco_search18", "tp_validation", "COCO-Search18 TP"), ("coco_search18", "ta_validation", "COCO-Search18 TA"),
         ("coco_freeview", "validation", "COCO-FreeView"), ("mit1003", "all", "MIT1003"), ("cat2000", "all", "CAT2000"),
         ("daemons", "validation", "DAEMONS"), ("figrim", "all", "FIGRIM (memorisation)")]
METRICS = [("ig_image", "IG over centre bias\n(bits/fix)"), ("ll_image", "LL over uniform\n(bits/fix)"),
           ("auc_image", "AUC"), ("nss_image", "NSS")]


# Published models drawn in the cross-dataset figure (reference_numbers.csv names -> (STYLE key or None, label)):
# DeepGaze3.5-VL's Table 1 covers all five free-viewing datasets and its search model COCO-Search18 TP/TA, with
# IG over each dataset's centre bias (helper_2, 2026-11-23). The csv holds many more; one figure shows these.
PUBLISHED = {"DeepGaze III-6B": ("deepgaze3", "DeepGaze III-6B, published"),
             "DeepGaze3.5-VL (combined)": (None, "DeepGaze3.5-VL, published"),
             "DeepGaze3.5-VL (search)": (None, "DeepGaze3.5-VL, published")}


def read_references(path: Path | None) -> list[dict]:
    """helper_2's reference_numbers.csv (model, dataset, split, condition, metric, value, unit, comparable,
    protocol_notes, source): the PUBLISHED models' rows with one of our metrics (ig, ll, auc, nss) whose numbers
    are at least defined like ours (``comparable`` yes / metric_only), mapped onto our EVALS rows."""
    import csv
    if path is None or not path.exists():
        return []
    norm_ = {e[0].replace("_", ""): e[0] for e in EVALS}
    out = []
    for row in csv.DictReader(path.open()):
        ds = norm_.get(row["dataset"].lower().replace("-", "").replace("_", "").replace(" ", ""))
        if (ds is None or row["model"] not in PUBLISHED or row["metric"] not in ("ig", "ll", "auc", "nss")
                or row["comparable"] not in ("yes", "metric_only")):
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


def cross_dataset_figure(recs: list[dict], out: Path, refs: list[dict] = ()) -> bool:
    """Rows = evaluated (dataset, split), columns = metrics (per-image means, pysaliency); one marker per model.
    Published numbers (``read_references``) are hollow markers: DeepGaze III in its colour, DeepGaze3.5-VL purple.
    None of them is on exactly our split (comparable = metric_only), so they place our numbers, not rank them."""
    models = [k for k in ("canvit", "canvit_heatmap", "deepgaze3") if any(r["kind"] == k for r in recs)]
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
                handles[label] = ax.plot(r["metrics"][key], row + (k - (len(kinds) - 1) / 2) * 0.15, marker=marker,
                                         ms=4, color=color, ls="none")[0]
        for f in refs:
            if f["metric"] != key:
                continue
            row = next(i for i, e in enumerate(EVALS) if e[:2] == (f["dataset"], f["split"]))
            kind, lab = PUBLISHED[f["model"]]
            c, mk = STYLE[kind][:2] if kind else (OKABE_ITO["purple"], "P")
            lab += "" if f["comparable"] == "yes" else " (other split)"
            handles[lab] = ax.plot(f["value"], row + 0.38, marker=mk, ms=4, mfc="none", mec=c, ls="none")[0]
        if key == "ig_image":
            ax.axvline(0, color=BASELINE_COLOR, lw=0.5)
        ax.set(xlabel=name)
        despine(ax)
    axes[0].set(yticks=range(len(EVALS)), yticklabels=[e[2] for e in EVALS], ylim=(len(EVALS) - 0.5, -0.5))
    fig.legend(handles=list(handles.values()), labels=list(handles), loc="outside upper center",
               ncol=min(len(handles), 3))
    save(fig, out / "cross_dataset.pdf")
    return True


def results_table(recs: list[dict], out: Path) -> dict[str, str]:
    """tables/results.tex: per evaluated (dataset, split), IG / LL / AUC / NSS (per-image means) of every model
    with a record; returns macros for the headline numbers (CanViT IG on COCO-Search18)."""
    kinds = [k for k in ("canvit", "canvit_heatmap", "deepgaze3", "centre_bias", "uniform")
             if any(r["kind"] == k for r in recs)]
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
            # tables keep short names; the caption names the checkpoint
            name = STYLE[kind][2].replace(f" ({CKPT})", "")
            if r["model"].endswith("cuemix"):  # 18 cues
                name = "CanViT policy, cue mixture"
            rows.append(f" & {name} & {int(m['n_fixations'])} & {vals} \\\\")
            tag = {"tp_validation": "TP", "ta_validation": "TA"}.get(split) if ds == "coco_search18" else \
                {"coco_freeview": "FV"}.get(ds)
            if kind in ("canvit", "canvit_heatmap", "deepgaze3") and tag:
                pre = {"canvit": "cv", "canvit_heatmap": "hm", "deepgaze3": "dg"}[kind]
                d = r.get("diagnostics", {}).get("tempered_global", {})
                vals = {f"{pre}IG{tag}": m["ig_image"], f"{pre}LL{tag}": m["ll_image"],
                        f"{pre}AUC{tag}": m["auc_image"],
                        f"{pre}NSS{tag}": m["nss_image"]} | {f"{pre}IG{tag}t{w}": m[f"ig_t{i}"]  # no digits in names
                                                          for i, w in ((1, "One"), (2, "Two"), (3, "Three"))
                                                          if f"ig_t{i}" in m}
                if d:  # the temperature diagnostic: global tau and the IG at that tau
                    vals |= {f"{pre}Tau{tag}": d["temperature"], f"{pre}IGtau{tag}": d["ig_image"]}
                fmt = {k: (".3f" if "AUC" in k else ".2f") for k in vals}
                macros |= {k: f"{v:{fmt[k]}}".replace("-", r"\ensuremath{-}") for k, v in vals.items()}
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


def scandiff_test_table(runs: Path, out: Path) -> None:
    """ScanDiff's evaluator on the full COCO-Search18 test split (unmodified, helper_2's driver): KL(human-pair
    || model-pair similarity distribution) in nats, lower = more human-like. 'human' is held-out subjects, the
    noise floor. Duration-free columns only (CanViT has no fixation durations)."""
    cols = [("MM_Vector", "vector"), ("MM_Direction", "direction"), ("MM_Length", "length"),
            ("MM_Position", "position"), ("ScanMatchNoDur", "ScanMatch"), ("SequenceScore", "SS"),
            ("SemanticSequenceScore", "SemSS")]
    models = [("human", "humans (held out)"), ("canvit_mr-r0-base-s0", "CanViT policy"),
              ("centre", "centre dummy"), ("random", "random dummy")]
    lines = [r"\begin{tabular}{ll" + "r" * len(cols) + "}", r"\toprule",
             r" & & \multicolumn{4}{c}{MultiMatch} & & & \\", r"\cmidrule(lr){3-6}",
             "condition & model & " + " & ".join(c for _, c in cols) + r" \\", r"\midrule"]
    found = False
    for cond in ("tp", "ta"):
        rows = {}
        for model, label in models:
            p = runs / f"cocosearch18_{cond}_test_{model}.json"
            if p.exists():
                rows[label] = json.loads(p.read_text())["kld"]
        if not rows:
            continue
        found = True
        best = {k: min(v[k] for lab, v in rows.items() if lab != "humans (held out)") for k, _ in cols}
        for label, kld in rows.items():
            vals = [(r"\textbf{%.2f}" if label != "humans (held out)" and kld[k] == best[k] else "%.2f") % kld[k]
                    for k, _ in cols]
            lines.append(f"{'present' if cond == 'tp' else 'absent'} & {label} & " + " & ".join(vals) + r" \\")
        if cond == "tp":
            lines.append(r"\midrule")
    if not found:
        print("ScanDiff test runs missing; skipped tables/scandiff.tex")
        return
    if lines[-1] == r"\midrule":
        lines.pop()
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "scandiff.tex").write_text("\n".join(lines) + "\n")


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
    ap.add_argument("--heatmap", action="store_true", help="also show the search heatmap (off since 2026-11-23)")
    a = ap.parse_args()
    global PNG_PREVIEW, SHOW_HEATMAP
    PNG_PREVIEW, SHOW_HEATMAP = a.png, a.heatmap
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
    scandiff_test_table(a.scandiff_runs, tabs)
    cb_macros = baselines_table(a.results, tabs)
    recs = load_records(a.results)
    ig_figure(recs, figs)
    policy_entropy_figure(recs, ent, ent["kde_range"], figs)
    qualitative_figure(recs, figs)
    cross_dataset_figure(recs, figs, read_references(a.references))
    res_macros = results_table(recs, tabs)
    res_macros |= policy_vs_human(recs, tabs)
    res_macros |= priors_figure(recs, a.results, figs, tabs)
    priors_index_figure(recs, a.results, figs)
    res_macros |= temperature_figure(recs, figs)
    res_macros |= retina_figure(a.results, figs)

    sacc = {}
    for slug, cond, title in PANELS:
        st, fx = dataset(slug, eval_part(slug))
        sacc[title] = human_saccades(select(fx, condition=cond, limit=None), st)
    cv_sacc = canvit_saccades(recs)
    summ = {k: saccade_summary(s) for k, s in sacc.items()}
    ppd = {title: float(r["protocol"]["deepgaze"]["ppd"]) for slug, cond, title in PANELS for r in recs
           if r["kind"] == "deepgaze3" and r["dataset"] == slug and r["protocol"].get("deepgaze", {}).get("ppd")}
    res_macros |= saccade_effects(ppd, figs, tabs)  # replaces saccades.pdf (manager, 2026-11-23)
    for ds, split in (("coco_search18", "tp_validation"), ("coco_search18", "ta_validation"),
                      ("coco_freeview", "validation")):
        for r in canvit_records(recs, ds, split)[:1]:
            title = next(t for s_, c, t in PANELS if s_ == ds and (ds != "coco_search18" or split.startswith(c)))
            saccade_compare(r, ppd.get(title), figs)
    sacc_macros = saccade_table(summ, tabs, ppd, {k: saccade_summary(s) for k, s in cv_sacc.items()})

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
