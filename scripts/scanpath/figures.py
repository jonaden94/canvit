"""Figures and tables for the scanpath working notes (ShareLaTeX `Getting_Info_scanpath_prediction`).

    source .envrc && .venv-cu126/bin/python scripts/scanpath/figures.py --out <ShareLaTeX project dir>

Writes figures/*.pdf and tables/*.tex there; every number in them comes from this script. Data: COCO-Search18
split1 validation from pysaliency (`canvit.scanpath.evaluate.load_coco_search18`), the ScanDiff driver's sanity
runs (`--scandiff-runs`), and the baseline records in `$SCANPATH_RESULTS_DIR` (table only when they exist).
Written by Claude (AI, Claude Opus 5.5), 2026-11-20.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import norm  # noqa: E402

from canvit.scanpath.evaluate import ScanpathConfig, load_coco_search18  # noqa: E402
from canvit.scanpath.generative import target_hit_index, tfp_auc, tfp_curve  # noqa: E402

# Colour follows the condition (reference categorical palette, fixed slots); humans/reference lines in ink.
TP, TA, INK, MUTED, GOOD = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#0ca30c"
G = 32  # the search policy's fixation grid
POLICY_TARGET = 2.11  # nats, the search policy's entropy target (search/config.py)
PDF_META = {"CreationDate": None}  # no timestamp: a rerun with the same data writes the same PDF
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e1",
                     "grid.linewidth": 0.6, "lines.linewidth": 2, "axes.edgecolor": MUTED})


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

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.1))
    for ax, mask, name in ((axes[0], tp, "target present"), (axes[1], ~tp, "target absent")):
        i, rows = pick(mask)
        ax.imshow(stimuli.stimuli[i])
        for r in rows:
            L = int(sp.length[r])
            ax.plot(sp.xs[r][:L], sp.ys[r][:L], "-", color="white", lw=0.8, alpha=0.85)
            ax.scatter(sp.xs[r][1:L], sp.ys[r][1:L], s=9, color="white", edgecolor=INK, lw=0.5, zorder=3)
        b = sp.scanpath_attributes["target_bbox"][rows[0]]
        if np.isfinite(b).all():
            ax.add_patch(plt.Rectangle(b[:2], b[2], b[3], fill=False, ec=GOOD, lw=1.6))
        ax.set_title(f"{name}: cue '{TASKS[int(stimuli.attributes['task'][i])]}', {len(rows)} subjects", fontsize=9)
        ax.axis("off")
    fig.savefig(out / "example_trials.pdf", metadata=PDF_META)
    plt.close(fig)


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
    tp = present_mask(fix)
    res = {"target present": human_entropy(stimuli, fix, tp), "target absent": human_entropy(stimuli, fix, ~tp)}
    floor, ceiling = kde_range(tuple(int(v) for v in stimuli.sizes[0]))
    fig, (ax, axn) = plt.subplots(2, 1, figsize=(5.2, 4.0), sharex=True, height_ratios=(3, 1))
    ax.axhspan(floor, ceiling, color="#e6e5e1", lw=0)
    ax.text(1, ceiling + 0.08, f"range of the estimator for {K_SUBJECTS} subjects (shaded)", color=MUTED,
            fontsize=8, va="bottom")
    for (name, e), c, dy in zip(res.items(), (TP, TA), (-0.25, 0.25)):  # present below, absent above its line
        t = np.array(list(e))
        med, q25, q75, n = (np.array([v[k] for v in e.values()]) for k in range(4))
        ax.fill_between(t, q25, q75, color=c, alpha=0.18, lw=0)
        ax.plot(t, med, marker="o", ms=4, color=c, label=name)
        ax.text(t[-1], med[-1] + dy, name, color=INK, ha="right", va="center", fontsize=8)
        axn.plot(t, n, marker="o", ms=3, color=c, lw=1.5)
    ax.axhline(POLICY_TARGET, color=MUTED, ls="--", lw=1)
    ax.text(1, POLICY_TARGET + 0.12, f"search policy target {POLICY_TARGET} nats", color=MUTED, fontsize=8)
    ax.axhline(np.log(G * 20), color=MUTED, ls=":", lw=1)
    ax.text(1, np.log(G * 20) + 0.12, "uniform on the image", color=MUTED, fontsize=8)
    ax.set(ylabel="entropy (nats)\nmedian and IQR", ylim=(0, 7.4))
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    axn.set(xlabel="fixation index t", ylabel="stimuli", xlim=(0.8, max(max(e) for e in res.values()) + 0.2),
            ylim=(0, None))
    fig.savefig(out / "human_entropy.pdf", metadata=PDF_META)
    plt.close(fig)
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
    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    ax.plot(range(len(curve)), curve, marker="o", ms=4, color=INK)
    ax.set(xlabel="fixations after the start", ylabel="P(target fixated)", ylim=(0, 1))
    fig.savefig(out / "human_tfp.pdf", metadata=PDF_META)
    plt.close(fig)
    return curve, auc


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


def baselines_table(results: Path, out: Path) -> dict[str, str]:
    """tables/baselines.tex from the baseline records; returns number macros for the text."""
    rows, macros = [], {}
    for cond in ("tp", "ta"):
        for model in ("uniform", "centre_bias"):
            p = results / f"{model}_{cond}_validation.json"
            if p.exists():
                m = json.loads(p.read_text())["metrics"]
                if model == "centre_bias":
                    tag = "TP" if cond == "tp" else "TA"
                    macros |= {f"cb{tag}": f"{m['ll_image']:.2f}", f"cb{tag}one": f"{m['ll_t1']:.2f}",
                               f"cb{tag}three": f"{m['ll_t3']:.2f}", f"cb{tag}nfix": f"{int(m['n_fixations'])}"}
                rows.append(f"{'present' if cond == 'tp' else 'absent'} & {model.replace('_', ' ')} & "
                            f"{int(m['n_fixations'])} & {m['ll_image']:.3f} & {m['ll_fix']:.3f} & "
                            f"{m['auc_image']:.3f} & {m['nss_image']:.3f} \\\\")
    lines = [r"\begin{tabular}{llrrrrr}", r"\toprule",
             r"condition & model & fixations & LL (image) & LL (fix.) & AUC & NSS \\", r"\midrule", *rows,
             r"\bottomrule", r"\end{tabular}"]
    (out / "baselines.tex").write_text("\n".join(lines) + "\n")
    return macros


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scandiff-runs", type=Path,
                    default=Path("/mnt/lustre-grete/usr/u13879/normative_modelling/external/scandiff_driver/runs"))
    ap.add_argument("--results", type=Path, default=Path(os.environ.get(
        "SCANPATH_RESULTS_DIR", "/mnt/lustre-grete/usr/u13879/normative_modelling/canvit_outputs/scanpath_eval")))
    a = ap.parse_args()
    (a.out / "figures").mkdir(parents=True, exist_ok=True)
    (a.out / "tables").mkdir(parents=True, exist_ok=True)
    _, _, sv, fv = load_coco_search18(ScanpathConfig())
    example_trials(sv, fv, a.out / "figures")
    ent = entropy_figure(sv, fv, a.out / "figures")
    curve, auc = tfp_figure(sv, fv, a.out / "figures")
    scandiff_table(a.scandiff_runs, a.out / "tables")
    cb_macros = baselines_table(a.results, a.out / "tables")
    tp = present_mask(fv)
    stats = {"validation_stimuli": len(sv), "scanpaths_tp": int(tp.sum()), "scanpaths_ta": int((~tp).sum()),
             "median_length_tp": float(np.median(np.asarray(fv.scanpaths.length)[tp])),
             "median_length_ta": float(np.median(np.asarray(fv.scanpaths.length)[~tp])),
             "entropy": {k: {t: v[0] for t, v in e.items()} for k, e in ent.items() if k != "kde_range"},
             "entropy_n_stimuli": {k: {t: v[3] for t, v in e.items()} for k, e in ent.items() if k != "kde_range"},
             "kde_range_nats": list(ent["kde_range"]),
             "tfp_curve": [round(float(c), 4) for c in curve], "tfp_auc": auc}
    (a.out / "tables" / "numbers.json").write_text(json.dumps(stats, indent=1))
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
    } | cb_macros
    (a.out / "tables" / "numbers.tex").write_text(
        "% Generated by canvit scripts/scanpath/figures.py -- do not edit by hand.\n"
        + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items()))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
