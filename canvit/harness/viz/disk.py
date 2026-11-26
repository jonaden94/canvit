"""Disk-side visualization sink: save figures under run_dir/visualization/."""

import gc
import logging
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

log = logging.getLogger(__name__)


def save_figure(fig: Figure, run_dir: Path, subdir: str, step: int) -> None:
    """Write a Figure to {run_dir}/visualization/{subdir}/step-{step}.png.

    Cleans up matplotlib state aggressively to prevent the leaks that the
    previous wandb-backed log_figure() had to defend against.
    """
    out_dir = run_dir / "visualization" / subdir
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"step-{step}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
    except Exception as e:
        log.exception(f"Failed to save figure to {out_dir} at step {step}: {e}")
    finally:
        for ax in fig.axes:
            ax.clear()
        fig.clf()
        plt.close(fig)
        gc.collect()


def plot_combined_curves(passes: dict[str, dict[str, float]]) -> Figure:
    """Distill validation curves over the eval-trajectory timestep, one line per validation
    pass (e.g. ``{"fixation_grid": series, "policy": series}``) so the passes can be compared.

    ``series`` holds the per-timestep scalars validation returns (``scene_cos_raw_t{t}``, ...,
    ``in1k_tts_top1_t{t}`` in percent, ``policy_entropy_t{t}`` from t = 1); the IN1k panel
    adds the teacher's accuracy (``in1k_teacher_top1``) as a dashed line. The y axis fits the
    data (shared by the passes in a panel). Layout:

        | scene_cos_raw | scene_cos_norm | in1k_tts_top1 (%) |
        | cls_cos_raw   | cls_cos_norm   | policy_entropy    |

    A panel no pass has data for is greyed out with "N/A", so the layout stays fixed.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    panels = [
        (axes[0, 0], "scene_cos_raw", "cosine"),
        (axes[0, 1], "scene_cos_norm", "cosine"),
        (axes[0, 2], "in1k_tts_top1", "top-1 (%)"),
        (axes[1, 0], "cls_cos_raw", "cosine"),
        (axes[1, 1], "cls_cos_norm", "cosine"),
        (axes[1, 2], "policy_entropy", "entropy (nats)"),
    ]
    for ax, key, ylabel in panels:
        ax.set_title(f"{key} vs timestep")
        ax.set_xlabel("timestep")
        drawn = False
        for label, series in passes.items():
            ts = sorted(int(k[len(key) + 2:]) for k in series
                        if k.startswith(f"{key}_t") and k[len(key) + 2:].isdigit())
            if ts:
                ax.plot(ts, [series[f"{key}_t{t}"] for t in ts], marker="o", label=label)
                drawn = True
        if key == "in1k_tts_top1" and drawn:
            teacher = next((s["in1k_teacher_top1"] for s in passes.values() if "in1k_teacher_top1" in s), None)
            if teacher is not None:
                ax.axhline(teacher, color="gray", linestyle="--", label="teacher")
        if drawn:
            ax.ticklabel_format(axis="y", useOffset=False)  # plain values, no "+6.929" offset
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                    transform=ax.transAxes, color="gray", fontsize=14)
            ax.set_facecolor((0.95, 0.95, 0.95))
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout()
    return fig
