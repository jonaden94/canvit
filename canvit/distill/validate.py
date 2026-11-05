"""The distill validation PHASE: streaming per-timestep metrics, the IN1k probe readout,
and the optional PCA/curve figures.

Lived under ``viz/`` until 2026-09-01, which read as though validation were a rendering
concern. Rendering is the part it delegates (to ``.viz``); the metrics are the point.

The glimpse loop is NOT here and never was: :meth:`CanViT.forward_reduce` owns it, in core.
See eval-merge doc §5 (Stage 1b) for why that is a different shape from
``harness/rollout/episode.py::run_episode`` and why the two are left separate.
"""

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from dinov3_in1k_probes import DINOv3LinearClassificationHead
from torch import Tensor

from canvit import CanViTForPretraining
from canvit.core import CanViTOutput, CLSStandardizer, PatchStandardizer, RecurrentState
from canvit.core.backbone.vit import NormFeatures
from canvit.core.teacher import DINOv3Teacher
from canvit.core.viewpoint import Viewpoint as CanvitViewpoint
from canvit.harness.infra.utils import assert_shape

from ..harness.infra.tracker import Tracker
from ..harness.rollout.eval_viewpoints import open_loop_viewpoints, resolve
from ..harness.viz.disk import plot_combined_curves, save_figure
from .probe import (
    compute_in1k_top1,
    get_imagenet_class_names,
    get_probe_resolution,
    get_top_k_predictions,
    labels_are_in1k,
)
from .viz.image import imagenet_denormalize_to_numpy
from .viz.plot import TimestepPredictions, plot_multistep_pca
from .viz.sample import VizSampleData, extract_sample0_viz

log = logging.getLogger(__name__)


@dataclass
class ValAccumulator:
    """Accumulator for streaming validation metrics.

    MEMORY OPTIMIZATION:
    - Metrics computed on full batch -> scalar -> discard tensors
    - PCA viz: sample 0 only -> O(T) not O(B×T)
    """

    scene_cos_raw: list[float] = field(default_factory=list)
    scene_cos_norm: list[float] = field(default_factory=list)
    cls_cos_raw: list[float] = field(default_factory=list)
    cls_cos_norm: list[float] = field(default_factory=list)
    in1k_accs: list[float] = field(default_factory=list)
    pca_predictions: list[TimestepPredictions] = field(default_factory=list)
    viz_samples: list[VizSampleData] = field(default_factory=list)
    initial_scene: np.ndarray | None = None
    initial_canvas_spatial: np.ndarray | None = None


def _log_pca(
    *,
    exp: Tracker,
    step: int,
    prefix: str,
    acc: ValAccumulator,
    full_img: np.ndarray,
    teacher_np: np.ndarray,
    boxes: list,
    names: list[str],
    canvas_grid_size: int,
    glimpse_grid_size: int,
    log_spatial_stats: bool,
    run_dir: Path,
) -> None:
    """Save PCA visualization to disk from accumulator data.

    The local-stream column (uniform g x g grid vs foveated patch-Voronoi)
    is selected inside `plot_multistep_pca` based on whether the
    `foveated_samples` field is populated.
    """
    assert acc.initial_scene is not None
    scenes = [vs.predicted_scene for vs in acc.viz_samples]
    glimpses = [vs.glimpse for vs in acc.viz_samples]
    canvas_spatials = [vs.canvas_spatial for vs in acc.viz_samples]

    # Extract local stream patches if available. plot_multistep_pca decides
    # how to render them (uniform g x g grid OR foveated patch-Voronoi)
    # based on whether `foveated_samples` is present.
    locals_avp: list[np.ndarray] | None = None
    locals_avp_raw = [vs.local_patches for vs in acc.viz_samples]
    has_locals = bool(locals_avp_raw) and all(lp is not None for lp in locals_avp_raw)
    if has_locals:
        locals_avp = [lp for lp in locals_avp_raw if lp is not None]

    foveated_samples_raw = [vs.foveated for vs in acc.viz_samples]
    foveated_samples = (
        foveated_samples_raw if any(s is not None for s in foveated_samples_raw) else None
    )
    square_samples_raw = [vs.square for vs in acc.viz_samples]
    square_samples = (
        square_samples_raw if any(s is not None for s in square_samples_raw) else None
    )

    fig_pca = plot_multistep_pca(
        full_img=full_img,
        teacher=teacher_np,
        scenes=scenes,
        glimpses=glimpses,
        boxes=boxes,
        names=names,
        scene_grid_size=canvas_grid_size,
        glimpse_grid_size=glimpse_grid_size,
        initial_scene=acc.initial_scene,
        locals_avp=locals_avp,
        hidden_spatials=canvas_spatials if canvas_spatials[0] is not None else None,
        initial_hidden_spatial=acc.initial_canvas_spatial,
        show_locals=has_locals,
        timestep_predictions=acc.pca_predictions if acc.pca_predictions else None,
        foveated_samples=foveated_samples,
        square_samples=square_samples,
    )
    save_figure(fig_pca, run_dir, f"pca_{prefix}", step)

    if log_spatial_stats and acc.viz_samples:
        target_stats = {"mean": float(np.mean(teacher_np)), "std": float(np.std(teacher_np))}
        pred_stats = {"mean": float(np.mean(scenes[-1])), "std": float(np.std(scenes[-1]))}
        exp.log_metrics(
            {
                f"{prefix}/target_spatial_mean": target_stats["mean"],
                f"{prefix}/target_spatial_std": target_stats["std"],
                f"{prefix}/pred_spatial_mean": pred_stats["mean"],
                f"{prefix}/pred_spatial_std": pred_stats["std"],
            },
            step=step,
        )


def _deploy_viewpoints(*, model, images, joint, n, canvas_grid_size):
    """The trajectory a trained scorer actually takes on this batch (argmax deploy).

    Distill validates through core's ``forward_reduce``, which consumes a precomputed
    viewpoint list, so the closed-loop selection runs FIRST and its result is then
    replayed through the unchanged metric machinery. That costs one extra backbone pass
    per glimpse — deliberate: distill validation is a 256-sample readout on a 1000-step
    cadence, and the alternative is restructuring ``_run_chunk``'s init_fn/step_fn
    around a per-glimpse loop, which is not worth the risk here. The replay is exact:
    argmax selection under ``no_grad`` consumes no RNG, so it revisits the same states.
    (ade20k/in1k collect their readout DURING selection and pay nothing extra.)
    """
    from canvit.harness.rollout.eval_viewpoints import deploy_rollout_viewpoints
    from canvit.harness.rollout.viewpoint import ViewpointType

    B = images.shape[0]

    def advance(vp, state, t):
        if state is None:  # t0: the rollout owns its own state init
            state = model.init_state(batch_size=B, canvas_grid_size=canvas_grid_size)
        return model(image=images, state=state, viewpoint=vp).state

    return deploy_rollout_viewpoints(joint=joint, advance=advance, t0_type=ViewpointType.FULL,
                                     batch_size=B, device=images.device, n=n)


def validate(
    *,
    exp: Tracker,
    step: int,
    model: CanViTForPretraining,
    compute_raw_targets: Callable[[Tensor, int], "NormFeatures"],
    scene_normalizer: PatchStandardizer,
    cls_normalizer: CLSStandardizer,
    val_batches: Iterable[tuple[Tensor, Tensor]],
    device: torch.device,
    canvas_grid_size: int,
    scene_size_px: int,
    glimpse_size_px: int,
    run_dir: Path,
    n_eval_viewpoints: int = 10,
    min_viewpoint_scale: float = 0.05,
    foveated_eval_scale: float = 1.0,
    eval_policy: str = "auto",
    override_scale: float | None = None,
    foveated_scale: Any = None,
    joint: Any = None,
    prefix: str = "val",
    probe: DINOv3LinearClassificationHead | None = None,
    log_curves: bool = False,
    log_pca: bool = False,
    teacher: DINOv3Teacher | None = None,
    log_spatial_stats: bool = False,
    teacher_name: str | None = None,
    non_blocking: bool = False,
) -> dict[str, float]:
    """Run validation over a fixed set of batches (rank 0 only); return its scalars.

    Iterates ``val_batches`` (the fixed N-sample subset, chunked by batch size),
    computes per-timestep streaming metrics per chunk, and aggregates them weighted
    by chunk size — so the reported metrics are over all N samples and identical
    regardless of the chunk (batch) size. PCA/viz uses sample 0 of the first chunk.
    """
    assert not log_pca or teacher is not None

    if probe is not None and teacher_name is not None:
        probe_res = get_probe_resolution(teacher_name)
        if scene_size_px != probe_res:
            log.warning(
                f"Resolution mismatch: model predicts teacher@{scene_size_px}, "
                f"but probe trained on teacher@{probe_res}. IN1k metrics may be unreliable."
            )

    is_foveated = getattr(model.cfg, "patcher_name", "uniform") in ("foveated", "square")
    policy = resolve(eval_policy, task="distill", is_foveated=is_foveated)
    has_cls = model.scene_cls_head is not None

    model_was_training = model.training
    model.eval()

    def _run_chunk(images: Tensor, labels: Tensor | None, *, want_viz: bool):
        """Evaluate one chunk. Returns (acc, batch_size, teacher_acc, target_sample0,
        viewpoints, gt_idx, gt_name); per-timestep entries in ``acc`` are batch means."""
        B = images.shape[0]
        if policy == "policy":
            viewpoints = _deploy_viewpoints(
                model=model, images=images, joint=joint, n=n_eval_viewpoints,
                canvas_grid_size=canvas_grid_size,
            )
        else:
            viewpoints = open_loop_viewpoints(
                policy, batch_size=B, device=images.device, n=n_eval_viewpoints,
                is_foveated=is_foveated, foveated_scale=foveated_scale,
                min_scale=min_viewpoint_scale, foveated_eval_scale=foveated_eval_scale,
                override_scale=override_scale,
            )
        has_probe = probe is not None and labels is not None and labels_are_in1k(labels)

        raw_feats = compute_raw_targets(images, scene_size_px)
        # Normalized targets for normalized cosine similarity and PCA
        target = scene_normalizer(raw_feats.patches)
        cls_target = cls_normalizer(raw_feats.cls.unsqueeze(1)).squeeze(1) if has_cls else None
        target_sample0 = target[0].cpu().float().numpy() if want_viz else None

        gt_idx = int(labels[0].item()) if has_probe and labels is not None else 0
        gt_name = get_imagenet_class_names()[gt_idx] if has_probe else ""

        teacher_acc: float | None = None
        if has_probe and teacher is not None:
            assert teacher_name is not None and probe is not None and labels is not None
            probe_res = get_probe_resolution(teacher_name)
            images_at_probe_res = F.interpolate(
                images, size=(probe_res, probe_res), mode="bilinear", align_corners=False
            )
            teacher_cls = teacher.forward_norm_features(images_at_probe_res).cls
            # Probe outside autocast: bf16 puts the 1000-class head at the noise floor
            # (~0.4% top-1 flips; upstream d525656). Same as in1k/ade20k heads.
            with torch.autocast(device_type=teacher_cls.device.type, enabled=False):
                teacher_logits = probe(teacher_cls.float())
            teacher_acc = compute_in1k_top1(teacher_logits, labels)

        def init_fn(state: RecurrentState) -> ValAccumulator:
            acc = ValAccumulator()
            if want_viz:
                n_canvas_tokens = model.n_canvas_registers + canvas_grid_size ** 2
                assert_shape(state.canvas, (B, n_canvas_tokens, model.canvas_dim))
                acc.initial_scene = (
                    model.predict_teacher_scene(state.canvas)[0].cpu().float().numpy()
                )
                acc.initial_canvas_spatial = (
                    model.get_spatial(state.canvas[0:1])[0].cpu().float().numpy()
                )
            return acc

        def step_fn(
            acc: ValAccumulator, out: CanViTOutput, vp: CanvitViewpoint,
        ) -> ValAccumulator:
            predicted_scene = model.predict_teacher_scene(out.state.canvas)
            predicted_cls = (
                model.predict_scene_teacher_cls(out.state.recurrent_cls) if has_cls else None
            )

            # Cosine similarity: both raw (stable across runs) and normalized
            scene_pred_raw = scene_normalizer.destandardize(predicted_scene)
            acc.scene_cos_raw.append(F.cosine_similarity(scene_pred_raw, raw_feats.patches, dim=-1).mean().item())
            acc.scene_cos_norm.append(F.cosine_similarity(predicted_scene, target, dim=-1).mean().item())

            if has_cls and predicted_cls is not None:
                assert cls_target is not None
                cls_pred_raw = cls_normalizer.destandardize(predicted_cls.unsqueeze(1)).squeeze(1)
                acc.cls_cos_raw.append(F.cosine_similarity(cls_pred_raw, raw_feats.cls, dim=-1).mean().item())
                acc.cls_cos_norm.append(F.cosine_similarity(predicted_cls, cls_target, dim=-1).mean().item())

                if has_probe:
                    assert probe is not None and labels is not None
                    with torch.autocast(device_type=cls_pred_raw.device.type, enabled=False):
                        logits = probe(cls_pred_raw.float())
                    acc.in1k_accs.append(compute_in1k_top1(logits, labels))

                    if want_viz:
                        top_k = get_top_k_predictions(logits[0:1], k=5)[0]
                        acc.pca_predictions.append(
                            TimestepPredictions(
                                predictions=top_k, gt_idx=gt_idx, gt_name=gt_name
                            )
                        )

            if want_viz:
                acc.viz_samples.append(
                    extract_sample0_viz(out, images, vp, predicted_scene, model, glimpse_size_px)
                )

            return acc

        acc, _final_state = model.forward_reduce(
            image=images,
            viewpoints=viewpoints,  # pyright: ignore[reportArgumentType]
            canvas_grid_size=canvas_grid_size,
            init_fn=init_fn,
            step_fn=step_fn,
        )
        return acc, B, teacher_acc, target_sample0, viewpoints, gt_idx, gt_name

    try:
        with torch.inference_mode():
            # Evaluate every chunk of the fixed subset; collect per-chunk accumulators.
            chunk_accs: list[tuple[ValAccumulator, int]] = []
            teacher_accs: list[tuple[float, int]] = []
            first: dict | None = None
            for ci, (images_cpu, labels_cpu) in enumerate(val_batches):
                images = images_cpu.to(device, non_blocking=non_blocking)
                labels = labels_cpu.to(device, non_blocking=non_blocking) if probe is not None else None
                want_viz = log_pca and ci == 0
                acc, B, t_acc, tgt0, viewpoints, _gt_idx, _gt_name = _run_chunk(
                    images, labels, want_viz=want_viz
                )
                chunk_accs.append((acc, B))
                if t_acc is not None:
                    teacher_accs.append((t_acc, B))
                if ci == 0:
                    first = {"acc": acc, "images": images, "target_sample0": tgt0, "viewpoints": viewpoints}

            assert chunk_accs, "validate() received no batches"

            def agg(attr: str) -> list[float]:
                """Per-timestep mean over all samples (weighted by chunk size)."""
                series = [(getattr(a, attr), b) for a, b in chunk_accs]
                total = sum(b for _, b in series)
                n_t = len(series[0][0])
                return [sum(vals[t] * b for vals, b in series) / total for t in range(n_t)]

            scene_cos_raw = agg("scene_cos_raw")
            scene_cos_norm = agg("scene_cos_norm")
            cls_cos_raw = agg("cls_cos_raw") if has_cls else []
            cls_cos_norm = agg("cls_cos_norm") if has_cls else []
            have_in1k = len(chunk_accs[0][0].in1k_accs) > 0
            in1k_accs = agg("in1k_accs") if have_in1k else []

            # RETURNED, not logged. The caller namespaces and logs them (run.py's
            # on_eval under the task's own prefix, so the wandb keys are unchanged), and
            # a standalone evaluation gets the whole series instead of a single scalar --
            # which is all it could see while this function logged and returned a float
            # (eval-merge doc §5, F8). Key names are character-for-character what this
            # used to log, so exp22-exp32 dashboards keep working.
            series: dict[str, float] = {
                "scene_cos_raw": scene_cos_raw[-1],
                "scene_cos_norm": scene_cos_norm[-1],
            }
            if teacher_accs:
                total = sum(b for _, b in teacher_accs)
                series["in1k_teacher_top1"] = sum(a * b for a, b in teacher_accs) / total
            for t, (raw, norm) in enumerate(zip(scene_cos_raw, scene_cos_norm)):
                series[f"scene_cos_raw_t{t}"] = raw
                series[f"scene_cos_norm_t{t}"] = norm
            if has_cls:
                series["cls_cos_raw"] = cls_cos_raw[-1]
                series["cls_cos_norm"] = cls_cos_norm[-1]
                for t, (raw, norm) in enumerate(zip(cls_cos_raw, cls_cos_norm)):
                    series[f"cls_cos_raw_t{t}"] = raw
                    series[f"cls_cos_norm_t{t}"] = norm
            if have_in1k:
                for t, ia in enumerate(in1k_accs):
                    series[f"in1k_tts_top1_t{t}"] = ia

            # Combined 5-subplot graphs figure saved to disk at log_curves cadence.
            # Replaces the per-curve wandb log_curve calls (scene_cos_*, cls_cos_*,
            # in1k_tts_top1_*). Per-timestep scalars above keep going to wandb.
            if log_curves:
                fig_graphs = plot_combined_curves(
                    scene_cos_raw=scene_cos_raw,
                    scene_cos_norm=scene_cos_norm,
                    cls_cos_raw=cls_cos_raw if has_cls else None,
                    cls_cos_norm=cls_cos_norm if has_cls else None,
                    in1k_accs=in1k_accs if have_in1k else None,
                )
                save_figure(fig_graphs, run_dir, "graphs", step)

            if log_pca:
                assert first is not None and first["target_sample0"] is not None
                images0 = first["images"]
                viewpoints = first["viewpoints"]
                H, W = images0.shape[-2], images0.shape[-1]
                boxes = [vp.to_pixel_box(0, H, W) for vp in viewpoints]
                names = [vp.name for vp in viewpoints]
                full_img = imagenet_denormalize_to_numpy(images0[0])
                # Uniform-grid local PCA panel needs g*g tokens; foveated tokens
                # are a point cloud so we disable the local stream panel.
                uniform_grid = getattr(model.cfg, "patcher_name", "uniform") == "uniform"
                # Conv output-size formula (W - kernel)/stride + 1: the number of
                # PatchEmbed tokens per side, which equals glimpse//patch for the
                # non-overlapping case (stride == patch) and stays correct when
                # patches overlap (stride < patch). patch_stride_px defaults to
                # patch_size_px, so non-overlap runs are unaffected.
                glimpse_grid_size = (
                    (glimpse_size_px - model.backbone.patch_size_px)
                    // model.backbone.patch_stride_px
                    + 1
                    if uniform_grid
                    else 0
                )

                _log_pca(
                    exp=exp,
                    step=step,
                    prefix=prefix,
                    acc=first["acc"],
                    full_img=full_img,
                    teacher_np=first["target_sample0"],
                    boxes=boxes,
                    names=names,
                    canvas_grid_size=canvas_grid_size,
                    glimpse_grid_size=glimpse_grid_size,
                    log_spatial_stats=log_spatial_stats,
                    run_dir=run_dir,
                )

            return series
    finally:
        if model_was_training:
            model.train()
