"""Standalone evaluation of a finished checkpoint — the counterpart of ``harness.run``.

    python -m canvit.harness.evaluate ade20k \
        --opts.ckpt logs/<group>/<run>/checkpoints/best.pt \
        --cfg.model-repo <the backbone the probe was trained on> \
        --cfg.eval-policy fixation_grid --cfg.foveated-scale.fixed-scale 2.0 \
        --opts.out results/exp34-fixgrid.json

A training run measures ONE protocol — whatever ``eval_policy`` resolved to for that run.
This re-measures a finished checkpoint under any protocol, without retraining, which is how
you answer "what would this model score under coarse-to-fine?".

**Same code as training-time validation, not a parallel implementation.** It builds the task
from its own config dataclass, restores the checkpoint's weights strictly, takes the val
loader from ``task.build_val_loader()`` — the same one ``build_loaders`` hands the training
loop — and calls ``task.evaluate``. The two entry points differ only in config: full val set
vs subset, glimpse count, single-GPU vs DDP. Nothing here can drift from the numbers a run
logs, which is the property the eval-merge exists to establish (doc §5, Stage 3).

**No default policy.** ``--cfg.eval-policy`` must be set. The ``auto`` table
(``HISTORICAL_DEFAULTS``) exists to keep exp22–exp36 training curves comparable; a standalone
measurement has no such history to protect and refuses to guess, because the guess that would
be wrong is silent — a fixed-scale foveated model under a scale-varying policy is out of
distribution and its metric FALLS as glimpses accumulate.

**One config per invocation, one artifact out.** Comparing several policies on one checkpoint
is a loop over :func:`evaluate` (see its docstring), not a flag: a list would need a second
CLI mode, a nested output schema, and would leave failure semantics murky when config 3 of 4
dies. The model and dataset reload per invocation; that is the price of one artifact per
measurement and it is the right trade.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import torch
import tyro

from canvit.ade20k.config import Ade20kConfig
from canvit.ade20k.dinov3_baseline import DinoV3BaselineOpts
from canvit.distill.config import Config as DistillConfig
from canvit.in1k.config import In1kConfig

log = logging.getLogger(__name__)


@dataclass
class EvalOpts:
    """Knobs with no task-config counterpart."""

    ckpt: Path
    """The checkpoint to evaluate: a harness ``.pt`` (``best.pt`` / ``step-N.pt`` /
    ``latest.pt``). Its weights are loaded with ``strict=True`` — a silently-missing head
    would still produce a plausible-looking metric, just from a freshly built one."""
    out: Path | None = None
    """Where to write the JSON record. Printed to stdout either way."""
    device: str = "cuda"


def _resolved_protocol(cfg: Any, task_name: str, is_foveated: bool) -> dict[str, Any]:
    """What the run actually measured, flattened for the record.

    Provenance is what protects comparability here, not restriction: any combination the
    code can execute is allowed, so the artifact has to say which one produced the number
    (doc §5, Stage 3). Answering "is this comparable to exp33?" a year from now needs this
    dict, not the command line someone happened to type.
    """
    from canvit.harness.rollout.eval_viewpoints import resolve

    fs = getattr(cfg, "foveated_scale", None)
    return {
        "eval_policy": resolve(cfg.eval_policy, task=task_name, is_foveated=is_foveated),
        "eval_policy_requested": cfg.eval_policy,
        "override_scale": getattr(cfg, "eval_override_scale", None),
        "is_foveated": is_foveated,
        "foveated_scale": None if fs is None else {
            "mode": fs.mode, "distribution": fs.distribution, "fixed_scale": fs.fixed_scale,
            "min_scale": fs.min_scale, "max_scale": fs.max_scale,
        },
        "n_timesteps": getattr(cfg, "n_timesteps", None) or getattr(cfg, "n_eval_viewpoints", None),
        "scene_size": getattr(cfg, "scene_size", None) or getattr(cfg, "scene_resolution", None),
        "limit_val_batches": getattr(cfg, "limit_val_batches", None),
    }


def adopt_checkpoint_provenance(cfg: Any, payload: dict, *, source: Any) -> list[str]:
    """Fill in what the checkpoint already knows, WHERE THE USER LEFT THE DEFAULT.

    A checkpoint records the view scale it was pretrained at and the teacher that supervised
    it. Requiring the user to retype either is how they get mistyped — and a mistyped view
    scale is the silent failure this whole merge exists to close: the metric falls as
    glimpses accumulate instead of raising.

    Deliberately NOT an auto-pin of the eval scale. Which trajectory to measure is the
    user's choice and this code does not make it (owner, 2026-09-02); pinning stays explicit
    via ``--cfg.eval-override-scale``. What this does is make the DEFAULT honest, so the
    off-scale warning in ``open_loop_viewpoints`` compares against the scale the model was
    really trained at rather than a config default of 1.0.

    Only touches fields still at their dataclass default — anything typed on the command
    line wins, always. Returns what it adopted, for the record.
    """
    from canvit.checkpoint.to_hf import read_pretraining_provenance
    from canvit.harness.config import FoveatedScaleConfig

    view_scale, teacher_name = read_pretraining_provenance(payload, source=source)
    adopted: list[str] = []

    fs = getattr(cfg, "foveated_scale", None)
    if fs is not None and view_scale is not None and fs == FoveatedScaleConfig():
        for field in ("mode", "distribution", "fixed_scale", "min_scale", "max_scale"):
            if view_scale.get(field) is not None:
                setattr(fs, field, view_scale[field])
        adopted.append(f"foveated_scale={view_scale['mode']}/{view_scale['fixed_scale']}")
        log.info("adopted the checkpoint's pretraining view scale: %s. Pass "
                 "--cfg.foveated-scale.* to override.", view_scale)

    default_teacher = type(cfg).__dataclass_fields__.get("teacher_name")
    if (default_teacher is not None and teacher_name
            and cfg.teacher_name == default_teacher.default
            and teacher_name != cfg.teacher_name):
        cfg.teacher_name = teacher_name
        adopted.append(f"teacher_name={teacher_name}")
        log.info("adopted the checkpoint's teacher_name=%s (picks the teacher and the IN1k "
                 "probe via distill/probe.py::PROBE_REGISTRY)", teacher_name)
    return adopted


def evaluate(task: Any, cfg: Any, opts: EvalOpts, *, task_name: str) -> dict[str, Any]:
    """Evaluate one checkpoint under one protocol; return the record that gets written.

    The library seam. Comparing protocols is a loop over this function::

        for policy in ("coarse_to_fine", "fixation_grid", "full"):
            rec = evaluate(build_ade20k(replace(cfg, eval_policy=policy)), ...)

    which is where a comparison table belongs — a notebook — rather than inside a CLI.
    """
    assert cfg.eval_policy != "auto", (
        "--cfg.eval-policy is required for a standalone evaluation; 'auto' is the "
        "TRAINING-time table (HISTORICAL_DEFAULTS), which exists to keep exp22-exp36 curves "
        "comparable and has no bearing on a fresh measurement. Pick the protocol you mean: "
        "a fixed-scale foveated model under a scale-varying policy such as coarse_to_fine is "
        "OUT OF DISTRIBUTION (fix_size = scale * H), and the symptom is a metric that FALLS "
        "as glimpses accumulate rather than an error."
    )
    device = torch.device(opts.device if torch.cuda.is_available() else "cpu")

    payload = torch.load(opts.ckpt, weights_only=False, map_location=device)
    adopted = adopt_checkpoint_provenance(cfg, payload, source=opts.ckpt)
    # distill rebuilds its architecture FROM the checkpoint (build_model's resume path);
    # ade20k/in1k rebuild from cfg.model_repo and ignore this.
    model, head = task.build_model(device, prior_model_config=payload.get("model_config"))
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()

    val_loader = task.build_val_loader()
    assert val_loader is not None, f"{task_name} has no val loader for this config"
    is_foveated = task.is_foveated(model)
    protocol = _resolved_protocol(cfg, task_name, is_foveated)
    protocol["adopted_from_checkpoint"] = adopted
    log.info("evaluating %s under %s", opts.ckpt, protocol)

    step = int(payload.get("step", 0))
    metrics = task.evaluate(model=model, head=head, val_loader=val_loader, device=device,
                            step=step)

    return {
        "task": task_name,
        "ckpt": str(opts.ckpt),
        "step": step,
        "protocol": protocol,
        "metrics": {k: float(v) for k, v in metrics.items()},
        "checkpoint_metadata": {
            k: v for k, v in (payload.get("metadata") or {}).items()
            # the histories are long and already in the .pt; keep the record readable
            if k not in ("training_config_history", "provenance_history")
        },
        "evaluated_at": datetime.now(UTC).isoformat(),
    }


@dataclass
class Ade20kEval:
    """ADE20K semantic segmentation: mIoU per timestep over the val set."""

    cfg: Ade20kConfig = field(default_factory=Ade20kConfig)
    opts: EvalOpts = field(default_factory=lambda: EvalOpts(ckpt=Path()))

    def build(self):
        from canvit.ade20k.task import Ade20kRunTask
        return Ade20kRunTask(self.cfg), "ade20k"


@dataclass
class In1kEval:
    """ImageNet-1k classification: top-1/5 at the final timestep."""

    cfg: In1kConfig = field(default_factory=In1kConfig)
    opts: EvalOpts = field(default_factory=lambda: EvalOpts(ckpt=Path()))

    def build(self):
        from canvit.in1k.task import In1kRunTask
        return In1kRunTask(self.cfg), "in1k"


@dataclass
class DistillEval:
    """Pretraining: cosine-to-teacher over the fixed val subset."""

    cfg: DistillConfig = field(default_factory=DistillConfig)
    opts: EvalOpts = field(default_factory=lambda: EvalOpts(ckpt=Path()))

    def build(self):
        from canvit.distill.task import DistillRunTask
        return DistillRunTask(self.cfg), "distill"


@dataclass
class Ade20kDinoV3Eval:
    """ADE20K with the DINOv3 TEACHER instead of a CanViT: one passive forward, mIoU at t0.

    The reference line the CanViT numbers are read against. No checkpoint and no episode --
    hence its own opts rather than ``EvalOpts`` -- but the same val images, transforms and
    mIoU reduction, so the baseline and the model it bounds are measured identically."""

    cfg: Ade20kConfig = field(default_factory=Ade20kConfig)
    opts: DinoV3BaselineOpts = field(
        default_factory=lambda: DinoV3BaselineOpts(probe_repo="", eval_resolution=0))
    out: Path | None = None

    def run(self) -> dict[str, Any]:
        from canvit.ade20k.dinov3_baseline import evaluate_dinov3
        assert self.opts.probe_repo and self.opts.eval_resolution, (
            "--opts.probe-repo and --opts.eval-resolution are both required: the probe was "
            "trained at one resolution and running the teacher at another degrades mIoU "
            "silently.")
        return evaluate_dinov3(self.cfg, self.opts)


Command = Annotated[Ade20kEval, tyro.conf.subcommand("ade20k")] | \
    Annotated[Ade20kDinoV3Eval, tyro.conf.subcommand("ade20k-dinov3")] | \
    Annotated[In1kEval, tyro.conf.subcommand("in1k")] | \
    Annotated[DistillEval, tyro.conf.subcommand("distill")]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cmd = tyro.cli(Command)  # pyright: ignore[reportArgumentType]
    if isinstance(cmd, Ade20kDinoV3Eval):
        record, out = cmd.run(), cmd.out
    else:
        task, task_name = cmd.build()
        record, out = evaluate(task, cmd.cfg, cmd.opts, task_name=task_name), cmd.opts.out
    text = json.dumps(record, indent=2, default=str)
    print(text)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        log.info("wrote %s", out)


if __name__ == "__main__":
    main()
