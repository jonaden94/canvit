"""Checkpoint serialization for CanViTForPretraining models."""

import logging
import os
import re
import socket
import subprocess
import sys
import tempfile
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict

import dacite
import torch
from torch import Tensor

from canvit import CanViTForPretraining, CanViTForPretrainingConfig

log = logging.getLogger(__name__)

class CheckpointData(TypedDict):
    """Checkpoint structure. All fields present; None where not applicable."""

    # --- Model reconstruction (required) ---
    state_dict: dict[str, Tensor]
    model_config: dict
    backbone_name: str
    canvas_patch_grid_sizes: list[int]
    teacher_dim: int
    teacher_repo_id: str
    teacher_name: str
    dataset: str

    # --- Training context ---
    glimpse_grid_size: int
    patch_stride: int | None  # uniform patch-embed conv stride (None = patch_size)
    scene_resolution: int
    step: int | None
    train_loss: float | None

    # --- Optimizer/scheduler state for resuming training ---
    optimizer_state: dict | None
    scheduler_state: dict | None
    training_config_history: dict[str, dict] | None
    # WebDataset job_index — index of the job that wrote this checkpoint.
    # On resume, the next job uses job_index + 1. None for legacy checkpoints
    # and for the sharded-features path (which uses scheduler.last_epoch).
    job_index: int | None
    # WebDataset shard-schedule invariants — the four values that determine
    # `compute_schedule_slice`'s offset (`job_index * shards_per_gpu * world_size`).
    # Asserted-equal at resume time so changing any of them between save and
    # resume cannot silently re-process or skip shards. None on the
    # sharded-features path (which doesn't use the WebDataset schedule).
    ddp_world_size: int | None
    batch_size_per_gpu: int | None
    steps_per_job: int | None
    samples_per_shard: int | None

    # --- Provenance (last save only — see provenance_history for full trail) ---
    timestamp: str
    git_commit: str | None
    git_dirty: bool
    comet_id: str | None
    wandb_run_id: str | None
    hostname: str | None
    slurm_job_id: str | None
    slurm_array_task_id: str | None
    cmdline: list[str] | None

    # --- Provenance history (accumulated across resumes) ---
    provenance_history: dict[str, dict] | None


SCALE_SENSITIVE_PATCHERS = ("foveated", "square")
"""Patchers whose out-of-distribution axis is the VIEW SCALE (``fix_size = scale * H``).
The uniform patcher's is the glimpse crop in pixels, so a view scale recorded for it means
nothing — mirrors ``canvit.core.checkpoint_schema.SCALE_SENSITIVE_PATCHERS``."""


def downstream_pretrain_view_scale(*, patcher_name: str | None, foveated_scale) -> dict | None:
    """The ``pretrain_view_scale`` an ade20k/in1k checkpoint should record, in the SAME
    dict shape ``extract_pretrain_view_scale`` builds for a distill checkpoint.

    One shape, so a downstream consumer has one thing to parse. Two things this fixes:

    * **ade20k recorded a bare float**, so a reader written against the published (dict)
      form silently saw "not recorded" — and every pre-2026-09-01 ade20k checkpoint still
      does; a reader must accept both.
    * **ade20k recorded it unconditionally**, i.e. ``1.0`` for a UNIFORM run, whose view
      scale is meaningless. ``extract_pretrain_view_scale``'s contract is that ``None``
      means "unknown, and never read it as 1.0"; emitting 1.0 for uniform breaks exactly
      that. Gated on the patcher here.
    * **in1k recorded nothing at all**, so a foveated finetune could not report the scale
      it was trained at (exp25/exp29/exp33 are all in that state — for those the only
      recovery is the backbone repo in ``model_config["model_repo"]``, which is what
      ``checkpoint/to_hf.py`` falls back to).

    Nothing in this repo reads the field yet; ``CanViT-eval`` does, and the standalone
    evaluator that replaces it will (eval-merge doc §5, Stage 3).
    """
    if patcher_name not in SCALE_SENSITIVE_PATCHERS or foveated_scale is None:
        return None
    return {
        "patcher_name": patcher_name,
        "mode": getattr(foveated_scale, "mode", None),
        "distribution": getattr(foveated_scale, "distribution", None),
        "fixed_scale": getattr(foveated_scale, "fixed_scale", None),
        "min_scale": getattr(foveated_scale, "min_scale", None),
        "max_scale": getattr(foveated_scale, "max_scale", None),
    }


def _git_info() -> tuple[str | None, bool]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = subprocess.call(["git", "diff", "--quiet"], stderr=subprocess.DEVNULL) != 0
        return commit, dirty
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None, False


def get_env_metadata() -> tuple[str | None, str | None, str | None, list[str] | None]:
    """Collect (hostname, slurm_job_id, slurm_array_task_id, cmdline). Best-effort."""
    hostname: str | None = None
    cmdline: list[str] | None = None
    try:
        hostname = socket.gethostname()
    except Exception as e:
        log.warning(f"Failed to get hostname: {e}")
    try:
        cmdline = sys.argv.copy()
    except Exception as e:
        log.warning(f"Failed to get cmdline: {e}")
    return (
        hostname,
        os.environ.get("SLURM_JOB_ID"),
        os.environ.get("SLURM_ARRAY_TASK_ID"),
        cmdline,
    )


def current_provenance() -> dict:
    """Snapshot of current environment provenance (git, host, slurm, cmdline)."""
    git_commit, git_dirty = _git_info()
    hostname, slurm_job_id, slurm_array_task_id, cmdline = get_env_metadata()
    return {
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "hostname": hostname,
        "slurm_job_id": slurm_job_id,
        "slurm_array_task_id": slurm_array_task_id,
        "cmdline": cmdline,
    }


def atomic_torch_save(data: CheckpointData, path: Path) -> None:
    """Save data to path atomically using tmp file + rename."""
    log.info(f"Saving checkpoint to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path_str = tempfile.mkstemp(
        suffix=".pt.tmp", prefix=path.stem, dir=path.parent
    )
    tmp_path = Path(tmp_path_str)
    log.debug(f"Writing to tmp file: {tmp_path}")
    try:
        os.close(fd)
        torch.save(data, tmp_path)
        tmp_path.rename(path)
    except Exception:
        log.exception(f"Failed to save checkpoint to {path}")
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def update_symlink(symlink_path: Path, target: Path) -> None:
    """Atomically update symlink to point to target."""
    log.info(f"Updating symlink {symlink_path} -> {target.name}")
    tmp_link = symlink_path.parent / f".{symlink_path.name}.tmp.{os.getpid()}"
    if tmp_link.is_symlink() or tmp_link.exists():
        tmp_link.unlink()
    tmp_link.symlink_to(target.name)
    tmp_link.rename(symlink_path)


def find_latest(run_dir: Path) -> Path | None:
    """Find latest.pt symlink in run_dir, return resolved path or None."""
    latest = run_dir / "latest.pt"
    if latest.is_symlink():
        resolved = latest.resolve()
        if resolved.exists():
            return resolved
        log.warning(f"latest.pt symlink broken: {latest} -> {resolved}")
    return None



def save(
    path: Path,
    model: CanViTForPretraining,
    backbone_name: str,
    *,
    teacher_repo_id: str,
    teacher_name: str,
    dataset: str,
    glimpse_grid_size: int,
    scene_resolution: int,
    patch_stride: int | None = None,
    step: int | None = None,
    train_loss: float | None = None,
    comet_id: str | None = None,
    wandb_run_id: str | None = None,
    optimizer_state: dict | None = None,
    scheduler_state: dict | None = None,
    training_config_history: dict[str, dict] | None = None,
    provenance_history: dict[str, dict] | None = None,
    job_index: int | None = None,
    ddp_world_size: int | None = None,
    batch_size_per_gpu: int | None = None,
    steps_per_job: int | None = None,
    samples_per_shard: int | None = None,
) -> None:
    """Save checkpoint with all info needed to reconstruct model and push to hub."""
    assert isinstance(model.cfg, CanViTForPretrainingConfig)
    git_commit, git_dirty = _git_info()
    hostname, slurm_job_id, slurm_array_task_id, cmdline = get_env_metadata()

    data: CheckpointData = {
        "state_dict": model.state_dict(),
        "model_config": asdict(model.cfg),
        "backbone_name": backbone_name,
        "canvas_patch_grid_sizes": model.canvas_patch_grid_sizes,
        "teacher_dim": model.cfg.teacher_dim,
        "teacher_repo_id": teacher_repo_id,
        "teacher_name": teacher_name,
        "dataset": dataset,
        "glimpse_grid_size": glimpse_grid_size,
        "patch_stride": patch_stride,
        "scene_resolution": scene_resolution,
        "step": step,
        "train_loss": train_loss,
        "optimizer_state": optimizer_state,
        "scheduler_state": scheduler_state,
        "training_config_history": training_config_history,
        "provenance_history": provenance_history,
        "job_index": job_index,
        "ddp_world_size": ddp_world_size,
        "batch_size_per_gpu": batch_size_per_gpu,
        "steps_per_job": steps_per_job,
        "samples_per_shard": samples_per_shard,
        "timestamp": datetime.now(UTC).isoformat(),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "comet_id": comet_id,
        "wandb_run_id": wandb_run_id,
        "hostname": hostname,
        "slurm_job_id": slurm_job_id,
        "slurm_array_task_id": slurm_array_task_id,
        "cmdline": cmdline,
    }

    atomic_torch_save(data, path)
    size_mb = path.stat().st_size / (1024 * 1024)

    log.info(f"Checkpoint saved: {path} ({size_mb:.1f} MB)")
    log.info(
        f"  backbone={backbone_name}, canvas_patch_grid_sizes={model.canvas_patch_grid_sizes},"
        f" teacher={teacher_name}, dataset={dataset},"
        f" glimpse={glimpse_grid_size}, scene={scene_resolution}px"
    )
    if step is not None:
        log.info(f"  step={step}, train_loss={train_loss:.4e}" if train_loss else f"  step={step}")
    if git_commit:
        log.info(f"  git={git_commit[:8]}{'*' if git_dirty else ''}")


def load(path: Path, device: torch.device | str = "cpu") -> CheckpointData:
    """Load checkpoint data. All fields are required."""
    log.info(f"Loading checkpoint: {path}")
    raw = torch.load(path, weights_only=False, map_location=device)

    # Required model fields — fail loudly if missing
    for key in (
        "state_dict", "model_config", "backbone_name", "canvas_patch_grid_sizes",
        "teacher_dim", "teacher_repo_id", "teacher_name", "dataset",
    ):
        assert key in raw, f"Checkpoint {path.name} missing required field: {key!r}"

    data: CheckpointData = {
        "state_dict": raw["state_dict"],
        "model_config": raw["model_config"],
        "backbone_name": raw["backbone_name"],
        "canvas_patch_grid_sizes": raw["canvas_patch_grid_sizes"],
        "teacher_dim": raw["teacher_dim"],
        "teacher_repo_id": raw["teacher_repo_id"],
        "teacher_name": raw["teacher_name"],
        "dataset": raw["dataset"],
        "glimpse_grid_size": raw["glimpse_grid_size"],
        "patch_stride": raw.get("patch_stride"),
        "scene_resolution": raw["scene_resolution"],
        "step": raw["step"],
        "train_loss": raw["train_loss"],
        "optimizer_state": raw["optimizer_state"],
        "scheduler_state": raw["scheduler_state"],
        "training_config_history": raw["training_config_history"],
        "provenance_history": raw.get("provenance_history"),
        "job_index": raw.get("job_index"),
        "ddp_world_size": raw["ddp_world_size"],
        "batch_size_per_gpu": raw["batch_size_per_gpu"],
        "steps_per_job": raw["steps_per_job"],
        "samples_per_shard": raw["samples_per_shard"],
        "timestamp": raw["timestamp"],
        "git_commit": raw["git_commit"],
        "git_dirty": raw["git_dirty"],
        "comet_id": raw["comet_id"],
        "wandb_run_id": raw.get("wandb_run_id"),
        "hostname": raw["hostname"],
        "slurm_job_id": raw["slurm_job_id"],
        "slurm_array_task_id": raw["slurm_array_task_id"],
        "cmdline": raw["cmdline"],
    }

    log.info(
        f"  backbone={data['backbone_name']}, grids={data['canvas_patch_grid_sizes']},"
        f" teacher={data['teacher_name']}, dataset={data['dataset']},"
        f" scene={data['scene_resolution']}px"
    )
    if data["step"] is not None:
        step = data["step"]
        msg = f"  step={step}, train_loss={data['train_loss']:.4e}" if data["train_loss"] else f"  step={step}"
        log.info(msg)
    if data["git_commit"]:
        log.info(f"  git={data['git_commit'][:8]}{'*' if data['git_dirty'] else ''}")

    return data


_STANDARDIZER_RE = re.compile(r"(cls|scene)_standardizers\.")


def load_state_dict_flexible(
    model: CanViTForPretraining,
    state_dict: dict[str, Tensor],
) -> None:
    """Load state dict, allowing standardizer key mismatches for grid-size changes.

    All non-standardizer keys must match exactly. Standardizer keys may differ
    when the canvas grid size changed between checkpoint and current model.
    """
    result = model.load_state_dict(state_dict, strict=False)
    bad_missing = [k for k in result.missing_keys if not _STANDARDIZER_RE.match(k)]
    bad_unexpected = [k for k in result.unexpected_keys if not _STANDARDIZER_RE.match(k)]
    assert not bad_missing, f"Missing core weights: {bad_missing}"
    assert not bad_unexpected, f"Unexpected core weights: {bad_unexpected}"
    if result.missing_keys or result.unexpected_keys:
        log.warning(f"Standardizer key mismatch (grid size change): "
                    f"missing={result.missing_keys}, unexpected={result.unexpected_keys}")
    else:
        log.info("Model state loaded (all keys matched)")


def load_model(
    path: Path, device: torch.device | str = "cpu",
) -> tuple[CanViTForPretraining, CheckpointData]:
    """Load CanViTForPretraining from checkpoint. Returns (model, checkpoint_data)."""
    from canvit.core import create_backbone

    ckpt = load(path, device)

    backbone_name = ckpt["backbone_name"]
    cfg = dacite.from_dict(CanViTForPretrainingConfig, ckpt["model_config"])

    # ``patch_stride`` absent for older runs -> None -> non-overlapping (patch_size).
    patch_stride = ckpt.get("patch_stride")
    backbone = create_backbone(backbone_name, patch_stride=patch_stride, recurrence=cfg.recurrence)
    # ``glimpse_grid_size`` is effectively required: load() above hard-indexes it
    # (KeyError for checkpoints without it), so no fallback here. With overlapping
    # patches the glimpse window is (grid-1)*stride + patch; patch_stride_px
    # defaults to patch_size, so this reduces to grid*patch for non-overlapping runs.
    glimpse_grid = ckpt["glimpse_grid_size"]
    glimpse_size_px = int(
        (glimpse_grid - 1) * backbone.patch_stride_px + backbone.patch_size_px
    )

    model = CanViTForPretraining(
        backbone=backbone,
        cfg=cfg,
        glimpse_size_px=glimpse_size_px,
        backbone_name=backbone_name,
        canvas_patch_grid_sizes=ckpt["canvas_patch_grid_sizes"],
    )
    load_state_dict_flexible(model, ckpt["state_dict"])

    if isinstance(device, str):
        device = torch.device(device)
    return model.to(device).eval(), ckpt
