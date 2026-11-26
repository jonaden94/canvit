"""Unified LOCAL checkpoint I/O for the harness (design D-G).

Persists everything needed to resume + (optionally, later) publish: the full model
state (backbone + head — both are submodules of the task wrapper, so one
``state_dict()`` captures them), the optimizer/scheduler, the step, the resolved
``TrainSpec``, the model config, and free-form metadata (incl. ``pretrain_view_scale``
for the foveated footgun, and the run's full config). The policy (scorer + reward
standardizers + PG dual), for every task that trains one, is in the same file
(``policy_state``); checkpoints before 2026-11-22 kept it in a ``.policy.pt`` sidecar, which
is still read.

**Never touches the network.** Training writes local files only — exactly as
this repo and the retired specialize repo do. HF publishing stays a separate, manual
``python -m canvit.checkpoint.to_hf`` step (owner decision D-G).
"""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from canvit.harness.spec import TrainSpec


def save_checkpoint(
    path: Path,
    *,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    step: int,
    task_name: str,
    spec: TrainSpec,
    model_config: dict | None = None,
    metadata: dict | None = None,
    joint: Any | None = None,
    extra: dict | None = None,
    update_latest: bool = True,
) -> Path:
    """Write a full local checkpoint: ONE file, with the policy (``joint``, any task) inside as
    ``policy_state``. Returns the path written. DDP-safe to call on rank 0 only (unwraps
    ``.module``)."""
    core = getattr(model, "module", model)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "task": task_name,
        "model_state": core.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "train_spec": asdict(spec),
        "model_config": model_config or {},
        "metadata": metadata or {},
        "extra": extra or {},
        "policy_state": joint.state_dict() if joint is not None else None,
    }
    # Atomic: a job killed mid-write (time limit) leaves only the .tmp, never a truncated
    # step-N.pt that find_latest's fallback (highest step file) would try to resume from.
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    if update_latest:
        update_latest_symlink(path.parent / "latest.pt", path)
    return path


def load_checkpoint(path: Path, device: torch.device | str) -> dict:
    """Load a checkpoint payload. Use ``restore_into`` to apply it to live modules."""
    return torch.load(path, map_location=device, weights_only=False)


def restore_into(
    payload: dict, *, model: Any, optimizer: Any = None, scheduler: Any = None,
    joint: Any | None = None, path: Path | None = None, device: torch.device | str = "cpu",
    new_param_prefixes: tuple[str, ...] = (),
) -> int:
    """Apply a loaded payload to live modules; return the saved step. Optimizer/scheduler
    are restored only when provided (resume mode); the policy when ``joint`` is given and
    the checkpoint has one (see :func:`policy_state_of`).

    ``new_param_prefixes`` (seed mode only): model params under these name prefixes may be
    ABSENT from the payload and keep their fresh init (e.g. policy tokens added to a
    pretrained model). Every other key must match exactly, as in the strict load."""
    core = getattr(model, "module", model)
    if not new_param_prefixes:
        core.load_state_dict(payload["model_state"])
    else:
        res = core.load_state_dict(payload["model_state"], strict=False)
        bad_missing = [k for k in res.missing_keys if not k.startswith(new_param_prefixes)]
        if res.unexpected_keys or bad_missing:
            raise RuntimeError(f"seed load: unexpected keys {res.unexpected_keys}, missing keys "
                               f"outside {new_param_prefixes}: {bad_missing}")
    if optimizer is not None and payload.get("optimizer_state") is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None and payload.get("scheduler_state") is not None:
        scheduler.load_state_dict(payload["scheduler_state"])
    if joint is not None:
        state = policy_state_of(payload, path=path, device=device)
        if state is not None:
            joint.load_state_dict(state)
    return int(payload.get("step", 0))


def policy_state_of(payload: dict, *, path: Path | None, device: torch.device | str = "cpu") -> dict | None:
    """The policy saved with a checkpoint, None if its run trained none: the payload's
    ``policy_state``, or -- for checkpoints written before 2026-11-22 -- the ``.policy.pt``
    sidecar file next to ``path``."""
    if payload.get("policy_state") is not None:
        return payload["policy_state"]
    sidecar = path.with_suffix(".policy.pt") if path is not None else None
    if sidecar is not None and sidecar.exists():
        return torch.load(sidecar, map_location=device, weights_only=False)
    return None


def update_latest_symlink(link: Path, target: Path) -> None:
    """Point ``link`` at ``target`` (atomic replace). Best-effort: falls back to a
    tiny pointer file on filesystems without symlinks."""
    try:
        tmp = link.with_suffix(".tmp")
        if tmp.exists() or tmp.is_symlink():
            tmp.unlink()
        os.symlink(target.name, tmp)
        os.replace(tmp, link)
    except OSError:
        link.write_text(target.name)


def find_latest(ckpt_dir: Path) -> Path | None:
    """Resolve the newest checkpoint: the ``latest.pt`` symlink/pointer if present,
    else the highest ``step-<n>.pt``. None if the dir has no checkpoints."""
    if not ckpt_dir.is_dir():
        return None
    latest = ckpt_dir / "latest.pt"
    if latest.is_symlink() or latest.is_file():
        target = ckpt_dir / (os.readlink(latest) if latest.is_symlink() else latest.read_text().strip())
        if target.is_file():
            return target
    steps = sorted(ckpt_dir.glob("step-*.pt"), key=lambda p: int(p.stem.split("-")[1]))
    return steps[-1] if steps else None


__all__ = ["find_latest", "load_checkpoint", "policy_state_of", "restore_into", "save_checkpoint",
           "update_latest_symlink"]
