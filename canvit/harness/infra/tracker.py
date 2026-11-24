"""Experiment tracker abstraction.

Wraps an optional Comet experiment OR an optional wandb run behind a single
interface that mirrors the surface used by `loop.py` and `viz/*.py`.
`make_tracker` selects the backend at job start based on `cfg.tracker`.
When neither backend is active (rank != 0 or `cfg.tracker == "none"`),
every method is a no-op so call sites stay unconditional.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import comet_ml
import wandb

log = logging.getLogger(__name__)


class Tracker:
    """Fans `log_*` calls out to whichever backend is active.

    `comet_exp` and `wandb_run` are mutually exclusive in practice (the
    selector exposes "comet", "wandb", or "none"), but both being None is
    a valid no-op and is what non-main ranks get.
    """

    def __init__(
        self,
        comet_exp: comet_ml.CometExperiment | None = None,
        wandb_run: Any | None = None,
    ) -> None:
        self._comet = comet_exp
        self._wandb = wandb_run

    def log_parameters(self, params: dict[str, Any]) -> None:
        if self._comet is not None:
            self._comet.log_parameters(params)
        if self._wandb is not None:
            self._wandb.config.update(params, allow_val_change=True)

    def log_metric(self, name: str, value: Any, step: int | None = None) -> None:
        if self._comet is not None:
            self._comet.log_metric(name, value, step=step)
        if self._wandb is not None:
            self._wandb.log({name: value}, step=step)

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if self._comet is not None:
            self._comet.log_metrics(metrics, step=step)
        if self._wandb is not None:
            self._wandb.log(metrics, step=step)

    def get_comet_id(self) -> str | None:
        return self._comet.get_key() if self._comet is not None else None

    def get_wandb_id(self) -> str | None:
        return self._wandb.id if self._wandb is not None else None

    def get_key(self) -> str:
        """Stable identifier for this run. Prefers wandb, falls back to comet."""
        return self.get_wandb_id() or self.get_comet_id() or "no-tracker"

    def end(self) -> None:
        if self._comet is not None:
            self._comet.end()
        if self._wandb is not None:
            self._wandb.finish()


def make_tracker(
    *,
    tracker: str,
    is_main: bool,
    is_seeding: bool,
    run_name: str,
    wandb_project: str | None,
    wandb_entity: str | None,
    wandb_dir: Path | None,
    prev_comet_id: str | None,
    prev_wandb_id: str | None,
    fork_from: str | None = None,
) -> Tracker:
    """Build the rank-0 tracker for this job. Non-main ranks get a no-op."""
    if not is_main or tracker == "none":
        return Tracker()

    if tracker == "comet":
        comet_cfg = comet_ml.ExperimentConfig(auto_metric_logging=False)
        if prev_comet_id is not None and not is_seeding:
            log.info(f"Continuing Comet experiment: {prev_comet_id}")
            exp = comet_ml.start(experiment_key=prev_comet_id, experiment_config=comet_cfg)
        else:
            if is_seeding and prev_comet_id:
                log.info(f"SEED mode: creating new Comet experiment (seed source had {prev_comet_id})")
            else:
                log.info("Creating NEW Comet experiment")
            exp = comet_ml.start(experiment_config=comet_cfg)
        return Tracker(comet_exp=exp)

    if tracker == "wandb":
        assert wandb_project, "tracker='wandb' requires --wandb-project"
        if wandb_dir is not None:
            wandb_dir.mkdir(parents=True, exist_ok=True)
        kwargs: dict[str, Any] = {
            "project": wandb_project,
            "name": run_name,
        }
        if wandb_dir is not None:
            kwargs["dir"] = str(wandb_dir)
        if wandb_entity:
            kwargs["entity"] = wandb_entity
        if prev_wandb_id is not None and not is_seeding:
            log.info(f"Resuming wandb run: {prev_wandb_id}")
            kwargs["id"] = prev_wandb_id
            kwargs["resume"] = "allow"
        elif fork_from is not None:
            log.info(f"Forking wandb run {fork_from} (new run continues its curves)")
            try:
                return Tracker(wandb_run=wandb.init(**kwargs, fork_from=fork_from))
            except Exception as e:  # fork not available for this entity/server: a plain new run
                log.warning(f"wandb fork_from={fork_from} failed ({e!r}); creating a NEW wandb run instead")
        else:
            if is_seeding and prev_wandb_id:
                log.info(f"SEED mode: creating new wandb run (seed source had {prev_wandb_id})")
            else:
                log.info("Creating NEW wandb run")
        run = wandb.init(**kwargs)
        return Tracker(wandb_run=run)

    raise ValueError(f"Unknown tracker: {tracker!r} (expected 'comet', 'wandb', or 'none')")
