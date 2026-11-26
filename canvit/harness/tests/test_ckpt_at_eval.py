"""Checkpoint at every validation (RunSettings.ckpt_at_eval / keep_ckpts), the run's total
(max_steps), atomic checkpoint writes and the per-task CLI defaults (2026-11-26).

The point: a job killed by its time limit must leave a resumable checkpoint of exactly the
last validated weights, so the next array task continues the SAME run (no restart from 0,
no duplicate W&B run) and loses at most one validation interval.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from canvit.harness.infra.checkpoint import find_latest, load_checkpoint, save_checkpoint
from canvit.harness.loop import prune_step_ckpts, read_last_val, run_training_loop
from canvit.harness.run import RunSettings, run
from canvit.harness.tests.test_loop_e2e import _batches, _probe_setup, _resume_kw


def _head_params(setup) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in setup["head"].state_dict().items()}


def test_each_validation_leaves_a_checkpoint_of_exactly_the_validated_weights(tmp_path):
    setup = _probe_setup()
    seen: dict[int, dict] = {}

    def on_eval(s):
        seen[s] = _head_params(setup)
        return {"miou_final": 0.1}
    run_training_loop(**setup, ckpt_dir=tmp_path, eval_every=2, on_eval=on_eval,
                      ckpt_at_eval=True, train_batches=_batches(), n_steps=5)   # vals 0, 2, 4, end 5
    assert sorted(seen) == [0, 2, 4, 5]
    files = sorted(p.name for p in tmp_path.glob("step-*.pt"))
    assert files == ["step-2.pt", "step-4.pt", "step-5.pt"]          # none at 0 (nothing trained yet)
    for s in (2, 4, 5):
        state = load_checkpoint(tmp_path / f"step-{s}.pt", "cpu")["model_state"]
        for k, v in seen[s].items():
            assert torch.equal(state[f"head.{k}"], v), (s, k)
    assert read_last_val(tmp_path)["step"] == 5


def test_keep_one_leaves_only_the_newest_step_checkpoint_and_best(tmp_path):
    scores = {0: 0.1, 2: 0.9, 4: 0.3, 5: 0.2}
    run_training_loop(**_probe_setup(), ckpt_dir=tmp_path, eval_every=2, best_metric="miou_final",
                      on_eval=lambda s: {"miou_final": scores[s]}, ckpt_at_eval=True, keep_ckpts=1,
                      train_batches=_batches(), n_steps=5)
    assert sorted(p.name for p in tmp_path.glob("*.pt")) == ["best.pt", "latest.pt", "step-5.pt"]
    assert find_latest(tmp_path).name == "step-5.pt"
    assert load_checkpoint(tmp_path / "best.pt", "cpu")["step"] == 2


def test_a_job_killed_mid_way_resumes_at_its_last_validation(tmp_path):
    """Killed during the validation at step 4: the checkpoint of step 2 (validated) is the
    resume point; the next job does not validate 2 again and continues the run."""
    evals: list[int] = []

    def on_eval(s):
        if s == 4 and not on_eval.died:
            on_eval.died = True
            raise TimeoutError("time limit")
        evals.append(s)
        return {"miou_final": 0.1 * s}
    on_eval.died = False
    kw = dict(_probe_setup(), ckpt_dir=tmp_path, eval_every=2, best_metric="miou_final",
              on_eval=on_eval, ckpt_at_eval=True, keep_ckpts=1)
    with pytest.raises(TimeoutError):
        run_training_loop(**kw, train_batches=_batches(), n_steps=6)
    assert find_latest(tmp_path).name == "step-2.pt" and read_last_val(tmp_path)["step"] == 2
    resume = _resume_kw(tmp_path)
    assert resume["start_step"] == 2 and resume["validated_step"] == 2
    run_training_loop(**kw, train_batches=_batches(), n_steps=4, **resume)      # 2 (skip), 4, end 6
    assert evals == [0, 2, 4, 6]
    assert sorted(p.name for p in tmp_path.glob("step-*.pt")) == ["step-6.pt"]


def test_prune_keeps_the_latest_target_even_if_not_the_highest_step(tmp_path):
    for s in (1, 2, 3):
        (tmp_path / f"step-{s}.pt").write_bytes(b"x")
    (tmp_path / "latest.pt").symlink_to("step-1.pt")      # e.g. a hand-repointed latest
    gone = prune_step_ckpts(tmp_path, keep=1)
    assert sorted(p.name for p in gone) == ["step-2.pt"]
    assert sorted(p.name for p in tmp_path.glob("step-*.pt")) == ["step-1.pt", "step-3.pt"]


def test_a_checkpoint_write_that_dies_leaves_no_truncated_file(tmp_path, monkeypatch):
    setup = _probe_setup()
    kw = dict(model=setup["model"], optimizer=setup["optimizer"], scheduler=setup["scheduler"],
              task_name="ade20k", spec=setup["spec"])
    save_checkpoint(tmp_path / "step-1.pt", step=1, **kw)

    def dying_save(obj, f, *a, **k):
        with open(f, "wb") as fh:
            fh.write(b"truncat")
        raise OSError("killed")
    monkeypatch.setattr(torch, "save", dying_save)
    with pytest.raises(OSError):
        save_checkpoint(tmp_path / "step-2.pt", step=1, **kw)
    assert not (tmp_path / "step-2.pt").exists()
    assert find_latest(tmp_path).name == "step-1.pt"


def test_ckpt_at_eval_is_refused_for_a_task_that_resumes_only_at_job_boundaries():
    task = SimpleNamespace(name="distill", resumes_mid_job=False)
    with pytest.raises(ValueError, match="job boundary"):
        run(task=task, spec=None, settings=RunSettings(ckpt_at_eval=True, device="cpu"))  # type: ignore[arg-type]


def test_cli_defaults_on_for_search_off_elsewhere(tmp_path):
    from canvit.distill.config import Config as DistillConfig
    from canvit.harness.cli import DistillCmd, HarnessOpts, SearchCmd
    from canvit.search.config import SearchConfig

    _, s = SearchCmd(cfg=SearchConfig(run_group="g", max_steps=123)).build()
    assert (s.ckpt_at_eval, s.keep_ckpts, s.max_steps) == (True, 1, 123)
    _, s = SearchCmd(cfg=SearchConfig(run_group="g"), opts=HarnessOpts(ckpt_at_eval=False, keep_ckpts=0)).build()
    assert (s.ckpt_at_eval, s.keep_ckpts) == (False, 0)
    _, s = DistillCmd(cfg=DistillConfig(run_group="g")).build()
    assert (s.ckpt_at_eval, s.keep_ckpts, s.max_steps) == (False, 0, None)
