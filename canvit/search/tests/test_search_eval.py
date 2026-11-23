"""A search checkpoint is enough to evaluate the model: ``harness.evaluate search --opts.ckpt``
rebuilds the run's config (saved in the checkpoint), its architecture (``model_config``, no
backbone file) and its learned policy (inside the same file), and reproduces the training-time
validation exactly. Tiny foveated ViT-S on the synthetic COCO-Search18 from conftest, CPU."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from canvit.harness.cli import command_from_run_config, run_config_of
from canvit.harness.evaluate import SearchEval, evaluate, parse
from canvit.harness.run import RunSettings, run
from canvit.search.task import SearchRunTask

from .test_search_task import _cfg, _rl, _TinyTask


class _RecordingTinyTask(_TinyTask):
    """Keeps the metrics of its last training-time validation."""

    def evaluate(self, **kw):
        self.last_eval = super().evaluate(**kw)
        return self.last_eval


def _argv(root: Path, run_dir: Path) -> list[str]:
    """A search command line matching `_cfg` + `_rl` (the run config the checkpoint will carry)."""
    return ["search", "--cfg.data-root", str(root), "--cfg.scene-size", "224", "--cfg.canvas-grid", "32",
            "--cfg.n-timesteps", "3", "--cfg.batch-size", "2", "--cfg.eval-batch-size", "2",
            "--cfg.num-workers", "0", "--cfg.head-width", "16", "--cfg.head-block-layers", "1",
            "--cfg.cond-dim", "8", "--cfg.max-steps", "4", "--cfg.warmup-steps", "1",
            "--cfg.limit-val-batches", "1", "--cfg.device", "cpu", "--cfg.no-amp", "--cfg.tracker", "none",
            "--cfg.aug.crop", "--rl.width", "16", "--rl.block-layers", "1",
            "--cfg.run-group", "g", "--cfg.logs-dir", str(run_dir.parent.parent)]


@pytest.fixture(scope="module")
def trained(fake_cs18, tmp_path_factory):
    """Train the tiny search model 2 steps (validation at step 2 writes best.pt)."""
    root, _ = fake_cs18
    run_dir = tmp_path_factory.mktemp("logs") / "g" / "tiny"
    argv = _argv(root, run_dir)
    cmd, drift = command_from_run_config({"argv": argv})
    task = _RecordingTinyTask(_cfg(root), rl=_rl())
    assert task.cfg == replace(cmd.cfg, run_group=task.cfg.run_group, logs_dir=task.cfg.logs_dir)
    assert task.rl == cmd.rl
    settings = RunSettings(n_steps=2, eval_every=2, log_every=1, grad_clip=1.0, amp=False, seed=0,
                           device="cpu", tracker="none", run_dir=run_dir,
                           run_config=run_config_of(cmd, argv))
    run(task=task, spec=task.default_spec(), settings=settings)
    return root, run_dir, task.last_eval, argv


def test_the_run_config_is_saved_in_the_checkpoint_and_the_run_dir(trained):
    _, run_dir, _, argv = trained
    meta = torch.load(run_dir / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)["metadata"]
    assert meta["run_config"]["argv"] == argv
    assert meta["run_config"]["resolved"]["cfg"]["head_width"] == 16
    assert json.loads((run_dir / "run_config.json").read_text()) == meta["run_config"]


def test_checkpoint_alone_reproduces_the_training_validation(trained):
    """ONE file, no --cfg.model-repo anywhere: the architecture, the learned policy and the run
    config all come from best.pt. Same metrics, exactly."""
    _, run_dir, trained_metrics, _ = trained
    ckpt = run_dir / "checkpoints" / "best.pt"
    assert list(ckpt.parent.glob("*.policy.pt")) == []     # no separate policy file any more
    cmd, source = parse(["search", "--opts.ckpt", str(ckpt), "--opts.device", "cpu"])
    assert isinstance(cmd, SearchEval) and cmd.cfg.model_repo is None and source["drift"] == []
    task, name = cmd.build()
    record = evaluate(task, cmd.cfg, cmd.opts, task_name=name)
    assert record["protocol"]["learned_policy"] is True
    assert any(k.endswith("_policy") for k in record["metrics"])
    assert record["metrics"] == {k: float(v) for k, v in trained_metrics.items()}


def test_command_line_overrides_the_checkpoints_config(trained):
    _, run_dir, _, _ = trained
    ckpt = str(run_dir / "checkpoints" / "best.pt")
    cmd, _ = parse(["search", "--opts.ckpt", ckpt, "--cfg.eval-seed", "7"])
    assert cmd.cfg.eval_seed == 7 and cmd.cfg.head_width == 16   # typed wins, the rest is the run's
    with pytest.raises(ValueError, match="harness.evaluate search"):
        parse(["ade20k", "--opts.ckpt", ckpt])


def test_a_changed_default_is_reported_as_drift(trained):
    _, run_dir, _, argv = trained
    rc = torch.load(run_dir / "checkpoints" / "best.pt", map_location="cpu",
                    weights_only=False)["metadata"]["run_config"]
    rc["resolved"]["cfg"]["presence_weight"] = 0.5          # "the run had another default"
    _, drift = command_from_run_config(rc)
    assert drift == ["cfg.presence_weight"]


def test_without_a_run_config_every_flag_is_typed_as_before(tmp_path):
    ckpt = tmp_path / "old.pt"
    torch.save({"model_state": {}, "metadata": {}}, ckpt)
    cmd, source = parse(["search", "--opts.ckpt", str(ckpt), "--cfg.head-width", "48"])
    assert source is None and cmd.cfg.head_width == 48 and cmd.cfg.cond_dim == 64


def test_search_builds_the_checkpoints_architecture_without_a_backbone_file(trained):
    """build_model(prior_model_config=...) needs no model_repo, and refuses a class list
    that would shift the cue indices."""
    root, run_dir, _, _ = trained
    payload = torch.load(run_dir / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)
    task = SearchRunTask(_cfg(root), rl=_rl())
    assert task.cfg.model_repo is None
    model, _ = task.build_model(torch.device("cpu"), prior_model_config=payload["model_config"])
    model.load_state_dict(payload["model_state"], strict=True)
    bad = dict(payload["model_config"], classes=list(reversed(payload["model_config"]["classes"])))
    with pytest.raises(ValueError, match="cue indices"):
        SearchRunTask(_cfg(root), rl=_rl()).build_model(torch.device("cpu"), prior_model_config=bad)
