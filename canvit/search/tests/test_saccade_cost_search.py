"""Stage 2 of the saccade cost on the search task: a beta-conditioned policy (beta drawn per
trial, appended to the FiLM cue), the eval beta, and the checkpoint round trip. Off (the
default) keeps the cue, the scorer and the metadata as before. Tiny foveated ViT-S on the
synthetic COCO-Search18 from conftest, CPU.

Written by Claude (AI, Claude Opus 5.5), 2026-11-25."""

from contextlib import nullcontext
from dataclasses import replace

import pytest
import torch

from canvit.harness.cli import command_from_run_config, run_config_of
from canvit.harness.evaluate import evaluate, parse
from canvit.harness.policy.joint import BETA_FEATURES, SaccadeCost
from canvit.harness.run import RunSettings, run

from .test_search_eval import _argv, _RecordingTinyTask
from .test_search_task import _cfg, _rl, _run, _TinyTask

STAGE2 = {"saccade_cost": "log", "saccade_cost_weight": 1.0, "saccade_cost_beta_mode": "conditioned"}
STAGE2_ARGV = ["--rl.saccade-cost", "log", "--rl.saccade-cost-weight", "1.0",
               "--rl.saccade-cost-beta-mode", "conditioned", "--rl.saccade-cost-beta-eval", "0.3"]


def _built(root, **rl):
    task = _TinyTask(_cfg(root), rl=_rl(**rl))
    model, _ = task.build_model(torch.device("cpu"))
    joint = task.build_policy(model, device=torch.device("cpu"), canvas_grid=32,
                              generator=torch.Generator().manual_seed(0))
    return task, model, joint


def _batch(task):
    return next(iter(task.build_val_loader()))


def test_off_keeps_the_cue_the_scorer_and_the_metadata(fake_cs18):
    root, _ = fake_cs18
    task, model, joint = _built(root)
    assert joint.scorer.film_cond_dim == 8 and not joint.saccade_cost.active
    b = _batch(task)
    assert torch.equal(task._policy_cond(model, b["target"]), task._policy_embed(model)(b["target"]))
    bound = task.bind(b, torch.device("cpu"), model=model, head=model.head)
    assert not hasattr(bound, "saccade_beta")
    assert "saccade_cost" not in task.checkpoint_metadata(model)


def test_conditioned_policy_sees_its_trials_beta_next_to_the_cue(fake_cs18):
    root, _ = fake_cs18
    task, model, joint = _built(root, **STAGE2, saccade_cost_beta_eval=0.3)
    assert joint.scorer.film_cond_dim == 8 + BETA_FEATURES
    b = _batch(task)
    torch.manual_seed(0)
    bound = task.bind(b, torch.device("cpu"), model=model, head=model.head)
    beta = bound.saccade_beta
    assert beta.shape == (len(b["target"]),) and bool(((beta >= 0) & (beta <= 1.0)).all())
    cond = bound.policy_cond()
    assert torch.equal(cond[:, :8], task._policy_embed(model)(b["target"]))
    assert torch.equal(cond[:, 8:], SaccadeCost.features(beta))
    # validation / deploy: every trial gets the eval beta
    ev = task._policy_cond(model, b["target"])
    assert torch.equal(ev[:, 8:], SaccadeCost.features(torch.full((len(b["target"]),), 0.3)))


def test_the_eval_beta_steers_the_deployed_policy(fake_cs18):
    """With the FiLM generators off their zero init, a different eval beta gives different
    policy maps on the same canvas (beta actually reaches the scorer)."""
    root, _ = fake_cs18
    task, model, joint = _built(root, **STAGE2)
    torch.manual_seed(1)
    with torch.no_grad():
        for p in joint.scorer.parameters():
            p.add_(0.1 * torch.randn_like(p))
    model.canvit.eval()
    b = _batch(task)
    maps = {}
    for beta_eval in (0.0, 1.0):
        joint.saccade_cost = replace(joint.saccade_cost, beta_eval=beta_eval)
        record: list = []
        with torch.no_grad():
            task._rollout_metrics(model=model, images=b["image"], targets=b["target"], boxes=b["boxes"],
                                  T=2, cg=32, amp=nullcontext(), joint=joint, record=record)
        maps[beta_eval] = record[0]["policy_next"]
    assert not torch.allclose(maps[0.0], maps[1.0])


def test_conditioned_search_trains_end_to_end(fake_cs18):
    root, _ = fake_cs18
    task = _TinyTask(_cfg(root), rl=_rl(**STAGE2))
    last = _run(task)
    assert torch.isfinite(torch.tensor(last["total_loss"]))
    assert task._joint.scorer.film_in.gen.net[0].in_features == 8 + BETA_FEATURES
    assert task._joint.scorer.film_in.gen.net[-1].weight.abs().sum() > 0  # trained off its zero init


@pytest.fixture(scope="module")
def trained_stage2(fake_cs18, tmp_path_factory):
    """Train the tiny conditioned search model 2 steps from a command line (best.pt at step 2)."""
    root, _ = fake_cs18
    run_dir = tmp_path_factory.mktemp("logs") / "g" / "tiny_beta"
    argv = _argv(root, run_dir) + STAGE2_ARGV
    cmd, _ = command_from_run_config({"argv": argv})
    task = _RecordingTinyTask(_cfg(root), rl=_rl(**STAGE2, saccade_cost_beta_eval=0.3))
    assert task.rl == cmd.rl
    settings = RunSettings(n_steps=2, eval_every=2, log_every=1, grad_clip=1.0, amp=False, seed=0,
                           device="cpu", tracker="none", run_dir=run_dir, run_config=run_config_of(cmd, argv))
    run(task=task, spec=task.default_spec(), settings=settings)
    return run_dir, task.last_eval


def test_stage2_checkpoint_records_the_cost_and_reproduces_its_validation(trained_stage2):
    run_dir, trained_metrics = trained_stage2
    ckpt = run_dir / "checkpoints" / "best.pt"
    meta = torch.load(ckpt, map_location="cpu", weights_only=False)["metadata"]
    assert meta["saccade_cost"] | {"beta_features": None} == {
        "form": "log", "weight": 1.0, "combine": "standardized", "beta_mode": "conditioned", "beta_eval": 0.3,
        "r": 1 / 32, "beta_features": None}
    assert meta["run_config"]["resolved"]["rl"]["saccade_cost_beta_mode"] == "conditioned"
    cmd, source = parse(["search", "--opts.ckpt", str(ckpt), "--opts.device", "cpu"])
    assert source["drift"] == [] and cmd.rl.saccade_cost_beta_mode == "conditioned"
    task, name = cmd.build()
    record = evaluate(task, cmd.cfg, cmd.opts, task_name=name)
    assert record["metrics"] == {k: float(v) for k, v in trained_metrics.items()}


def test_stage2_checkpoint_is_scored_at_another_eval_beta(trained_stage2):
    run_dir, _ = trained_stage2
    ckpt = str(run_dir / "checkpoints" / "best.pt")
    cmd, _ = parse(["search", "--opts.ckpt", ckpt, "--opts.device", "cpu", "--rl.saccade-cost-beta-eval", "0.8"])
    assert cmd.rl.saccade_cost_beta_eval == 0.8 and cmd.rl.saccade_cost_weight == 1.0
    task, name = cmd.build()
    record = evaluate(task, cmd.cfg, cmd.opts, task_name=name)
    assert any(k.endswith("_policy") for k in record["metrics"])
