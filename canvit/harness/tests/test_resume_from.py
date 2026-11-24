"""``RunSettings.resume_from`` (LR decay as a continuation, 2026-11-24): another run's full state is
restored, then this run's lr / schedule replaces the source's while AdamW's moments and the step
count stay; W&B forks the source run instead of resuming it."""

import pytest
import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from canvit.harness.infra import tracker as tracker_mod
from canvit.harness.run import apply_configured_lr


def _warmup_constant(warm: int):
    return lambda step: (step / warm if step < warm else 1.0) if warm > 0 else 1.0


def _opt(model: nn.Module, lr: float, warm: int):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4, betas=(0.9, 0.999))
    return opt, LambdaLR(opt, lr_lambda=[_warmup_constant(warm)])


def test_lr_is_this_runs_and_the_rest_of_the_state_is_the_sources():
    torch.manual_seed(0)
    src = nn.Linear(4, 3)
    opt, sched = _opt(src, lr=8e-4, warm=5)
    for _ in range(12):  # past the source's warmup
        src(torch.randn(2, 4)).sum().backward()
        opt.step()
        opt.zero_grad()
        sched.step()
    payload = {"opt": opt.state_dict(), "sched": sched.state_dict(), "model": src.state_dict()}

    dst = nn.Linear(4, 3)
    opt2, sched2 = _opt(dst, lr=8e-5, warm=0)
    configured = [{k: g[k] for k in ("lr", "initial_lr", "weight_decay", "betas")} for g in opt2.param_groups]
    dst.load_state_dict(payload["model"])
    opt2.load_state_dict(payload["opt"])
    sched2.load_state_dict(payload["sched"])
    assert opt2.param_groups[0]["lr"] == pytest.approx(8e-4)  # without the reset: the source's lr
    apply_configured_lr(opt2, sched2, configured)

    assert sched2.last_epoch == 12 and sched2.base_lrs == [8e-5]
    assert opt2.param_groups[0]["lr"] == pytest.approx(8e-5)
    assert opt2.param_groups[0]["initial_lr"] == pytest.approx(8e-5)
    for p_src, p_dst in zip(src.parameters(), dst.parameters()):
        a, b = opt.state[p_src], opt2.state[p_dst]
        assert torch.equal(a["exp_avg"], b["exp_avg"]) and torch.equal(a["exp_avg_sq"], b["exp_avg_sq"])
        assert torch.equal(a["step"], b["step"])
    # the next steps keep this run's lr (the schedule continues at the restored step)
    dst(torch.randn(2, 4)).sum().backward()
    opt2.step()
    sched2.step()
    assert sched2.last_epoch == 13 and opt2.param_groups[0]["lr"] == pytest.approx(8e-5)


def test_refuses_a_group_count_mismatch():
    m = nn.Linear(2, 2)
    opt, sched = _opt(m, 1e-3, 0)
    with pytest.raises(ValueError, match="param groups"):
        apply_configured_lr(opt, sched, [])


def _capture_init(monkeypatch, fail_fork=False):
    calls = []

    class _Run:
        id = "new"

    def fake_init(**kw):
        calls.append(kw)
        if fail_fork and "fork_from" in kw:
            raise RuntimeError("fork not enabled")
        return _Run()

    monkeypatch.setattr(tracker_mod.wandb, "init", fake_init)
    return calls


def _make(**kw):
    return tracker_mod.make_tracker(tracker="wandb", is_main=True, is_seeding=False, run_name="r",
                                    wandb_project="p", wandb_entity=None, wandb_dir=None,
                                    prev_comet_id=None, **kw)


def test_wandb_forks_the_source_run(monkeypatch):
    calls = _capture_init(monkeypatch)
    _make(prev_wandb_id=None, fork_from="33gy6d4a?_step=360448")
    assert calls == [{"project": "p", "name": "r", "fork_from": "33gy6d4a?_step=360448"}]


def test_wandb_fork_failure_falls_back_to_a_new_run(monkeypatch):
    calls = _capture_init(monkeypatch, fail_fork=True)
    _make(prev_wandb_id=None, fork_from="33gy6d4a?_step=360448")
    assert calls[-1] == {"project": "p", "name": "r"}


def test_wandb_resume_unchanged(monkeypatch):
    calls = _capture_init(monkeypatch)
    _make(prev_wandb_id="own")
    assert calls == [{"project": "p", "name": "r", "id": "own", "resume": "allow"}]
