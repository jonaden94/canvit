"""The learned policy is part of the ONE checkpoint file, for every task: ``save_checkpoint`` is
task-neutral (the harness loop calls it with whatever ``joint`` the task built), so these tests use
stand-in modules. Checkpoints written before 2026-11-22 kept the policy in a ``.policy.pt`` sidecar;
those still restore."""

import torch
from torch import nn

from canvit.harness.infra.checkpoint import (
    load_checkpoint,
    policy_state_of,
    restore_into,
    save_checkpoint,
)
from canvit.harness.spec import TrainSpec


def _save(path, *, joint):
    model = nn.Linear(3, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    save_checkpoint(path, model=model, optimizer=opt, scheduler=sched, step=7, task_name="any",
                    spec=TrainSpec.probe(), joint=joint, update_latest=False)
    return model


def test_the_policy_is_saved_inside_the_checkpoint_and_restored_from_it(tmp_path):
    torch.manual_seed(0)
    policy = nn.Linear(4, 4)
    path = tmp_path / "best.pt"
    _save(path, joint=policy)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["best.pt"]        # one file
    fresh = nn.Linear(4, 4)
    restore_into(load_checkpoint(path, "cpu"), model=nn.Linear(3, 2), joint=fresh, path=path)
    assert all(torch.equal(a, b) for a, b in zip(fresh.state_dict().values(), policy.state_dict().values()))


def test_a_run_without_a_policy_saves_none(tmp_path):
    path = tmp_path / "best.pt"
    _save(path, joint=None)
    payload = load_checkpoint(path, "cpu")
    assert payload["policy_state"] is None and policy_state_of(payload, path=path) is None


def test_an_old_checkpoint_with_a_policy_sidecar_still_restores(tmp_path):
    torch.manual_seed(1)
    policy = nn.Linear(4, 4)
    path = tmp_path / "best.pt"
    _save(path, joint=None)
    payload = load_checkpoint(path, "cpu")
    del payload["policy_state"]                                          # the pre-2026-11-22 format
    torch.save(payload, path)
    torch.save(policy.state_dict(), path.with_suffix(".policy.pt"))
    fresh = nn.Linear(4, 4)
    restore_into(load_checkpoint(path, "cpu"), model=nn.Linear(3, 2), joint=fresh, path=path)
    assert torch.equal(fresh.weight, policy.weight)
