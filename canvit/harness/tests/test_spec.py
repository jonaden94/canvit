"""CPU unit tests for the harness TrainSpec + validation (design §3.3/§8)."""

import pytest

from canvit.harness.spec import (
    BpttSpec,
    GroupOptim,
    ScheduleSpec,
    TaskCaps,
    TrainSpec,
    check_spec,
)

HEAD_POLICY = TaskCaps(has_head=True, supports_policy=True)
HEAD_ONLY = TaskCaps(has_head=True, supports_policy=False)


def go(**kw) -> GroupOptim:
    return GroupOptim(lr=kw.pop("lr", 1e-3), **kw)


# --------------------------------------------------------------------------- #
# Valid presets — the four canonical configs must pass for a capable task.
# --------------------------------------------------------------------------- #
def test_probe_preset_valid():
    spec = TrainSpec.probe(optim={"head": go()})
    r = check_spec(spec, HEAD_ONLY)
    assert r.ok, r.errors
    assert spec.trainable_modules() == ("head",)
    assert not spec.task_grad_to_backbone


def test_finetune_preset_valid():
    spec = TrainSpec.finetune(optim={"backbone": go(lr=1e-4), "head": go()})
    r = check_spec(spec, HEAD_ONLY)
    assert r.ok, r.errors
    assert spec.trainable_modules() == ("backbone", "head")


def test_policy_only_frozen_valid():
    spec = TrainSpec.policy_only(freeze_model=True, optim={"policy": go(lr=2e-4)})
    r = check_spec(spec, HEAD_POLICY)
    assert r.ok, r.errors
    assert spec.feats_detached is True  # frozen => policy grad does not reach backbone
    assert not spec.task_loss_active and spec.policy_loss_active


def test_policy_only_unfrozen_backbone_valid():
    # Train the policy AND let its loss reshape the backbone (no task loss).
    spec = TrainSpec.policy_only(
        freeze_model=False, optim={"backbone": go(lr=1e-5), "policy": go(lr=2e-4)}
    )
    r = check_spec(spec, HEAD_POLICY)
    assert r.ok, r.errors
    assert spec.policy_grad_to_backbone and not spec.feats_detached
    # single-GPU allowed; warns that the backbone is trained solely by policy
    assert any("solely by the policy" in x for x in r.warnings)


def test_joint_preset_valid_and_maps_to_p4b():
    spec = TrainSpec.joint(optim={"backbone": go(lr=1e-4), "head": go(), "policy": go(lr=2e-4)})
    r = check_spec(spec, HEAD_POLICY)
    assert r.ok, r.errors
    assert spec.feats_detached is True  # P4b default: policy-net-only
    assert spec.bptt.stochastic and spec.bptt.chunk_size == 2


# --------------------------------------------------------------------------- #
# Hard errors (reject).
# --------------------------------------------------------------------------- #
def test_nothing_trainable_errors():
    spec = TrainSpec(train_backbone=False, train_head=False, train_policy=False)
    assert not check_spec(spec, HEAD_POLICY).ok


def test_both_weights_zero_errors():
    spec = TrainSpec(train_head=True, task_weight=0.0, policy_weight=0.0, optim={"head": go()})
    assert any("no loss is active" in x for x in check_spec(spec, HEAD_POLICY).errors)


def test_policy_loss_without_train_policy_errors():
    spec = TrainSpec(train_head=True, policy_weight=1.0, train_policy=False, optim={"head": go()})
    assert any("train_policy is False" in x for x in check_spec(spec, HEAD_POLICY).errors)


def test_train_policy_without_policy_loss_errors():
    spec = TrainSpec(train_head=True, train_policy=True, policy_weight=0.0,
                     optim={"head": go(), "policy": go()})
    assert any("policy would never learn" in x for x in check_spec(spec, HEAD_POLICY).errors)


def test_head_without_capability_errors():
    spec = TrainSpec.probe(optim={"head": go()})
    assert any("no head" in x for x in check_spec(spec, TaskCaps(has_head=False, supports_policy=False)).errors)


def test_policy_without_capability_errors():
    spec = TrainSpec.policy_only(optim={"policy": go()})
    assert any("does not support a policy" in x for x in check_spec(spec, HEAD_ONLY).errors)


def test_task_grad_into_frozen_backbone_errors():
    spec = TrainSpec(train_backbone=False, train_head=True, task_grad_to_backbone=True,
                     optim={"head": go()})
    assert any("routing task loss into a frozen backbone" in x for x in check_spec(spec, HEAD_POLICY).errors)


def test_policy_grad_into_frozen_backbone_errors():
    spec = TrainSpec(train_head=True, train_policy=True, policy_weight=1.0,
                     train_backbone=False, policy_grad_to_backbone=True,
                     optim={"head": go(), "policy": go()})
    errs = check_spec(spec, HEAD_POLICY).errors
    assert any("routing policy loss into a frozen backbone" in x for x in errs)


def test_coupled_policy_backbone_allowed_under_ddp():
    """Was refused under DDP until 2026-11-22; the loop syncs every trainable grad by hand,
    so the policy gradient into the backbone is averaged like the task gradient (2-GPU
    cross-rank identity verified, see spec.py)."""
    spec = TrainSpec.joint(policy_grad_to_backbone=True,
                           optim={"backbone": go(lr=1e-4), "head": go(), "policy": go(lr=2e-4)})
    assert check_spec(spec, HEAD_POLICY, is_dist=False).ok
    assert check_spec(spec, HEAD_POLICY, is_dist=True).ok


def test_task_without_ddp_support_refuses_multi_gpu():
    """A task whose loader cannot shard by rank (ade20k) must be REFUSED under DDP, not
    silently run on overlapping samples. Fires before the model is built."""
    no_ddp = TaskCaps(has_head=True, supports_policy=True, supports_ddp=False)
    spec = TrainSpec.probe(optim={"head": go()})
    assert check_spec(spec, no_ddp, is_dist=False).ok
    assert any("does not support DDP" in x for x in check_spec(spec, no_ddp, is_dist=True).errors)
    assert check_spec(spec, HEAD_POLICY, is_dist=True).ok  # default caps: DDP allowed


def test_missing_optim_group_errors():
    spec = TrainSpec.finetune(optim={"head": go()})  # missing backbone group
    assert any("optim[backbone] missing" in x for x in check_spec(spec, HEAD_ONLY).errors)


def test_validate_raises_on_error():
    with pytest.raises(ValueError, match="invalid TrainSpec"):
        TrainSpec(train_backbone=False, train_head=False).validate(HEAD_POLICY)


# --------------------------------------------------------------------------- #
# BpttSpec validity.
# --------------------------------------------------------------------------- #
def test_bptt_requires_exactly_one_length():
    assert BpttSpec(mode="full", horizon=5, continue_prob=0.5).errors()
    assert BpttSpec(mode="full").errors()  # neither set


def test_bptt_continue_prob_requires_chunked():
    assert any("requires mode='chunked'" in x for x in BpttSpec(mode="full", continue_prob=0.5).errors())
    assert not BpttSpec(mode="chunked", chunk_size=2, continue_prob=0.5).errors()


def test_bptt_ranges():
    assert BpttSpec(mode="chunked", chunk_size=0, horizon=1).errors()
    assert any("continue_prob" in x for x in BpttSpec(mode="chunked", continue_prob=1.5).errors())


def test_schedule_requires_total_steps():
    go_bad = GroupOptim(lr=1e-3, schedule=ScheduleSpec(kind="warmup_cosine"))
    assert any("requires total_steps" in x for x in go_bad.errors(group="head"))


def test_schedule_warmup_must_be_below_total():
    """warmup >= total pins the LR in warmup for the whole run (no anneal). Rejected for
    the decaying kinds, matching the standalone warmup_cosine_scheduler's assert."""
    bad = ScheduleSpec(kind="warmup_cosine", warmup_steps=10_000, total_steps=800)
    assert any("must be < total_steps" in x for x in bad.errors(group="head"))
    ok = ScheduleSpec(kind="warmup_cosine", warmup_steps=50, total_steps=1000)
    assert not ok.errors(group="head")


# --------------------------------------------------------------------------- #
# Warnings (run anyway).
# --------------------------------------------------------------------------- #
def test_warn_optim_for_non_trainable_group():
    spec = TrainSpec.probe(optim={"head": go(), "backbone": go()})
    r = check_spec(spec, HEAD_ONLY)
    assert r.ok
    assert any("backbone] given but backbone is not trainable" in x for x in r.warnings)


def test_warn_backbone_trainable_but_no_loss_routed():
    spec = TrainSpec(train_backbone=True, train_head=True,
                     task_grad_to_backbone=False, task_weight=1.0,
                     optim={"backbone": go(lr=1e-4), "head": go()})
    r = check_spec(spec, HEAD_ONLY)
    assert r.ok
    assert any("no loss is routed to the backbone" in x for x in r.warnings)
