"""End-to-end CPU smoke of the harness vertical (design §Loop): the neutral
``run_training_loop`` drives an ADE20K probe on synthetic data through the engine +
per-group optimizer, trains only the head, and the local checkpoint round-trips.

Proves loop → task.bind → run_rollout → optimizer/scheduler → checkpoint wiring on
CPU. Real-data ``build_loaders``/``evaluate`` + the ``python -m canvit_pretrain.train``
dispatch are validated against the datasets separately.
"""

import torch

from canvit.ade20k.data import IGNORE_LABEL, NUM_CLASSES
from canvit.ade20k.task import BoundAde20kTask
from canvit.core import CanViTForSemanticSegmentation
from canvit.harness.config import FoveatedScaleConfig
from canvit.harness.infra.checkpoint import find_latest, load_checkpoint, restore_into
from canvit.harness.loop import apply_requires_grad, run_training_loop
from canvit.harness.optim import build_optimizer_and_scheduler
from canvit.harness.rollout.selector import RandomSelector
from canvit.harness.rollout.viewpoint import ViewpointType
from canvit.harness.spec import BpttSpec, GroupOptim, ScheduleSpec, TrainSpec

_B, _G, _IMG = 2, 8, 224


class _StubAdeTask:
    """Minimal Task surface the loop needs (the run-level Task wrapper adds
    build_model/build_loaders/evaluate; here we supply a synthetic batch stream)."""

    def batch_images(self, batch, device):
        return batch[0].to(device)

    def bind(self, batch, device, *, model, head):
        return BoundAde20kTask(seg=model, masks=batch[1].to(device), canvas_grid=_G)


def _batches():
    torch.manual_seed(2)
    while True:
        m = torch.randint(0, NUM_CLASSES, (_B, _IMG, _IMG))
        m[:, :8] = IGNORE_LABEL
        yield (torch.randn(_B, 3, _IMG, _IMG), m)


def _seg():
    torch.manual_seed(0)
    return CanViTForSemanticSegmentation(backbone_name="vits16", model_config={}, num_classes=NUM_CLASSES)


def test_loop_trains_probe_and_checkpoint_roundtrips(tmp_path):
    seg = _seg()
    spec = TrainSpec.probe(
        bptt=BpttSpec(mode="none", horizon=2),
        optim={"head": GroupOptim(lr=1e-2, schedule=ScheduleSpec(kind="warmup_constant", warmup_steps=1))},
    )
    apply_requires_grad(model=seg, head=seg.head, joint=None, spec=spec)
    opt, sched = build_optimizer_and_scheduler(spec, {"head": list(seg.head.parameters())})
    selector = RandomSelector(is_foveated=False, foveated_scale=FoveatedScaleConfig(), min_viewpoint_scale=0.05)

    before = next(seg.head.parameters()).detach().clone()
    last = run_training_loop(
        task=_StubAdeTask(), model=seg, head=seg.head, optimizer=opt, scheduler=sched,
        selector=selector, spec=spec, branches=[ViewpointType.FULL], canvas_grid=_G,
        device=torch.device("cpu"), train_batches=_batches(), n_steps=4,
        task_name="ade20k", model_config={"num_classes": NUM_CLASSES}, ckpt_dir=tmp_path,
    )
    assert torch.isfinite(torch.tensor(last["total_loss"])) and last["step"] == 3
    assert not torch.allclose(before, next(seg.head.parameters()))       # head trained
    assert all(p.grad is None for p in seg.canvit.parameters())          # backbone frozen throughout

    # end-of-run checkpoint round-trips into a fresh model.
    latest = find_latest(tmp_path)
    assert latest is not None and latest.name == "step-4.pt"
    payload = load_checkpoint(latest, "cpu")
    assert payload["task"] == "ade20k" and payload["step"] == 4
    assert payload["train_spec"]["train_head"] is True and payload["train_spec"]["train_backbone"] is False
    assert payload["model_config"] == {"num_classes": NUM_CLASSES}

    fresh = _seg()
    restore_into(payload, model=fresh)
    for (n, p), (_, q) in zip(seg.head.named_parameters(), fresh.head.named_parameters()):
        assert torch.allclose(p, q), f"restored head mismatch at {n}"


def test_loop_adds_revit_aux_loss_when_backbone_trains(tmp_path):
    """A reViT backbone (recurrence + expert bank) that trains gets its auxiliary losses
    from the loop (once per step) and logs them; the router learns."""
    from canvit.core import CanViTConfig
    from canvit.core.backbone import RecurrenceConfig
    from canvit.core.model.base.config import serialize_canvit_config

    rec = RecurrenceConfig(enabled=True, n_experts=4)
    torch.manual_seed(0)
    seg = CanViTForSemanticSegmentation(
        backbone_name="vits16", model_config=serialize_canvit_config(CanViTConfig(recurrence=rec)),
        num_classes=NUM_CLASSES)
    go = GroupOptim(lr=1e-3, schedule=ScheduleSpec(kind="warmup_constant", warmup_steps=1))
    spec = TrainSpec.finetune(bptt=BpttSpec(mode="full", horizon=2), optim={"backbone": go, "head": go})
    apply_requires_grad(model=seg, head=seg.head, joint=None, spec=spec)
    opt, sched = build_optimizer_and_scheduler(
        spec, {"backbone": list(seg.canvit.parameters()), "head": list(seg.head.parameters())})
    selector = RandomSelector(is_foveated=False, foveated_scale=FoveatedScaleConfig(), min_viewpoint_scale=0.05)
    router = seg.canvit.backbone.router
    assert router is not None
    before = router.fc2.weight.detach().clone()
    logged: list[dict] = []
    run_training_loop(
        task=_StubAdeTask(), model=seg, head=seg.head, optimizer=opt, scheduler=sched,
        selector=selector, spec=spec, branches=[ViewpointType.FULL], canvas_grid=_G,
        device=torch.device("cpu"), train_batches=_batches(), n_steps=2,
        task_name="ade20k", model_config={}, ckpt_dir=tmp_path, log_every=1,
        on_log=lambda step, m: logged.append(dict(m)),
    )
    assert not torch.equal(before, router.fc2.weight)
    for k in ("recurrence/aux_bal", "recurrence/aux_z", "recurrence/aux_div",
              "recurrence/gate_max", "recurrence/w_cos_max"):
        assert k in logged[-1], k


def test_best_checkpoint_tracks_max_and_leaves_latest_alone(tmp_path):
    """`best.pt` follows the maximum of the task's best_metric, and must NOT steal the
    `latest.pt` pointer — otherwise a resumed array task restarts from the best step."""
    seg = _seg()
    spec = TrainSpec.probe(
        bptt=BpttSpec(mode="none", horizon=2),
        optim={"head": GroupOptim(lr=1e-2, schedule=ScheduleSpec(kind="warmup_constant", warmup_steps=1))},
    )
    apply_requires_grad(model=seg, head=seg.head, joint=None, spec=spec)
    opt, sched = build_optimizer_and_scheduler(spec, {"head": list(seg.head.parameters())})
    selector = RandomSelector(is_foveated=False, foveated_scale=FoveatedScaleConfig(), min_viewpoint_scale=0.05)

    # mIoU rises then falls: the best is at step 2, which is NOT the last eval.
    scores = {0: 0.10, 1: 0.30, 2: 0.50, 3: 0.20}
    seen: list[tuple[int, float]] = []
    run_training_loop(
        task=_StubAdeTask(), model=seg, head=seg.head, optimizer=opt, scheduler=sched,
        selector=selector, spec=spec, branches=[ViewpointType.FULL], canvas_grid=_G,
        device=torch.device("cpu"), train_batches=_batches(), n_steps=4,
        task_name="ade20k", model_config={"num_classes": NUM_CLASSES}, ckpt_dir=tmp_path,
        eval_every=1, best_metric="miou_final",
        on_eval=lambda s: {"miou_final": scores[s]},
        on_best=lambda s, n, v: seen.append((s, v)),
    )

    best = load_checkpoint(tmp_path / "best.pt", "cpu")
    assert best["step"] == 2, "best.pt should hold the step-2 peak, not the last eval"
    # the running max is monotone and ends at the peak, not at the final (worse) score
    assert [v for _, v in seen] == [0.10, 0.30, 0.50, 0.50]
    # resume still resolves to the NEWEST checkpoint, not the best one
    assert find_latest(tmp_path).name == "step-4.pt"


def test_no_best_metric_writes_no_best_checkpoint(tmp_path):
    seg = _seg()
    spec = TrainSpec.probe(
        bptt=BpttSpec(mode="none", horizon=2),
        optim={"head": GroupOptim(lr=1e-2, schedule=ScheduleSpec(kind="warmup_constant", warmup_steps=1))},
    )
    apply_requires_grad(model=seg, head=seg.head, joint=None, spec=spec)
    opt, sched = build_optimizer_and_scheduler(spec, {"head": list(seg.head.parameters())})
    selector = RandomSelector(is_foveated=False, foveated_scale=FoveatedScaleConfig(), min_viewpoint_scale=0.05)
    run_training_loop(
        task=_StubAdeTask(), model=seg, head=seg.head, optimizer=opt, scheduler=sched,
        selector=selector, spec=spec, branches=[ViewpointType.FULL], canvas_grid=_G,
        device=torch.device("cpu"), train_batches=_batches(), n_steps=2,
        task_name="ade20k", model_config={"num_classes": NUM_CLASSES}, ckpt_dir=tmp_path,
        eval_every=1, on_eval=lambda s: {"miou_final": 0.5},
    )
    assert not (tmp_path / "best.pt").exists()
