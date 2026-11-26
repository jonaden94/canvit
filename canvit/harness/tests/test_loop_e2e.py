"""End-to-end CPU smoke of the harness vertical (design §Loop): the neutral
``run_training_loop`` drives an ADE20K probe on synthetic data through the engine +
per-group optimizer, trains only the head, and the local checkpoint round-trips.

Proves loop → task.bind → run_rollout → optimizer/scheduler → checkpoint wiring on
CPU. Real-data ``build_loaders``/``evaluate`` + the ``python -m canvit_pretrain.train``
dispatch are validated against the datasets separately.
"""

import pytest
import torch

from canvit.ade20k.data import IGNORE_LABEL, NUM_CLASSES
from canvit.ade20k.task import BoundAde20kTask
from canvit.core import CanViTForSemanticSegmentation
from canvit.harness.config import FoveatedScaleConfig
from canvit.harness.infra.checkpoint import find_latest, load_checkpoint, restore_into
from canvit.harness.loop import apply_requires_grad, read_last_val, run_training_loop
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

    # mIoU rises then falls: the best is at step 2, which is NOT the last eval (the job's
    # final weights, step 4, are validated after the loop).
    scores = {0: 0.10, 1: 0.30, 2: 0.50, 3: 0.20, 4: 0.25}
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
    assert [s for s, _ in seen] == [0, 1, 2, 3, 4]
    assert [v for _, v in seen] == [0.10, 0.30, 0.50, 0.50, 0.50]
    # resume still resolves to the NEWEST checkpoint, not the best one
    assert find_latest(tmp_path).name == "step-4.pt"


def _probe_setup():
    seg = _seg()
    spec = TrainSpec.probe(
        bptt=BpttSpec(mode="none", horizon=2),
        optim={"head": GroupOptim(lr=1e-2, schedule=ScheduleSpec(kind="warmup_constant", warmup_steps=1))},
    )
    apply_requires_grad(model=seg, head=seg.head, joint=None, spec=spec)
    opt, sched = build_optimizer_and_scheduler(spec, {"head": list(seg.head.parameters())})
    selector = RandomSelector(is_foveated=False, foveated_scale=FoveatedScaleConfig(), min_viewpoint_scale=0.05)
    return dict(task=_StubAdeTask(), model=seg, head=seg.head, optimizer=opt, scheduler=sched,
                selector=selector, spec=spec, branches=[ViewpointType.FULL], canvas_grid=_G,
                device=torch.device("cpu"), task_name="ade20k", model_config={"num_classes": NUM_CLASSES})


def _resume_kw(ckpt_dir):
    """What run() passes a resumed job: the record of the previous job's final validation
    if it covers the start step, else the best stored in the checkpoint."""
    latest = load_checkpoint(find_latest(ckpt_dir), "cpu")
    rec = read_last_val(ckpt_dir)
    if rec is not None and rec["step"] != latest["step"]:
        rec = None
    best = (rec or {}).get("best") or (latest.get("extra") or {}).get("best")
    return dict(start_step=latest["step"], best_init=best,
                validated_step=rec["step"] if rec is not None else None)


def test_best_checkpoint_survives_a_resume(tmp_path):
    """A resumed array task continues the run's best so far: a later, worse validation must
    not replace best.pt (it used to: best was per job). Every job validates its final
    weights; the next job does not validate them again."""
    scores = {0: 0.10, 1: 0.60, 2: 0.30, 3: 0.40, 4: 0.20, 5: 0.25, 6: 0.70}
    evals: list[int] = []
    kw = dict(_probe_setup(), ckpt_dir=tmp_path, eval_every=1, best_metric="miou_final",
              on_eval=lambda s: (evals.append(s), {"miou_final": scores[s]})[1])
    run_training_loop(**kw, train_batches=_batches(), n_steps=3)        # job 1: 0-2 + end 3; best at 1
    assert read_last_val(tmp_path) == {"step": 3, "best": {"metric": "miou_final", "value": 0.60, "step": 1}}
    run_training_loop(**kw, train_batches=_batches(), n_steps=2, **_resume_kw(tmp_path))  # job 2: 4 + end 5
    assert load_checkpoint(tmp_path / "best.pt", "cpu")["step"] == 1, "a worse resumed eval replaced best.pt"
    run_training_loop(**kw, train_batches=_batches(), n_steps=1, **_resume_kw(tmp_path))  # job 3: end 6 beats it
    assert load_checkpoint(tmp_path / "best.pt", "cpu")["step"] == 6
    assert evals == [0, 1, 2, 3, 4, 5, 6], "every step validated exactly once across the jobs"
    # without a stored best (an old checkpoint), the loop starts over as before
    run_training_loop(**kw, train_batches=_batches(), n_steps=1, start_step=4, best_init=None)
    assert load_checkpoint(tmp_path / "best.pt", "cpu")["step"] == 5


def test_final_validation_that_did_not_complete_is_redone_at_the_next_start(tmp_path):
    """The end-of-job checkpoint is written BEFORE the final validation: if that validation
    dies (e.g. the job's time limit), the chunk is kept and the next job validates the
    same weights at its start, as before 2026-11-26."""
    evals: list[int] = []

    def flaky(s):
        if s == 2 and 2 not in evals and not flaky.armed:
            flaky.armed = True
            raise TimeoutError("killed during the final validation")
        evals.append(s)
        return {"miou_final": 0.1 * s}
    flaky.armed = False
    kw = dict(_probe_setup(), ckpt_dir=tmp_path, eval_every=1, best_metric="miou_final", on_eval=flaky)
    with pytest.raises(TimeoutError):
        run_training_loop(**kw, train_batches=_batches(), n_steps=2)    # job 1: 0, 1, end 2 dies
    assert find_latest(tmp_path).name == "step-2.pt", "the chunk's checkpoint must survive"
    assert read_last_val(tmp_path) is None
    resume = _resume_kw(tmp_path)
    assert resume["validated_step"] is None
    run_training_loop(**kw, train_batches=_batches(), n_steps=1, **resume)  # job 2: 2 at start + end 3
    assert evals == [0, 1, 2, 3]
    assert read_last_val(tmp_path)["step"] == 3


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
