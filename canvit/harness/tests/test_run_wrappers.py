"""CPU unit tests for the run-level Task wrappers + the ``harness.run`` CLI glue.

Covers the pure-config surface (caps / default_spec / branches / feature groups /
RunTask protocol conformance), the trainable-param-group routing on tiny CPU models
(no HF download), the IN1k head=norm+head wrinkle, and the ``harness.cli`` command
dataclasses (preset matrix, nested-config parsing, config-derived RunSettings). The
model-loading + real-data training path is covered by the GPU integration script
(``canvit_claude/canvit/unification/harness_run_integration.py``).
"""

from pathlib import Path

import pytest
import tyro

from canvit import CanViTForPretraining, CanViTForPretrainingConfig
from canvit.ade20k.config import Ade20kConfig
from canvit.ade20k.data import NUM_CLASSES as ADE_CLASSES
from canvit.ade20k.task import Ade20kRunTask
from canvit.core import (
    CanViTForImageClassification,
    CanViTForSemanticSegmentation,
    create_backbone,
)
from canvit.distill.config import Config
from canvit.distill.task import DistillRunTask
from canvit.harness.cli import (
    Ade20kCmd,
    Command,
    DistillCmd,
    HarnessOpts,
    In1kCmd,
    resolve_spec,
)
from canvit.harness.loop import apply_requires_grad
from canvit.harness.run import RunTask
from canvit.harness.spec import TrainSpec
from canvit.in1k.config import In1kConfig
from canvit.in1k.task import In1kRunTask

_G, _D, _C = 8, 384, 10


def _distill_cfg():
    return Config(webdataset_dir="/nonexistent", canvas_patch_grid_size=_G)


def _tiny_distill():
    return CanViTForPretraining(
        backbone=create_backbone("vits16"), cfg=CanViTForPretrainingConfig(teacher_dim=_D),
        glimpse_size_px=128, backbone_name="vits16", canvas_patch_grid_sizes=[_G],
    )


def _tiny_seg():
    return CanViTForSemanticSegmentation(backbone_name="vits16", model_config={}, num_classes=ADE_CLASSES)


def _tiny_clf():
    return CanViTForImageClassification(backbone_name="vits16", model_config={}, n_classes=_C, glimpse_grid_size=_G)


def _wrappers():
    return [
        Ade20kRunTask(Ade20kConfig(tracker="none")),
        In1kRunTask(In1kConfig(tracker="none")),
        DistillRunTask(_distill_cfg()),
    ]


# --- pure-config surface ---------------------------------------------------
def test_wrappers_satisfy_runtask_protocol():
    for t in _wrappers():
        assert isinstance(t, RunTask), t.name


def test_default_specs_validate():
    for t in _wrappers():
        spec = t.default_spec()
        spec.validate(t.caps())  # raises on incoherent spec
        # every trainable module has an optimizer group
        for m in spec.trainable_modules():
            assert m in spec.optim, (t.name, m)


def test_caps_and_feature_groups():
    a, i, d = _wrappers()
    assert a.caps().has_head and i.caps().has_head
    assert not d.caps().has_head  # distill heads live in the forward
    # ade20k is the only probe-aware (spatial-entropy) feature set
    assert "ent" in a.policy_feature_groups() or "ent_delta" in a.policy_feature_groups()
    assert "ent" not in i.policy_feature_groups()
    assert "ent" not in d.policy_feature_groups()


def test_branches_default():
    a, i, d = _wrappers()
    assert len(a.branches()) == 1 and len(i.branches()) == 1
    assert len(d.branches()) == 2  # 1 full + 1 random by default


# --- trainable param-group routing (tiny CPU models) ----------------------
def test_ade20k_param_groups_probe_vs_finetune():
    t = Ade20kRunTask(Ade20kConfig(tracker="none"))
    seg = _tiny_seg()
    probe = t.default_spec()  # frozen backbone, train head
    apply_requires_grad(model=seg, head=seg.head, joint=None, spec=probe)
    g = t.trainable_param_groups(model=seg, head=seg.head, joint=None, spec=probe)
    assert set(g) == {"head"}
    assert all(not p.requires_grad for p in seg.canvit.parameters())
    assert all(p.requires_grad for p in seg.head.parameters())

    ft = TrainSpec.finetune(optim=probe.optim | {"backbone": probe.optim["head"]})
    apply_requires_grad(model=seg, head=seg.head, joint=None, spec=ft)
    g = t.trainable_param_groups(model=seg, head=seg.head, joint=None, spec=ft)
    assert set(g) == {"backbone", "head"}
    assert all(p.requires_grad for p in seg.canvit.parameters())


def test_in1k_head_group_is_norm_plus_head():
    t = In1kRunTask(In1kConfig(tracker="none"))
    clf = _tiny_clf()
    spec = t.default_spec()
    g = t.trainable_param_groups(model=clf, head=clf.head, joint=None, spec=spec)
    head_ids = {id(p) for p in g["head"]}
    assert head_ids == {id(p) for p in list(clf.norm.parameters()) + list(clf.head.parameters())}
    assert head_ids  # non-empty


def test_distill_param_group_is_whole_model():
    t = DistillRunTask(_distill_cfg())
    model = _tiny_distill()
    spec = t.default_spec()
    g = t.trainable_param_groups(model=model, head=None, joint=None, spec=spec)
    assert set(g) == {"backbone"}
    assert len(g["backbone"]) == len(list(model.parameters()))


# --- CLI glue: command dataclasses + resolve_spec preset matrix ------------
def test_cli_preset_matrix_head_aware():
    ade, _ = Ade20kCmd(cfg=Ade20kConfig(run_group="g")).build()
    in1k, _ = In1kCmd(cfg=In1kConfig(run_group="g"), opts=HarnessOpts(n_steps=10)).build()
    distill, _ = DistillCmd(cfg=Config(webdataset_dir=Path("/x"), run_group="g")).build()
    # head-bearing tasks: all presets that make sense validate
    for t in (ade, in1k):
        for preset in ("default", "probe", "finetune", "policy_only", "joint"):
            resolve_spec(t, preset, 3e-4, 1e-3).validate(t.caps())
    # headless distill: train_head is dropped, so finetune/joint still validate
    for preset in ("default", "finetune", "policy_only", "joint"):
        spec = resolve_spec(distill, preset, 3e-4, 1e-3)
        spec.validate(distill.caps())
        assert not spec.train_head


def test_cli_parses_nested_config_trees():
    """The CLI must reach the nested model/foveated-scale trees — the fovi configs are
    unreproducible without them (the hand-rolled argparse could not express these)."""
    cmd = tyro.cli(Command, args=[
        "distill", "--cfg.model.patcher-name", "foveated",
        "--cfg.foveated-scale.mode", "per_rollout", "--cfg.foveated-scale.min-scale", "0.25",
        "--cfg.patch-stride", "8", "--cfg.run-group", "fovi", "--cfg.steps-per-job", "512",
    ])
    assert cmd.cfg.model.patcher_name == "foveated"
    assert cmd.cfg.foveated_scale.mode == "per_rollout"
    assert cmd.cfg.foveated_scale.min_scale == 0.25
    assert cmd.cfg.patch_stride == 8
    # RunSettings is DERIVED from the task config: the job length is steps_per_job,
    # not the harness default (the n_steps footgun).
    _, settings = cmd.build()
    assert settings.n_steps == 512
    assert settings.eval_every == cmd.cfg.val_every
    assert settings.run_dir == cmd.cfg.logs_dir / "fovi" / settings.run_name


def test_cli_task_config_drives_settings():
    """Every RunSettings knob that has a task-config counterpart comes FROM the config,
    so there is no second place to set the same thing."""
    cfg = Config(webdataset_dir=Path("/x"), compile=False, amp=False, grad_clip=0.5,
                 log_every=7, val_every=13, seed=5, steps_per_job=64, tracker="none",
                 run_group="g")
    _, s = DistillCmd(cfg=cfg).build()
    assert (s.compile, s.amp, s.grad_clip, s.log_every, s.eval_every, s.seed, s.n_steps) == \
        (False, False, 0.5, 7, 13, 5, 64)


def test_ade20k_refuses_ddp_and_probes_refuse_wrapper_compile():
    """ade20k is single-GPU by construction: `make_ade20k_loaders` takes no world_size/rank,
    so multi-GPU would draw overlapping samples instead of disjoint shards. Refused twice —
    via caps (before the model is built) and at the loader itself.

    Wrapper-level torch.compile is likewise refused for both probes: they step
    .canvit/.head directly, so compiling the wrapper's forward changes nothing; only
    distill calls `model(image=…)`."""
    ade = Ade20kRunTask(Ade20kConfig(tracker="none"))
    assert not ade.caps().supports_ddp
    with pytest.raises(RuntimeError, match="does not support DDP"):
        ade.build_loaders(world_size=2, rank=0)
    assert In1kRunTask(In1kConfig(tracker="none")).caps().supports_ddp  # shards by rank

    assert not ade.caps().supports_compile
    assert not In1kRunTask(In1kConfig(tracker="none")).caps().supports_compile
    assert DistillRunTask(_distill_cfg()).caps().supports_compile


def test_run_identity_is_uniform_across_tasks():
    """Every task resolves its run identity the same way: the wandb name IS cfg.run_name
    and the artifact root IS cfg.logs_dir/run_group/run_name. ade20k used to hardcode the
    name "ade20k" (so exp24's three probes were indistinguishable in the UI) and its
    checkpoints always went to a flat cfg.probe_ckpt_dir under the launch cwd, where a
    second run would overwrite the first's best.pt."""
    logs = Path("/logs")
    cmds = {
        "ade20k": Ade20kCmd(cfg=Ade20kConfig(tracker="none", run_group="exp24",
                                             run_name="probe-a", logs_dir=logs)),
        "in1k": In1kCmd(cfg=In1kConfig(tracker="none", run_group="exp25",
                                       run_name="clf-a", logs_dir=logs)),
        "distill": DistillCmd(cfg=Config(webdataset_dir=Path("/x"), tracker="none",
                                         run_group="exp26", run_name="d-a", logs_dir=logs)),
    }
    for name, cmd in cmds.items():
        _, s = cmd.build()
        assert s.run_name == cmd.cfg.run_name, name
        assert s.run_dir == logs / cmd.cfg.run_group / cmd.cfg.run_name, name
        # ckpt_dir unset => run() derives run_dir/checkpoints (per-run, never shared)
        assert s.ckpt_dir is None, name
        assert s.tracker == "none" and s.wandb_project == cmd.cfg.wandb_project, name

    # A run_group is MANDATORY, for every task. It used to be optional, and then
    # ade20k/in1k wrote to a flat ./checkpoints relative to the launch cwd (3 GB of it
    # accumulated at the repo root) while distill wrote no checkpoints whatsoever.
    for cmd in (Ade20kCmd(cfg=Ade20kConfig(tracker="none")),
                In1kCmd(cfg=In1kConfig(tracker="none")),
                DistillCmd(cfg=Config(webdataset_dir=Path("/x"), tracker="none"))):
        with pytest.raises(ValueError, match="--cfg.run-group is required"):
            cmd.build()

    # An unnamed ARM is still fine — only the group is required; the name is derived.
    _, s = Ade20kCmd(cfg=Ade20kConfig(tracker="none", run_group="g", logs_dir=logs)).build()
    assert s.run_name.startswith("ade20k_") and s.run_dir == logs / "g" / s.run_name

    # --opts overrides still win, for every task.
    opts = HarnessOpts(run_dir=Path("/elsewhere"), ckpt_dir=Path("/ckpts"), ema_alpha=0.0)
    for cmd_cls, cfg in ((Ade20kCmd, Ade20kConfig(tracker="none", run_group="g")),
                         (In1kCmd, In1kConfig(tracker="none", run_group="g")),
                         (DistillCmd, Config(webdataset_dir=Path("/x"), tracker="none",
                                             run_group="g"))):
        _, s = cmd_cls(cfg=cfg, opts=opts).build()
        assert (s.run_dir, s.ckpt_dir, s.ema_alpha) == (Path("/elsewhere"), Path("/ckpts"), 0.0)


def test_in1k_derives_n_steps_from_max_steps():
    """in1k is now step-based like ade20k: n_steps/eval_every come from cfg.max_steps/
    val_every when --opts are unset, and --opts.n-steps overrides."""
    cfg = In1kConfig(tracker="none", max_steps=1234, val_every=321, run_group="g")
    task, s = In1kCmd(cfg=cfg).build()
    assert s.n_steps == 1234 and s.eval_every == 321
    assert task.total_steps == 1234  # single job: horizon == run length
    s2 = In1kCmd(cfg=cfg, opts=HarnessOpts(n_steps=10)).build()[1]
    assert s2.n_steps == 10
    # array mode: n_steps = per-job window (steps_per_job); LR horizon = full run (max_steps)
    at, asettings = In1kCmd(cfg=In1kConfig(tracker="none", max_steps=192_000,
                                           steps_per_job=6_400, run_group="g")).build()
    assert asettings.n_steps == 6_400 and at.total_steps == 192_000 and at._steps_per_job == 6_400


def test_in1k_shard_schedule_resume_roundtrip():
    """in1k carries distill's shard-schedule resume state: job_index advances by one, the
    derived start_step is the shard window × the new job index, and a checkpoint missing
    job_index or written mid-job is refused."""
    cfg = In1kConfig(tracker="none", max_steps=192_000, steps_per_job=6_400, batch_size=64)
    task = In1kRunTask(cfg)
    assert task.resume_state() == {}  # loaders not built yet
    task._train_loader = type("L", (), {"samples_per_shard": 4096})()
    task._world_size = 1
    rs = task.resume_state()
    assert rs["job_index"] == 0 and rs["steps_per_job"] == 6_400 and rs["samples_per_shard"] == 4096

    sched = type("S", (), {"last_epoch": 6_400})()  # ckpt written at end of job 0
    assert In1kRunTask(cfg).resume_start_step({"metadata": {"resume_state": rs}}, sched) == 6_400
    with pytest.raises(RuntimeError, match="job_index"):  # no schedule state in ckpt
        In1kRunTask(cfg).resume_start_step({"metadata": {"resume_state": {}}}, sched)
    bad = type("S", (), {"last_epoch": 999})()  # mid-job / wrong step count
    with pytest.raises(RuntimeError, match="end-of-job"):
        In1kRunTask(cfg).resume_start_step({"metadata": {"resume_state": rs}}, bad)


def test_in1k_shard_schedule_invariant_mismatch_refused():
    """Refuse resume when a shard-schedule input (here world_size) changed — the slice
    offset would silently re-process or skip shards."""
    cfg = In1kConfig(tracker="none", max_steps=192_000, steps_per_job=6_400, batch_size=64)
    task = In1kRunTask(cfg)
    task._resume_saved = {"ddp_world_size": 2, "batch_size": 64, "steps_per_job": 6_400,
                          "samples_per_shard": 4096}
    loader = type("L", (), {"samples_per_shard": 4096})()
    with pytest.raises(RuntimeError, match="mismatch"):
        task._check_schedule_invariants(loader, world_size=1)


def test_comet_tracker_rejected_loudly():
    with pytest.raises(NotImplementedError, match="comet"):
        DistillCmd(cfg=Config(webdataset_dir=Path("/x"), tracker="comet",
                              run_group="g")).build()


# --- LR-schedule reproduction (task default_spec vs the standalone recipes) ---
# test_optim.py checks the schedule PRIMITIVES against the real schedulers; these
# check the wiring, i.e. that each task's default_spec actually selects and
# parameterizes the primitive its standalone entry point used.

def _lrs_from_spec(spec, group, n):
    import torch
    from torch import nn

    from canvit.harness.optim import build_optimizer_and_scheduler

    opt, sched = build_optimizer_and_scheduler(
        spec, {m: [nn.Parameter(torch.zeros(2))] for m in spec.trainable_modules()})
    idx = spec.trainable_modules().index(group)
    out = []
    for _ in range(n):
        out.append(opt.param_groups[idx]["lr"])
        sched.step()
    return out


def test_ade20k_default_spec_reproduces_standalone_lr_schedule():
    import math

    import torch
    from torch import nn

    from canvit.ade20k.data import make_optimizer_and_scheduler

    cfg = Ade20kConfig(tracker="none", max_steps=300, warmup_steps=20)
    got = _lrs_from_spec(Ade20kRunTask(cfg).default_spec(), "head", cfg.max_steps)
    ref_opt, ref_sched = make_optimizer_and_scheduler(
        [nn.Parameter(torch.zeros(2))],
        lr=cfg.peak_lr, weight_decay=cfg.weight_decay, max_steps=cfg.max_steps,
        warmup_steps=cfg.warmup_steps, warmup_lr_ratio=cfg.warmup_lr_ratio)
    want = []
    for _ in range(cfg.max_steps):
        want.append(ref_opt.param_groups[0]["lr"])
        ref_sched.step()
    for step, (g, w) in enumerate(zip(got, want, strict=True)):
        assert math.isclose(g, w, rel_tol=1e-9, abs_tol=1e-15), f"step {step}: {g} != {w}"


def test_in1k_default_spec_reproduces_standalone_lr_schedule():
    import math

    import torch
    from torch import nn

    from canvit.harness.optim.scheduler import warmup_cosine_scheduler

    cfg = In1kConfig(tracker="none", max_steps=300, warmup_steps=15)
    total = cfg.max_steps
    got = _lrs_from_spec(In1kRunTask(cfg, total_steps=total).default_spec(), "head", total)

    # in1k/train.py: warmup_steps = max(1, cfg.warmup_steps), cosine to 0 over max_steps
    warmup = max(1, cfg.warmup_steps)
    ref_opt = torch.optim.AdamW([nn.Parameter(torch.zeros(2))], lr=cfg.peak_lr)
    ref_sched = warmup_cosine_scheduler(ref_opt, warmup, total, cfg.peak_lr,
                                        start_lr=cfg.peak_lr * cfg.warmup_lr_ratio)
    want = []
    for _ in range(total):
        want.append(ref_opt.param_groups[0]["lr"])
        ref_sched.step()
    for step, (g, w) in enumerate(zip(got, want, strict=True)):
        assert math.isclose(g, w, rel_tol=1e-9, abs_tol=1e-15), f"step {step}: {g} != {w}"


def test_opts_seed_overrides_task_seed():
    """--opts.seed wins over the task config's own seed; ade20k (no seed field)
    defaults to 0 and is settable — otherwise every harness ade20k run is byte-identical."""
    # ade20k: no cfg.seed field -> default 0, overridable
    ade_cfg = Ade20kConfig(tracker="none", run_group="g")
    assert Ade20kCmd(cfg=ade_cfg).build()[1].seed == 0
    assert Ade20kCmd(cfg=ade_cfg, opts=HarnessOpts(seed=7)).build()[1].seed == 7
    # distill: cfg.seed unless overridden
    d = Config(webdataset_dir=Path("/x"), seed=3, run_group="g")
    assert DistillCmd(cfg=d).build()[1].seed == 3
    assert DistillCmd(cfg=d, opts=HarnessOpts(seed=7)).build()[1].seed == 7


def test_resume_default_is_per_task():
    """resume defaults per task: distill True (array jobs must continue across tasks),
    ade20k/in1k False (single-job probes mirror the no-resume standalone, so a re-run
    into a populated dir starts fresh instead of silently continuing). --opts.resume
    overrides either way."""
    d = Config(webdataset_dir=Path("/x"), run_group="g")
    ade_cfg = Ade20kConfig(tracker="none", run_group="g")
    assert DistillCmd(cfg=d).build()[1].resume is True
    assert Ade20kCmd(cfg=ade_cfg).build()[1].resume is False
    assert In1kCmd(cfg=In1kConfig(run_group="g"), opts=HarnessOpts(n_steps=10)).build()[1].resume is False
    # explicit --opts.resume wins in both directions
    assert Ade20kCmd(cfg=ade_cfg, opts=HarnessOpts(resume=True)).build()[1].resume is True
    assert DistillCmd(cfg=d, opts=HarnessOpts(resume=False)).build()[1].resume is False


def test_ade20k_build_model_warns_on_foveated_view_scale(monkeypatch, caplog):
    """The foveated OOD warning branch must actually be reachable. It only fires for
    full-image (foveated/square) models, so a plain uniform run — including every GPU
    smoke so far — never executes it; a bare `log.warning` there once meant a NameError
    on exactly the foveated path."""
    import canvit.ade20k.task as adetask
    from canvit.harness.config import FoveatedScaleConfig

    class _StubSeg:
        head = "HEAD"

        def to(self, device):
            return self

    monkeypatch.setattr(adetask.CanViTForSemanticSegmentation, "from_pretrained_with_new_probe",
                        classmethod(lambda cls, **kw: _StubSeg()), raising=True)
    monkeypatch.setattr(adetask, "consumes_full_image", lambda m: True)

    cfg = Ade20kConfig(tracker="none")
    cfg.foveated_scale = FoveatedScaleConfig(mode="fixed", fixed_scale=2.0)
    with caplog.at_level("WARNING"):
        model, head = Ade20kRunTask(cfg).build_model("cpu")
    assert head == "HEAD" and isinstance(model, _StubSeg)
    assert "out of distribution" in caplog.text and "fixed_scale=2.0" in caplog.text


def test_in1k_without_total_steps_has_no_decay():
    """A bare In1kRunTask (no runner) can't know the step budget — hold at peak
    rather than invent a decay horizon."""
    sched = In1kRunTask(In1kConfig(tracker="none")).default_spec().optim["head"].schedule
    assert sched.kind == "warmup_constant" and sched.total_steps is None


def test_presets_inherit_the_tasks_lr_schedule():
    """A `--preset` says WHAT trains; it must not silently reset HOW it is scheduled.

    Before this, resolve_spec filled new groups with a bare GroupOptim, i.e.
    ScheduleSpec()'s default `warmup_constant, warmup_steps=0` — so
    `ade20k --preset finetune` threw away warmup_onecycle and `in1k --preset finetune`
    its warmup_cosine, running a flat LR with no warmup and no anneal.
    """
    from canvit.harness.cli import resolve_spec

    cfg = Ade20kConfig(tracker="none")
    task = Ade20kRunTask(cfg)
    want = task.default_spec().optim["head"].schedule
    assert want.kind == "warmup_onecycle"
    for preset in ("probe", "finetune", "joint"):
        spec = resolve_spec(task, preset, cfg.peak_lr, cfg.weight_decay)
        for group in ("backbone", "head"):
            if group in spec.optim:
                got = spec.optim[group].schedule
                assert got.kind == want.kind, f"{preset}/{group} lost the schedule: {got.kind}"
                assert got.total_steps == want.total_steps

    in1k_cfg = In1kConfig(tracker="none")
    in1k_task = In1kRunTask(in1k_cfg, total_steps=in1k_cfg.max_steps)
    in1k_want = in1k_task.default_spec().optim["head"].schedule
    assert in1k_want.kind == "warmup_cosine"
    ft = resolve_spec(in1k_task, "finetune", in1k_cfg.peak_lr, in1k_cfg.weight_decay)
    assert ft.optim["backbone"].schedule.kind == "warmup_cosine"


def test_ade20k_mode_selects_probe_or_finetune_like_in1k():
    """ade20k must express frozen/finetune through `cfg.mode`, the same first-class
    knob in1k uses — not only through the generic `--preset`."""
    frozen = Ade20kRunTask(Ade20kConfig(tracker="none", mode="frozen")).default_spec()
    assert not frozen.train_backbone and frozen.train_head
    assert frozen.bptt.mode == "none" and not frozen.task_grad_to_backbone

    ft = Ade20kRunTask(Ade20kConfig(tracker="none", mode="finetune")).default_spec()
    assert ft.train_backbone and ft.train_head and ft.task_grad_to_backbone
    assert ft.bptt.mode == "full"          # loss must reach the trunk
    assert set(ft.optim) == {"backbone", "head"}
    assert ft.optim["backbone"].schedule.kind == "warmup_onecycle"


def test_distill_derives_teacher_dim_from_the_real_teacher(monkeypatch):
    """`cfg.model.teacher_dim` is a PLACEHOLDER (train/config.py:113) that
    train/loop.py overrides with `teacher.embed_dim`. The harness passed the config
    value straight back into create_model, where `cfg.model.teacher_dim = teacher_dim`
    made it a self-assignment — hardwiring every harness run to the 768 default. A
    dinov3-vitl16 teacher is 1024 and 5 exp21 launchers use it.
    """
    from canvit.distill.config import Config
    from canvit.distill.task import DistillRunTask

    task = DistillRunTask(Config(teacher_repo_id="facebook/dinov3-vitl16-pretrain-lvd1689m"))

    class _Cfg:
        hidden_size = 1024

    class _AutoConfig:
        @staticmethod
        def from_pretrained(repo):
            assert "vitl16" in repo, f"asked for the wrong teacher: {repo}"
            return _Cfg()

    import transformers
    monkeypatch.setattr(transformers, "AutoConfig", _AutoConfig, raising=True)
    assert task._teacher_embed_dim("cpu") == 1024, "harness must follow the actual teacher"
    # and the placeholder it would have used instead:
    assert Config().model.teacher_dim == 768


def test_distill_scene_size_uses_the_teacher_patch_size(monkeypatch):
    """The scene must tokenize into G x G TEACHER patches (train/loop.py:307-308) —
    the teacher produces the targets. The harness used the STUDENT's patch size:
    identical while both are /16, wrong for a mixed pair."""
    from types import SimpleNamespace

    import torch

    from canvit.distill.config import Config
    from canvit.distill.task import DistillRunTask

    task = DistillRunTask(Config(canvas_patch_grid_size=32))
    task._device = "cpu"
    task._model = SimpleNamespace(backbone=SimpleNamespace(patch_size_px=8))  # student /8
    task._teacher = SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(patch_size=16)))
    # 32 * 16 (teacher), not 32 * 8 (student)
    assert task._scene_size_px() == 512


def test_teacher_init_is_not_broken_by_compile(monkeypatch):
    """REGRESSION: `compile_teacher` rewraps teacher.model, renaming every parameter with
    an `_orig_mod.` prefix. Backbone teacher-init looks weights up BY NAME, so compiling
    BEFORE `load_student_backbone` made `init_backbone_from_teacher=True` + `compile=True`
    — the production config — crash at startup with "Teacher has fewer than 12 transformer
    layers". train/loop.py orders it load(284) -> init(291) -> compile(303); the task must
    too, hence `_load_teacher` (never compiles) vs `_teacher_for_forward` (compiles)."""
    from types import SimpleNamespace

    import torch

    from canvit.distill.config import Config
    from canvit.distill.task import DistillRunTask

    compiled: list[str] = []
    monkeypatch.setattr("canvit.distill.model.compile_teacher",
                        lambda t: compiled.append("compiled"))
    fake = SimpleNamespace(embed_dim=768, model=SimpleNamespace(config=SimpleNamespace(patch_size=16)))
    monkeypatch.setattr("canvit.distill.model.load_teacher", lambda cfg: fake)

    t = DistillRunTask(Config(webdataset_dir="/nonexistent", compile=True,
                              init_backbone_from_teacher=True))
    dev = torch.device("cpu")

    # what build_model / _scene_size_px use: must hand back an UNCOMPILED teacher
    assert t._load_teacher(dev) is fake
    assert compiled == [], "teacher must NOT be compiled on the structural path"

    # forward path: compiles, once
    assert t._teacher_for_forward(dev) is fake
    assert compiled == ["compiled"]
    t._teacher_for_forward(dev)
    assert compiled == ["compiled"], "compile must happen at most once"
