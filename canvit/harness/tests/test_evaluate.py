"""``harness.evaluate``: the standalone entry point.

The load-bearing claim is proven on the cluster, not here — standalone reproduces
training-time validation BIT-IDENTICALLY on all three tasks (eval-merge doc §5, Stage 3,
`stage0_baseline/gate3b_*.json`). What unit tests can pin is the surface: that a missing
policy is refused rather than guessed, that every flag actually reaches its config, and that
the record says which protocol produced the number.
"""

from dataclasses import dataclass
from pathlib import Path

import pytest
import tyro

from canvit.harness.evaluate import Command, EvalOpts, _resolved_protocol, evaluate


@dataclass
class _Dummy:
    """Stands in for a task: the policy assertion fires before anything is built."""

    def build_model(self, device, prior_model_config=None):  # pragma: no cover
        raise AssertionError("must not be reached")


def test_auto_is_refused_and_the_message_names_the_hazard():
    """`auto` is the TRAINING-time table, kept for exp22-exp36 comparability. A standalone
    measurement has no history to protect, and the wrong guess here is SILENT — the metric
    falls as glimpses accumulate instead of raising."""
    from canvit.ade20k.config import Ade20kConfig

    cfg = Ade20kConfig()
    assert cfg.eval_policy == "auto", "the task default must stay 'auto' for training"
    with pytest.raises(AssertionError) as e:
        evaluate(_Dummy(), cfg, EvalOpts(ckpt=Path("nope.pt")), task_name="ade20k")
    msg = str(e.value)
    assert "--cfg.eval-policy is required" in msg
    assert "OUT OF DISTRIBUTION" in msg and "FALLS" in msg


@pytest.mark.parametrize("sub,extra", [
    ("ade20k", ["--cfg.resize-mode", "squish"]),
    ("in1k", ["--cfg.mode", "finetune"]),
    ("distill", []),
])
def test_every_flag_reaches_its_config(sub, extra):
    """tyro over each task's OWN dataclass, the same idiom as harness.run. If a subcommand
    stopped exposing `--cfg.*`, the flags would be silently unavailable rather than error."""
    cmd = tyro.cli(Command, args=[
        sub, "--opts.ckpt", "/tmp/x.pt", "--cfg.eval-policy", "fixation_grid", *extra,
    ])
    assert cmd.opts.ckpt == Path("/tmp/x.pt")
    assert cmd.cfg.eval_policy == "fixation_grid"
    task, name = cmd.build()
    assert name == sub and task.cfg is cmd.cfg


def test_override_scale_is_reachable_on_all_three():
    """F4: it used to be ade20k-only, and in1k is the task whose historical default is the
    off-scale one."""
    for sub in ("ade20k", "in1k", "distill"):
        cmd = tyro.cli(Command, args=[
            sub, "--opts.ckpt", "/tmp/x.pt", "--cfg.eval-policy", "full",
            "--cfg.eval-override-scale", "2.0",
        ])
        assert cmd.cfg.eval_override_scale == 2.0, sub


def test_the_record_says_which_protocol_produced_the_number():
    """Provenance, not restriction, is what protects comparability: any executable
    combination is allowed, so the artifact must name the one that ran."""
    from canvit.ade20k.config import Ade20kConfig

    cfg = Ade20kConfig(eval_policy="full", eval_override_scale=2.0)
    cfg.foveated_scale.fixed_scale = 2.0
    p = _resolved_protocol(cfg, "ade20k", is_foveated=True)
    assert p["eval_policy"] == "full" and p["eval_policy_requested"] == "full"
    assert p["override_scale"] == 2.0 and p["is_foveated"] is True
    assert p["foveated_scale"]["fixed_scale"] == 2.0
    assert p["n_timesteps"] == cfg.n_timesteps and p["scene_size"] == cfg.scene_size


def test_auto_is_recorded_as_what_it_resolved_to():
    """A record saying 'auto' would be useless a year later; it must say the trajectory."""
    from canvit.in1k.config import In1kConfig

    p = _resolved_protocol(In1kConfig(), "in1k", is_foveated=True)
    assert p["eval_policy"] == "coarse_to_fine"      # HISTORICAL_DEFAULTS["in1k"], foveated
    assert p["eval_policy_requested"] == "auto"


# --- reading provenance off the checkpoint ------------------------------------
# Deferred here from Stage 2, where the readers would have had no caller. Their job is NOT
# to pin the eval scale -- which trajectory to measure is the user's choice and this code
# does not make it -- but to stop the user having to retype what the checkpoint records, so
# the off-scale warning compares against the real training scale instead of a default of 1.0.

def _payload(*, patcher="foveated", scale=2.0, teacher="dinov3_vitl16"):
    md = {"task": "ade20k", "teacher_name": teacher}
    if scale is not None:
        md["pretrain_view_scale"] = scale          # ade20k's legacy BARE FLOAT form
    return {"model_config": {"canvit": {"patcher_name": patcher}}, "metadata": md}


def test_the_checkpoints_scale_is_adopted_when_the_user_left_the_default():
    from canvit.ade20k.config import Ade20kConfig
    from canvit.harness.evaluate import adopt_checkpoint_provenance

    cfg = Ade20kConfig()
    assert cfg.foveated_scale.fixed_scale == 1.0, "the dataclass default we are detecting"
    adopted = adopt_checkpoint_provenance(cfg, _payload(), source="x.pt")
    assert cfg.foveated_scale.fixed_scale == 2.0
    assert any("foveated_scale" in a for a in adopted)


def test_an_explicit_scale_always_wins():
    """Anything typed on the command line must survive. Silently replacing it would be the
    same class of bug as silently ignoring it."""
    from canvit.ade20k.config import Ade20kConfig
    from canvit.harness.evaluate import adopt_checkpoint_provenance

    cfg = Ade20kConfig()
    cfg.foveated_scale.fixed_scale = 0.75
    adopted = adopt_checkpoint_provenance(cfg, _payload(scale=2.0), source="x.pt")
    assert cfg.foveated_scale.fixed_scale == 0.75
    assert not any("foveated_scale" in a for a in adopted)


def test_adoption_never_pins_the_eval_scale():
    """The distinction the owner asked for: make the default honest, do not choose the
    protocol. Pinning stays explicit via --cfg.eval-override-scale."""
    from canvit.ade20k.config import Ade20kConfig
    from canvit.harness.evaluate import adopt_checkpoint_provenance

    cfg = Ade20kConfig()
    adopt_checkpoint_provenance(cfg, _payload(), source="x.pt")
    assert cfg.eval_override_scale is None


def test_a_uniform_checkpoint_adopts_no_scale():
    """A uniform model has no view scale; its OOD axis is the glimpse crop in pixels."""
    from canvit.ade20k.config import Ade20kConfig
    from canvit.harness.evaluate import adopt_checkpoint_provenance

    cfg = Ade20kConfig()
    adopted = adopt_checkpoint_provenance(cfg, _payload(patcher="uniform"), source="x.pt")
    assert cfg.foveated_scale.fixed_scale == 1.0
    assert not any("foveated_scale" in a for a in adopted)


def test_distill_adopts_the_teacher_that_supervised_it():
    """teacher_name picks both the teacher and the IN1k probe (PROBE_REGISTRY). Retyping it
    is how a vitl16 run gets evaluated against a vitb16 probe."""
    from canvit.distill.config import Config
    from canvit.harness.evaluate import adopt_checkpoint_provenance

    cfg = Config()
    assert cfg.teacher_name == "dinov3_vitb16"
    adopted = adopt_checkpoint_provenance(cfg, _payload(teacher="dinov3_vitl16"), source="x.pt")
    assert cfg.teacher_name == "dinov3_vitl16"
    assert "teacher_name=dinov3_vitl16" in adopted


# --- F8: the series comes back, and keeps its wandb namespace -----------------

def test_distill_keeps_its_val_namespace():
    """`validate` used to LOG the per-timestep series and return one float; it now returns
    them and the caller logs. The prefix has to stay `val/` or every exp22-exp32 dashboard
    breaks — while ade20k/in1k keep the harness default they have always used."""
    from canvit.ade20k.task import Ade20kRunTask
    from canvit.distill.task import DistillRunTask
    from canvit.in1k.task import In1kRunTask

    assert DistillRunTask.metrics_prefix == "val"
    for t in (Ade20kRunTask, In1kRunTask):
        assert getattr(t, "metrics_prefix", "eval") == "eval", t.__name__


def test_validate_returns_a_mapping_not_a_scalar():
    """The signature F8 is about: a standalone distill evaluation could only ever see one
    number while this returned a float."""
    import inspect

    from canvit.distill.validate import validate

    assert inspect.signature(validate).return_annotation == dict[str, float]


# --- the DINOv3 teacher baseline ---------------------------------------------
# Ported from canvit_eval's `ade20k-seg-dinov3`. Verified bit-identical against it over the
# full 2000-image val set with a shared synthetic probe (0.0006143441136078409 both sides) --
# a random probe is the right instrument for an equivalence test, since it proves the two
# implementations compute the same function without needing a good one.
#
# Worth recording: NO probe in this stack can actually read DINOv3 features. Both cached
# ADE20K probes are 1024-d `canvas_hidden` probes and DINOv3-B/16 patches are 768-d, so a
# real baseline number needs a probe nobody here has trained yet.

def test_dinov3_baseline_requires_probe_and_resolution():
    """`eval_resolution` has no default on purpose: the probe was trained at one resolution
    and running the teacher at another degrades mIoU silently."""
    cmd = tyro.cli(Command, args=["ade20k-dinov3", "--opts.probe-repo", "", 
                                  "--opts.eval-resolution", "0"])
    with pytest.raises(AssertionError, match="both required"):
        cmd.run()


def test_dinov3_baseline_flags_reach_their_config():
    cmd = tyro.cli(Command, args=[
        "ade20k-dinov3", "--opts.probe-repo", "some/probe", "--opts.eval-resolution", "512",
        "--cfg.resize-mode", "squish", "--opts.teacher-repo", "facebook/dinov3-vitl16-pretrain-lvd1689m",
    ])
    assert cmd.opts.probe_repo == "some/probe" and cmd.opts.eval_resolution == 512
    assert cmd.cfg.resize_mode == "squish"
    assert "vitl16" in cmd.opts.teacher_repo


def test_dinov3_baseline_shares_the_canvit_reduction():
    """Same val loader and same upsample-then-argmax as the CanViT path, so the baseline and
    the model it bounds are measured identically rather than merely similarly."""
    import inspect

    from canvit.ade20k import dinov3_baseline

    src = inspect.getsource(dinov3_baseline)
    assert "make_ade20k_val_loader" in src, "must not build its own val set"
    assert "eval_probe_on_batch" in src, "must not reimplement the mIoU reduction"


# --- view-scale resolution: the cases canvit_eval/tests/test_view_scale.py covered ---------
# Ported by CASE, not verbatim: its `resolve_scale_from_metadata` returned a scale to PIN,
# whereas this repo adopts the training scale into the config and leaves pinning explicit
# (owner, 2026-09-02). The situations it enumerated still all have to be right.

def test_square_patcher_scale_is_adopted_too():
    """`square` is scale-sensitive for the same reason as `foveated` (fix_size = scale * H),
    and it is the patcher this repo is likeliest to forget."""
    from canvit.ade20k.config import Ade20kConfig
    from canvit.harness.evaluate import adopt_checkpoint_provenance

    cfg = Ade20kConfig()
    adopt_checkpoint_provenance(cfg, _payload(patcher="square", scale=1.41), source="x.pt")
    assert cfg.foveated_scale.fixed_scale == pytest.approx(1.41)


@pytest.mark.parametrize("mode", ["per_rollout", "per_glimpse"])
def test_multiscale_models_adopt_their_mode_and_are_not_treated_as_fixed(mode):
    """A model trained across a RANGE of scales is scale-robust; the trajectory's own scales
    are in distribution for it. Adopting `mode` is what keeps the off-scale warning quiet."""
    from canvit.ade20k.config import Ade20kConfig
    from canvit.harness.evaluate import adopt_checkpoint_provenance

    cfg = Ade20kConfig()
    payload = {"model_config": {"canvit": {"patcher_name": "foveated"}},
               "metadata": {"pretrain_view_scale": {
                   "patcher_name": "foveated", "mode": mode, "fixed_scale": None,
                   "min_scale": 0.5, "max_scale": 1.0}}}
    adopt_checkpoint_provenance(cfg, payload, source="x.pt")
    assert cfg.foveated_scale.mode == mode
    assert cfg.foveated_scale.fixed_scale == 1.0, "no fixed_scale recorded -> leave the default"


def test_a_checkpoint_with_no_metadata_at_all_is_a_noop():
    """Every pre-metadata checkpoint. Unknown must stay unknown, never become 1.0."""
    from canvit.ade20k.config import Ade20kConfig
    from canvit.harness.evaluate import adopt_checkpoint_provenance

    cfg = Ade20kConfig()
    assert adopt_checkpoint_provenance(cfg, {}, source="x.pt") == []
    assert cfg.foveated_scale == type(cfg.foveated_scale)()


def test_mode_fixed_without_a_value_adopts_nothing():
    """`_as_view_scale_dict` keeps the dict (mode is set) but there is no scale to take."""
    from canvit.ade20k.config import Ade20kConfig
    from canvit.harness.evaluate import adopt_checkpoint_provenance

    cfg = Ade20kConfig()
    payload = {"model_config": {"canvit": {"patcher_name": "foveated"}},
               "metadata": {"pretrain_view_scale": {
                   "patcher_name": "foveated", "mode": "fixed", "fixed_scale": None}}}
    adopt_checkpoint_provenance(cfg, payload, source="x.pt")
    assert cfg.foveated_scale.fixed_scale == 1.0


def test_the_cli_hands_the_run_its_full_config(monkeypatch):
    """`harness.run` records what it was started with: the exact argv and every resolved field."""
    import canvit.harness.cli as cli

    seen = {}
    monkeypatch.setattr(cli, "run", lambda *, task, spec, settings: seen.setdefault("s", settings))
    argv = ["search", "--cfg.run-group", "g", "--cfg.model-repo", "x.pt", "--cfg.retina", "r0",
            "--rl.width", "16"]
    cli.main(argv)
    rc = seen["s"].run_config
    assert rc["argv"] == argv
    assert rc["resolved"]["cfg"]["retina"] == "r0" and rc["resolved"]["rl"]["width"] == 16
    cmd, drift = cli.command_from_run_config(rc)
    assert drift == [] and cmd.cfg.model_repo == "x.pt"
