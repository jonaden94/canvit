"""Generate the cross-task capability matrix FROM THE LIVE CODE.

Why this exists: a hand-written "what can each task do" doc drifts, and a stale one is
worse than none — it reads as authoritative while being wrong. Everything below is
introspected from the actual task objects, so the answer to "can task X do Y?" is
derived from the producer rather than from anyone's recollection.

What it reports, per task (distill / ade20k / in1k):
  * ``caps()``            — has_head / supports_policy / supports_ddp / supports_compile
  * ``default_spec()``    — the spec you get with ``--preset default``
  * preset table          — the spec each ``--preset`` yields, and whether the task's own
                            LR schedule survives (non-default presets historically did not)
  * spec-selecting knobs  — auto-discovered: every bool/Literal config field is flipped and
                            ``default_spec()`` re-read; fields that change the spec are the
                            ones that actually modulate training
  * head construction     — which ``from_pretrained_*`` entry ``build_model`` calls

Run:  .venv-cu126/bin/python scripts/capability_matrix.py [--check]

``--check`` regenerates and diffs against the committed docs/capability_matrix.md, exiting
non-zero on drift (used by test_capability_matrix.py) instead of rewriting it.
"""

from __future__ import annotations

import argparse
import dataclasses
import inspect
import os
import re
import sys
import typing
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

OUT = Path(__file__).resolve().parents[1] / "docs" / "capability_matrix.md"


def _tasks():
    """Construct each RunTask from its default config. No model, no GPU, no data."""
    from canvit.ade20k.config import Ade20kConfig
    from canvit.ade20k.task import Ade20kRunTask
    from canvit.distill.config import Config
    from canvit.distill.task import DistillRunTask
    from canvit.in1k.config import In1kConfig
    from canvit.in1k.task import In1kRunTask

    return [
        ("distill", Config, lambda c: DistillRunTask(c)),
        ("ade20k", Ade20kConfig, lambda c: Ade20kRunTask(c)),
        # total_steps drives in1k's schedule branch; use the config's own horizon.
        ("in1k", In1kConfig, lambda c: In1kRunTask(c, total_steps=c.max_steps)),
    ]


def _fmt_spec(spec) -> str:
    b = spec.bptt
    length = f"horizon={b.horizon}" if b.horizon is not None else f"continue_prob={b.continue_prob}"
    trains = ",".join(m for m in ("backbone", "head", "policy")
                      if getattr(spec, f"train_{m}"))
    groups = ", ".join(
        f"{g}(lr={o.lr:g}, wd={o.weight_decay:g}, sched={o.schedule.kind}"
        f"{'' if o.schedule.warmup_steps == 0 else f'/warmup={o.schedule.warmup_steps}'})"
        for g, o in sorted(spec.optim.items())
    )
    return (f"trains=[{trains or '-'}] task->bb={spec.task_grad_to_backbone} "
            f"pol->bb={spec.policy_grad_to_backbone} "
            f"bptt={b.mode}/{length}/chunk={b.chunk_size} | {groups or 'no groups'}")


def _spec_key(spec) -> tuple:
    """Identity of a spec for change-detection (optimizer numbers excluded: we care
    about which modules train under which loss, not the LR value)."""
    b = spec.bptt
    return (spec.train_backbone, spec.train_head, spec.train_policy,
            spec.task_weight, spec.policy_weight,
            spec.task_grad_to_backbone, spec.policy_grad_to_backbone,
            b.mode, b.chunk_size, b.horizon, b.continue_prob,
            tuple(sorted(spec.optim)),
            tuple(sorted((g, o.schedule.kind) for g, o in spec.optim.items())))


def _candidate_values(ftype):
    """Alternative values to try for a config field, or None if not enumerable."""
    if ftype is bool:
        return [True, False]
    origin = typing.get_origin(ftype)
    if origin is typing.Literal:
        return list(typing.get_args(ftype))
    return None


def _spec_selecting_fields(cfg_cls, make_task) -> list[str]:
    """Flip every enumerable field; report those that change default_spec()."""
    base_cfg = cfg_cls()
    try:
        base = _spec_key(make_task(base_cfg).default_spec())
    except Exception as e:  # a task that cannot spec its own default is itself a finding
        return [f"!! default_spec() raised: {type(e).__name__}: {e}"]
    hits = []
    hints = typing.get_type_hints(cfg_cls)
    for f in dataclasses.fields(cfg_cls):
        vals = _candidate_values(hints.get(f.name, f.type))
        if not vals:
            continue
        for v in vals:
            if v == getattr(base_cfg, f.name):
                continue
            try:
                alt = dataclasses.replace(base_cfg, **{f.name: v})
                if _spec_key(make_task(alt).default_spec()) != base:
                    hits.append(f"`{f.name}` (e.g. ={v!r})")
                    break
            except Exception:
                continue  # field combination invalid; not a spec selector
    return hits


def _head_construction(task) -> str:
    try:
        src = inspect.getsource(type(task).build_model)
    except (OSError, TypeError):
        return "?"
    calls = sorted(set(re.findall(r"from_pretrained\w*|build_classifier|create_model", src)))
    # Must be cfg.mode specifically: ade20k's build_model contains `fs.mode == "fixed"`
    # (foveated view-scale), which is NOT a train-mode branch.
    branching = bool(re.search(r"cfg\.mode\b", src))
    return ", ".join(calls) + (" (mode-dependent)" if branching else " (unconditional)")


def render() -> str:
    from canvit.harness.cli import resolve_spec
    from canvit.harness.spec import check_spec

    presets = ["default", "probe", "finetune", "policy_only", "joint"]
    out: list[str] = [
        "# Cross-task capability matrix",
        "",
        "**GENERATED — do not hand-edit.** Regenerate with",
        "`.venv-cu126/bin/python scripts/capability_matrix.py`;",
        "`test_capability_matrix.py` fails if this file drifts from the code.",
        "",
        "Answers \"can task X do Y?\" from the live task objects. A capability is reachable",
        "if ANY row below offers it — a task config lacking a field does not mean the",
        "capability is absent, because `--preset` is an independent entry point.",
        "",
        "## Capabilities (`TaskCaps`)",
        "",
        "| task | has_head | supports_policy | supports_ddp | supports_compile |",
        "|---|---|---|---|---|",
    ]
    built = []
    for name, cfg_cls, make in _tasks():
        cfg = cfg_cls()
        task = make(cfg)
        built.append((name, cfg_cls, make, task))
        c = task.caps()
        out.append(f"| {name} | {c.has_head} | {c.supports_policy} | "
                   f"{c.supports_ddp} | {c.supports_compile} |")

    out += ["", "## `--preset` → resulting spec", "",
            "`default` uses the task's own `default_spec()` (task-tuned LR schedule).",
            "Other presets are built generically by `harness/cli.py::resolve_spec`.",
            ""]
    for name, cfg_cls, make, task in built:
        out += [f"### {name}", ""]
        default_sched = {g: o.schedule.kind for g, o in task.default_spec().optim.items()}
        for p in presets:
            try:
                spec = resolve_spec(task, p, cfg_cls().peak_lr, cfg_cls().weight_decay)
            except Exception as e:
                out.append(f"- `--preset {p}` → **rejected**: {type(e).__name__}: {e}")
                continue
            # A preset that produces an incoherent spec is REJECTED by run() (which
            # raises on check_spec's report), so report validity here rather than
            # printing a spec that cannot actually run.
            report = check_spec(spec, task.caps(), is_dist=False)
            if not report.ok:
                out.append(f"- `--preset {p}` → **rejected by check_spec**: "
                           + "; ".join(report.errors))
                continue
            got = {g: o.schedule.kind for g, o in spec.optim.items()}
            lost = [g for g, k in got.items()
                    if g in default_sched and default_sched[g] != k]
            warn = (f"  ⚠️ LR schedule replaced with `{got[lost[0]]}` "
                    f"(default_spec uses `{default_sched[lost[0]]}`)" if lost else "")
            out.append(f"- `--preset {p}` → {_fmt_spec(spec)}{warn}")
        out.append("")

    out += ["## Config knobs that change the spec (auto-discovered)", "",
            "Every bool/Literal config field was flipped and `default_spec()` re-read;",
            "these are the fields that actually modulate what trains.", ""]
    for name, cfg_cls, make, _ in built:
        hits = _spec_selecting_fields(cfg_cls, make)
        out.append(f"- **{name}**: " + (", ".join(hits) if hits else
                                        "_none_ — spec is fixed; use `--preset` to change it"))

    out += ["", "## Head construction in `build_model`", ""]
    for name, _, _, task in built:
        out.append(f"- **{name}**: {_head_construction(task)}")
    out.append("")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="diff against the committed file instead of rewriting it")
    args = ap.parse_args()
    text = render()
    if not args.check:
        OUT.write_text(text)
        print(f"wrote {OUT}")
        return 0
    if not OUT.exists():
        print(f"MISSING: {OUT} — run without --check", file=sys.stderr)
        return 1
    if OUT.read_text() != text:
        print(f"DRIFT: {OUT} is stale — regenerate it", file=sys.stderr)
        return 1
    print(f"{OUT} is current")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
