"""Evaluate a finished ADE20K policy run's checkpoint standalone, through the TRAINING-time
validation path: `task.evaluate(..., joint=joint)` with the scorer restored from the
checkpoint's `.policy.pt` sidecar. `harness.evaluate` cannot do this (it passes no joint, so
`--cfg.eval-policy policy` has no scorer to deploy).

Used for the train-vs-val check (2026-11-16): `--cfg.ade20k-root` pointing at a symlink root
whose `validation` split holds a fixed subset of TRAINING images evaluates the policy on
images it was trained on; the real root gives val. Built exactly as `harness.run` builds a
fresh run (tyro over the same Ade20kCmd, build_model from cfg, build_policy,
apply_requires_grad); the checkpoint is then restored like `--cfg.seed-ckpt` would.

    python policy_eval.py --ckpt RUN/checkpoints/step-9000.pt --out OUT.json -- \\
        ade20k <the run's own training flags> [--cfg.ade20k-root ROOT]

Gate: on the real val root, `best.pt` must reproduce the run's logged eval at its step.
"""

import argparse
import json
import logging
from pathlib import Path

import torch
import tyro


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("run_args", nargs=argparse.REMAINDER)
    a = p.parse_args()
    run_args = a.run_args[1:] if a.run_args[:1] == ["--"] else a.run_args

    from canvit.harness.cli import Command, resolve_spec
    from canvit.harness.infra.checkpoint import load_checkpoint, restore_into
    from canvit.harness.loop import apply_requires_grad

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    torch.set_float32_matmul_precision("high")  # as cli.main
    cmd = tyro.cli(Command, args=run_args)
    task, settings = cmd.build()
    spec = resolve_spec(task, cmd.preset, *cmd.lr_wd(), getattr(cmd, "spec", None))
    assert spec.train_policy, "a policy run's flags are expected (--preset policy_only)"
    device = torch.device(settings.device)
    torch.manual_seed(settings.seed)

    model, head = task.build_model(device, prior_model_config=None)  # a fresh run's build
    gen = torch.Generator(device=device).manual_seed(settings.seed)
    joint = task.build_policy(model, device=device, canvas_grid=task.canvas_grid(model), generator=gen)
    apply_requires_grad(model=model, head=head, joint=joint, spec=spec)
    sidecar = a.ckpt.with_suffix(".policy.pt")
    assert sidecar.exists(), f"no policy sidecar {sidecar}"  # restore_into would skip it silently
    payload = load_checkpoint(a.ckpt, device)
    step = restore_into(payload, model=model, joint=joint, path=a.ckpt, device=device)

    val_loader = task.build_val_loader()
    n_images = len(val_loader.dataset)
    metrics = task.evaluate(model=model, head=head, val_loader=val_loader, device=device,
                            step=step, tracker=None, run_dir=None, joint=joint)
    rec = {"ckpt": str(a.ckpt), "step": step, "ade20k_root": str(task.cfg.ade20k_root),
           "n_images": n_images, "metrics": {k: float(v) for k, v in metrics.items()},
           "run_args": run_args}
    logging.info("eval %s step %d on %s (%d images): %s", a.ckpt, step, task.cfg.ade20k_root,
                 n_images, rec["metrics"])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
