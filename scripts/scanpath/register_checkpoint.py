"""Add (or replace) one checkpoint's entry in the list of scored checkpoints, $LOGS_DIR/scanpath_eval/checkpoints.json,
which scripts/scanpath/figures.py reads (per-checkpoint summaries, comparisons across checkpoints).

    source .envrc && .venv-cu126/bin/python scripts/scanpath/register_checkpoint.py <best.pt> <name> \
        [--label "exp39 s0"] [--group main] [--retina cmf2.2_res65_cps6] [--notes "..."]

<name> is the records' directory under scanpath_eval/ (as for eval_checkpoint.sh). Entry format (agreed with the
figures side, 2026-11-24): name, label, group, in_distribution (false = evaluated through a retina other than the
trained one), ckpt, task, retina, notes, props (free-form x values: step, seed, run, entropy_target, and cmf / res /
cps of the retina evaluated through). Everything in props comes from the checkpoint's own metadata.

> Written by Claude (AI, Claude Opus 5.5), 2026-11-24.
"""
import argparse
import json
import os
import re
from pathlib import Path

import torch


def find_key(tree, key):
    """The first value of ``key`` anywhere in a nested dict/list (None if absent)."""
    if isinstance(tree, dict):
        if key in tree:
            return tree[key]
        tree = list(tree.values())
    if isinstance(tree, list):
        for v in tree:
            found = find_key(v, key)
            if found is not None:
                return found
    return None


def retina_props(retina: str | None) -> dict:
    m = re.fullmatch(r"cmf([\d.]+)_res(\d+)_cps(\d+)", retina or "")
    return {"cmf": float(m[1]), "res": int(m[2]), "cps": int(m[3])} if m else {}


def entry_of(ckpt: Path, name: str, *, label: str | None, group: str, retina: str | None, notes: str) -> dict:
    payload = torch.load(ckpt, map_location="cpu", weights_only=False, mmap=True)
    meta = payload.get("metadata") or {}
    cfg = ((meta.get("run_config") or {}).get("resolved") or {}).get("cfg") or {}
    trained = cfg.get("retina")
    run = f"{cfg['run_group']}/{cfg['run_name']}" if "run_group" in cfg and "run_name" in cfg else None
    props = {"step": payload.get("step"), "seed": cfg.get("seed"), "run": run,
             "entropy_target": find_key(meta, "entropy_target")} | retina_props(retina or trained)
    return {"name": name, "label": label or name, "group": group,
            "in_distribution": retina is None or retina == trained,
            "ckpt": str(ckpt), "task": payload.get("task") or meta.get("task"), "retina": retina, "notes": notes,
            "props": {k: v for k, v in props.items() if v is not None}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("name")
    ap.add_argument("--label")
    ap.add_argument("--group", default="main")
    ap.add_argument("--retina", help="the retina evaluated through, if not the trained one")
    ap.add_argument("--notes", default="")
    a = ap.parse_args()
    path = Path(os.environ.get("SCANPATH_RESULTS_DIR") or Path(os.environ["LOGS_DIR"]) / "scanpath_eval") / "checkpoints.json"
    entries = json.loads(path.read_text()) if path.exists() else []
    new = entry_of(a.ckpt, a.name, label=a.label, group=a.group, retina=a.retina, notes=a.notes)
    old = [e for e in entries if e["name"] == a.name]
    entries = [e for e in entries if e["name"] != a.name] + [new]
    path.write_text(json.dumps(entries, indent=1) + "\n")
    print(("replaced " if old else "added ") + json.dumps(new))


if __name__ == "__main__":
    main()
