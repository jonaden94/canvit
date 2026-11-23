"""Where did ``canvit.core`` come from?

The 2026-09-03 core merge moved ``canvit_pytorch`` into ``canvit/core/`` but left the old
``CanViT-PyTorch`` clone on disk on purpose — 116 launchers ``git archive`` a core commit out
of it (``canvit_claude/canvit/unification/21-core-merge.md`` §4). Both venvs still carry a stale
``_editable_impl_canvit_pytorch.pth`` pointing at that clone, so the old top-level
``canvit_pytorch`` package remains importable.

That is harmless only as long as nothing imports it. Both copies are functionally identical
today, so every other test in this suite passes whichever one wins — asking where the module
came from is the only way to catch a shadow. Hence these two tests rather than a checklist
item (§5, Trap A).
"""

import re
from pathlib import Path

import canvit
import canvit.core

_REPO_ROOT = Path(__file__).resolve().parents[3]
# canvit_pytorch_rl is the RL repo's package, not core's; it is referenced in prose and must
# not be mistaken for a live import of the retired top-level name.
_OLD_NAME = re.compile(r"canvit_pytorch(?!_rl)")


def test_core_resolves_inside_this_repo() -> None:
    """``canvit.core`` must come from this working tree, not the retired clone."""
    core_file = Path(canvit.core.__file__).resolve()
    assert core_file.is_relative_to(_REPO_ROOT), (
        f"canvit.core resolved to {core_file}, outside {_REPO_ROOT}. A stale editable install "
        f"or a PYTHONPATH entry is shadowing this repo's copy."
    )


def test_core_and_canvit_are_the_same_tree() -> None:
    """The two halves of the merged package must not come from different checkouts."""
    canvit_dir = Path(canvit.__file__).resolve().parent
    core_dir = Path(canvit.core.__file__).resolve().parent
    assert core_dir.parent == canvit_dir, (
        f"canvit is at {canvit_dir} but canvit.core is at {core_dir}; expected the latter to be "
        f"a direct subpackage. Two checkouts are being mixed."
    )


# Every tree in the repo that holds LIVE code. Deliberately excludes
# slurm/runs + slurm/archive, whose launchers pin pre-merge commits on purpose.
_LIVE_TREES = ("canvit", "bench", "demos", "scripts")


def test_no_source_file_imports_the_retired_name() -> None:
    """No live tree may import ``canvit_pytorch``; that name is the shadow."""
    offenders = []
    paths = sorted(q for tree in _LIVE_TREES for q in (_REPO_ROOT / tree).rglob("*.py"))
    for path in paths:
        if path == Path(__file__).resolve():
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            stripped = line.lstrip()
            if not (stripped.startswith("import ") or stripped.startswith("from ")):
                continue
            if _OLD_NAME.search(line):
                offenders.append(f"{path.relative_to(_REPO_ROOT)}:{lineno}: {stripped}")
    assert not offenders, "these import the retired top-level name instead of canvit.core:\n" + "\n".join(offenders)
