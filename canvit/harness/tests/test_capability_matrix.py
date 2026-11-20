"""The generated capability matrix must stay in sync with the code.

`docs/capability_matrix.md` is the answer to "can task X do Y?". It is only
trustworthy while it matches the live task objects, so adding a task, a preset, a
`TaskCaps` field or a spec-selecting config knob must fail here until it is regenerated:

    .venv-cu126/bin/python scripts/capability_matrix.py
"""

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "capability_matrix.py"


def _load():
    spec = importlib.util.spec_from_file_location("capability_matrix", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.skipif(not _SCRIPT.exists(), reason="capability_matrix.py not present")
def test_capability_matrix_is_current():
    mod = _load()
    assert mod.OUT.exists(), f"{mod.OUT} missing — run capability_matrix.py"
    assert mod.OUT.read_text() == mod.render(), (
        f"{mod.OUT.name} is stale. Regenerate:\n"
        f"    .venv-cu126/bin/python scripts/capability_matrix.py"
    )
