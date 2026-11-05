# `claude_dev` — the engineering record Claude reads

**Audience rule.** This tree is written primarily for Claude: it is what an agent reads to
make a decision about this repo — what was tried, what was measured, what was decided and
why, and which comparisons are invalid. It is kept in git so a session can be handed the
reasoning rather than re-deriving it.

The human-facing documentation is the root [`README.md`](../README.md) and the pages it
links under [`docs/`](../docs/). If a document is meant to be read by someone *using*
CanViT, it belongs there, not here.

`CLAUDE.md` at the repo root is the operating guide — the short, always-loaded rules. This
tree is the long form those rules were derived from.

## Projects

| directory | what it is |
|---|---|
| [`unification/`](unification/) | the five-repo merge: design docs, per-phase notes, the gate/parity instruments, and the equivalence audits. **Finished** — `21-core-merge.md` closed it, `22-april-equivalence-audit.md` verified pretraining was unchanged by it. Read newer docs over older ones; they are chronological and later ones correct earlier ones in place (see `21` §10.2). |
| [`dataloading/`](dataloading/) | the original data-loading and DDP design notes (April 2026), from before the unification. `canvit/harness/infra/dist.py` and `unification/ddp_transport_probe.py` both cite them. |
| [`search/`](search/) | the visual-search task (COCO-Search18), started 2026-11-05: design notes, and under `external_info_provided_by_jonathan/` the material the owner handed over. **In progress.** |
| [`papers/`](papers/) | material handed to Claude to read. Notes (`*.md`) are tracked; the sources are not — see its `.gitignore`. |

## Conventions

* **A new project gets a new subdirectory here**, not a new top-level folder, and not a
  file added to `unification/` — that name means the five-repo merge, which is done.
* **These are dated records.** They keep the names things had when they were written
  (`canvit_pytorch`, `CanViT-train`), because the record is what makes an old commit
  legible. *Executable* code is the exception: scripts get renamed, because a script that
  cannot run preserves no history git does not already hold (`unification/22` §7).
* **Scripts here are runnable from the repo root**, e.g.
  `.venv-cu126/bin/python claude_dev/unification/capability_matrix.py`. One of them,
  `capability_matrix.py`, is pinned by a test — `canvit/harness/tests/test_capability_matrix.py`
  fails if its generated output drifts.
