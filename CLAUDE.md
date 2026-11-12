# Coding guidelines

**Invoke the `andrej-karpathy-skills:karpathy-guidelines` skill at the start of every
session, before planning or writing code**, and follow it throughout. Short form: state
assumptions and ask when unsure; simplest code that works; surgical changes only; define a
verifiable success criterion and check it before claiming done. Never call work
"complete"/"a drop-in" without saying what the criterion was — and when porting or
replacing code, read the target end to end rather than planning from a `grep`.

# What this repo is

**CanViT** — the Canvas Vision Transformer, plus everything that trains and evaluates it.
One repo, one package.

Two entry points, and only two:

```bash
python -m canvit.harness.run      <distill|ade20k|in1k> --preset <default|probe|finetune|policy_only|joint>
python -m canvit.harness.evaluate <distill|ade20k|ade20k-dinov3|in1k> --cfg.eval-policy <policy>
```

| task | trains | data |
|---|---|---|
| `distill` | passive→active dense distillation from DINOv3 — the pretraining objective | IN21k WebDataset shards + a val image folder |
| `ade20k` | ADE20K segmentation probe / finetune | ADE20K (`$ADE20K_ROOT`) |
| `in1k` | ImageNet-1k linear probe / full finetune | IN1k WebDataset + val image folder |

On top of any of the three, the **viewpoint-selection policy** can be trained by RL, alone
against a frozen model (`policy_only`) or jointly with the task (`joint`).

`--cfg.eval-policy` is **required** for standalone eval — no `auto` guess. A foveated model
is only in-distribution at its training view scale, so pin it
(`--cfg.foveated-scale.fixed-scale`); off-scale glimpses make the metric *fall* as glimpses
accumulate.

## Package layout

One rule: **`harness/` holds the entry point and everything shared by more than one task;
each task folder holds only what is specific to that task.**

```
canvit/
├── core/        THE MODEL — canvas ViT, patchers, HF-hub classes, probes, teacher,
│                standardizers, viewpoint, rope, foveation (the geometry ported from fovi).
│                Imports nothing from the layers below.
├── harness/     entry point + shared primitives
│                six flat files: run evaluate cli loop spec config;
│                subpackages: rollout/ policy/ optim/ infra/ viz/
├── distill/     DINOv3 latent distillation: loss, model, probe, data/, viz/
├── ade20k/      ADE20K segmentation: data, metrics, rollout, viz
├── in1k/        ImageNet-1k: data, metrics, model, eval, rollout
└── checkpoint/  checkpoint I/O and `to_hf` / `probe_to_hf` — publishing to the HF layout
```

`core/` must not import from `harness/`, `distill/`, `ade20k/` or `in1k/`. One grep checks it:

```bash
grep -rn "from canvit\.\(harness\|distill\|ade20k\|in1k\)" canvit/core/
```

Also at the repo root, split by AUDIENCE:

- `README.md` + `docs/` — **for humans.** The front page GitHub renders, plus the campaign
  procedures it links.
- `claude_dev/` — **primarily for Claude:** the engineering record you read to make a
  decision. `unification/` (the finished five-repo merge — design notes, per-phase notes,
  gate instruments, `capability_matrix.py`), `dataloading/` (April 2026 data + DDP design
  notes), `papers/` (reading material; notes tracked, sources not). **A new project gets a
  new subdirectory there** — not a new top-level folder, and not a file in `unification/`,
  whose name means that one finished merge. See `claude_dev/README.md`.

Plus `slurm/` (launchers), `scripts/bench/pt/` (inference benchmark), `scripts/demos/`, `test_data/`.

# The other repos on disk

```
repos/
├── canvit/   this repo                       LIVE — self-contained since 2026-11-12
└── fovi/     foveated-vision geometry        kept for pre-port pinned runs only
```

**fovi was ported into `canvit/core/foveation/` on 2026-11-12** (fovi 27b1bde, planar
geometry, bit-identical: `claude_dev/fovi_port/`). canvit's code no longer imports fovi; new
launchers pin `TRAIN_COMMIT` only. fovi stays on disk and in the venv (`fovi-legacy`
dependency group) because launchers pinning a pre-port `TRAIN_COMMIT` + `FOVI_COMMIT` still
`import fovi` — do not remove either. **Change the foveated geometry in
`canvit/core/foveation/`, never in the fovi repo**: an edit there no longer reaches canvit.
Older notes and memories that say "fovi commit X" or point at fovi source describe the
pre-port state.

Four clones are kept **read-only, as fallback references. Do not edit them**, and prefer
this repo's equivalent:

| clone | superseded by | why it is kept |
|---|---|---|
| `CanViT-PyTorch/` | `canvit.core` | 106 launchers `git archive` a core commit out of its `.git`. **Must stay on disk permanently.** |
| `CanViT-eval/` | `canvit.harness.evaluate` | its `results/` are the historical record — but see its `ARCHIVED.md`: the `reconstruction` task's `*_cos_raw` numbers are wrong |
| `CanViT-specialize/` | `canvit.{ade20k,in1k}` | pre-unification reference for the downstream recipes |
| `CanViT-PyTorch-RL/` | `canvit.core.policy` + `canvit.harness.policy` | pre-unification reference for the RL recipes; has its own `CLAUDE.md` governing work inside it |

# Gotchas

Each of these has cost someone real time.

- **There is no `canvit_pytorch` and no `canvit_train` package.** The model is
  `canvit.core`. Code importing the old names is pre-2026-09-03 and belongs to a pinned
  snapshot, not to this tree.
- **Historical launchers under `slurm/runs/` and `slurm/archive/` say `canvit_pretrain` or
  `canvit_train` ON PURPOSE** — they pin pre-rename commits whose snapshot carries that
  package directory. Do not "modernize" them; `harness_train.sbatch` detects which name a
  snapshot holds.
- **Checkpoint paths in launchers are ABSOLUTE, and that is deliberate.** `logs/` is
  gitignored, so each artifact exists in exactly one place — mode 640 under group
  `HPC_nib00021`, every parent group-traversable — and any project member reads it there
  without copying. A path derived from the reader's own clone would find nothing. The
  2026-09-03 rename left the old `repos/CanViT-train/...` prefix dangling; the 29 launchers
  under `slurm/runs/` were repointed on 2026-09-09 (all 22 primary ones verified to reach
  `sbatch` with a stub). **The ~199 under `slurm/archive/` are still stale, on purpose** —
  they are the record of what those jobs ran; repoint one by hand if you ever re-run it.
  Nothing fails silently: a guarded launcher refuses to submit, and the rest die on a
  missing file.
- **`PYTORCH_COMMIT` is load-bearing for those old launchers but must NOT be set by new
  ones.** `TRAIN_COMMIT` now pins model and trainer together; setting `PYTORCH_COMMIT` on a
  post-merge pin has no effect. The launcher warns in both failure directions.
- **Use `.venv-cu126` for the test suite**, always. The eight `test_task_digests.py` digests
  pin *CPU* numerics against hashes recorded under that torch build; `.venv` (cu130) fails
  exactly those for that reason alone.
- **A failing digest is a FINDING, not a chore. Never re-record it to make the suite
  green.** `test_rollout_parity.py` (`9a0100a1a3de3acd`) and the eight in
  `test_task_digests.py` exist to tell you that a change moved the numbers. Re-recording the
  expected value destroys the only guard you had and the change ships silently — which is how
  the harness policy gradient sat at 0.8x the reference for weeks. Find out WHY it moved
  first; record a new value only once you can say what changed and why that is intended, and
  say so in the commit. The two kinds differ: parity is an EQUIVALENCE digest (it certifies
  agreement with the old trainer), the eight are PINNING digests ("today equals what was
  recorded"), so a pre-existing bug is pinned in with everything else and neither is a
  correctness claim.
- **Run the whole suite before pinning a commit for a real job.** For the inner loop, select
  by name — `-k foveated` (36 tests), `-k policy` (58), `-k digest` (9), or a single test with
  `path::test_name`. `test_rollout_parity.py` is one test and ~20% of the runtime, so
  `--ignore=canvit/harness/tests/test_rollout_parity.py` is the cheap partial run. Treat all
  of these as convenience filters, never as a gate: `-k` matches the test id INCLUDING its
  path, so it both over- and under-selects (`test_init_reference_mode.py` is a policy test
  that `-k policy` misses). Every expensive bug here was a cross-cutting surprise that "only
  touched" something else.
- **Every test lives in a `<package>/tests/` directory** (`core/`, `harness/`, `ade20k/`,
  `in1k/`, `distill/`, `checkpoint/`). A test file placed beside the code it tests will still
  be collected, but it breaks the convention — and moving one later means fixing its relative
  imports, INCLUDING the ones inside function bodies, which a `^from \.` grep does not see.
- **Ask for a short `--time`; never mention QOS.** [owner, standing] A short job gets
  scheduled sooner than a long one, which is why the long runs are chunked into 2 h array
  tasks. Whether it formally lands in Grete's `2h` QOS or in `normal` does not matter — do
  not use `--qos=2h` in new launchers and do not explain QOS selection in `docs/`. To buy
  margin, lower `CFG_STEPS_PER_JOB` in a NEW run group; it cannot change on a run already
  in progress (`_check_schedule_invariants` refuses to resume).
- **The DINOv3 teacher is gated.** `facebook/dinov3-vitb16-pretrain-lvd1689m` is
  `gated: manual`, so any `distill` run needs an HF account that has been granted access
  plus `hf auth login`. It bites at startup, where the teacher's width is read. The
  `canvit/*` checkpoints are public.
- **Eval gates are machine-local.** Bit-identity holds only within one GPU (≈1e-5 across
  GPU types, and MIG slice size counts). Old job logs cannot gate a refactor.
- **The wandb entity `cidas_goettingen` is shared** with other people's projects. Pin runs
  by id, assert on names, never iterate `api.projects(entity)` and act.
- **`git mv` on a directory carries untracked `__pycache__` with it**, and preserves mtimes,
  so Python keeps using stale bytecode whose paths no longer exist. Purge after any rename.
- **The pretraining model's `forward` takes `image=`; the classification and segmentation
  wrappers take `glimpse=`.** Both are keyword-only, so a mix-up is a `TypeError` — which
  only helps if someone RUNS the file. It has now silently broken two of the three files
  that touch those wrappers: `scripts/bench/pt/run.py` (passed `glimpse=` to bare CanViT;
  found 2026-09-02, the benchmark had not run for some time) and `scripts/demos/classify.py`
  (passed `image=` to the classifier; broken 2026-09-03 to 2026-09-08). **Run the file after
  touching either call.**
- **Launchers live in `slurm/`; one-off instruments live beside their record.** `slurm/runs/`
  holds the reusable, copyable launchers (all of them submit; `scripts/` contains no
  `sbatch` at all). The 8 `.sbatch` gate scripts and the parity probes under
  `claude_dev/unification/` belong to the investigation that used them, not to `slurm/`.
  `scripts/` is for what you run yourself now — an eval driver, a figure generator, the
  benchmark, the demos.

# Environments

Each repo has its **own** uv-managed venv; a venv is an editable install of its own repo, so
running it directly picks up your edits — no `PYTHONPATH` gymnastics.

```bash
.venv-cu126/bin/python -m pytest canvit          # 526 tests
```

torch uses the GPU only if its CUDA build is `<=` the node's driver. Many compute nodes here
cap at **CUDA 12.8** (A100/MIG, driver 570.x), where a `cu130` build is GPU-dead.

| venv | torch | use |
|---|---|---|
| `.venv` | cu130 | CUDA-13-capable nodes |
| `.venv-cu126` | cu126 | 12.8 nodes (A100/MIG) — **and all test runs** |

Rebuild the cu126 env with:
`UV_PROJECT_ENVIRONMENT=.venv-cu126 uv sync --no-group cuda --group cu126`

`[tool.uv.sources]` links `fovi` (only for the default-on `fovi-legacy` group, see above) as a
**relative-path editable install** (`fovi = { path = "../fovi", editable = true }`), so the
two repos are cloned as siblings. Swap to `{ git = ... }` to install without it.

# Datasets and env variables

Dataset roots, caches and the run-artifact directory come from the environment. For project
`nib00021` on Grete they are filled in already in **`.envrc.grete`**; `slurm/env.sh` sources
it directly, so submitting needs no `direnv`. For interactive work,
`source .envrc.grete` (or `cp .envrc.grete .envrc && direnv allow`).

The variables that matter: `LOGS_DIR`, `WEBDATASET_DIR`, `VAL_DIR`, `VAL_INDEX_DIR`,
`ADE20K_ROOT`, `IN1K_TRAIN_DIR`, `IN1K_VAL_DIR`, `HF_HOME`, `HF_TOKEN`, `WANDB_DIR`,
`WANDB_PROJECT`, `WANDB_ENTITY`. Full table and first-time setup: `README.md`.

`IN1K_TRAIN_DIR` / `IN1K_VAL_DIR` are the exception: `.envrc.grete` does NOT set them, and
`in1k/config.py`'s `_default_in1k_{train,val}_dir` fall back to hardcoded project paths.

# Commit pinning — why editing the repo is safe during live jobs

The venvs are editable installs, so in principle a running job would pick up a mid-run edit.
`slurm/harness_train.sbatch` neutralizes that: each run pins exact commits via
**`TRAIN_COMMIT`** and **`FOVI_COMMIT`**, and the sbatch `git archive`s them **offline** into
the per-job `$TMPDIR/canvit_src`, prepending that snapshot to `PYTHONPATH` with
`PYTHONSAFEPATH=1` so it wins over the cwd and the editable install.

- You can **edit the repo freely** while pinned jobs are in flight. Only newly submitted
  jobs that pin a new commit pick up changes.
- `git archive` reads the local `.git` only — no network, no SSH, works for private repos,
  and does not touch HEAD or the working tree.
- Launcher scripts and `.envrc.grete` are read at submit time, so editing them is always
  safe.
- Legacy spellings still accepted: `PRETRAIN_COMMIT` (88 launchers set it) and
  `PYTORCH_COMMIT` (106) — see Gotchas.
- **Workflow:** edit → commit/push → submit runs pinning the new hash.
- `FOVI_COMMIT` matters only for a `TRAIN_COMMIT` from before the fovi port (2026-11-12);
  on a later one it has no effect and the sbatch warns.

# Guardrails

- **Claude does not run `git commit` / `push` / `pull`, and does not submit or cancel SLURM
  jobs, unless the user explicitly asks.** (In practice the user asks often; just don't do
  it unprompted.)
- **Never persist a `PYTHONPATH` override** in `~/.bashrc`, `~/.profile`, a SLURM script or
  any sourced file, and never `pip install -e` one repo into another's venv in a way that
  rewrites the editable `.pth`. Both leak in-progress edits into unrelated runs.
- Stage explicit paths. No `git add -A`, `git commit -a`, `git checkout`, `git stash`.
- HuggingFace: cached downloads fine, **no uploads**.

# History

This repo is the result of merging five repos into one. The narrative lives in
`claude_dev/unification/`, not here:

| doc | what happened |
|---|---|
| `17-harness-consolidation.md` | specialize + RL folded in; one trainer; `CanViT-pretrain` → `CanViT-train` |
| `18-package-restructure.md` | `train/` + `tasks/` dissolved into `harness/` + per-task dirs |
| `20-eval-merge.md` | `CanViT-eval` folded in; one eval entry point |
| `21-core-merge.md` | `CanViT-PyTorch` folded in as `canvit/core`; repo renamed `canvit` |

`00-master-plan.md` §5 lists what was deliberately **not** ported.

# Notes

- Persistent memory for these sessions lives under
  `~/.claude/projects/<encoded-repo-path>/memory/` (`MEMORY.md` is the index).
