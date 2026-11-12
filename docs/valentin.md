# CanViT: a quick overview

## What it is

The model is the **Canvas Vision Transformer (CanViT)**. It sees an image through a sequence of glimpses (viewpoints) and accumulates what it sees in a spatial memory, the "canvas". Glimpses are either uniform patches or **foveated** ones (dense at the fixation point, coarser towards the periphery). Where to look next is either fixed (random, coarse-to-fine, …) or chosen by a **learned policy** trained with RL.

This one repo holds the model and everything that trains and evaluates it. It needs no other repo: the foveated geometry, originally from the `fovi` library, now lives in `canvit/core/foveation/`.

## Tasks

| task | what it does |
|---|---|
| `distill` | pretraining: dense distillation from DINOv3 on IN21k |
| `ade20k` | segmentation (probe or finetune) |
| `in1k` | classification (probe or finetune) |
| `search` | class-cued visual search on COCO-Search18 (new, see below) |

## How you use it

**Two command-line entry points**, one for training and one for evaluating a finished checkpoint:

```bash
python -m canvit.harness.run ade20k --preset probe                     # train
python -m canvit.harness.evaluate ade20k --opts.ckpt <best.pt> --cfg.eval-policy random   # evaluate
```

**Inference** (just running a model on images) needs no entry point: load a model in Python, e.g. `canvit.core.model_source.load_pretraining(<.pt file or HF Hub id>)`. `scripts/demos/` has two small examples.

**Presets** (the `--preset` flag) choose *which parts are trained*:

| `--preset` | trains |
|---|---|
| `probe` | only the task head; the backbone is frozen |
| `finetune` | backbone and task head |
| `policy_only` | only the policy (where to look); backbone and head frozen |
| `joint` | task (backbone + head) and policy together |
| `default` (if omitted) | the task's own usual choice |

**Everything else is a config flag**, e.g. `--cfg.peak-lr 3e-4` or `--rl.reward difference`: patcher, view scale, number of glimpses, how glimpses are chosen at eval, RL objective, reward, and so on. There is one code path per feature with switches, not copies of it, and validation during training calls the same evaluation code as the standalone entry point, so the two give the same numbers.

**Cluster jobs run a snapshot of the code, not your working copy.** A launcher in `slurm/runs/` names a commit (`TRAIN_COMMIT=<sha>`); each job extracts exactly that commit into a private directory and runs it. So after submitting you can change the *code* (edit, switch branches, pull) without affecting queued or running jobs. The other side of this: commit (and push) your changes first, and put that commit in the launcher, or the job runs the old code.

A few things are *not* in the snapshot and are read from your clone when a job starts: the environment setup (`slurm/env.sh`, `.envrc.grete`: dataset paths, caches), the Python environment (`.venv-cu126`), and `slurm/harness_train.sbatch` if a queue script calls it per run. Don't change or re-install these while your jobs are queued or running, and don't switch to a branch where they differ (usually not the case). Details: `slurm/README.md`.

## Layout

```
canvit/
  core/        the model: canvas ViT, patchers (incl. the foveated geometry), policy networks
  harness/     the two entry points + everything shared by the tasks (training loop,
               glimpse rollout, policy/RL, optimiser, checkpointing, logging)
  distill/ ade20k/ in1k/ search/   one folder per task: data, loss, metrics, figures
  checkpoint/  export to the HuggingFace format
slurm/         cluster launchers, one folder per experiment (slurm/runs/<exp>/)
claude_dev/    design notes and engineering records (mostly for Claude); search: claude_dev/search/
docs/          docs for humans
```

**Tests** live in a `tests/` folder inside each package (about 600). They check shapes and behaviour, compare against reference implementations, and include fingerprints of results that fail when a change moves the numbers. Run them with `.venv-cu126/bin/python -m pytest canvit` before using a commit for a real job.

`CLAUDE.md` holds the working rules for Claude Code, which reads it automatically; you don't need to read it. Ask Claude to summarise what in it matters for what you are doing.

## Search (where you start)

- **Setup:** a frozen foveated CanViT backbone. On top sit two small U-Nets, both told which object class to search for: a **task head** (a heatmap of where the target is, plus whether it is present at all) and a **policy** (where to look next). Head and policy are trained together in one run; the backbone stays fixed.
- **Data:** COCO-Search18 (target-present and target-absent images), following Haoze's recipe.
- **Metrics**, at every glimpse, for three ways of choosing fixations (random, greedy, learned policy):
  - `found`: the heatmap's peak lies inside the target box;
  - `hit`: a fixation so far landed on the target;
  - presence accuracy: target present/absent predicted correctly.
- **Current state:** the learned policy localises the target as well as **greedy** fixation (always look at the current heatmap's peak, never twice at the same spot), but not better: `found` after the last glimpse is about 0.87 for both, against about 0.80 for random fixations. Its scanpaths, however, are more varied than greedy's. For the documented results (W&B runs, notes), ask Jonathan.

**Missing so far:**
- No standalone evaluation for search yet: `harness.evaluate` has no `search` subcommand, so search is evaluated only during training.
- No comparison with **human scanpaths** (e.g. scanpath similarity, fixation-map metrics). COCO-Search18 has the human fixations; the other datasets don't.
