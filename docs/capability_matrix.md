# Cross-task capability matrix

**GENERATED — do not hand-edit.** Regenerate with
`.venv-cu126/bin/python scripts/capability_matrix.py`;
`test_capability_matrix.py` fails if this file drifts from the code.

Answers "can task X do Y?" from the live task objects. A capability is reachable
if ANY row below offers it — a task config lacking a field does not mean the
capability is absent, because `--preset` is an independent entry point.

## Capabilities (`TaskCaps`)

| task | has_head | supports_policy | supports_ddp | supports_compile |
|---|---|---|---|---|
| distill | False | True | True | True |
| ade20k | True | True | False | False |
| in1k | True | True | True | False |

## `--preset` → resulting spec

`default` uses the task's own `default_spec()` (task-tuned LR schedule).
Other presets are built generically by `harness/cli.py::resolve_spec`.

### distill

- `--preset default` → trains=[backbone] task->bb=True pol->bb=False bptt=chunked/continue_prob=0.5/chunk=2 | backbone(lr=0.0004, wd=0.0001, sched=warmup_constant/warmup=100000)
- `--preset probe` → **rejected by check_spec**: nothing is trainable (train_backbone/head/policy all False)
- `--preset finetune` → trains=[backbone] task->bb=True pol->bb=False bptt=full/horizon=10/chunk=1 | backbone(lr=0.0004, wd=0.0001, sched=warmup_constant/warmup=100000)
- `--preset policy_only` → trains=[policy] task->bb=False pol->bb=False bptt=none/horizon=10/chunk=1 | policy(lr=0.0002, wd=0.01, sched=warmup_constant)
- `--preset joint` → trains=[backbone,policy] task->bb=True pol->bb=False bptt=chunked/continue_prob=0.5/chunk=2 | backbone(lr=0.0004, wd=0.0001, sched=warmup_constant/warmup=100000), policy(lr=0.0002, wd=0.01, sched=warmup_constant)

### ade20k

- `--preset default` → trains=[head] task->bb=False pol->bb=False bptt=none/horizon=10/chunk=1 | head(lr=0.0003, wd=0.001, sched=warmup_onecycle/warmup=1500)
- `--preset probe` → trains=[head] task->bb=False pol->bb=False bptt=none/horizon=10/chunk=1 | head(lr=0.0003, wd=0.001, sched=warmup_onecycle/warmup=1500)
- `--preset finetune` → trains=[backbone,head] task->bb=True pol->bb=False bptt=full/horizon=10/chunk=1 | backbone(lr=0.0003, wd=0.001, sched=warmup_onecycle/warmup=1500), head(lr=0.0003, wd=0.001, sched=warmup_onecycle/warmup=1500)
- `--preset policy_only` → trains=[policy] task->bb=False pol->bb=False bptt=none/horizon=10/chunk=1 | policy(lr=0.0002, wd=0.01, sched=warmup_constant/warmup=5000)
- `--preset joint` → trains=[backbone,head,policy] task->bb=True pol->bb=False bptt=chunked/continue_prob=0.5/chunk=2 | backbone(lr=0.0003, wd=0.001, sched=warmup_onecycle/warmup=1500), head(lr=0.0003, wd=0.001, sched=warmup_onecycle/warmup=1500), policy(lr=0.0002, wd=0.01, sched=warmup_constant/warmup=5000)

### in1k

- `--preset default` → trains=[head] task->bb=False pol->bb=False bptt=none/horizon=10/chunk=1 | head(lr=0.0003, wd=0.001, sched=warmup_cosine/warmup=10000)
- `--preset probe` → trains=[head] task->bb=False pol->bb=False bptt=none/horizon=10/chunk=1 | head(lr=0.0003, wd=0.001, sched=warmup_cosine/warmup=10000)
- `--preset finetune` → trains=[backbone,head] task->bb=True pol->bb=False bptt=full/horizon=10/chunk=1 | backbone(lr=0.0003, wd=0.001, sched=warmup_cosine/warmup=10000), head(lr=0.0003, wd=0.001, sched=warmup_cosine/warmup=10000)
- `--preset policy_only` → trains=[policy] task->bb=False pol->bb=False bptt=none/horizon=10/chunk=1 | policy(lr=0.0002, wd=0.01, sched=warmup_constant/warmup=25000)
- `--preset joint` → trains=[backbone,head,policy] task->bb=True pol->bb=False bptt=chunked/continue_prob=0.5/chunk=2 | backbone(lr=0.0003, wd=0.001, sched=warmup_cosine/warmup=10000), head(lr=0.0003, wd=0.001, sched=warmup_cosine/warmup=10000), policy(lr=0.0002, wd=0.01, sched=warmup_constant/warmup=25000)

## Config knobs that change the spec (auto-discovered)

Every bool/Literal config field was flipped and `default_spec()` re-read;
these are the fields that actually modulate what trains.

- **distill**: _none_ — spec is fixed; use `--preset` to change it
- **ade20k**: `mode` (e.g. ='finetune')
- **in1k**: `mode` (e.g. ='finetune')

## Head construction in `build_model`

- **distill**: create_model, from_pretrained (unconditional)
- **ade20k**: from_pretrained_with_new_probe, from_pretrained_with_probe (mode-dependent)
- **in1k**: build_classifier (unconditional)
