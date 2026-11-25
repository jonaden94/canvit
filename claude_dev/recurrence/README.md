# Recurrent backbone (bViT / reViT) — design note, 2026-11-19

Jonathan asked (2026-11-15) for the two single-block-recurrence papers to be implemented
and tested on ImageNet, standalone; that is `~/caglar/repos/recurrent-vit` (package `rvit`,
not part of this repo). This note records how the same two methods were then put into the
CanViT backbone as an **opt-in** switch.

* bViT — Byra et al., arXiv:2605.10661. One block looped L times.
* reViT — Bulat et al., arXiv:2610.12448. Shared block, FFN = bank of E experts whose
  WEIGHTS are merged per step by a router that sees only the depth `s_t = t/(L-1)`;
  three parameter-only aux losses (balance, z, diversity).

## Switch and fields

`CanViTConfig.recurrence: RecurrenceConfig` (`canvit/core/backbone/recurrence.py`).
`enabled=False` (default) = the old backbone: same modules, same state_dict keys, same
numbers (`test_recurrence.py::test_off_is_the_old_model`, and every digest test unchanged).

| field | meaning |
|---|---|
| `enabled` | master switch |
| `depth` | computational depth L; `None` = registry depth (12 for vits16/vitb16) |
| `n_templates` | parameter depth; step t runs `blocks[t % n_templates]` (1 = full sharing) |
| `n_experts` | 1 = bViT; >1 = reViT (paper default 4) |
| `router_hidden` | reViT router width (16) |
| `step_embed` | bViT-TE per-step embedding |
| `lambda_bal/z/div` | reViT aux-loss weights (1e-2 / 1e-3 / 1e-3, as in rvit) |

## Decisions

* **`backbone.n_blocks` stays the COMPUTATIONAL depth.** So `compute_rw_positions`, the
  per-block `TokenModulation` and the forward loop needed no change in meaning;
  `backbone.blocks` holds only the templates, `backbone.run_block(t, ...)` picks one.
* **Only the ViT trunk recurs.** Canvas read/write cross-attention keeps one module per
  position, as before. Sharing those too would be a separate experiment.
* **Config lives in `CanViTConfig`, not in the backbone name.** It is then serialized in
  every checkpoint's `model_config` and rebuilt by all loaders (`coerce_nested_configs`,
  dacite); `in1k`'s probe lookup by `backbone_name` keeps working. Every `create_backbone`
  call that feeds a `CanViT` passes `recurrence=cfg.recurrence`; `CanViT.__init__` asserts
  the two agree, so a missed call site fails loudly instead of building the wrong model.
* **Aux losses are added in `harness/loop.py`,** once per optimizer step after the rollout
  backward and before the DDP AllReduce (they are parameter-only and identical on every
  rank). Only when the backbone has a router and it trains. The loop emits
  `recurrence/aux_{bal,z,div}`, `recurrence/gate_max` and `recurrence/w_cos_max`; the
  tracker prefixes them, so in wandb they are **`train/recurrence/*`**.
* **DINOv3 init refuses a recurrent backbone** (`dinov3_init.py`): there is no canonical
  layer for a shared block, and an expert bank has no `mlp.fc1` keys, which the loader's
  key-dropping would have skipped SILENTLY.
* Kept CanViT's block (LayerNorm + LayerScale 1e-5, qkv without K bias, RoPE). rvit's bViT
  used RMSNorm and no LayerScale; that is a norm choice, not the recurrence.
* Weight decay: CanViT's optimizer applies one decay per module group to all params
  (biases included). rvit excluded `b1/b2`; noted because rvit's reViT-S collapse
  (2026-11-18) involved the undecayed biases growing under L_div.

## Use

```bash
# distill from scratch, bViT-B
EXTRA_ARGS="--cfg.model.recurrence.enabled"
# reViT-B, 4 experts
EXTRA_ARGS="--cfg.model.recurrence.enabled --cfg.model.recurrence.n-experts 4"
```

Not with `--cfg.init-backbone-from-teacher`, and not seeded from a non-recurrent
checkpoint (strict load fails on the key mismatch, by design). Downstream `ade20k`/`in1k`
inherit the setting from the pretrained checkpoint's `model_config`.

## exp48 results, seed 0, 2026-11-25

Launcher `slurm/runs/exp48_recurrence/exp48.sh`, pin `3e42f3b`. The ViT baseline is
jon_exp32_pretrain_lrdrop/exp32-uniform16, which has the same config, the same data order
(schedule seed 0, 3211 shards) and the same 256-image val subset, on an older code pin.
Its val history comes from its local wandb files
(`/mnt/vast-nhr/projects/nib00021/jonathan/wandb/run-*-q5m790pl/*.wandb`, read-only:
copy them before opening with `wandb.sdk.internal.datastore`, which opens r+b). Its wandb
run is not reachable through the API.

Val at step 196,608 (the last step validated on all three; ours validate at each job
START, so 204,800 comes with the next job):

| | scene_cos_raw | cls_cos_raw | in1k top1 t0 | in1k top1 t9 | trunk params |
|---|---:|---:|---:|---:|---:|
| ViT (12 blocks) | 0.877 | 0.725 | 59.0 | 65.2 | 85.1 M |
| bViT (1 block) | 0.828 | 0.558 | 37.5 | 41.8 | 7.1 M |
| reViT (1 block, E=4) | 0.836 | 0.590 | 39.5 | 46.1 | 21.3 M |

* The gap opens after ~16k steps and widens monotonically on every metric, e.g.
  scene_cos ViT − bViT: 0.020 at 32k, 0.029 at 65k, 0.040 at 98k, 0.049 at 197k.
* reViT ≥ bViT on the cos metrics from ~100k on (+0.008 scene, +0.032 cls at 197k).
* top-1 is on 256 images (SE ≈ 3 points at 45 %), so bViT vs reViT top-1 is within noise.
  The cos metrics are the reliable ones.
* reViT did not collapse: train/recurrence/w_cos_max 0.40 (25k) → 0.16 (204k), and
  gate_max 0.25 → 0.98 (each depth picks ~one expert).
* 204,800 steps × batch 64 = 13.1 M samples ≈ 1 pass over IN21k (13.15 M). Only ~105k
  steps at peak lr (warmup 100k, then constant). exp32 continues to 1.43 M.

Follow-ups submitted 2026-11-25: s0 of both arms continued to 409,600 (jobs 17551441,
17551444); seeds 1 and 2 of both arms to 204,800 (17551442/3, 17551445/6). The ViT
baseline still has one seed only.
