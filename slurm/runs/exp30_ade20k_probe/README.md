# exp30 — ADE20K frozen-probe training (harness, current code)

Four frozen-probe runs, one per exp22 pretrained source. Recipe copied from **exp24** = the
original `canvit_specialize` ade20k probe, reproduced by the harness `ade20k` defaults
(frozen backbone via the default `probe` preset, 40k steps, random-view training,
`n_timesteps 10`, scene 512, `canvas_grid 32`).

Sources are the same four as exp29 (see its README; `ade20k-fovi-1901k` is the new one).
The foveated arms pass `--cfg.foveated-scale.fixed-scale 2.0`.

**`resize_mode=squish` for every arm, including foveated.** That is the protocol every earlier
CanViT / specialize number was measured under, so it is what makes these comparable to exp24
and to the published values. It distorts aspect ratio, so it is *not* the right choice for a
human-viewing comparison — `center_crop` preserves the geometry foveated sampling assumes, at
the cost of cropping the long side. Whichever is used has to be reported with the number.

Note: ADE20K train-mIoU (`train_miou_mean`, `best_val_miou_t{t}`) is **deliberately not
logged** by the harness (owner decision, 2026-07-31 — see
`canvit_claude/canvit/unification/17-harness-consolidation.md`). Train loss and per-timestep val mIoU are
unaffected. Old pre-2026-07-31 ade20k wandb runs have `train_miou_mean` panels these will not.

## How to judge these results — mIoU is NOT directly comparable to exp24

**exp24's pin (`24a8500`) PREDATES `68b635f` "ade20k: fix the mIoU reduction order (RE-BASES
all ADE20K mIoU)" (2026-07-29).** exp30 pins `455bdae`, which includes it. The old path took
argmax at the 64x64 probe grid then nearest-upsampled, locking every label into an 8x8 pixel
block; the fix upsamples logits bilinearly and argmaxes at full resolution. The old order is
strictly coarser — a bug, not an alternative convention.

Measured gain (full 2000-image ADE20K val, both reductions from the same logits):
**+0.15 / +0.17 / +0.18 / +0.19 pp at t1..t4**, growing with t because the segmentation
sharpens. Reproducible to 0.01 pp across seeds. CE is unchanged (it has its own bilinear path).

So exp30's mIoU should come out **above** exp24's by roughly +0.2 pp at late timesteps — and
that is a metric-definition change, **not** an improvement from the unification. Do not read it
as one. The gain beyond t4 was never measured, so for t9 treat +0.2 pp as a lower bound.

| exp24 reference (STALE basis) | miou_final | miou_mean | exp30 expectation |
|---|---|---|---|
| `ade20k-uni16ti-803k` | 0.44443 | 0.43305 | ~0.446 |
| `ade20k-uni16-1516k` | 0.41953 | 0.40769 | ~0.421 |
| `ade20k-fovi-ti-1196k` | 0.43838 | 0.41877 | ~0.440 |
| `ade20k-fovi-1901k` | — | — | no reference (new arm) |

Two further reasons not to expect equality: exp24's probe runs were **unseeded** (`24a8500`'s
ade20k config has no `seed` field at all; the harness added `seed: int = 0`), and exp24 logged
**no CE at all**, so there is no re-basing-immune metric to fall back on.

**Therefore the meaningful checks here are relative, not absolute:** the ordering
`uni16ti > fovi-ti > uni16` should hold, all four should sit in the 0.41-0.45 band, and the
four exp30 runs are mutually comparable because they share code and a seed. exp30 is the FIRST
ade20k probe set on the corrected mIoU basis — it becomes the reference for what follows.

## Finding exp24's runs in wandb: they are all named `ade20k`

exp24 predates the run-naming fix, so **every** exp24 ade20k run appears in wandb literally as
`ade20k`. Identify them by id, not name:

| source | wandb id | `miou_final` last / max |
|---|---|---|
| `ade20k-uni16ti-803k` | `t32oc3x1` | 0.44443 / 0.44479 |
| `ade20k-uni16-1516k` | `n6gw2xoi` | 0.41953 / 0.42321 |
| `ade20k-fovi-ti-1196k` | `pkq5lpyc` | 0.43838 / 0.43997 |

(A fourth run `aa4dtxg6`, correctly named `ade20k-fovi-ti-1196k`, is a later CRASHED retry with
no metrics — ignore it.) All three finished at step 39,500.

## Submitted job IDs (2026-07-31)

`ade20k-uni16ti-803k` 15113358 · `ade20k-fovi-ti-1196k` 15113359 · `ade20k-uni16-1516k` 15113360 · `ade20k-fovi-1901k` 15113361
