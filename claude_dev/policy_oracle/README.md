# Greedy ground-truth oracle for the fixation policy (2026-11-12)

How close are the trained policies to the best a policy could do? exp36_1 found QReg,
on-policy PG, Q-Prop and canvas-only Q-Prop tied (t4 ≈ 0.452 mIoU), so the next question
is the ceiling, not another policy.

`oracle.py` answers it greedily, per image: from the t0 anchor (centre, view scale 2.0),
try all 256 cells of the 16×16 fixation grid, keep the one with the lowest per-image
probe CE against the ground truth (the policies' own reward, 128 px), repeat to t4. Same
model, probe, data protocol and pins (canvit f3d6e6b, fovi c399d3b) as exp36_1; the
docstring lists each setting. `--criterion random` takes a random cell of the same grid:
the matched lower reference.

Saved per image: chosen cells and centres, the CE of all 256 candidates at every step
(`ce_map`), per-step reward and full-resolution CE, and per-class intersection/union
counts, so dataset mIoU is computed by `merge` exactly as the eval's `per_row_iou` path.

Checks run before submission (CPU, 1 image, 4 candidates): imports resolve to the pinned
snapshots; argmin selection correct; batch-1 replay of the chosen glimpse matches its CE
in the candidate sweep to 6e-8. Checks for the real run: merged t0 mIoU must equal the
V100 policy eval's t0 (0.3767); the replay residual in the log stays at bf16 noise.

    bash claude_dev/policy_oracle/submit.sh         # 2 jupyter V100 jobs, 1000 images each
    bash claude_dev/policy_oracle/submit.sh merge   # results, once both finish

Output: `logs/jon_exp36_1_oracle/`.
