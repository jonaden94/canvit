"""Greedy ground-truth oracle for the foveated fixation policy on ADE20K val.

The question: how good can ANY glimpse policy get with this frozen model? Per image,
start from the run's t0 anchor (centre fixation at the training view scale), then at
each step try every candidate fixation of the policy's action space, score each against
that image's ground truth, keep the best, and continue from it. Greedy, per image.

Everything is the exp36 / exp36_1 setup, so the numbers are directly comparable to the
trained policies: exp22 backbone + exp34 probe, loaded the way `Ade20kRunTask.build_model`
loads them; ADE20K val, squish, 512 px; canvas grid 32; 5 glimpses; fixed view scale 2.0;
the 16x16 `fixation_candidates` grid (`--rl.centers-per-axis` default); bf16 autocast
for the backbone and fp32 for the head, as the eval does. Selection criterion: the
policy's own training reward, per-image probe CE at 128 px (`reward_ce`). Reported:
dataset mIoU per t (from per-image counts, the eval's `per_row_iou` path) and per-image
full-resolution CE (the eval's `_full_res_ce`).

Must run under the PINNED snapshot -- TRAIN_COMMIT f3d6e6b + FOVI_COMMIT c399d3b, the
exp36_1 pins -- not today's tree: the exp22 checkpoint was trained on fovi c399d3b's
geometry, and today's ported geometry (fovi 27b1bde) differs. `oracle.sbatch` does that.

    oracle.py run   --out DIR --start 0 --end 1000 [--criterion ce|random] [--split-block 64]
    oracle.py merge DIR [DIR ...]
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

log = logging.getLogger("oracle")

MODEL = ("/mnt/vast-nhr/projects/nib00021/jonathan/repos/canvit/logs/jon_exp22_full_runs/"
         "exp22-fovi-teacherinit-lrdrop-1196k/checkpoints/step-155648.pt")
PROBE = ("/mnt/vast-nhr/projects/nib00021/jonathan/repos/canvit/logs/jon_exp34_ade20k_probe/"
         "ade20k-fovi-ti-1196k/checkpoints/best.pt")


def build(args, device):
    from canvit.ade20k.config import Ade20kConfig
    from canvit.ade20k.data import ADE20kDataset, make_val_transforms
    from canvit.ade20k.task import Ade20kRunTask
    from canvit.harness.config import FoveatedScaleConfig

    cfg = Ade20kConfig(model_repo=args.model_repo, probe_repo=args.probe_repo,
                       canvas_grid=args.canvas_grid, n_timesteps=args.n_timesteps,
                       resize_mode="squish",
                       foveated_scale=FoveatedScaleConfig(mode="fixed", fixed_scale=args.scale))
    seg, _ = Ade20kRunTask(cfg).build_model(device)
    seg.eval()  # = the eval's state: trunk.eval() (frozen) + head.eval()
    img_tf, mask_tf = make_val_transforms(cfg.scene_size, cfg.resize_mode)
    ds = ADE20kDataset(root=cfg.ade20k_root, split="validation",
                       img_transform=img_tf, mask_transform=mask_tf)
    end = min(args.end, len(ds)) if args.end is not None else len(ds)
    idx = list(range(args.start, end))
    loader = DataLoader(Subset(ds, idx), batch_size=1, num_workers=args.num_workers,
                        pin_memory=device.type == "cuda")
    names = [ds.images[i].stem for i in idx]
    return seg, cfg, loader, idx, names


@torch.no_grad()
def run(args):
    from canvit.ade20k.data import IGNORE_LABEL, NUM_CLASSES
    from canvit.ade20k.metrics import preds_from_logits, reward_ce
    from canvit.ade20k.per_row_iou import batch_confusion
    from canvit.core import RecurrentState
    from canvit.core.policy import fixation_candidates
    from canvit.core.viewpoint import Viewpoint

    device = torch.device(args.device)
    seg, cfg, loader, idx, names = build(args, device)
    G, T, s = args.canvas_grid, args.n_timesteps, args.scale
    cand = fixation_candidates(args.centers_per_axis).reshape(-1, 3)[:, :2].to(device)  # [A, 2]
    A = cand.shape[0]
    amp = (torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda"
           else torch.autocast("cpu", enabled=False))
    gen = torch.Generator().manual_seed(args.seed)

    def step(img, state, centers):
        """One glimpse for n = len(centers) copies of one image/state -> (state, logits)."""
        n = centers.shape[0]
        st = RecurrentState(canvas=state.canvas.expand(n, -1, -1),
                            recurrent_cls=state.recurrent_cls.expand(n, -1, -1))
        vp = Viewpoint(centers=centers, scales=torch.full((n,), s, device=device))
        with amp:
            out = seg.canvit(image=img.expand(n, -1, -1, -1), state=st, viewpoint=vp)
            hidden = seg.canvit.get_spatial(out.state.canvas).view(n, G, G, -1)
        return out.state, seg.head(hidden.float())  # head in fp32, as the eval

    N = len(idx)
    cand_idx = torch.full((N, T - 1), -1, dtype=torch.long)
    ce_map = torch.full((N, T - 1, A), float("nan"))      # reward CE of EVERY candidate
    ce_reward = torch.zeros(N, T)                          # 128 px CE along the trajectory
    ce_full = torch.zeros(N, T)                            # full-res CE (the eval's ce_t)
    inter = torch.zeros(T, N, NUM_CLASSES)
    union = torch.zeros(T, N, NUM_CLASSES)
    b = args.split_block
    if b:  # held-out split: checkerboard of b x b px blocks, A = choose on, B = held out
        inter_ab = torch.zeros(2, T, N, NUM_CLASSES)
        union_ab = torch.zeros(2, T, N, NUM_CLASSES)
        ce_full_ab = torch.zeros(2, N, T)
    max_resid = 0.0  # |CE of the chosen candidate in the sweep - CE of its batch-1 replay|
    t_start = time.time()
    for i, (img, mask) in enumerate(loader):
        img, mask = img.to(device), mask.to(device)
        if b:
            H, W = mask.shape[1:]
            yy, xx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device),
                                    indexing="ij")
            in_a = ((yy // b + xx // b) % 2 == 0)[None]
            ignore = torch.full_like(mask, IGNORE_LABEL)
            masks_ab = [torch.where(in_a, mask, ignore), torch.where(in_a, ignore, mask)]
        sel_mask = masks_ab[0] if b else mask  # the labels the oracle may look at
        state = seg.init_state(batch_size=1, canvas_grid_size=G)
        for t in range(T):
            if t == 0:
                k = None
                centers = torch.zeros(1, 2, device=device)  # the FULL anchor: centre, scale s
            else:
                if args.criterion == "ce":
                    ces = []
                    for c0 in range(0, A, args.chunk):
                        c = cand[c0:c0 + args.chunk]
                        _, logits = step(img, state, c)
                        ces.append(reward_ce(logits, sel_mask.expand(len(c), -1, -1),
                                             score_res=args.score_res))
                    ce_map[i, t - 1] = torch.cat(ces).cpu()
                    k = int(ce_map[i, t - 1].argmin())
                else:  # random cell of the same grid: the matched lower reference
                    k = int(torch.randint(A, (1,), generator=gen))
                cand_idx[i, t - 1] = k
                centers = cand[k:k + 1]
            state, logits = step(img, state, centers)  # batch-1 replay of the chosen glimpse
            ce_reward[i, t] = reward_ce(logits, sel_mask, score_res=args.score_res).item()
            if k is not None and args.criterion == "ce":
                max_resid = max(max_resid, abs(ce_reward[i, t].item() - ce_map[i, t - 1, k].item()))
            ce_full[i, t] = _full_res_ce(logits, mask)
            preds = preds_from_logits(logits, mask.shape[1], mask.shape[2])
            it, un, _ = batch_confusion(preds, mask, NUM_CLASSES)
            inter[t, i], union[t, i] = it[0].cpu(), un[0].cpu()
            if b:
                for h, m in enumerate(masks_ab):
                    it, un, _ = batch_confusion(preds, m, NUM_CLASSES)
                    inter_ab[h, t, i], union_ab[h, t, i] = it[0].cpu(), un[0].cpu()
                    ce_full_ab[h, i, t] = _full_res_ce(logits, m)
        if (i + 1) % args.log_every == 0 or i + 1 == N:
            el = time.time() - t_start
            log.info("%d/%d images  %.2f s/img  ETA %.2f h  mean CE t0..t%d: %s  max replay resid %.2e",
                     i + 1, N, el / (i + 1), el / (i + 1) * (N - i - 1) / 3600, T - 1,
                     " ".join(f"{v:.4f}" for v in ce_full[: i + 1].mean(0).tolist()), max_resid)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = args.criterion + (f"_split{b}" if b else "")
    f = out / f"oracle-{tag}-{idx[0]:04d}-{idx[-1] + 1:04d}.pt"
    torch.save({
        "image_idx": torch.tensor(idx), "image_name": names,
        "cand_idx": cand_idx, "centers": cand.cpu()[cand_idx.clamp(min=0)],
        "candidates": cand.cpu(), "ce_map": ce_map if args.criterion == "ce" else None,
        "ce_reward": ce_reward, "ce_full": ce_full, "inter": inter, "union": union,
        "max_replay_resid": max_resid, "args": vars(args),
        **({"inter_ab": inter_ab, "union_ab": union_ab, "ce_full_ab": ce_full_ab} if b else {}),
    }, f)
    log.info("wrote %s", f)


def _full_res_ce(logits, masks):
    """The eval's per-image full-resolution CE (`Ade20kRunTask._full_res_ce`), from logits."""
    import torch.nn.functional as F

    from canvit.ade20k.data import IGNORE_LABEL
    up = F.interpolate(logits, size=masks.shape[1:], mode="bilinear", align_corners=False)
    per_px = F.cross_entropy(up, masks, ignore_index=IGNORE_LABEL, reduction="none")
    valid = (masks != IGNORE_LABEL).float()
    return ((per_px * valid).flatten(1).sum(1) / valid.flatten(1).sum(1).clamp_min(1.0)).item()


def miou_from_rows(inter, union):
    """Copy of `canvit.ade20k.per_row_iou.miou_from_rows` (f3d6e6b), so `merge` needs no
    snapshot: per class sum over images, mean over classes with a nonzero union."""
    ti, tu = inter.double().sum(0), union.double().sum(0)
    return (ti[tu > 0] / tu[tu > 0]).mean().item()


def merge(dirs):
    shards = sorted(f for d in dirs for f in Path(d).glob("oracle-*.pt"))
    by_crit: dict[str, list] = {}
    for f in shards:
        by_crit.setdefault(f.name.split("-")[1], []).append(torch.load(f, weights_only=False))  # e.g. ce_split64
    for crit, parts in by_crit.items():
        idx = torch.cat([p["image_idx"] for p in parts])
        assert len(set(idx.tolist())) == len(idx), f"{crit}: overlapping shards"
        inter = torch.cat([p["inter"] for p in parts], dim=1)
        union = torch.cat([p["union"] for p in parts], dim=1)
        ce = torch.cat([p["ce_full"] for p in parts])
        T = inter.shape[0]
        print(f"criterion={crit}  images={len(idx)}  "
              f"max replay resid={max(p['max_replay_resid'] for p in parts):.2e}")
        print("  mIoU " + "  ".join(f"t{t} {miou_from_rows(inter[t], union[t]):.4f}" for t in range(T)))
        print("  CE   " + "  ".join(f"t{t} {ce[:, t].mean():.4f}" for t in range(T)))
        if "inter_ab" in parts[0]:
            i_ab = torch.cat([p["inter_ab"] for p in parts], dim=2)
            u_ab = torch.cat([p["union_ab"] for p in parts], dim=2)
            c_ab = torch.cat([p["ce_full_ab"] for p in parts], dim=1)
            for h, name in enumerate(["A (selection pixels)", "B (held-out pixels)"]):
                print(f"  {name}")
                print("    mIoU " + "  ".join(f"t{t} {miou_from_rows(i_ab[h, t], u_ab[h, t]):.4f}" for t in range(T)))
                print("    CE   " + "  ".join(f"t{t} {c_ab[h, :, t].mean():.4f}" for t in range(T)))


def main():
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--start", type=int, default=0)
    r.add_argument("--end", type=int, default=None)
    r.add_argument("--criterion", choices=["ce", "random"], default="ce")
    r.add_argument("--model-repo", default=MODEL)
    r.add_argument("--probe-repo", default=PROBE)
    r.add_argument("--canvas-grid", type=int, default=32)
    r.add_argument("--n-timesteps", type=int, default=5)
    r.add_argument("--scale", type=float, default=2.0)
    r.add_argument("--centers-per-axis", type=int, default=16)
    r.add_argument("--score-res", type=int, default=128)
    r.add_argument("--chunk", type=int, default=128, help="candidates per forward")
    r.add_argument("--seed", type=int, default=0, help="random criterion only")
    r.add_argument("--split-block", type=int, default=0,
                   help="held-out test: choose on the A cells of a checkerboard of this many px, "
                        "report A and B separately (0 = off, the original oracle)")
    r.add_argument("--device", default="cuda")
    r.add_argument("--num-workers", type=int, default=4)
    r.add_argument("--log-every", type=int, default=20)
    m = sub.add_parser("merge")
    m.add_argument("dirs", nargs="+")
    args = p.parse_args()
    if args.cmd == "run":
        log.info("args: %s", vars(args))
        run(args)
    else:
        merge(args.dirs)


if __name__ == "__main__":
    main()
