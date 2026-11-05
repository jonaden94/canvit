"""Visual-search task: the per-batch engine seam and the run-level ``RunTask``.

Readout = per-glimpse canvas tokens ``[B, G, G, D]``; the target-conditioned head is
applied OUTSIDE the backbone forward (so a frozen backbone still trains it). Foveated
backbones only: the patcher consumes the full image and foveates it itself.

Design: ``claude_dev/search/00-design.md``. Default regime: backbone frozen, task head and
policy trained JOINTLY (``default_spec``).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import nullcontext
from typing import Any

import torch
from torch import Tensor, nn

from canvit.core import RecurrentState, Viewpoint
from canvit.harness.rollout import GlimpseOut, TaskLoss
from canvit.harness.rollout.episode import consumes_full_image
from canvit.harness.rollout.selector import RandomSelector
from canvit.harness.rollout.viewpoint import ViewpointType

from .config import SearchConfig, search_policy_config
from .loss import hit, per_image_loss, target_map

log = logging.getLogger(__name__)

POLICY_FEATURE_GROUPS: tuple[str, ...] = ("ln_feat",)  # the plain canvas, no handcrafted maps


def _canvas_hidden(model: Any, state: RecurrentState, canvas_grid: int) -> Tensor:
    b = state.canvas.shape[0]
    return model.canvit.get_spatial(state.canvas).view(b, canvas_grid, canvas_grid, -1)


class BoundSearchTask:
    """Per-batch :class:`RolloutTask`: binds this batch's targets and boxes.

    Optional engine hooks, present only when they apply: ``step_weight`` (``supervise='last'``)
    and ``policy_cond`` (a policy is trained)."""

    def __init__(self, *, model: Any, targets: Tensor, boxes: Tensor, canvas_grid: int,
                 cfg: SearchConfig, policy_embed: Callable[[Tensor], Tensor] | None = None):
        self.model = model
        self.targets = targets
        self.boxes = boxes
        self.canvas_grid = canvas_grid
        self.target = target_map(boxes, grid=canvas_grid, kind=cfg.loss, sigma=cfg.target_sigma)
        if cfg.supervise == "last":
            self.step_weight = self._last_only
        if policy_embed is not None:
            self.policy_cond = lambda: policy_embed(targets)

    @staticmethod
    def _last_only(t: int, n_glimpses: int) -> float:
        # The engine divides the summed per-glimpse loss by n_glimpses, so weight n at the
        # last glimpse makes the task loss exactly that glimpse's loss.
        return float(n_glimpses) if t == n_glimpses - 1 else 0.0

    def forward_glimpse(self, *, model: Any, images: Tensor, state: RecurrentState,
                        viewpoint: Any, backbone_no_grad: bool) -> GlimpseOut:
        core = getattr(model, "module", model)
        ctx = torch.no_grad() if backbone_no_grad else nullcontext()
        with ctx:
            out = core.canvit(image=images, state=state, viewpoint=viewpoint)
            hidden = _canvas_hidden(core, out.state, self.canvas_grid)
        return GlimpseOut(readout=hidden, state=out.state, vpe=out.vpe)

    def logits(self, readout: Tensor) -> Tensor:
        return self.model.head(readout.float(), self.targets)

    def per_image_loss(self, readout: Tensor) -> Tensor:
        """Per-sample KL to the target map [B] -- also the policy reward's raw material."""
        return per_image_loss(self.logits(readout), self.target)

    def step_loss(self, readout: Tensor) -> TaskLoss:
        return TaskLoss(combined=self.per_image_loss(readout).mean())


class SearchRunTask:
    """Run-level visual-search task for ``harness.run``."""

    name = "search"

    def __init__(self, cfg: SearchConfig, *, rl=None):
        self.cfg = cfg
        self.rl = rl if rl is not None else search_policy_config()
        self._joint = None
        self.classes: list[str] | None = None

    @property
    def best_metric(self) -> str:
        """Last-glimpse hit rate under the learned policy, or under random fixations when no
        policy is trained (``--preset probe``)."""
        return "hit_final_policy" if self._joint is not None else "hit_final_random"

    # --- capabilities & defaults ------------------------------------------
    def caps(self):
        from canvit.harness.spec import TaskCaps
        # Map-style dataset behind a shuffling DataLoader, no rank sharding (as ade20k).
        return TaskCaps(has_head=True, supports_policy=True, supports_ddp=False)

    def _policy_warmup_steps(self) -> int:
        pol = self.rl
        if pol.policy_warmup_steps > 0:
            return pol.policy_warmup_steps
        return int(max(pol.policy_warmup_frac, 0.0) * self.cfg.max_steps)

    def default_spec(self):
        """Frozen backbone; task head and policy trained JOINTLY; no BPTT (the backbone runs
        under no_grad, so there is no cross-timestep graph to keep)."""
        from canvit.harness.spec import BpttSpec, GroupOptim, ScheduleSpec, TrainSpec
        head = GroupOptim(lr=self.cfg.peak_lr, weight_decay=self.cfg.weight_decay,
                          schedule=ScheduleSpec(kind="warmup_cosine", warmup_steps=self.cfg.warmup_steps,
                                                total_steps=self.cfg.max_steps))
        pol = self.rl
        policy = GroupOptim(lr=pol.policy_lr, weight_decay=pol.policy_weight_decay, betas=pol.policy_betas,
                            schedule=ScheduleSpec(kind="warmup_constant", warmup_steps=self._policy_warmup_steps()))
        return TrainSpec(
            train_backbone=False, train_head=True, train_policy=True, task_weight=1.0, policy_weight=1.0,
            task_grad_to_backbone=False, policy_grad_to_backbone=False,
            bptt=BpttSpec(mode="none", horizon=self.cfg.n_timesteps),
            optim={"head": head, "policy": policy},
        )

    # --- construction ------------------------------------------------------
    def _classes(self) -> list[str]:
        if self.classes is None:
            from .data import coco_search18_classes
            self.classes = coco_search18_classes(self.cfg.data_root)
        return self.classes

    def build_model(self, device, prior_model_config=None):
        from .model import CanViTForVisualSearch
        if not self.cfg.model_repo:
            raise ValueError("--cfg.model-repo is required: a pretrained FOVEATED CanViT (HF id, local HF "
                             "dir or training .pt).")
        c = self.cfg
        model = CanViTForVisualSearch.from_pretrained_backbone(
            c.model_repo, n_classes=len(self._classes()), cond_dim=c.cond_dim, kind=c.head,
            width=c.head_width, block_layers=c.head_block_layers, dropout=c.dropout,
        ).to(device)
        if not consumes_full_image(model):
            raise ValueError(f"the search task supports FOVEATED backbones only; {c.model_repo} has patcher "
                             f"{getattr(model.canvit.cfg, 'patcher_name', '?')!r}.")
        fs = c.foveated_scale
        log.warning("foveated view scale: mode=%s fixed_scale=%s -- this MUST match the backbone's "
                    "pretraining scale or every glimpse is out of distribution.", fs.mode, fs.fixed_scale)
        return model, model.head

    def canvas_grid(self, model):
        if self.cfg.canvas_grid is not None:
            return self.cfg.canvas_grid
        return self.cfg.scene_size // model.canvit.backbone.patch_size_px

    def is_foveated(self, model):
        return consumes_full_image(model)

    def branches(self):
        return [ViewpointType.FULL]  # t0 = the centred fixation, as in the human trials and Haoze

    def build_val_loader(self):
        from .data import make_search_val_loader
        loader, self.classes = make_search_val_loader(self.cfg)
        return loader

    def build_loaders(self, *, world_size, rank):
        if world_size > 1:
            raise RuntimeError("search does not support DDP yet: its loader cannot shard by rank.")
        from .data import make_search_loaders
        train, val, self.classes = make_search_loaders(self.cfg)
        return train, val

    def _random_selector(self) -> RandomSelector:
        return RandomSelector(is_foveated=True, foveated_scale=self.cfg.foveated_scale, min_viewpoint_scale=0.05)

    def build_selector(self, *, device, canvas_grid, is_foveated):
        return self._random_selector()

    def build_policy(self, model, *, device, canvas_grid, generator):
        from canvit.harness.policy import build_policy
        c, rl = self.cfg, self.rl
        if rl.objective == "vpg":
            raise ValueError("search does not support objective='vpg' in v1: its learned baseline cannot "
                             "see the target (design doc §5.1). Use objective='pg'.")
        jp = build_policy(
            canvit=model.canvit, rl=rl, feature_groups=POLICY_FEATURE_GROUPS, device=device,
            canvas_grid=canvas_grid, min_viewpoint_scale=0.05, foveated_scale=c.foveated_scale,
            generator=generator, encode_model=model, scorer_input_norm="none", film_cond_dim=c.cond_dim,
        )
        if c.supervise == "last" and jp.reward.kind in ("fractional", "difference"):
            raise ValueError(
                f"reward={jp.reward.kind!r} needs the task loss at EVERY glimpse, but supervise='last' "
                "trains the head on the last glimpse only, so the early losses (and the rewards built "
                "from them) are noise. Use --cfg.supervise every, or --rl.reward terminal.")
        if c.embedding == "separate":
            # The policy's own embedding rides on the scorer module, so the harness treats it as
            # policy parameters everywhere (requires_grad, grad clipping, DDP sync, checkpoint).
            jp.scorer.add_module("target_embed", nn.Embedding(len(self._classes()), c.cond_dim).to(device))
        self._joint = jp
        return jp

    def _policy_embed(self, model) -> Callable[[Tensor], Tensor] | None:
        jp = self._joint
        if jp is None:
            return None
        if self.cfg.embedding == "shared":
            core = getattr(model, "module", model)
            return lambda t: core.head.policy_embedding(t, detach=self.cfg.detach_policy_embedding)
        return lambda t: jp.scorer.target_embed(t)

    def policy_feature_groups(self):
        return POLICY_FEATURE_GROUPS

    def trainable_param_groups(self, *, model, head, joint, spec):
        groups: dict[str, list] = {}
        if spec.train_backbone:
            groups["backbone"] = list(model.canvit.parameters())
        if spec.train_head:
            groups["head"] = list(model.head.parameters())
        if spec.train_policy:
            assert joint is not None
            groups["policy"] = list(joint.scorer.parameters())
        return groups

    def resume_start_step(self, payload, scheduler):
        return scheduler.last_epoch

    def resume_state(self):
        return {}

    # --- per-batch ---------------------------------------------------------
    def batch_images(self, batch, device):
        return batch["image"].to(device, non_blocking=True)

    def bind(self, batch, device, *, model, head):
        return BoundSearchTask(
            model=getattr(model, "module", model), targets=batch["target"].to(device),
            boxes=batch["boxes"].to(device), canvas_grid=self.canvas_grid(model), cfg=self.cfg,
            policy_embed=self._policy_embed(model))

    # --- evaluation --------------------------------------------------------
    def _rollout_metrics(self, *, model, images, targets, boxes, T, cg, amp, choose=None, joint=None):
        """One closed-loop episode of ``T`` glimpses; returns per-timestep (loss [B], hit [B]).

        t0 is the centred fixation. Later fixations come from ``choose(t, logits, visited)``
        (random / greedy) or, with ``joint``, from the learned scorer's argmax."""
        from canvit.harness.rollout.eval_viewpoints import deploy_rollout_viewpoints
        from canvit.harness.rollout.engine import _to_vp

        B = images.shape[0]
        tmap = target_map(boxes, grid=cg, kind=self.cfg.loss, sigma=self.cfg.target_sigma)
        losses: list[Tensor] = []
        hits: list[Tensor] = []
        visited = torch.zeros(B, cg * cg, dtype=torch.bool, device=images.device)
        last_logits: list[Tensor] = []

        def mark(vp: Viewpoint) -> None:
            cy, cx = vp.centers[:, 0], vp.centers[:, 1]
            iy = ((cy + 1) / 2 * cg).floor().clamp(0, cg - 1).long()
            ix = ((cx + 1) / 2 * cg).floor().clamp(0, cg - 1).long()
            visited[torch.arange(B, device=images.device), iy * cg + ix] = True

        def advance(vp: Viewpoint, state, t: int):
            if state is None:
                state = model.init_state(batch_size=B, canvas_grid_size=cg)
            with amp:
                out = model.canvit(image=images, state=state, viewpoint=vp)
            logits = model.head(_canvas_hidden(model, out.state, cg).float(), targets).float()
            losses.append(per_image_loss(logits, tmap))
            hits.append(hit(logits, boxes, box_frac=self.cfg.hit_box_frac))
            mark(vp)
            last_logits[:] = [logits]
            return out.state

        if joint is not None:
            embed = self._policy_embed(model)
            assert embed is not None
            deploy_rollout_viewpoints(joint=joint, advance=advance, t0_type=ViewpointType.FULL,
                                      batch_size=B, device=images.device, n=T, cond=lambda: embed(targets))
        else:
            assert choose is not None
            sel = self._random_selector()
            ctx = sel.start_rollout(t0_type=ViewpointType.FULL, batch_size=B, device=images.device)
            vp = _to_vp(sel.select(vp_type=ViewpointType.FULL, ctx=ctx, t=0, batch_size=B,
                                   device=images.device, state=None))  # type: ignore[arg-type]
            state = None
            for t in range(T):
                state = advance(vp, state, t)
                if t < T - 1:
                    vp = choose(sel, ctx, last_logits[0], visited)
        return losses, hits

    def _greedy(self, cg: int):
        """Fixate the current heatmap's argmax among cells not yet visited."""
        def choose(sel, ctx, logits, visited):
            B = logits.shape[0]
            idx = logits.flatten(1).masked_fill(visited, float("-inf")).argmax(-1)
            c = (torch.arange(cg, device=logits.device, dtype=torch.float32) + 0.5) * (2.0 / cg) - 1.0
            centers = torch.stack([c[idx // cg], c[idx % cg]], dim=-1)  # (y, x)
            scales, _ = sel.view_scales(rollout_scales=ctx.rollout_scales, batch_size=B, device=logits.device)
            return Viewpoint(centers=centers, scales=scales)
        return choose

    @staticmethod
    def _random(sel, ctx, logits, visited):
        from canvit.harness.rollout.engine import _to_vp
        B = logits.shape[0]
        return _to_vp(sel.select(vp_type=ViewpointType.RANDOM, ctx=ctx, t=1, batch_size=B,
                                 device=logits.device, state=None))  # type: ignore[arg-type]

    @torch.no_grad()
    def evaluate(self, *, model, head, val_loader, device, step, tracker=None, run_dir=None, joint=None):
        """Per-timestep KL and hit rate on the val set under each of ``cfg.eval_policies``."""
        T = self.cfg.n_timesteps
        cg = self.canvas_grid(model)
        amp = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
        was_training = model.head.training
        model.head.eval()
        policies = [p for p in self.cfg.eval_policies if p != "policy" or joint is not None]
        sums = {p: ([0.0] * T, [0.0] * T) for p in policies}
        n = 0
        # Random fixations draw from the global torch RNG: fork it, so eval is seeded and
        # cannot perturb the training stream.
        with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
            torch.manual_seed(self.cfg.eval_seed)
            for vb, batch in enumerate(val_loader):
                if self.cfg.limit_val_batches is not None and vb >= self.cfg.limit_val_batches:
                    break
                images = batch["image"].to(device)
                targets, boxes = batch["target"].to(device), batch["boxes"].to(device)
                for p in policies:
                    kw = dict(model=model, images=images, targets=targets, boxes=boxes, T=T, cg=cg, amp=amp)
                    if p == "policy":
                        losses, hits = self._rollout_metrics(**kw, joint=joint)
                    elif p == "greedy":
                        losses, hits = self._rollout_metrics(**kw, choose=self._greedy(cg))
                    elif p == "random":
                        losses, hits = self._rollout_metrics(**kw, choose=self._random)
                    else:
                        raise ValueError(f"unknown eval policy {p!r}")
                    for t in range(T):
                        sums[p][0][t] += losses[t].sum().item()
                        sums[p][1][t] += hits[t].float().sum().item()
                n += images.shape[0]
        if was_training:
            model.head.train()
        out: dict[str, float] = {}
        for p in policies:
            for t in range(T):
                out[f"loss_t{t}_{p}"] = sums[p][0][t] / max(n, 1)
                out[f"hit_t{t}_{p}"] = sums[p][1][t] / max(n, 1)
            out[f"loss_final_{p}"] = out[f"loss_t{T - 1}_{p}"]
            out[f"hit_final_{p}"] = out[f"hit_t{T - 1}_{p}"]
        return out

    # --- checkpoint ----------------------------------------------------------
    def model_config(self, model):
        from dataclasses import asdict
        core = getattr(model, "module", model)
        return {"task": "search", "backbone_name": core.backbone_name, "canvit": asdict(core.canvit.cfg),
                "glimpse_grid_size": core.glimpse_grid_size, "n_classes": len(self._classes()),
                "classes": list(self._classes()), "head_kw": dict(core.head_kw),
                "model_repo": self.cfg.model_repo, "canvas_grid": self.canvas_grid(model)}

    def checkpoint_metadata(self, model):
        from canvit.checkpoint import downstream_pretrain_view_scale
        return {"task": "search", "scene_size": self.cfg.scene_size, "n_timesteps": self.cfg.n_timesteps,
                "model_repo": str(self.cfg.model_repo),
                "pretrain_view_scale": downstream_pretrain_view_scale(
                    patcher_name=getattr(model.canvit.cfg, "patcher_name", None),
                    foveated_scale=self.cfg.foveated_scale)}


__all__ = ["BoundSearchTask", "SearchRunTask", "POLICY_FEATURE_GROUPS"]
