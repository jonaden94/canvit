"""ADE20K semantic-segmentation task — engine-facing core (design §3.1, §0 table).

Readout = per-glimpse ``canvas_hidden`` [B,G,G,D]; the segmentation probe head is
applied OUTSIDE the backbone forward (so a frozen backbone still trains the head —
the probe cell). Glimpse routing follows canvit_eval/the existing ade20k rollout:
uniform → pre-crop at the training-matched pixel size; foveated/square → full image.

Reuses the ported, tested helpers (``consumes_full_image`` / ``derive_glimpse_px`` /
``ce_loss``) rather than duplicating them. The run-level ``Task`` wrapper
(build_model via ``from_pretrained_with_new_probe`` / ``from_pretrained_with_probe``,
loaders, mIoU eval) lands with the neutral loop.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from canvit.ade20k.data import IGNORE_LABEL
from canvit.ade20k.metrics import ce_loss
from canvit.core import CanViTForSemanticSegmentation, RecurrentState, sample_at_viewpoint
from canvit.core.policy.features import FEATURE_GROUPS
from canvit.harness.rollout import GlimpseOut, TaskLoss
from canvit.harness.rollout.episode import consumes_full_image, derive_glimpse_px
from canvit.harness.rollout.viewpoint import ViewpointType

log = logging.getLogger(__name__)

# ade20k has a spatial segmentation probe, so its scorer reads the full feature set
# INCLUDING probe entropy (ent / ent_delta) — the RL repo's canonical seg-policy features.
POLICY_FEATURE_GROUPS: tuple[str, ...] = FEATURE_GROUPS


class BoundAde20kTask:
    """Per-batch ADE20K :class:`RolloutTask`. Binds this batch's masks; holds the
    seg wrapper (``.canvit`` + ``.head``) and the glimpse routing derived once."""

    def __init__(
        self, *, seg: CanViTForSemanticSegmentation, masks: Tensor, canvas_grid: int,
        glimpse_px: int | None = None, reward_score_res: int | None = 128,
    ):
        self.seg = seg
        self.masks = masks  # [B, H, W] long
        self.canvas_grid = canvas_grid
        self.reward_score_res = reward_score_res
        self.full_image = consumes_full_image(seg)
        self.glimpse_px = None if self.full_image else derive_glimpse_px(seg, glimpse_px)

    def forward_glimpse(
        self, *, model: Any, images: Tensor, state: RecurrentState,
        viewpoint: Any, backbone_no_grad: bool,
    ) -> GlimpseOut:
        seg = getattr(model, "module", model)  # unwrap DDP; ade20k steps .canvit directly
        B = images.shape[0]
        model_input = images if self.full_image else sample_at_viewpoint(
            spatial=images, viewpoint=viewpoint, glimpse_size_px=self.glimpse_px,
        )
        ctx = torch.no_grad() if backbone_no_grad else nullcontext()
        with ctx:
            # collect_write_snapshots is a pure side effect (extra tensor refs kept in a
            # list, no RNG/compute difference) -- always on, like local_patches below,
            # so the "depth_feat" policy group needs no per-task opt-in flag.
            out = seg.canvit(
                image=model_input, state=state, viewpoint=viewpoint, collect_write_snapshots=True
            )
            hidden = seg.canvit.get_spatial(out.state.canvas).view(B, self.canvas_grid, self.canvas_grid, -1)
        return GlimpseOut(
            readout=hidden, state=out.state, vpe=out.vpe,
            local_patches=out.local_patches, write_snapshots=out.write_snapshots,
        )

    def _logits(self, readout: Tensor) -> Tensor:
        return self.seg.head(readout.float())  # [B, C, G, G]

    def step_loss(self, readout: Any) -> TaskLoss:
        return TaskLoss(combined=ce_loss(self._logits(readout), self.masks))

    def per_image_loss(self, readout: Any) -> Tensor:
        """The POLICY REWARD's raw material (rollout.py:268/310/327) — nothing else reads
        it. Shared with `rl_train.ce_from_logits` via `reward_ce`, so the reward cannot
        depend on which trainer computes it (doc 15 §A gap #5, closed 2026-07-30)."""
        from canvit.ade20k.metrics import reward_ce
        return reward_ce(self._logits(readout), self.masks, score_res=self.reward_score_res)


class Ade20kRunTask:
    """Run-level ADE20K :class:`~canvit.harness.run.RunTask` — the full seam
    ``harness.run`` drives: model construction (pretrained backbone + fresh probe),
    ADE20K loaders, mIoU eval, joint-policy assembly, and per-batch ``bind`` into the
    :class:`BoundAde20kTask` engine core. Config is composed (design D-B): the task holds
    its ``Ade20kConfig``; the policy config for joint runs is passed in (``rl``) so the
    live ade20k config stays untouched.
    """

    name = "ade20k"

    def __init__(self, cfg, *, rl=None):
        self.cfg = cfg
        self.rl = rl  # JointPolicyConfig | None; only needed for train_policy runs

    @property
    def best_metric(self):
        """Eval key the harness MAXIMIZES for `best.pt`, and it follows the eval policy.

        Probe runs: last-timestep mIoU, what `ade20k/train.py` selects on
        (`probe.best_last_miou`). Policy-deploy runs: ``neg_ce_mean`` = -mean(t1..tH)
        val CE, because that is the rule the CanViT-PyTorch-RL qband band is defined by
        (`docs/qband_results.md`) — selecting a policy on mIoU instead would put our
        checkpoints on a different axis from the reference band we are trying to match.
        Negated because the harness loop only maximizes."""
        return "neg_ce_mean" if self.cfg.eval_policy == "policy" else "miou_final"

    # --- capabilities & defaults ------------------------------------------
    def caps(self):
        from canvit.harness.spec import TaskCaps
        # supports_ddp=False: see build_loaders — ADE20K is a map-style dataset behind a
        # plain shuffling DataLoader, with no rank sharding, so multi-GPU is refused rather
        # than silently run on overlapping samples.
        return TaskCaps(has_head=True, supports_policy=True, supports_ddp=False)

    def default_spec(self):
        """``frozen`` (default) = the historical frozen-backbone probe; ``finetune`` =
        the whole model end to end. Fixed horizon = n_timesteps either way.

        The LR schedule is ``warmup_onecycle``, which reproduces the standalone probe's
        AdamW + ``WarmupOneCycleLR`` step for step (``ade20k/data.make_optimizer_and_scheduler``
        with the same max_steps / warmup_steps / warmup_lr_ratio). Structured like
        ``tasks/in1k/task.py::default_spec`` so both downstream tasks map ``cfg.mode``
        onto a spec identically."""
        from canvit.harness.spec import GroupOptim, ScheduleSpec, TrainSpec, fixed_horizon_bptt
        go = GroupOptim(
            lr=self.cfg.peak_lr, weight_decay=self.cfg.weight_decay,
            schedule=ScheduleSpec(kind="warmup_onecycle", warmup_steps=self.cfg.warmup_steps,
                                  total_steps=self.cfg.max_steps,
                                  warmup_lr_ratio=self.cfg.warmup_lr_ratio))
        # The task loss must reach the trunk in finetune, so the rollout keeps a graph
        # (one over all glimpses, or one per cfg.bptt_chunk_size); probe mode runs the
        # backbone under no_grad, where bptt is a no-op — see fixed_horizon_bptt.
        finetune = self.cfg.mode == "finetune"
        bptt = fixed_horizon_bptt(frozen=not finetune, horizon=self.cfg.n_timesteps,
                                  chunk_size=self.cfg.bptt_chunk_size)
        if finetune:
            return TrainSpec.finetune(bptt=bptt, optim={"backbone": go, "head": go})
        return TrainSpec.probe(bptt=bptt, optim={"head": go})

    # --- construction ------------------------------------------------------
    def build_model(self, device, prior_model_config=None):
        # prior_model_config is unused: the backbone arch comes from the HF repo the probe
        # was built on, so a resume rebuilds the same model from cfg.model_repo already.
        if self.cfg.probe_repo:
            # Finetune from a PUBLISHED probe (specialize's `init_probe_repo`). A
            # finetune from a random head at the finetune LR crawls — the in1k twin of
            # this was a real, costly bug (8f780ba).
            #
            # Honoured in FROZEN mode too (was: finetune-only). POLICY training needs it:
            # the reward is the fraction of the PROBE's CE a glimpse removes, so a random
            # head makes the reward pure noise — and `--preset policy_only` runs in frozen
            # mode, where this used to fall through to a fresh head silently. `rl_train.py`
            # always loads a trained probe (`from_pretrained_with_probe`), so this is what
            # reproducing its recipe requires. Probe TRAINING leaves probe_repo unset and
            # is unaffected.
            log.info("Initialising head from published probe %s (mode=%s)",
                     self.cfg.probe_repo, self.cfg.mode)
            seg = CanViTForSemanticSegmentation.from_pretrained_with_probe(
                pretrained_repo=self.cfg.model_repo, probe_repo=self.cfg.probe_repo,
            ).to(device)
        else:
            if self.cfg.mode == "finetune":
                log.warning(
                    "FINETUNE mode with a FRESH RANDOM head (no --cfg.probe-repo). The head "
                    "starts at chance and the backbone is already being updated at the "
                    "finetune LR, so early steps drag a trained trunk toward a random "
                    "readout. Pass --cfg.probe-repo to fuse a published probe instead."
                )
            seg = CanViTForSemanticSegmentation.from_pretrained_with_new_probe(
                pretrained_repo=self.cfg.model_repo, num_classes=self._num_classes(),
                dropout=self.cfg.dropout, use_ln=True,
            ).to(device)
        # The OOD footgun that silently ruined run 15025338 (ade20k/train.py:97): a
        # foveated backbone derives its fixation window as `fix_size = scale * H`, so a
        # probe rollout at a scale the backbone never saw makes EVERY glimpse
        # out-of-distribution. It does not crash — mIoU just falls as glimpses
        # accumulate. Warn loudly, exactly as the standalone does.
        if consumes_full_image(seg):
            fs = self.cfg.foveated_scale
            detail = (f"fixed_scale={fs.fixed_scale}" if fs.mode == "fixed"
                      else f"{fs.distribution} in [{fs.min_scale}, {fs.max_scale}]")
            log.warning(
                "  foveated view scale: mode=%s, %s — this MUST match the backbone's "
                "pretraining scale or every glimpse is out of distribution "
                "(symptom: mIoU falls as glimpses accumulate).", fs.mode, detail)
        return seg, seg.head

    def _num_classes(self):
        from canvit.ade20k.data import NUM_CLASSES
        return NUM_CLASSES

    def canvas_grid(self, model):
        if self.cfg.canvas_grid is not None:
            return self.cfg.canvas_grid
        return self.cfg.scene_size // model.canvit.backbone.patch_size_px

    def is_foveated(self, model):
        return consumes_full_image(model)

    def branches(self):
        return [ViewpointType.FULL if self.cfg.train_start_full else ViewpointType.RANDOM]

    def build_val_loader(self):
        """The val loader alone — the seam standalone evaluation uses, so it cannot drift
        from what training-time validation measures."""
        from canvit.ade20k.data import make_ade20k_val_loader
        return make_ade20k_val_loader(self.cfg)

    def build_loaders(self, *, world_size, rank):
        # SINGLE-GPU ONLY, asserted here as well as via caps().supports_ddp (which fires
        # earlier, in check_spec): `make_ade20k_loaders` builds a map-style
        # DataLoader(shuffle=True) and takes NO world_size/rank, so under DDP each rank
        # would sample independently from the whole dataset — overlapping draws, not
        # disjoint shards, with no error to show for it. A real multi-GPU ade20k needs a
        # DistributedSampler plus set_epoch plumbing in run.py::_infinite.
        if world_size > 1:
            raise RuntimeError(
                f"ade20k does not support DDP (world_size={world_size}): its loader cannot "
                "shard by rank. Run it on one GPU (NGPU=1, --ntasks-per-node=1).")
        from canvit.ade20k.data import make_ade20k_loaders
        return make_ade20k_loaders(self.cfg)

    def build_selector(self, *, device, canvas_grid, is_foveated):
        from canvit.harness.rollout.selector import RandomSelector
        return RandomSelector(is_foveated=is_foveated, foveated_scale=self.cfg.foveated_scale,
                              min_viewpoint_scale=self.cfg.min_vp_scale)

    def build_policy(self, model, *, device, canvas_grid, generator):
        from canvit.harness.config import JointPolicyConfig
        from canvit.harness.policy import build_policy
        # Only reached on a policy run. The reward is the fraction of the PROBE's CE a
        # glimpse removes, so an untrained head makes it noise — and nothing downstream
        # would fail, the scorer would just learn from garbage.
        if self.cfg.mode == "frozen" and not self.cfg.probe_repo:
            log.warning(
                "POLICY training with a FRESH RANDOM segmentation head (no --cfg.probe-repo). "
                "The scorer's reward is the fraction of the probe's CE each glimpse removes, "
                "so with an untrained probe that reward is noise and the run will look "
                "healthy while learning nothing. Pass --cfg.probe-repo (rl_train.py's default "
                "is probe-ade20k-40k-s512-c%d-in21k).", canvas_grid)
        # The published qband band / EG-C2F numbers were measured under SQUISH — it is
        # CanViT-PyTorch-RL's whole measurement contract. center_crop is a valid protocol
        # and the right default for new work, but it shifts CE by ~0.016 (exp27 arm A),
        # which silently dwarfs the band's 0.0007 seed spread. Warn rather than force:
        # the mode stays the user's choice.
        if self.cfg.resize_mode != "squish":
            log.warning(
                "POLICY training with resize_mode=%r. The CanViT-PyTorch-RL qband band "
                "(0.6853 +- 0.0007 mean t1-t4 CE) and the EG-C2F baselines were measured "
                "under SQUISH; under %r the numbers are internally consistent but NOT "
                "comparable to them (center_crop measured ~0.016 CE lower in exp27). Pass "
                "--cfg.resize-mode squish for band-comparable results.",
                self.cfg.resize_mode, self.cfg.resize_mode)
        rl = self.rl or JointPolicyConfig(feature_groups=POLICY_FEATURE_GROUPS)
        return build_policy(
            canvit=model.canvit, rl=rl, feature_groups=POLICY_FEATURE_GROUPS, device=device,
            canvas_grid=canvas_grid, min_viewpoint_scale=self.cfg.min_vp_scale,
            foveated_scale=self.cfg.foveated_scale, generator=generator, encode_model=model,
        )

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
            params = list(joint.scorer.parameters())
            if joint.critic is not None:  # Q-Prop: critic rides the same param group/LR
                params += list(joint.critic.parameters())
            groups["policy"] = params
        return groups

    def resume_start_step(self, payload, scheduler):
        return scheduler.last_epoch  # map dataset: steps == scheduler.step() calls

    def resume_state(self):
        return {}  # re-iterable map dataset: nothing to carry across jobs

    # --- per-batch (engine-facing) ----------------------------------------
    def batch_images(self, batch, device):
        return batch[0].to(device, non_blocking=True)

    def bind(self, batch, device, *, model, head):
        _, masks = batch
        return BoundAde20kTask(seg=model, masks=masks.to(device),
                               canvas_grid=self.canvas_grid(model), glimpse_px=self.cfg.glimpse_px,
                               reward_score_res=self.cfg.reward_score_res)

    # --- visualization (specialize's segmentation overlay, restored) -------
    def viz_frame(self, *, model, images, gout, viewpoint, loss):
        """Branch-0 capture for the segmentation figure: the first ``viz_samples`` canvas
        features and this glimpse's argmax prediction, both moved to CPU. The engine fires
        this per glimpse; only t0 and the last one get drawn."""
        seg = getattr(model, "module", model)
        n = min(self.cfg.viz_samples, gout.readout.shape[0])
        hidden = gout.readout[:n].detach().float()
        with torch.no_grad():
            pred = seg.head(hidden).argmax(1)
        return hidden.cpu(), pred.cpu()

    def render_viz(self, viz, *, batch, run_dir, step):
        """Training-batch segmentation figure -> ``{run_dir}/visualization/seg_train/``."""
        from canvit.ade20k.viz import make_seg_viz_figure
        from canvit.harness.viz.disk import save_figure

        if not viz.frames:
            return
        hidden, preds = zip(*viz.frames)
        n = hidden[0].shape[0]
        images, masks = batch
        fig = make_seg_viz_figure(hidden=hidden, preds=preds, images=images[:n], masks=masks[:n])
        save_figure(fig, run_dir, "seg_train", step)

    def _render_val_viz(self, head, hidden, images, masks, run_dir, step):
        """Same figure for the first val batch -> ``{run_dir}/visualization/seg_val/``.
        Diagnostic only: a plotting failure must never abort validation."""
        from canvit.ade20k.viz import make_seg_viz_figure
        from canvit.harness.viz.disk import save_figure

        try:
            n = min(self.cfg.viz_samples, images.shape[0])
            pair = [hidden[0][:n].float(), hidden[-1][:n].float()]
            preds = [head(h).argmax(1).cpu() for h in pair]
            fig = make_seg_viz_figure(hidden=[h.cpu() for h in pair], preds=preds,
                                      images=images[:n].cpu(), masks=masks[:n].cpu())
            save_figure(fig, run_dir, "seg_val", step)
        except Exception:
            log.exception("ade20k val viz failed at step %d (validation continues)", step)

    def _policy_rollout(self, *, model, images, joint, T, canvas_grid, amp):
        """Closed-loop deploy rollout for the LEARNED policy: the scorer picks each glimpse
        by argmax from the live canvas."""
        from canvit.harness.rollout.eval_viewpoints import deploy_rollout_viewpoints

        return self._closed_loop_rollout(
            model=model, images=images, T=T, canvas_grid=canvas_grid, amp=amp,
            drive=lambda advance: deploy_rollout_viewpoints(
                joint=joint, advance=advance, t0_type=ViewpointType.FULL,
                batch_size=images.shape[0], device=images.device, n=T))

    def _entropy_c2f_rollout(self, *, model, images, T, canvas_grid, amp):
        """Closed-loop rollout for EG-C2F — the paper's strongest heuristic baseline. A
        fresh chooser per batch: it carries per-rollout `visited` state."""
        from canvit.harness.rollout.eval_viewpoints import closed_loop_rollout, entropy_c2f_chooser

        chooser = entropy_c2f_chooser(seg=model, batch_size=images.shape[0],
                                      device=images.device, canvas_grid=canvas_grid)
        return self._closed_loop_rollout(
            model=model, images=images, T=T, canvas_grid=canvas_grid, amp=amp,
            drive=lambda advance: closed_loop_rollout(chooser=chooser, advance=advance, n=T))

    def _closed_loop_rollout(self, *, model, images, T, canvas_grid, amp, drive):
        """Shared glimpse-stepping for every closed-loop eval policy. Returns
        canvas_hidden per timestep, exactly like the open-loop ``rollout_canvas_hidden``
        it mirrors, so all the metric code below is shared and never branches on which
        policy produced the trajectory. ``drive(advance)`` supplies the trajectory."""
        B = images.shape[0]
        full_image = consumes_full_image(model)
        px = None if full_image else derive_glimpse_px(model, self.cfg.glimpse_px)
        hidden: list[Tensor] = []

        def advance(vp, state, t):
            if state is None:  # t0: the rollout owns its own state init
                state = model.init_state(batch_size=B, canvas_grid_size=canvas_grid)
            model_input = images if full_image else sample_at_viewpoint(
                spatial=images, viewpoint=vp, glimpse_size_px=px)
            with amp:
                out = model.canvit(image=model_input, state=state, viewpoint=vp)
                hidden.append(model.canvit.get_spatial(out.state.canvas)
                              .view(B, canvas_grid, canvas_grid, -1))
            return out.state

        drive(advance)
        return hidden

    # --- eval & checkpoint -------------------------------------------------
    @torch.no_grad()
    def evaluate(self, *, model, head, val_loader, device, step, tracker=None, run_dir=None,
                 joint=None):
        # tracker unused: this task returns its scalars for the caller to log. run_dir is
        # the sink for the segmentation figure (specialize's `viz_val`), rendered from the
        # FIRST val batch on cfg.viz_every — the rollout it already runs, no recomputation.
        """mIoU per timestep over the val set (the historical ade20k eval), reusing the
        tested rollout + probe-eval helpers. Returns t0 / final / mean mIoU."""
        from canvit.ade20k.data import IGNORE_LABEL, NUM_CLASSES
        from canvit.ade20k.metrics import eval_probe_on_batch, mIoUAccumulator
        from canvit.ade20k.rollout import rollout_canvas_hidden
        from canvit.harness.rollout.eval_viewpoints import open_loop_viewpoints, resolve
        T = self.cfg.n_timesteps
        cg = self.canvas_grid(model)
        is_fov = consumes_full_image(model)
        eval_policy = resolve(self.cfg.eval_policy, task="ade20k", is_foveated=is_fov)
        was_training = model.head.training
        model.head.eval()
        ious = [mIoUAccumulator(NUM_CLASSES, IGNORE_LABEL, device) for _ in range(T)]
        # Optional per-(image, class, timestep) counts, riding the preds the accumulator
        # already computes (see per_row_iou.py). Off by default; nothing below runs then.
        rows: list[list] | None = [[] for _ in range(T)] if self.cfg.per_row_iou_out else None
        # CE is accumulated ONLY for a deployed policy: it is the qband selection metric
        # (see best_metric), and it needs full-resolution logits, which for B x 150 x 512
        # x 512 is a multi-GB tensor the mIoU path never materializes (it upsamples the
        # argmax instead). Probe runs keep paying nothing for it.
        ce_sums = [0.0] * T
        n_images = 0
        amp = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
        do_viz = (run_dir is not None and self.cfg.viz_every
                  and step % self.cfg.viz_every == 0)
        for vb, (vi, vm) in enumerate(val_loader):
            if self.cfg.limit_val_batches is not None and vb >= self.cfg.limit_val_batches:
                break
            vi, vm = vi.to(device), vm.to(device)
            if eval_policy == "policy":
                hidden = self._policy_rollout(model=model, images=vi, joint=joint, T=T,
                                              canvas_grid=cg, amp=amp)
            elif eval_policy == "entropy_coarse_to_fine":
                hidden = self._entropy_c2f_rollout(model=model, images=vi, T=T,
                                                   canvas_grid=cg, amp=amp)
            else:
                vps = open_loop_viewpoints(
                    eval_policy, batch_size=vi.shape[0], device=device, n=T, is_foveated=is_fov,
                    foveated_scale=self.cfg.foveated_scale, min_scale=self.cfg.min_vp_scale,
                    max_scale=self.cfg.max_vp_scale,
                    foveated_eval_scale=getattr(self.cfg.foveated_scale, "fixed_scale", 1.0),
                    override_scale=self.cfg.eval_override_scale,
                )
                with amp:
                    hidden = rollout_canvas_hidden(seg=model, images=vi, viewpoints=vps,
                                                   canvas_grid=cg, glimpse_px=self.cfg.glimpse_px)
            for t in range(T):
                preds = eval_probe_on_batch(model.head, hidden[t], vm, ious[t])
                if rows is not None:
                    from canvit.ade20k.per_row_iou import batch_confusion
                    rows[t].append([x.cpu() for x in batch_confusion(preds, vm, NUM_CLASSES)])
                if eval_policy == "policy":
                    ce_sums[t] += self._full_res_ce(model.head, hidden[t], vm).sum().item()
            if eval_policy == "policy":
                n_images += vi.shape[0]
            if do_viz:  # first batch only
                do_viz = False
                self._render_val_viz(model.head, hidden, vi, vm, run_dir, step)
        if rows is not None:
            import torch as _torch

            from canvit.ade20k.per_row_iou import write_rows
            # rows[t] is a list over BATCHES of [inter, union, gt_area], each [B, C], and
            # the last batch is ragged (2000 val images, batch 32) -- so concatenate along
            # the image axis, then stack the timesteps: [T, N, C] for each of the three.
            stacked = [_torch.stack([_torch.cat([b[i] for b in rows[t]]) for t in range(T)])
                       for i in range(3)]
            write_rows(self.cfg.per_row_iou_out, *stacked,
                       mask_resolution_px=self.cfg.scene_size,
                       resize_mode=self.cfg.resize_mode,
                       extra={"step": step, "eval_policy": eval_policy, "n_timesteps": T})
        mious = [m.compute() for m in ious]
        if was_training:
            model.head.train()
        # EVERY timestep, not just the endpoints: mIoU-vs-glimpse-count is the whole
        # point of a canvas probe (and how the foveated OOD symptom shows up — mIoU
        # FALLING as glimpses accumulate). ade20k/train.py:179 logs val_miou_t{t} for
        # all t; the caller namespaces these as eval/miou_t{t}.
        out = {f"miou_t{t}": v for t, v in enumerate(mious)}
        out["miou_final"] = mious[-1]   # the best-checkpoint key for probe runs
        out["miou_mean"] = sum(mious) / T
        if eval_policy == "policy":
            # t indices follow the reference: t0 is the full-scene anchor, so the band is
            # the mean over t1..t{T-1} — the glimpses the policy actually chose.
            ces = [s / max(n_images, 1) for s in ce_sums]
            out.update({f"ce_t{t}": v for t, v in enumerate(ces)})
            ce_mean = sum(ces[1:]) / max(len(ces) - 1, 1)
            out["ce_mean"] = ce_mean
            out["neg_ce_mean"] = -ce_mean   # the best-checkpoint key (loop maximizes)
        return out

    @staticmethod
    def _full_res_ce(head, hidden, masks):
        """Per-image CE at the MASK resolution — the reference's deploy metric
        (`ade20k/rl_train.py::ce_from_logits` with score_res=None). Logits are upsampled
        to the mask rather than the mask downsampled to the logits, so this is the
        paper-protocol number and not the coarse training-grid one."""
        logits = head(hidden.float())
        logits = F.interpolate(logits, size=masks.shape[1:], mode="bilinear", align_corners=False)
        per_px = F.cross_entropy(logits, masks, ignore_index=IGNORE_LABEL, reduction="none")
        valid = (masks != IGNORE_LABEL).float()
        return (per_px * valid).flatten(1).sum(1) / valid.flatten(1).sum(1).clamp_min(1.0)

    def model_config(self, model):
        from dataclasses import asdict

        # `model_repo` alone is a POINTER, and a pointer is not a description: it can move,
        # be unreadable by someone else, or -- for a finetune -- no longer describe the
        # model that was trained. So record the architecture too, as exactly the
        # constructor arguments CanViTForSemanticSegmentation.from_checkpoint needs. Read
        # off the model rather than the config, so it cannot disagree with the weights.
        core = getattr(model, "module", model)
        return {"task": "ade20k", "num_classes": self._num_classes(),
                "canvas_grid": self.canvas_grid(model), "model_repo": self.cfg.model_repo,
                "backbone_name": core.backbone_name, "canvit": asdict(core.canvit.cfg),
                "glimpse_grid_size": core.glimpse_grid_size,
                "dropout": core.head.dropout_p, "use_ln": core.head.use_ln}

    def checkpoint_metadata(self, model):
        from canvit.checkpoint import downstream_pretrain_view_scale
        return {"task": "ade20k", "scene_size": self.cfg.scene_size,
                "n_timesteps": self.cfg.n_timesteps,
                "model_repo": str(self.cfg.model_repo),
                "pretrain_view_scale": downstream_pretrain_view_scale(
                    patcher_name=getattr(model.canvit.cfg, "patcher_name", None),
                    foveated_scale=self.cfg.foveated_scale)}


__all__ = ["POLICY_FEATURE_GROUPS", "BoundAde20kTask", "Ade20kRunTask"]
