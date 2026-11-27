"""Visual-search task: the per-batch engine seam and the run-level ``RunTask``.

Readout = per-glimpse canvas tokens ``[B, G, G, D]``; the target-conditioned head is
applied OUTSIDE the backbone forward (so a frozen backbone still trains it). Foveated
backbones only: the patcher consumes the full image and foveates it itself.

Design: ``canvit_claude/canvit/search/00-design.md``. Default regime: backbone frozen, task head and
policy trained JOINTLY (``default_spec``).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import nullcontext
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from canvit.core import RecurrentState, Viewpoint
from canvit.core.patcher.foveated import fix_backbone_retina
from canvit.harness.rollout import GlimpseOut, TaskLoss
from canvit.harness.rollout.episode import consumes_full_image
from canvit.harness.rollout.selector import RandomSelector
from canvit.harness.rollout.viewpoint import ViewpointType

from .config import SearchConfig, search_policy_config
from .loss import (
    argmax_centres,
    box_centres,
    inside_box,
    per_image_loss,
    percent_label,
    presence_loss,
    square_overlaps_box,
    target_map,
)

log = logging.getLogger(__name__)


POLICY_FEATURE_GROUPS: tuple[str, ...] = ("ln_feat",)  # the plain canvas, no handcrafted maps
ALL_RETINAS = "all"  # --cfg.retina: use a multi-retina backbone through every retina
TRAIN_LENGTH_UNIT = 2  # prompt-cued search's random training length grows in distill's chunks of 2 glimpses


def _canvas_hidden(model: Any, state: RecurrentState, canvas_grid: int) -> Tensor:
    b = state.canvas.shape[0]
    return model.canvit.get_spatial(state.canvas).view(b, canvas_grid, canvas_grid, -1)


class BoundSearchTask:
    """Per-batch :class:`RolloutTask`: binds this batch's targets and boxes.

    Optional engine hooks, present only when they apply: ``step_weight`` (``supervise='last'``),
    ``policy_cond`` (a policy is trained), ``reward_mask`` (``condition='both'``: only the
    target-present trials carry a policy reward, Haoze's ``valid_mask``), ``saccade_beta``
    (a beta-conditioned saccade cost: each trial's beta [B], already inside ``policy_cond``) and
    ``rollout_length`` (prompt-cued search: this step's training glimpse count, set by ``SearchRunTask.bind``).

    Losses (Haoze's, with ``condition='both'``): the heatmap KL averaged over the
    target-present trials only, plus ``presence_weight`` x the presence loss over all trials."""

    def __init__(self, *, model: Any, targets: Any, boxes: Tensor, canvas_grid: int,
                 cfg: SearchConfig, policy_embed: Callable[[Any], Tensor] | None = None,
                 present: Tensor | None = None, saccade_beta: Tensor | None = None,
                 scene_target: Tensor | None = None, grad_touch: list[Tensor] | None = None):
        """``targets``: the batch's cue, class indices [B] or a ``prompt.Prompt`` (prompt-cued search).
        ``scene_target``: standardized DINOv3 patch features [B, G*G, T] of the scene (distill loss on).
        ``grad_touch`` (DDP only): parameters that must get a gradient TENSOR in this step's backward, even where
        this batch does not use them (``SearchRunTask._grad_touch``)."""
        self.model = model
        self.grad_touch = grad_touch
        self.scene_target = scene_target
        self.cfg = cfg
        self.targets = targets
        self.boxes = boxes
        self.canvas_grid = canvas_grid
        self.target = target_map(boxes, grid=canvas_grid, kind=cfg.loss, sigma=cfg.target_sigma)
        has_box = box_centres(boxes)[1].any(-1)
        self.present = has_box if present is None else present.bool()
        assert torch.equal(self.present, has_box), "present/absent flags disagree with the boxes"
        if cfg.condition == "both":
            self.reward_mask = self.present
        if cfg.supervise == "last":
            self.step_weight = self._last_only
        if policy_embed is not None:
            self.policy_cond = lambda: policy_embed(self.cue())
        if saccade_beta is not None:
            self.saccade_beta = saccade_beta

    def cue(self):
        """The cue for one use: class indices, or the prompt encoded afresh (``prompt.PromptInputs``)."""
        return resolve_cue(self.model, self.targets)

    @staticmethod
    def _last_only(t: int, n_glimpses: int) -> float:
        # The engine divides the summed per-glimpse loss by n_glimpses, so weight n at the
        # last glimpse makes the task loss exactly that glimpse's loss.
        return float(n_glimpses) if t == n_glimpses - 1 else 0.0

    def forward_glimpse(self, *, model: Any, images: Tensor, state: RecurrentState,
                        viewpoint: Any, backbone_no_grad: bool) -> GlimpseOut:
        core = getattr(model, "module", model)
        ctx = torch.no_grad() if backbone_no_grad else nullcontext()
        with ctx, prompt_context(core, self.cue()):
            out = core.canvit(image=images, state=state, viewpoint=viewpoint)
            hidden = _canvas_hidden(core, out.state, self.canvas_grid)
        return GlimpseOut(readout=hidden, state=out.state, vpe=out.vpe)

    def logits(self, readout: Tensor) -> Tensor:
        return self.model.search_head(readout.float(), self.cue())[0]

    def per_image_loss(self, readout: Tensor) -> Tensor:
        """Per-sample heatmap KL to the target map [B] -- also the policy reward's raw
        material. 0 on target-absent trials (no target map; masked out of the reward)."""
        return per_image_loss(self.logits(readout), self.target)

    def step_loss(self, readout: Tensor) -> TaskLoss:
        heat, pres = self.model.search_head(readout.float(), self.cue())
        kl = per_image_loss(heat, self.target)
        tp = self.present
        if bool(tp.all()):
            loss = kl.mean()  # target-present only: the expression from before the TA option
        else:
            loss = kl[tp].mean() if bool(tp.any()) else kl.sum() * 0.0
        if pres is not None:
            c = self.cfg
            pl = presence_loss(pres, tp, kind=c.presence_loss, gamma=c.presence_focal_gamma)
            loss = loss + c.presence_weight * pl.mean()
        if self.scene_target is not None:  # decision 7: the pretraining objective as a second loss
            scene = self.model.predict_scene(readout.float())
            loss = loss + self.cfg.prompt.distill_weight * F.mse_loss(scene.float(), self.scene_target)
        if self.grad_touch is not None:  # once per step: a zero term that reaches every listed parameter
            loss = loss + 0.0 * sum(p.sum() for p in self.grad_touch)
            self.grad_touch = None
        return TaskLoss(combined=loss)


def sdpa_guard(capability: tuple[int, int] | None, *, backprop_through_backbone: bool, math_sdpa: bool) -> bool:
    """On a V100 (sm_70) torch's memory-efficient SDPA backward returns garbage gradients (~1e12 at the first
    block, exp38 step-360448, fp32; CPU and the math kernel give ~10; measured 2026-11-26,
    ``canvit_claude/canvit/search/prompt_conditioning/distill_grad_probe.py``). Refuse such a run unless
    ``math_sdpa``; returns True when SDPA must be switched to the math kernel. Only sm_70 was measured."""
    if capability != (7, 0) or not backprop_through_backbone:
        return False
    if not math_sdpa:
        raise RuntimeError(
            "V100 (sm_70): gradients through the backbone's attention are garbage with the default SDPA kernels "
            "(~1e12). Run on an A100/H100, freeze the backbone in class-cued mode, or pass "
            "--cfg.prompt.v100-math-sdpa (math-only SDPA: slower, more memory).")
    return True


def resolve_cue(model: Any, cue: Any):
    """``prompt.PromptInputs`` -> a freshly encoded ``prompt.Prompt``; anything else unchanged."""
    from .prompt import PromptInputs
    return cue.encode(model.prompt.encoder) if isinstance(cue, PromptInputs) else cue


def prompt_context(model: Any, cue: Any):
    """The backbone conditioned on this batch's prompt (``prompt.PromptConditioning.use``); a no-op for a
    class-cued model or a class-index cue."""
    from .prompt import Prompt
    if isinstance(cue, Prompt) and getattr(model, "prompt", None) is not None:
        return model.prompt.use(cue)
    return nullcontext()


class SearchRunTask:
    """Run-level visual-search task for ``harness.run``."""

    name = "search"
    display_frame = "1680x1050"
    """How the training images sit in the scene (read by the scanpath evaluation): COCO-Search18's 1680x1050
    displays letterboxed into the scene (``SearchDataset``; the crop augmentation keeps the aspect)."""

    def __init__(self, cfg: SearchConfig, *, rl=None):
        self.cfg = cfg
        self.rl = rl if rl is not None else search_policy_config()
        self._joint = None
        self._bound: BoundSearchTask | None = None
        self.classes: list[str] | None = None
        self._start_step = 0  # release: where the trial stream resumes (resume_start_step)
        self._resume_saved: dict | None = None
        self._world_size: int | None = None
        self._text_encoder = None  # prompt-cued search: Talk2DINO, built on first use
        self._teacher = None  # prompt-cued search with the distill loss: DINOv3, built on first use
        self._ddp = False  # world_size > 1 (build_loaders): every trainable parameter gets a grad tensor per step
        self._train_step = 0  # the step bind() is called for (resumed in resume_start_step): seeds the train length

    @property
    def _prompt_mode(self) -> bool:
        return self.cfg.prompt.cue == "prompt"

    @property
    def best_metric(self) -> str:
        """Last-glimpse ``found`` (the task head's argmax inside the target box) under the
        learned policy, or under random fixations when no policy is trained (``--preset probe``)."""
        last = self.cfg.n_timesteps - 1
        if self._prompt_mode:  # decision 8: the mean found rate over glimpses t1..t(T-1)
            return "found_mean_policy" if self._joint is not None else "found_mean_random"
        return f"found_t{last}_policy" if self._joint is not None else f"found_t{last}_random"

    # --- capabilities & defaults ------------------------------------------
    def caps(self):
        from canvit.harness.spec import TaskCaps
        # COCO-Search18: a map-style dataset behind a shuffling DataLoader, no rank sharding (as ade20k); the
        # release's stream is sharded by rank (``_ddp_ok``).
        return TaskCaps(has_head=True, supports_policy=True, supports_ddp=self._ddp_ok())

    def _ddp_ok(self) -> bool:
        """Whether this run may train under DDP. Decision 9: only after a test of disjoint rank samples and exact
        resume for the whole task. Passed 2026-11-26 for the curated release (job 17581598, 2x A100, prompt-cued,
        visual share 0.9; ``canvit_claude/canvit/search/prompt_conditioning/ddp_check.{py,sbatch}``, record
        ``canvit_claude/canvit/search/05-prompt-conditioning.md``): each rank trains on rows rank::world of the
        pruned epoch (disjoint, union = the epoch), parameters stay identical across ranks, no trainable parameter
        reaches the all-reduce without a grad (``_grad_touch``), and a stopped + resumed run trains on exactly the
        same trials. The resume is DATA-exact, not RNG-exact (augmentation, policy sampling and the per-rank policy
        state restart; parameters at 2k differ from an uninterrupted run about as much as two uninterrupted runs
        differ from each other); the same guarantee as the validated distill DDP. COCO-Search18 has no rank
        sharding: refused."""
        return self.cfg.dataset == "release"

    def _policy_warmup_steps(self) -> int:
        pol = self.rl
        if pol.policy_warmup_steps > 0:
            return pol.policy_warmup_steps
        return int(max(pol.policy_warmup_frac, 0.0) * self.cfg.max_steps)

    def default_spec(self):
        """Frozen backbone; task head and policy trained JOINTLY; no BPTT (the backbone runs
        under no_grad, so there is no cross-timestep graph to keep). Prompt-cued search:
        ``_prompt_spec``."""
        from canvit.harness.spec import BpttSpec, GroupOptim, ScheduleSpec, TrainSpec
        if self._prompt_mode:
            return self._prompt_spec()
        head = GroupOptim(lr=self.cfg.peak_lr, weight_decay=self.cfg.weight_decay, schedule=self._task_schedule())
        pol = self.rl
        policy = GroupOptim(lr=pol.policy_lr, weight_decay=pol.policy_weight_decay, betas=pol.policy_betas,
                            schedule=ScheduleSpec(kind="warmup_constant", warmup_steps=self._policy_warmup_steps()))
        return TrainSpec(
            train_backbone=False, train_head=True, train_policy=True, task_weight=1.0, policy_weight=1.0,
            task_grad_to_backbone=False, policy_grad_to_backbone=False,
            bptt=BpttSpec(mode="none", horizon=self.cfg.n_timesteps),
            optim={"head": head, "policy": policy},
        )

    def _task_schedule(self):
        """The task groups' LR schedule after the warmup (``--cfg.lr-schedule``): ``cosine`` (Haoze) decays to 0 at
        ``max_steps``; ``constant`` holds the peak (the policy's schedule)."""
        from canvit.harness.spec import ScheduleSpec
        if self.cfg.lr_schedule == "cosine":
            return ScheduleSpec(kind="warmup_cosine", warmup_steps=self.cfg.warmup_steps,
                                total_steps=self.cfg.max_steps)
        return ScheduleSpec(kind="warmup_constant", warmup_steps=self.cfg.warmup_steps)

    def _prompt_spec(self):
        """Prompt-cued search (decision 5): the WHOLE model trains (backbone fine-tuned, one graph over the
        rollout), the policy with exp43's PG recipe (``--rl``), its gradient kept out of the backbone."""
        from canvit.harness.spec import BpttSpec, GroupOptim, ScheduleSpec, TrainSpec
        c, p, pol = self.cfg, self.cfg.prompt, self.rl
        # LEARNING RATES (owner decision 6, 2026-11-26): by default ONE LR (--cfg.peak-lr) for every group;
        # --cfg.prompt.{backbone,head,policy}-lr set them apart. For search fine-tuning a LOWER BACKBONE LR is
        # probably wise: exp40 put LR 4e-4 into the pretrained model and its canvas degraded. Claude: remind the
        # owner of this whenever he sets these values.
        lr = c.peak_lr
        backbone_lr = p.backbone_lr if p.backbone_lr is not None else lr
        head_lr = p.head_lr if p.head_lr is not None else lr
        policy_lr = p.policy_lr if p.policy_lr is not None else lr
        sched = self._task_schedule()  # backbone and heads: --cfg.lr-schedule
        # The training length is not this horizon: bind() hands each step's own length to the engine
        # (``BoundSearchTask.rollout_length``, ``_train_length``), also when --spec.train-backbone False rebuilds
        # the BPTT spec (mode 'none').
        return TrainSpec(
            train_backbone=True, train_head=True, train_policy=True, task_weight=1.0, policy_weight=1.0,
            task_grad_to_backbone=True, policy_grad_to_backbone=False,
            bptt=(BpttSpec(mode="full", horizon=c.n_timesteps) if p.bptt_chunk is None
                  else BpttSpec(mode="chunked", chunk_size=p.bptt_chunk, horizon=c.n_timesteps)),
            optim={"backbone": GroupOptim(lr=backbone_lr, weight_decay=c.weight_decay, schedule=sched),
                   "head": GroupOptim(lr=head_lr, weight_decay=c.weight_decay, schedule=sched),
                   "policy": GroupOptim(lr=policy_lr, weight_decay=pol.policy_weight_decay, betas=pol.policy_betas,
                                        schedule=ScheduleSpec(kind="warmup_constant",
                                                              warmup_steps=self._policy_warmup_steps()))},
        )

    # --- construction ------------------------------------------------------
    def _classes(self) -> list[str]:
        if self._prompt_mode:
            raise RuntimeError("prompt-cued search has no class list (no class index anywhere, decision 1)")
        if self.cfg.dataset == "release":
            raise ValueError("dataset='release' has no class-index cue (owner 2026-11-25): use --cfg.prompt.cue prompt")
        if self.classes is None:
            from .data import coco_search18_classes
            self.classes = coco_search18_classes(self.cfg.data_root)
        return self.classes

    def build_model(self, device, prior_model_config=None):
        from .model import CanViTForVisualSearch
        c = self.cfg
        if self._prompt_mode:
            model = self._build_prompt_model(device, prior_model_config)
        elif prior_model_config and "canvit" in prior_model_config:
            # A resume or a standalone eval: the checkpoint's OWN architecture (backbone + head);
            # the caller loads all its weights (strict), so cfg.model_repo is not read.
            saved = prior_model_config.get("classes")
            if saved is not None and list(saved) != list(self._classes()):
                raise ValueError(f"the checkpoint was trained on the classes {saved}, the data gives "
                                 f"{self._classes()}: the cue indices would not match.")
            model = CanViTForVisualSearch.from_model_config(prior_model_config).to(device)
        elif not c.model_repo:
            raise ValueError("--cfg.model-repo is required: a pretrained FOVEATED CanViT (HF id, local HF "
                             "dir or training .pt).")
        else:
            model = CanViTForVisualSearch.from_pretrained_backbone(
                c.model_repo, n_classes=len(self._classes()), cond_dim=c.cond_dim, kind=c.head,
                width=c.head_width, block_layers=c.head_block_layers, dropout=c.dropout,
                presence=c.condition == "both", levels=c.head_levels,
            ).to(device)
        if not consumes_full_image(model):
            raise ValueError(f"the search task supports FOVEATED backbones only; {c.model_repo} has patcher "
                             f"{getattr(model.canvit.cfg, 'patcher_name', '?')!r}.")
        self._apply_retina(model, source=c.model_repo)
        fs = c.foveated_scale
        log.warning("foveated view scale: mode=%s fixed_scale=%s -- this MUST match the backbone's "
                    "pretraining scale or every glimpse is out of distribution.", fs.mode, fs.fixed_scale)
        return model, model.head

    def _apply_retina(self, model, *, source) -> None:
        """``--cfg.retina``: one named retina of a multi-retina backbone (``fix_backbone_retina``),
        or ``all``: every retina, as distill trains -- the rollout engine draws each training
        sample's retina, ``evaluate`` validates each one (``_retina_names``)."""
        c = self.cfg
        if c.retina != ALL_RETINAS:
            fix_backbone_retina(model, c.retina, source)
            return
        n = getattr(model.canvit.patcher, "n_retinas", 1)
        if n < 2:
            raise ValueError(f"--cfg.retina {ALL_RETINAS} needs a multi-retina backbone; {source} has one retina.")
        log.info("multi-retina backbone: all %d retinas (drawn per training sample, each validated)", n)

    def _build_prompt_model(self, device, prior_model_config):
        from .model import CanViTForVisualSearch
        c, p = self.cfg, self.cfg.prompt
        if prior_model_config and "canvit" in prior_model_config:
            if prior_model_config.get("prompt_kw") is None:
                raise ValueError("--cfg.prompt.cue prompt, but the checkpoint is a class-cued search model")
            return CanViTForVisualSearch.from_model_config(prior_model_config).to(device)
        if not c.model_repo:
            raise ValueError("--cfg.model-repo is required: the pretrained FOVEATED CanViT (training .pt).")
        if not (p.adaln or p.tokens != "none" or p.condition_heads):
            raise ValueError("prompt-cued search needs at least one of: --cfg.prompt.adaln, prompt tokens, "
                             "--cfg.prompt.condition-heads (otherwise the heatmap never sees the prompt)")
        keep_scene = p.distill_weight > 0 or p.eval_cosine_baseline
        prompt_kw = {"adaln": p.adaln, "tokens": p.tokens, "token_mode": p.token_mode,
                     "condition_heads": p.condition_heads, "scene_grid": c.scene_size // 16 if keep_scene else None}
        return CanViTForVisualSearch.from_pretrained_backbone(
            c.model_repo, n_classes=None, prompt_kw=prompt_kw, **self._head_kw()).to(device)

    def _head_kw(self) -> dict:
        """The head's constructor arguments in prompt-cued search (``head_kw`` of the checkpoint)."""
        c, p = self.cfg, self.cfg.prompt
        kw = dict(cond_dim=c.cond_dim, kind=c.head, width=c.head_width, block_layers=c.head_block_layers,
                  dropout=c.dropout, presence=c.condition == "both", levels=c.head_levels)
        if c.head == "xattn":
            if not p.condition_heads:
                raise ValueError("--cfg.head xattn reads the prompt as its query: --cfg.prompt.no-condition-heads "
                                 "leaves it nothing to read")
            if c.dropout > 0:
                raise ValueError("--cfg.dropout is the U-Net head's Dropout2d; the xattn readout has none")
            kw |= {"readout_layers": p.readout_layers, "readout_heads": p.readout_heads}
        return kw

    def canvas_grid(self, model):
        if self.cfg.canvas_grid is not None:
            return self.cfg.canvas_grid
        return self.cfg.scene_size // model.canvit.backbone.patch_size_px

    def is_foveated(self, model):
        return consumes_full_image(model)

    def branches(self):
        return [ViewpointType.FULL]  # t0 = the centred fixation, as in the human trials and Haoze

    def build_val_loader(self):
        if self.cfg.dataset == "release":
            from .release import make_release_val_loader
            return make_release_val_loader(self.cfg)
        from .data import make_search_val_loader
        loader, self.classes = make_search_val_loader(self.cfg)
        return loader

    def build_loaders(self, *, world_size, rank):
        if world_size > 1 and not self._ddp_ok():
            raise RuntimeError("search supports DDP only on the curated release (--cfg.dataset release): the "
                               "COCO-Search18 loader has no rank sharding (SearchRunTask._ddp_ok).")
        self._ddp = world_size > 1
        if self._prompt_mode and self.cfg.dataset != "release":
            raise ValueError("prompt-cued TRAINING reads the curated release (--cfg.dataset release); "
                             "COCO-Search18 is its zero-shot test (harness.evaluate).")
        if self.cfg.dataset == "release":
            from .release import make_release_loaders
            self._check_release_resume(world_size)
            self._world_size = world_size
            return make_release_loaders(self.cfg, world_size=world_size, rank=rank, start_step=self._start_step)
        from .data import make_search_loaders
        train, val, self.classes = make_search_loaders(self.cfg)
        return train, val

    def _release_schedule(self, world_size: int) -> dict:
        """What fixes the release's trial stream: a resume continues it only under the same values."""
        from .release import release_meta
        return {"release": release_meta(self.cfg.release_dir)["name"], "seed": self.cfg.seed,
                "ddp_world_size": world_size, "batch_size": self.cfg.batch_size}

    def _check_release_resume(self, world_size: int) -> None:
        if self._resume_saved is None:
            return
        current = self._release_schedule(world_size)
        saved = {k: self._resume_saved.get(k) for k in current}
        if saved != current:
            raise RuntimeError(f"search release resume: the trial stream was {saved}, this run gives {current}; "
                               "resuming would skip or repeat trials. Seed from the checkpoint to start a new stream.")

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
        conditioned = c.prompt.condition_policy if self._prompt_mode else True
        jp = build_policy(
            canvit=model.canvit, rl=rl, feature_groups=POLICY_FEATURE_GROUPS, device=device,
            canvas_grid=canvas_grid, min_viewpoint_scale=0.05, foveated_scale=c.foveated_scale,
            generator=generator, encode_model=model, scorer_input_norm="none",
            film_cond_dim=c.cond_dim if conditioned else None,
        )
        if c.supervise == "last" and jp.reward.kind in ("fractional", "difference"):
            raise ValueError(
                f"reward={jp.reward.kind!r} needs the task loss at EVERY glimpse, but supervise='last' "
                "trains the head on the last glimpse only, so the early losses (and the rewards built "
                "from them) are noise. Use --cfg.supervise every, or --rl.reward terminal.")
        if self._prompt_mode:
            if conditioned:  # the prompt's 1x1 vector (detached: the policy gradient stays out of the prompt)
                jp.scorer.add_module("prompt_cond", nn.Linear(model.canvit.backbone.embed_dim, c.cond_dim).to(device))
        elif c.embedding == "separate":
            # The policy's own embedding rides on the scorer module, so the harness treats it as
            # policy parameters everywhere (requires_grad, grad clipping, DDP sync, checkpoint).
            jp.scorer.add_module("target_embed", nn.Embedding(len(self._classes()), c.cond_dim).to(device))
        self._joint = jp
        return jp

    def _policy_embed(self, model) -> Callable[[Tensor], Tensor] | None:
        jp = self._joint
        if jp is None:
            return None
        if self._prompt_mode:
            if not hasattr(jp.scorer, "prompt_cond"):
                return None
            return lambda cue: jp.scorer.prompt_cond(cue.vec.detach())
        if self.cfg.embedding == "shared":
            core = getattr(model, "module", model)
            return lambda t: core.head.policy_embedding(t, detach=self.cfg.detach_policy_embedding)
        return lambda t: jp.scorer.target_embed(t)

    def _policy_cond(self, model, targets: Any, beta: Tensor | None = None) -> Tensor | None:
        """The scorer's FiLM input for ``targets`` (class indices, or a ``prompt.Prompt``): the cue embedding,
        plus, for a beta-conditioned saccade cost, the features of each row's beta (None: the eval beta
        ``saccade_cost_beta_eval`` for every row). None: an unconditioned policy (prompt-cued,
        ``--cfg.prompt.no-condition-policy``; the harness's ``build_policy`` refuses that with a beta-conditioned
        saccade cost)."""
        embed = self._policy_embed(model)
        assert self._joint is not None
        if embed is None:
            return None
        cue = embed(targets)
        sc = self._joint.saccade_cost
        if not sc.conditioned:
            return cue
        if beta is None:
            beta = torch.full((cue.shape[0],), sc.beta_eval, device=cue.device)
        return torch.cat([cue, sc.features(beta).to(cue.dtype)], dim=-1)

    def policy_feature_groups(self):
        return POLICY_FEATURE_GROUPS

    def trainable_param_groups(self, *, model, head, joint, spec):
        dev = next(model.parameters()).device
        cap = torch.cuda.get_device_capability(dev) if dev.type == "cuda" else None
        # The prompt modules hook into the backbone, so their gradients pass through its attention too.
        if sdpa_guard(cap, backprop_through_backbone=spec.train_backbone or self._prompt_mode,
                      math_sdpa=self.cfg.prompt.v100_math_sdpa):
            for enable in (torch.backends.cuda.enable_flash_sdp, torch.backends.cuda.enable_mem_efficient_sdp,
                           torch.backends.cuda.enable_cudnn_sdp):
                enable(False)
            log.warning("V100: scaled_dot_product_attention restricted to the math kernel for this run")
        groups: dict[str, list] = {}
        if spec.train_backbone:
            groups["backbone"] = list(model.canvit.parameters())
        if spec.train_head:
            groups["head"] = list(model.head.parameters())
            if self._prompt_mode:  # the new prompt modules (and the kept scene head) train with the heads
                for name in ("prompt", "head_cond", "scene"):
                    m = getattr(model, name, None)
                    if m is not None:
                        groups["head"] += [q for q in m.parameters() if q.requires_grad]
        if spec.train_policy:
            assert joint is not None
            groups["policy"] = joint.policy_parameters()  # actor (+ SAC critics)
        return groups

    def resume_start_step(self, payload, scheduler):
        self._train_step = scheduler.last_epoch  # the training-length draw continues at this step
        if self.cfg.dataset == "release":
            # The release's trial stream resumes at position step x batch_size (release.EpochStream).
            self._resume_saved = (payload.get("metadata") or {}).get("resume_state") or {}
            self._start_step = scheduler.last_epoch
        return scheduler.last_epoch

    def resume_state(self):
        if self.cfg.dataset == "release" and self._world_size is not None:
            return self._release_schedule(self._world_size)
        return {}

    # --- per-batch ---------------------------------------------------------
    def batch_images(self, batch, device):
        return batch["image"].to(device, non_blocking=True)

    def bind(self, batch, device, *, model, head):
        core = getattr(model, "module", model)
        targets = self._cue_of(batch, device, core)
        policy_embed, beta = self._policy_embed(model), None
        jp = self._joint
        if jp is not None and jp.saccade_cost.conditioned:  # stage 2: one beta per trial
            beta = jp.saccade_cost.sample_beta(batch["present"].shape[0], device)
            policy_embed = lambda t: self._policy_cond(model, t, beta)  # noqa: E731
        bound = BoundSearchTask(
            model=core, targets=targets,
            boxes=batch["boxes"].to(device), canvas_grid=self.canvas_grid(model), cfg=self.cfg,
            policy_embed=policy_embed, present=batch["present"].to(device), saccade_beta=beta,
            scene_target=self._scene_target(batch, device, core), grad_touch=self._grad_touch(core))
        if self._prompt_mode:  # engine hook: this step's training length (validation keeps n_timesteps)
            bound.rollout_length = self._train_length(self._train_step)
        self._train_step += 1
        self._bound = bound  # this step's targets, for viz_frame
        return bound

    def _train_length(self, step: int) -> int:
        """Prompt-cued search's training glimpse count at ``step`` (owner 2026-11-26): ``prompt.train_timesteps``,
        or (None) canvit's distill paradigm unchanged: chunks of 2 glimpses, after each chunk continue with
        probability ``prompt.train_length_continue`` (0.5): P(n = 2k) = 0.5^k, mean 4, no cap (validation keeps
        ``n_timesteps``). A PRIVATE ``random.Random`` seeded by (seed, step), so the draw is the same at a resume and
        on every DDP rank, and Python's global ``random`` is never touched (distill draws from the global one,
        unseeded)."""
        import random
        c, p = self.cfg, self.cfg.prompt
        if p.train_timesteps is not None:
            return p.train_timesteps
        rng = random.Random(f"search-train-length/{c.seed}/{step}")
        n = TRAIN_LENGTH_UNIT
        while rng.random() < p.train_length_continue:
            n += TRAIN_LENGTH_UNIT
        return n

    def _grad_touch(self, model) -> list[Tensor] | None:
        """DDP: every trainable parameter (model and policy scorer) gets a gradient tensor in every step.
        ``harness`` all-reduces only the grads that are not None, rank by rank, so a parameter one rank's batch
        does not use (no visual row -> the visual prompt Linear; no text row -> the text one; ...) would make the
        ranks' all_reduce sequences differ (a hang or mixed-up tensors). A zero term (``BoundSearchTask.step_loss``)
        gives such parameters an exact zero gradient instead. Single GPU (None): nothing changes."""
        if not self._ddp:
            return None
        params = [p for p in model.parameters() if p.requires_grad]
        if self._joint is not None:
            params += [p for p in self._joint.policy_parameters() if p.requires_grad]
        return params

    # --- prompts (prompt-cued search) ------------------------------------------
    def _text_vectors(self, texts: list[str], device) -> Tensor:
        if self._text_encoder is None:
            from .text_encoder import TextEncoder
            self._text_encoder = TextEncoder(cache=True).to(device)
        return self._text_encoder(texts)

    def _cue_of(self, batch, device, model, prompt_type: str | None = None):
        """The batch's cue: class indices, or a ``prompt.Prompt``. ``prompt_type`` None = as the loader assigned
        it (``vp_valid``: visual), 'text' / 'visual' = that type for every row (val)."""
        if not self._prompt_mode:
            return batch["target"].to(device)
        n = batch["present"].shape[0]
        if prompt_type is None:
            visual = batch["vp_valid"] if "vp_valid" in batch else torch.zeros(n, dtype=torch.bool)
        else:
            visual = torch.full((n,), prompt_type == "visual")
        visual = visual.to(device)
        text = self._text_vectors(batch["text"], device) if bool((~visual).any()) else None
        vp = {k: batch[k].to(device, non_blocking=True) for k in ("vp_feat", "vp_cov") if k in batch}
        from .prompt import PromptInputs
        return PromptInputs(visual=visual, text=text, **vp)

    def _scene_target(self, batch, device, model) -> Tensor | None:
        """Standardized DINOv3 patch features of the scene (the distill loss's target), or None when off."""
        if not (self._prompt_mode and self.cfg.prompt.distill_weight > 0):
            return None
        return self._teacher_scene(batch["image"].to(device, non_blocking=True), model)

    @torch.no_grad()
    def _teacher_scene(self, images: Tensor, model) -> Tensor:
        if self._teacher is None:
            from canvit.core.teacher import load_teacher
            self._teacher = load_teacher("facebook/dinov3-vitb16-pretrain-lvd1689m", images.device)
        amp = (torch.autocast("cuda", dtype=torch.bfloat16) if images.device.type == "cuda" else nullcontext())
        with amp:
            raw = self._teacher.forward_norm_features(images).patches.float()
        return model.scene["std"](raw)

    @staticmethod
    def _prompt_rows(batch: dict, prompt_type: str) -> dict | None:
        """The val batch for one prompt type: all rows for 'text', the rows with a visual prompt for 'visual'
        (None if there is none)."""
        if prompt_type == "text":
            return batch
        keep = batch["vp_valid"].bool()
        if not bool(keep.any()):
            return None
        idx = keep.nonzero().flatten()
        return {k: ([v[i] for i in idx.tolist()] if isinstance(v, list) else v[idx]) for k, v in batch.items()}

    def _eval_prompt_types(self) -> list[str]:
        s = self.cfg.prompt.visual_share
        return ["text"] if s == 0 else ["visual"] if s == 1 else ["text", "visual"]

    # --- visualization (canvit/search/viz.py) ------------------------------
    def _viz_index(self, boxes: Tensor) -> Tensor:
        """The batch rows the figure shows: up to ``viz_samples // 2`` target-present rows,
        then up to as many target-absent rows (the figure's top and bottom half); a kind
        the batch lacks is filled up from the other. ``condition='tp'``: the first
        ``viz_samples`` rows."""
        n = min(self.cfg.viz_samples, boxes.shape[0])
        present = box_centres(boxes)[1].any(-1)
        tp = present.nonzero().flatten()
        ta = (~present).nonzero().flatten()
        if self.cfg.condition != "both":
            return torch.arange(n, device=boxes.device)
        k_tp = min(len(tp), max(n // 2, n - len(ta)))
        return torch.cat([tp[:k_tp], ta[:n - k_tp]])

    def _policy_logits(self, model, joint, state: RecurrentState, targets: Tensor, idx: Tensor,
                       beta: Tensor | None = None) -> Tensor:
        """[n, Gp, Gp] scorer logits on ``state`` for the batch rows ``idx``: the
        distribution the policy draws the NEXT fixation from. The scorer is deterministic
        (no BatchNorm, no dropout), so this equals the selector's own forward. The encoder's
        rolling reference is restored, so a mid-rollout call cannot shift later features."""
        st = RecurrentState(canvas=state.canvas[idx], recurrent_cls=state.recurrent_cls[idx],
                            policy_tokens=None if state.policy_tokens is None else state.policy_tokens[idx])
        enc = joint.policy_selector.encoder
        prev = enc.prev
        # the encoder's rolling reference holds the whole batch (or the batch-1 init
        # template at t0, which broadcasts); select the same rows
        enc.prev = tuple(None if x is None else x if x.shape[0] == 1 else x[idx] for x in prev)
        try:
            with torch.no_grad():
                cond = self._policy_cond(model, targets[idx], None if beta is None else beta[idx])
                scores = joint.scorer(enc(st).float(), cond=cond)
        finally:
            enc.prev = prev
        g = int(round(scores[0].numel() ** 0.5))
        return scores.reshape(len(idx), g, g).float()

    def _viz_capture(self, model, *, hidden, logits, presence, state, centers, targets, joint,
                     idx: Tensor, beta: Tensor | None = None) -> dict:
        """One glimpse of the figure for the batch rows ``idx`` (``_viz_index``), on CPU.
        ``logits`` / ``presence`` are already restricted to those rows."""
        return {"idx": idx.cpu(),
                "center": centers[idx].detach().float().cpu(),
                "logits": logits.detach().float().cpu(),
                "presence": presence.detach().float().cpu() if presence is not None else None,
                "canvas": hidden[idx].detach().float().cpu(),
                "policy_next": (self._policy_logits(model, joint, state, targets, idx, beta).cpu()
                                if joint is not None else None)}

    def _episode(self, frames: list[dict], *, images, boxes, targets, present, labels: list[str] | None = None):
        from .viz import SearchEpisode
        idx = frames[0]["idx"]
        images, boxes, present = (x.cpu()[idx] for x in (images, boxes, present))
        policy = None
        if frames[0]["policy_next"] is not None:
            nxt = torch.stack([f["policy_next"] for f in frames])  # nxt[t] chose fixation t+1
            policy = torch.cat([torch.full_like(nxt[:1], float("nan")), nxt[:-1]])
        if labels is not None:  # prompt-cued: the trials' texts
            names = [labels[int(i)] for i in idx]
        else:
            names = [self._classes()[int(k)] for k in targets.cpu()[idx]]
        return SearchEpisode(
            images=images, boxes=boxes, targets=names,
            centers=torch.stack([f["center"] for f in frames]), logits=torch.stack([f["logits"] for f in frames]),
            canvas=torch.stack([f["canvas"] for f in frames]), policy=policy,
            present=present.bool(),
            presence=(torch.stack([f["presence"] for f in frames]) if frames[0]["presence"] is not None
                      else None),
            overlap_box_frac=self.cfg.overlap_box_frac, loss_kind=self.cfg.loss, sigma=self.cfg.target_sigma)

    def _save_episode(self, frames, *, batch, run_dir, subdir, step, title) -> None:
        from canvit.harness.viz.disk import save_figure

        from .viz import make_search_figure
        ep = self._episode(frames, images=batch["image"], boxes=batch["boxes"], targets=batch.get("target"),
                           present=batch["present"], labels=batch.get("text") if self._prompt_mode else None)
        save_figure(make_search_figure(ep, title=title), run_dir, subdir, step)

    def viz_frame(self, *, model, images, gout, viewpoint, loss):
        """Branch-0 capture of the TRAINING rollout (sampled fixations, augmented images)."""
        core = getattr(model, "module", model)
        bound = self._bound
        idx = self._viz_index(bound.boxes)
        with torch.no_grad():
            logits, pres = core.search_head(gout.readout[idx].float(), bound.cue()[idx])
        return self._viz_capture(core, hidden=gout.readout, logits=logits, presence=pres, state=gout.state,
                                 centers=viewpoint.centers, targets=bound.cue(), joint=self._joint, idx=idx,
                                 beta=getattr(bound, "saccade_beta", None))

    def render_viz(self, viz, *, batch, run_dir, step):
        """Training-rollout figure -> ``{run_dir}/visualization/search_train/``."""
        if not viz.frames:
            return
        self._save_episode(viz.frames, batch=batch, run_dir=run_dir, subdir="search_train", step=step,
                           title=f"train step {step} (sampled fixations)")

    # --- evaluation --------------------------------------------------------
    def _rollout_metrics(self, *, model, images, targets, boxes, T, cg, amp, choose=None, joint=None,
                         record: list | None = None, scene_target: Tensor | None = None):
        """One closed-loop episode of ``T`` glimpses.

        t0 is the centred fixation. Later fixations come from ``choose(t, logits, visited)``
        (random / greedy) or, with ``joint``, from the learned scorer's argmax. ``record``
        (a list) collects the figure's per-glimpse frames. Returns per-timestep lists of the
        heatmap KL [B], the presence logit [B] (None without a presence head), and a dict of
        per-timestep [B] bool checks: ``argmax_in`` / ``argmax_overlap`` (the task head's
        argmax inside the target box / its ``overlap_box_frac`` square overlapping it) and
        ``fix_in`` / ``fix_overlap`` (the same for fixation t)."""
        from canvit.harness.rollout.eval_viewpoints import deploy_rollout_viewpoints
        from canvit.harness.rollout.engine import _to_vp

        B = images.shape[0]
        tmap = target_map(boxes, grid=cg, kind=self.cfg.loss, sigma=self.cfg.target_sigma)
        losses: list[Tensor] = []
        presences: list[Tensor | None] = []
        checks: dict[str, list[Tensor]] = {k: [] for k in ("argmax_in", "argmax_overlap", "fix_in", "fix_overlap",
                                                            "cos_in", "cos_overlap", "scene_cos")}
        cosine = self._prompt_mode and self.cfg.prompt.eval_cosine_baseline
        frac = self.cfg.overlap_box_frac
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
            with amp, prompt_context(model, targets):
                out = model.canvit(image=images, state=state, viewpoint=vp)
            hidden = _canvas_hidden(model, out.state, cg)
            logits, pres = model.search_head(hidden.float(), targets)
            logits = logits.float()
            losses.append(per_image_loss(logits, tmap))
            presences.append(pres)
            guess, fix = argmax_centres(logits), vp.centers.flip(-1)  # both (x, y)
            checks["argmax_in"].append(inside_box(guess, boxes))
            checks["argmax_overlap"].append(square_overlaps_box(guess, boxes, box_frac=frac))
            checks["fix_in"].append(inside_box(fix, boxes))
            checks["fix_overlap"].append(square_overlaps_box(fix, boxes, box_frac=frac))
            if cosine or scene_target is not None:
                scene = model.predict_scene(hidden.float()).float()  # standardized DINOv3 patch features
                if scene_target is not None:
                    checks["scene_cos"].append(F.cosine_similarity(scene, scene_target, dim=-1).mean(-1))
                if cosine:  # zero-shot baseline: cosine of the raw predicted features and the raw prompt
                    cos = F.cosine_similarity(model.scene["std"].destandardize(scene), targets.raw[:, None], dim=-1)
                    g_cos = argmax_centres(cos.view(B, cg, cg))
                    checks["cos_in"].append(inside_box(g_cos, boxes))
                    checks["cos_overlap"].append(square_overlaps_box(g_cos, boxes, box_frac=frac))
            if record is not None:
                idx = self._viz_index(boxes)
                record.append(self._viz_capture(model, hidden=hidden, logits=logits[idx],
                                                presence=pres[idx] if pres is not None else None,
                                                state=out.state, centers=vp.centers, targets=targets,
                                                joint=joint, idx=idx))
            mark(vp)
            last_logits[:] = [logits]
            return out.state

        if joint is not None:
            deploy_rollout_viewpoints(joint=joint, advance=advance, t0_type=ViewpointType.FULL,
                                      batch_size=B, device=images.device, n=T,
                                      cond=lambda: self._policy_cond(model, targets))
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
        return losses, presences, checks

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
        """Validation; writes ``eval.jsonl`` and the tracker's ``overview/``. Metrics: :meth:`_evaluate_pass`.

        Prompt-cued search (decision 8): one pass per prompt type -- text on every val trial, visual on the
        trials with a visual prompt; only the trained type when the visual share is 0 or 1. Keys of a type are
        prefixed ``text/`` / ``visual/``; the plain keys are the mean over the evaluated types. The metric of
        record ``found_mean_<policy>`` = the mean of ``found`` over glimpses t1..t(T-1), averaged over the types
        (``prompt.best_prompt_type``). Only the rules in ``cfg.eval_policies`` run (owner 2026-11-26: no automatic
        random passes; a baseline is a listed rule or a separate ``harness.evaluate``).

        ``--cfg.retina all`` (a multi-retina backbone through every retina): each retina is
        validated on the same images (same seeded random fixations), the prompt types inside each retina; the MEAN
        over retinas goes under the usual keys (so ``best_metric`` and best.pt follow it), each retina's
        values under ``<retina>/<key>`` (prompt-cued: ``<retina>/<type>/<key>`` too; all of them in
        ``eval.jsonl``; the tracker gets the last-glimpse ``found`` / ``hit`` / ``correct_found`` /
        ``presence_acc`` (and ``found_mean``) per retina), and its figures under ``visualization/<retina>/``."""
        T = self.cfg.n_timesteps
        pct = percent_label(self.cfg.overlap_box_frac)
        policies = [p for p in self.cfg.eval_policies if p != "policy" or joint is not None]
        if not policies:
            raise ValueError(f"no fixation rule to validate: --cfg.eval-policies {self.cfg.eval_policies} without a "
                             "trained policy; list random and/or greedy")
        kw = dict(model=model, val_loader=val_loader, device=device, step=step, run_dir=run_dir, joint=joint,
                  policies=policies)
        names_r = self._retina_names(model)
        per: dict[str, dict[str, float]] = {}
        if names_r is None:
            full, out = self._evaluate_types(**kw)
        else:
            patcher = model.canvit.patcher
            outs: dict[str, dict[str, float]] = {}
            try:
                for i, name in enumerate(names_r):
                    patcher.set_retinas(i)
                    per[name], outs[name] = self._evaluate_types(**kw, viz_prefix=f"{name}/")
            finally:
                patcher.set_retinas(None)
            keys = set.intersection(*(set(v) for v in per.values()))
            full = {k: sum(v[k] for v in per.values()) / len(per) for k in sorted(keys)}
            first = next(iter(outs.values()))
            out = {k: sum(o[k] for o in outs.values()) / len(outs) for k in first if all(k in o for o in outs.values())}
            L = T - 1
            for name, d in per.items():
                out.update({f"{name}/{k}_t{L}_{p}": d[f"{k}_t{L}_{p}"]
                            for k in ("found", "hit", "correct_found", "presence_acc") for p in policies
                            if f"{k}_t{L}_{p}" in d})
                out.update({f"{name}/found_mean_{p}": d[f"found_mean_{p}"] for p in policies
                            if f"found_mean_{p}" in d})
        if run_dir is not None:
            import json
            from pathlib import Path
            Path(run_dir).mkdir(parents=True, exist_ok=True)  # an empty tmp dir can vanish (tests)
            flat = {**full, **{f"{name}/{k}": v for name, d in per.items() for k, v in d.items()}}
            with open(Path(run_dir) / "eval.jsonl", "a") as f:
                f.write(json.dumps({"step": step, **flat}) + "\n")
        if tracker is not None:
            tracker.log_metrics({f"overview/{k}": v for k, v in self._overview(full, T, pct).items()}, step=step)
        return out

    def _evaluate_types(self, *, policies, viz_prefix: str = "", **kw) -> tuple[dict, dict]:
        """One validation (one retina): a single :meth:`_evaluate_pass`, or for prompt-cued search one per prompt
        type, with ``<type>/`` keys and the plain keys = their mean (see :meth:`evaluate`)."""
        T = self.cfg.n_timesteps
        if not self._prompt_mode:
            return self._evaluate_pass(**kw, policies=policies, viz_prefix=viz_prefix)
        per = {ty: self._evaluate_pass(**kw, policies=policies, prompt_type=ty, viz_prefix=viz_prefix)
               for ty in self._eval_prompt_types()}
        for ty, (f, o) in per.items():
            for p in policies:
                f[f"found_mean_{p}"] = o[f"found_mean_{p}"] = sum(
                    f[f"found_t{t}_{p}"] for t in range(1, T)) / max(T - 1, 1)
        full, out = {}, {}
        for ty, (f, o) in per.items():
            full |= {f"{ty}/{k}": v for k, v in f.items()}
            out |= {f"{ty}/{k}": v for k, v in o.items()}
        for src, dst in ((0, full), (1, out)):
            first = next(iter(per.values()))[src]
            for k in first:
                if all(k in d[src] for d in per.values()):
                    dst[k] = sum(d[src][k] for d in per.values()) / len(per)
        sel = self.cfg.prompt.best_prompt_type
        if sel != "mean" and sel in per:
            for p in policies:
                out[f"found_mean_{p}"] = full[f"found_mean_{p}"] = per[sel][0][f"found_mean_{p}"]
        return full, out

    @staticmethod
    def _retina_names(model) -> list[str] | None:
        """The retinas to validate one by one: all of a multi-retina patcher that no task
        fixed to one retina (``--cfg.retina all``); None otherwise (one pass, as before)."""
        patcher = model.canvit.patcher
        if getattr(patcher, "n_retinas", 1) > 1 and getattr(patcher, "fixed_retina", None) is None:
            return list(patcher.retina_names)
        return None

    def _evaluate_pass(self, *, model, val_loader, device, step, run_dir=None, joint=None, policies,
                       prompt_type: str | None = None, viz_prefix: str = "") -> tuple[dict, dict]:
        """One validation pass under each of ``policies`` (random / greedy / policy fixations); for
        prompt-cued search under one prompt type. Returns (every metric, the ones the tracker gets).
        ``viz_prefix`` (e.g. ``<retina>/``) is put in front of the figure folders.

        Localisation, over the target-present trials, after glimpse t (X = ``percent_label``
        of ``cfg.overlap_box_frac``, 10 by default):

        - ``found``: the task head's argmax lies inside the target box (strict).
        - ``found_X_percent``: a square of side ``overlap_box_frac`` x scene around that argmax
          overlaps the box (GVSM's fixation box / Haoze's rule).
        - ``hit`` / ``hit_X_percent``: the same two criteria on the FIXATIONS: some fixation
          0..t (t0, the fixed centre, included) satisfied it.

        ``correct_<criterion>``, over all trials: a target-present trial counts if it is
        located by that criterion AND judged "present", a target-absent trial if judged
        "absent" (without a presence head: located). ``loss`` is the heatmap KL over the
        target-present trials. With a presence head, at p(present) > 0.5: ``presence_acc``,
        ``recall`` (present trials judged present), ``specificity`` (absent judged absent),
        ``precision``, ``f1``.

        Keys are ``<metric>_t<t>_<policy>``. :meth:`evaluate` writes every value to
        ``run_dir/eval.jsonl`` (one line per validation) and returns -- the harness logs it under
        ``eval/`` -- ``loss`` / ``found`` / ``hit`` / ``correct_found`` / ``presence_acc`` at every
        glimpse and the rest at the last glimpse only; ``tracker`` additionally gets a short
        ``overview/`` section (``_overview``). Prompt-cued extras: ``found_cos`` (zero-shot cosine
        baseline) and ``scene_cos`` (distill), when on."""
        T = self.cfg.n_timesteps
        cg = self.canvas_grid(model)
        amp: Any = nullcontext()
        if device.type == "cuda" and self.cfg.amp:  # the training loop's rule (harness/run.py)
            dt = torch.bfloat16 if torch.cuda.get_device_capability(device) >= (8, 0) else torch.float16
            amp = torch.autocast("cuda", dtype=dt)
        was_training = model.head.training
        model.head.eval()
        pct = percent_label(self.cfg.overlap_box_frac)
        # criterion -> (rollout check, cumulative over the fixations so far?)
        crit = {"found": ("argmax_in", False), f"found_{pct}": ("argmax_overlap", False),
                "hit": ("fix_in", True), f"hit_{pct}": ("fix_overlap", True)}
        if self._prompt_mode and self.cfg.prompt.eval_cosine_baseline:
            crit |= {"found_cos": ("cos_in", False), f"found_cos_{pct}": ("cos_overlap", False)}
        distill = self._prompt_mode and self.cfg.prompt.distill_weight > 0
        scene_cos = {p: 0.0 for p in policies}
        keys = ("kl", "tpos", "tneg", "fpos", *crit, *(f"correct_{c}" for c in crit))
        sums = {p: {k: [0.0] * T for k in keys} for p in policies}
        n_tp = n_ta = 0
        # Release val only (batch["band"]): the last glimpse's metrics per class-size band (release spec 5).
        from .release import BANDS
        band_sums = {p: [dict.fromkeys(("n", "n_tp", "presence_ok", *crit, *(f"correct_{c}" for c in crit)), 0.0)
                         for _ in BANDS] for p in policies}
        has_bands = False
        has_presence = False
        do_viz = run_dir is not None and bool(self.cfg.viz_every) and step % self.cfg.viz_every == 0
        n = 0
        # Random fixations draw from the global torch RNG: fork it, so eval is seeded and
        # cannot perturb the training stream.
        with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
            torch.manual_seed(self.cfg.eval_seed)
            for vb, batch in enumerate(val_loader):
                if self.cfg.limit_val_batches is not None and vb >= self.cfg.limit_val_batches:
                    break
                if prompt_type is not None:
                    batch = self._prompt_rows(batch, prompt_type)
                    if batch is None:
                        continue
                images = batch["image"].to(device)
                targets = resolve_cue(model, self._cue_of(batch, device, model, prompt_type))  # no grad: once
                boxes = batch["boxes"].to(device)
                present = batch["present"].to(device)
                scene_target = self._teacher_scene(images, model) if distill else None
                n_tp += int(present.sum())
                n_ta += int((~present).sum())
                for p in policies:
                    record = [] if (do_viz and vb == 0) else None
                    kw = dict(model=model, images=images, targets=targets, boxes=boxes, T=T, cg=cg, amp=amp,
                              record=record, scene_target=scene_target)
                    if p == "policy":
                        losses, pres, chk = self._rollout_metrics(**kw, joint=joint)
                    elif p == "greedy":
                        losses, pres, chk = self._rollout_metrics(**kw, choose=self._greedy(cg))
                    elif p == "random":
                        losses, pres, chk = self._rollout_metrics(**kw, choose=self._random)
                    else:
                        raise ValueError(f"unknown eval policy {p!r}")
                    s = sums[p]
                    if distill:
                        scene_cos[p] += chk["scene_cos"][T - 1].sum().item()
                    reached = {c: torch.zeros_like(present) for c in crit}
                    for t in range(T):
                        said = pres[t] > 0 if pres[t] is not None else None  # p(present) > 0.5
                        s["kl"][t] += losses[t][present].sum().item()
                        for c, (k, cumulative) in crit.items():
                            ok = chk[k][t]
                            if cumulative:  # some fixation 0..t
                                reached[c] = reached[c] | ok
                                ok = reached[c]
                            ok = ok & present
                            s[c][t] += ok.float().sum().item()
                            corr = ok if said is None else (ok & said) | (~said & ~present)
                            s[f"correct_{c}"][t] += corr.float().sum().item()
                        if said is None:
                            continue
                        has_presence = True
                        s["tpos"][t] += (said & present).float().sum().item()
                        s["tneg"][t] += (~said & ~present).float().sum().item()
                        s["fpos"][t] += (said & ~present).float().sum().item()
                    if "band" in batch:
                        has_bands = True
                        self._accumulate_bands(band_sums[p], batch["band"].to(device), present=present,
                                               pres=pres[T - 1], chk=chk, reached=reached, crit=crit, last=T - 1)
                    if record:
                        try:  # a diagnostic: a plotting failure must never abort validation
                            self._save_episode(record, batch=batch, run_dir=run_dir,
                                               subdir=(f"{viz_prefix}search_val_{p}"
                                                       + (f"_{prompt_type}" if prompt_type else "")),
                                               step=step,
                                               title=f"val step {step}, fixations: {p}")
                        except Exception:
                            log.exception("search val viz failed at step %d (validation continues)", step)
                n += images.shape[0]
        if was_training:
            model.head.train()
        names = ["loss", *crit, *(f"correct_{c}" for c in crit)]
        if has_presence:
            names += ["presence_acc", "recall", "specificity", "precision", "f1"]
        full: dict[str, float] = {}
        for p in policies:
            s = sums[p]
            for t in range(T):
                full[f"loss_t{t}_{p}"] = s["kl"][t] / max(n_tp, 1)
                for c in crit:
                    full[f"{c}_t{t}_{p}"] = s[c][t] / max(n_tp, 1)
                    full[f"correct_{c}_t{t}_{p}"] = s[f"correct_{c}"][t] / max(n, 1)
                if has_presence:
                    tpos, tneg, fpos = s["tpos"][t], s["tneg"][t], s["fpos"][t]
                    rec, prec = tpos / max(n_tp, 1), tpos / max(tpos + fpos, 1)
                    full[f"presence_acc_t{t}_{p}"] = (tpos + tneg) / max(n, 1)
                    full[f"recall_t{t}_{p}"] = rec
                    full[f"specificity_t{t}_{p}"] = tneg / max(n_ta, 1)
                    full[f"precision_t{t}_{p}"] = prec
                    full[f"f1_t{t}_{p}"] = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
            if distill:
                full[f"scene_cos_t{T - 1}_{p}"] = scene_cos[p] / max(n, 1)
        band_keys: list[str] = []
        if has_bands:
            for p in policies:
                for name, bs in zip(BANDS, band_sums[p]):
                    for c in crit:
                        full[f"{c}_t{T - 1}_{p}_band{name}"] = bs[c] / max(bs["n_tp"], 1)
                        full[f"correct_{c}_t{T - 1}_{p}_band{name}"] = bs[f"correct_{c}"] / max(bs["n"], 1)
                    if has_presence:
                        full[f"presence_acc_t{T - 1}_{p}_band{name}"] = bs["presence_ok"] / max(bs["n"], 1)
                    full[f"n_trials_band{name}"] = bs["n"]
            band_keys = [k for k in full if "_band" in k and k.split("_t")[0] in ("found", "correct_found",
                                                                                  "presence_acc", "n")]
        per_glimpse = {"loss", "found", "hit", "correct_found", "presence_acc"}
        # ordered by metric, then fixation rule, then glimpse (W&B shows panels in log order)
        out = {f"{k}_t{t}_{p}": full[f"{k}_t{t}_{p}"]
               for k in names for p in policies for t in range(T) if k in per_glimpse or t == T - 1}
        out |= {k: v for k, v in full.items() if k.startswith(("found_cos", "scene_cos"))}
        return full, out | {k: full[k] for k in band_keys}

    @staticmethod
    def _accumulate_bands(bsums: list[dict], band: Tensor, *, present: Tensor, pres: Tensor | None,
                          chk: dict, reached: dict, crit: dict, last: int) -> None:
        """Per class-size band, the counts behind the last glimpse's ``found`` / ``hit`` (both criteria),
        ``correct_*`` and ``presence_acc``, by the same rules as the overall metrics in ``evaluate``."""
        said = pres > 0 if pres is not None else None
        for g, bs in enumerate(bsums):
            m = band == g
            bs["n"] += m.sum().item()
            bs["n_tp"] += (m & present).sum().item()
            for c, (k, cumulative) in crit.items():
                ok = (reached[c] if cumulative else chk[k][last]) & present
                bs[c] += (ok & m).sum().item()
                corr = ok if said is None else (ok & said) | (~said & ~present)
                bs[f"correct_{c}"] += (corr & m).sum().item()
            if said is not None:
                bs["presence_ok"] += ((said == present) & m).sum().item()

    @staticmethod
    def _overview(full: dict[str, float], T: int, pct: str) -> dict[str, float]:
        """The short W&B section, under the same names as in ``eval/``: last-glimpse values,
        the policy-minus-random gain of ``found`` / ``found_X_percent``, and the development
        over glimpses of three curated series (``found`` under the policy and under random
        fixations, ``hit`` under the policy). Keys whose policy or metric this run lacks are
        left out."""
        L = T - 1
        last = [f"t{L}_{p}" for p in ("policy", "random", "greedy")]
        every = [f"t{t}_policy" for t in range(T)] + [f"t{t}_random" for t in range(T)]
        # in display order: metric by metric, each in glimpse order (W&B shows panels in log order)
        wanted: list[str] = []
        for c in ("found", "hit"):
            wanted += [f"{c}_{s}" for s in every if c == "found" or s.endswith("policy")]
            wanted += [f"{c}_{s}" for s in last]
            wanted.append(f"{c}_gain")
        for c in (f"found_{pct}", f"hit_{pct}"):
            wanted += [f"{c}_{s}" for s in last] + [f"{c}_gain"]
        for c in ("correct_found", f"correct_found_{pct}"):
            wanted += [f"{c}_t{L}_policy", f"{c}_t{L}_random"]
        wanted += [f"presence_acc_t{L}_policy", f"loss_t{L}_policy"]
        ov: dict[str, float] = {}
        for k in wanted:
            if k.endswith("_gain"):  # policy minus random, found and found_X_percent only
                c = k.removesuffix("_gain")
                if c.startswith("found") and f"{c}_t{L}_policy" in full and f"{c}_t{L}_random" in full:
                    ov[f"{c}_gain_t{L}_policy_vs_random"] = full[f"{c}_t{L}_policy"] - full[f"{c}_t{L}_random"]
            elif k in full and k not in ov:
                ov[k] = full[k]
        return ov

    # --- checkpoint ----------------------------------------------------------
    def model_config(self, model):
        from dataclasses import asdict
        core = getattr(model, "module", model)
        mc = {"task": "search", "backbone_name": core.backbone_name, "canvit": asdict(core.canvit.cfg),
              "glimpse_grid_size": core.glimpse_grid_size, "head_kw": dict(core.head_kw),
              "model_repo": self.cfg.model_repo, "canvas_grid": self.canvas_grid(model)}
        if self._prompt_mode:
            return mc | {"n_classes": None, "classes": None, "prompt_kw": dict(core.prompt_kw)}
        return mc | {"n_classes": len(self._classes()), "classes": list(self._classes())}

    def checkpoint_metadata(self, model):
        from canvit.checkpoint import downstream_pretrain_view_scale
        meta = {"task": "search", "scene_size": self.cfg.scene_size, "n_timesteps": self.cfg.n_timesteps,
                "model_repo": str(self.cfg.model_repo),
                "pretrain_view_scale": downstream_pretrain_view_scale(
                    patcher_name=getattr(model.canvit.cfg, "patcher_name", None),
                    foveated_scale=self.cfg.foveated_scale)}
        rl = self.rl
        if rl.saccade_cost != "none":  # absent when off: the historical metadata
            meta["saccade_cost"] = {
                "form": rl.saccade_cost, "weight": rl.saccade_cost_weight, "combine": rl.saccade_cost_combine,
                "beta_mode": rl.saccade_cost_beta_mode, "beta_eval": rl.saccade_cost_beta_eval,
                "r": 1.0 / rl.centers_per_axis if rl.saccade_cost_r is None else rl.saccade_cost_r,
                "beta_features": "[beta, log1p(beta / 0.1)] appended to the FiLM cue (conditioned only)"}
        return meta


__all__ = ["BoundSearchTask", "SearchRunTask", "POLICY_FEATURE_GROUPS"]
