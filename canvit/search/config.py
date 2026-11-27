"""Visual-search task config (class-cued search, COCO-Search18 first).

Design: ``canvit_claude/canvit/search/00-design.md``. Haoze's defaults (autoregressive_detection,
branch haoze-wip) unless canvit has a natural equivalent; every formulation choice the
owner named is an option here or in the policy config (``search_policy_config``).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from canvit.core.policy.net import N_UNET_ENC
from canvit.harness.config import FoveatedScaleConfig, JointPolicyConfig


def _default_data_root() -> Path:
    if root := os.environ.get("COCO_SEARCH18_ROOT"):
        return Path(root)
    return Path("/user/henrich1/u25995/jonathan/datasets/coco_search18")


def _default_wandb_project() -> str | None:
    return os.environ.get("WANDB_PROJECT")


def _default_wandb_entity() -> str | None:
    return os.environ.get("WANDB_ENTITY") or None


def _default_wandb_dir() -> Path | None:
    if d := os.environ.get("WANDB_DIR"):
        return Path(d)
    return None


def _fixed_scale_2() -> FoveatedScaleConfig:
    """exp22/exp32/exp34/exp36 all pretrain the foveated backbone at view scale 2.0."""
    return FoveatedScaleConfig(mode="fixed", fixed_scale=2.0)


@dataclass
class SearchAugConfig:
    """Train-time augmentation (torchvision v2, boxes transformed with the image), applied
    to the original image before letterboxing. Defaults per the design doc §2.3."""

    hflip_p: float = 0.5
    crop: bool = True
    """Random scaled crop with the image's aspect kept (output = the letterboxed content size)."""
    crop_scale: tuple[float, float] = (0.6, 1.0)
    color_jitter: bool = False
    """Haoze: ColorJitter(0.4, 0.4, 0.4, 0.1) p 0.8, then greyscale p 0.1."""
    rotate: bool = False
    """Haoze: rotation +-60 deg, p 0.5 (the box becomes the axis-aligned box of the rotated box)."""
    affine: bool = False
    """Haoze: affine rot +-45 deg, translate +-10 %, scale 0.8-1.2, shear +-15 deg, p 0.5."""
    autocontrast: bool = False
    """Haoze: autocontrast p 0.5."""
    min_box_fraction: float = 1.0
    """A sample is kept only if at least this fraction of the transformed target box lies
    inside the frame; otherwise the geometric chain is redrawn."""
    max_tries: int = 10
    """Redraws before falling back to the un-augmented sample."""


@dataclass
class PromptConfig:
    """Prompt-cued search on the curated release (owner decisions 2026-11-26,
    ``canvit_claude/canvit/search/04-prompt-conditioning-decisions.md``; ``prompt.py``). ``cue='class'`` = the
    class-index path of COCO-Search18 (everything below unused); ``cue='prompt'`` = text and visual prompts, no
    class index anywhere."""

    cue: Literal["class", "prompt"] = "class"
    visual_share: float = 0.5
    """PROPOSAL (not decided by the owner). Share of trials cued by their visual prompt, in [0, 1]: 0 = text only,
    1 = visual only. Trials are dropped only when the epoch cannot meet it (``prompt.assign_prompt_types``); > 0
    needs a release with visual prompts."""
    adaln: bool = True
    """AdaLN of the prompt's 1x1 vector in every backbone block, zero at init."""
    tokens: Literal["none", "1x1", "2x2", "3x3"] = "2x2"
    """PROPOSAL (not decided by the owner). Prompt tokens read by gated cross-attention (tanh gate, 0 at init)
    before every backbone block: none / the 1x1 cell / the 4 cells of 2x2 / the 9 cells of 3x3 (empty cells
    masked); a text prompt is one token."""
    token_mode: Literal["xattn", "prefix"] = "xattn"
    """How the prompt tokens enter the backbone: 'xattn' = gated cross-attention before every block; 'prefix' =
    LLaMA-Adapter style, every block's self-attention also attends to the tokens in a separate softmax scaled by a
    per-head gate (0 at init). Both leave the pretrained model unchanged at init (``prompt.py``)."""
    condition_heads: bool = True
    """The task head is FiLM-conditioned on the prompt (else on a zero vector: the backbone alone carries it)."""
    condition_policy: bool = True
    """The policy is FiLM-conditioned on the prompt (else unconditioned)."""
    backbone_lr: float | None = None
    head_lr: float | None = None
    policy_lr: float | None = None
    """Optional separate LRs (backbone; heads incl. the prompt modules; policy). None = ``peak_lr``, the ONE LR of
    all groups by default (decision 6). See the comment in ``SearchRunTask._prompt_spec``."""
    bptt_chunk: int | None = 2
    """PROPOSAL (memory). The backbone trains through the rollout in chunks of this many glimpses (truncated
    BPTT, as distill); None = one graph over all ``n_timesteps`` glimpses (did not fit a V100-32GB at batch 16 with
    10 glimpses)."""
    train_timesteps: int | None = None
    """Glimpses per TRAINING episode (validation keeps ``n_timesteps``; owner 2026-11-26). None = a random length per
    batch, canvit's distill paradigm unchanged: chunks of 2 glimpses, after each chunk continue with probability
    ``train_length_continue`` (P(n = 2k) = 0.5^k at 0.5, mean 4, no cap), seeded by (``seed``, step) so a resume
    draws the same lengths (``SearchRunTask._train_length``); frozen runs draw the same. An int = that fixed length."""
    train_length_continue: float = 0.5
    """The random training length's continue probability (distill's ``continue_prob``)."""
    distill_weight: float = 0.0
    """Weight of the distillation loss (DINOv3 scene features from the canvas, as in pretraining) next to the
    search loss. 0 = off: no teacher is built or called."""
    best_prompt_type: Literal["mean", "text", "visual"] = "mean"
    """Which prompt type's mean found rate selects the best checkpoint (open for the owner)."""
    v100_math_sdpa: bool = False
    """Allow gradients through the backbone on a V100 (sm_70) by switching SDPA to its math kernel for the run.
    Without it such a run is refused (``task.sdpa_guard``): there the memory-efficient SDPA backward returns
    garbage gradients (~1e12 at block 0; CPU and math-only give ~10, 2026-11-26). Applies to every search run
    (class-cued with a trained backbone too), not only prompt-cued ones."""
    readout_layers: int = 2
    """``--cfg.head xattn``: decoder layers (self-attn among the prompt queries, cross-attn to the canvas, MLP)."""
    readout_heads: int = 8
    """``--cfg.head xattn``: attention heads per layer."""
    eval_cosine_baseline: bool = False
    """Also report the zero-shot baseline: heatmap = cosine of the canvas's predicted DINOv3 patch features and the
    raw prompt vector (needs the pretraining scene head, kept for it)."""


@dataclass
class SearchConfig:
    # --- data ------------------------------------------------------------------
    dataset: Literal["coco_search18", "release"] = "coco_search18"
    """``coco_search18``: the class-cued COCO-Search18 path (``data_root``, ``split``, ``condition``).
    ``release``: a curated search release of get_datasets (``release_dir``; ``release.py``): text-cued
    trials, half of them target-absent; its training order is fixed by (``seed``, epoch, position)."""
    release_dir: Path = Path("/user/henrich1/u25995/jonathan/datasets/visual_search/search_release/basic-v1")
    """``dataset='release'``: the release folder (its ``README.md`` is the spec). A new release = a new folder."""
    visual_prompts: bool = False
    """Samples also carry the trial's visual prompt (``vp_feat`` [14, 768], ``vp_cov`` [14], ``vp_valid``; spec
    section 7 of a release with visual prompts, read from ``release_dir``). ``release``: the sampler's / val's
    ``vp_idx``. ``coco_search18``: evaluation only, a fixed prompt per trial from ``release_dir``'s ``vp_test``
    (training on COCO-Search18 with it is refused). Off: samples exactly as without visual prompts."""
    prompt: PromptConfig = field(default_factory=PromptConfig)
    data_root: Path = field(default_factory=_default_data_root)
    """COCO-Search18 root in the restructured layout: ``images/<name>.jpg`` (flat) +
    ``fixations/*.json`` (``data.py``). The padded release (Haoze's setting), not ``_nopadding``."""
    split: Literal[1, 2] = 1
    """The dataset's ``split1`` (the paper's) or ``split2`` train/valid assignment."""
    condition: Literal["tp", "both"] = "both"
    """``both`` (Haoze's default): every target-present AND every target-absent (image, cue)
    pair, about 50/50; adds the presence readout and its loss. ``tp``: target-present only.
    All images live in ``data_root/images/`` (the restructured layout, see ``data.py``)."""
    scene_size: int = 512
    """Square scene side in px; the 1680x1050 images are letterboxed into it."""
    aug: SearchAugConfig = field(default_factory=SearchAugConfig)

    # --- model -----------------------------------------------------------------
    model_repo: str | None = None
    """The pretrained FOVEATED CanViT: an HF repo id, a local HF dir or a training ``.pt``.
    Required (there is no published foveated search backbone to default to)."""
    retina: str | None = None
    """A MULTI-RETINA backbone is used through this one retina, by name as the backbone's patcher
    lists it (e.g. ``cmf0.75_res65_cps6``), or through ALL of them with ``all`` (as distill
    trains: each training sample draws its retina, held for its glimpses; validation runs every
    retina on the same images and reports the mean under the usual keys, so the best checkpoint
    follows the mean, plus each retina's metrics under ``<retina>/``). Required for such a
    backbone, refused for a single-retina one."""
    canvas_grid: int | None = None
    """None = scene_size // patch_size (32 for ViT-B/16 at 512 px)."""
    foveated_scale: FoveatedScaleConfig = field(default_factory=_fixed_scale_2)
    """Must match the backbone's pretraining view scale, or every glimpse is out of distribution."""
    head: Literal["unet", "linear", "xattn"] = "unet"
    """``unet`` / ``linear``: the FiLM-conditioned conv head (``head.SearchHead``). ``xattn`` (prompt-cued only): the
    cross-attention readout ``head.PromptQueryHead``, the prompt as the query of a small decoder over the canvas
    tokens (``prompt.readout_layers`` / ``readout_heads``)."""
    head_width: int | None = None
    """None = 64 (unet / linear) or 256 (xattn)."""
    head_block_layers: int = 2
    head_levels: int = N_UNET_ENC
    """Depth of the head's U-Net (resolutions; 6 pools 32x32 down to 1x1, 3 stops at 8x8) --
    the head's counterpart of ``--cfg.rl.policy-levels``. ``unet`` head only."""
    cond_dim: int = 64
    """Width of the target-class embedding(s) (Haoze: 64)."""
    embedding: Literal["separate", "shared"] = "separate"
    """``separate``: the task head and the policy each learn their own class embedding.
    ``shared``: one embedding (in the head) read by both, each with its own FiLM projections."""
    detach_policy_embedding: bool = False
    """``shared`` only: the policy reads a detached copy, so only the task loss trains it."""
    dropout: float = 0.0
    """Dropout2d before the head's output conv. Off by default."""

    # --- loss --------------------------------------------------------------------
    loss: Literal["kl_gaussian", "nll_point"] = "kl_gaussian"
    """``kl_gaussian`` (Haoze): KL(target || softmax over the canvas grid), target = a
    normalised Gaussian per box centre. ``nll_point``: cross-entropy on the cell holding the
    box centre (the sigma -> 0 limit; Tom's comparison)."""
    target_sigma: float = 2.5 / 64
    """Gaussian sigma as a fraction of the scene side (Haoze: 2.5 cells on a 64x64 heatmap)."""
    supervise: Literal["last", "every"] = "last"
    """``last`` (Haoze): only the last glimpse's loss trains the head. ``every``: all glimpses.
    Applies to the heatmap loss AND the presence loss (Haoze: both last)."""
    presence_loss: Literal["focal", "ce"] = "focal"
    """``condition='both'`` only. Loss of the present/absent readout: Haoze's focal loss
    (gamma ``presence_focal_gamma``) or plain binary cross-entropy."""
    presence_focal_gamma: float = 2.0
    presence_weight: float = 0.75
    """Weight of the presence loss. 0.75 reproduces Haoze's focal alpha 0.75, which in his
    implementation scales every sample alike (it is not class balancing)."""

    # --- rollout / eval ----------------------------------------------------------------
    n_timesteps: int | None = None
    """Glimpses per episode: the centred t0 fixation plus n-1 chosen ones. None = 4 (Haoze) for
    ``prompt.cue='class'``, 10 for ``'prompt'`` (the metric of record averages t1..t9). Prompt-cued search trains
    with its own length (``prompt.train_timesteps``); validation always runs this many."""
    eval_policies: tuple[str, ...] | None = None
    """Fixation policies run at every validation. ``random``: seeded random fixations.
    ``greedy``: the argmax of the current task heatmap, never revisiting a cell.
    ``policy``: the learned scorer's argmax (skipped when no policy is trained).
    None = all three for ``prompt.cue='class'``; ``('policy',)`` for ``'prompt'`` (a full release validation per
    rule takes long; owner 2026-11-26: no automatic random passes, list a baseline rule or run ``harness.evaluate``)."""
    overlap_box_frac: float = 0.1
    """Side of the square drawn around a point, as a fraction of the scene side, for the
    lenient metrics ``found_X_percent`` (the task head's argmax) and ``hit_X_percent`` (the
    fixations): a success if the square overlaps the target box. X follows this value
    (0.1 -> ``found_10_percent``). 0.1 = GVSM's fixation box / Haoze (64 px at 640). The
    strict ``found`` / ``hit`` (point inside the box) do not depend on it."""
    eval_seed: int = 1234
    limit_val_batches: int | None = None

    # --- optimisation -----------------------------------------------------------------
    batch_size: int = 16
    eval_batch_size: int = 32
    num_workers: int = 4
    peak_lr: float = 1e-4
    """Task head (Haoze: AdamW lr 1e-4)."""
    weight_decay: float = 0.01
    """Task head (Haoze's search config: 0.01; his 0.05 is the COCO-pretraining config's)."""
    warmup_steps: int = 500
    lr_schedule: Literal["cosine", "constant"] = "cosine"
    """Task head LR after the warmup: ``cosine`` (Haoze) decays to 0 at ``max_steps``;
    ``constant`` holds ``peak_lr`` (the policy's schedule)."""
    max_steps: int = 20000
    grad_clip: float = 1.0

    # --- run identity / logging (same trio as every task) ------------------------------
    run_group: str | None = None
    run_name: str | None = None
    logs_dir: Path = Path("logs")
    log_every: int = 20
    val_every: int = 500
    viz_every: int = 500
    """Figures (``viz.py``) of a training rollout every N steps, and of the first val batch
    under each eval policy at validations that fall on a multiple of N. 0 = off."""
    viz_samples: int = 6
    """Samples per figure: half target-present (top row of blocks), half target-absent."""
    device: str = "cuda"
    amp: bool = True
    seed: int = 0
    resume_from: Path | None = None
    """Continue ANOTHER run's checkpoint with its full state (RunSettings.resume_from): weights,
    AdamW, policy (incl. reward normalisers and entropy coefficient), step count; the LRs are
    this config's (e.g. an LR decay of a finished run). Only used while this run has no
    checkpoint of its own."""
    tracker: Literal["comet", "wandb", "none"] = "wandb"
    wandb_project: str | None = field(default_factory=_default_wandb_project)
    wandb_entity: str | None = field(default_factory=_default_wandb_entity)
    wandb_dir: Path | None = field(default_factory=_default_wandb_dir)

    def __post_init__(self) -> None:
        prompt = self.prompt.cue == "prompt"
        if self.head_width is None:
            self.head_width = 256 if self.head == "xattn" else 64
        if self.n_timesteps is None:
            self.n_timesteps = 10 if prompt else 4
        t = self.prompt.train_timesteps
        if prompt and t is not None and t < 1:
            raise ValueError(f"--cfg.prompt.train-timesteps {t} must be >= 1")
        if self.eval_policies is None:
            self.eval_policies = ("policy",) if prompt else ("random", "greedy", "policy")


# FoAViS's entropy target was 3.5 nats on a 64x64 policy grid. The same spatial spread on
# the 32x32 grid is 3.5 - ln(4096/1024) = 3.5 - ln 4 ~ 2.11 nats (e^H ~ 8.3 effective cells,
# ~0.8 % of the image, as in FoAViS).
SEARCH_ENTROPY_TARGET = 3.5 - math.log(4.0)


def search_policy_config() -> JointPolicyConfig:
    """The search task's policy defaults (design doc §5): PG on the plain canvas, a 32x32
    fixation grid, Haoze's terminal reward with FoAViS's per-step batch standardisation and
    no learned baseline, FoAViS's entropy target and dual step adapted to 32x32, and a
    scorer as small as the task head (64 wide, 2 blocks per stage)."""
    return JointPolicyConfig(
        objective="pg",
        select_bn_eval=False,  # the search scorer has no BatchNorm; keep it explicit
        feature_groups=("ln_feat",),
        centers_per_axis=32,
        width=64,
        block_layers=2,
        reward="terminal",
        reward_center="batch_mean",
        reward_scale="batch_std",
        entropy_target=SEARCH_ENTROPY_TARGET,
        alpha_lr=0.01,
        entropy_alpha_init=1e-3,
        entropy_alpha_min=1e-6,
        entropy_alpha_max=0.1,
        entropy_update="per_rollout",
        # Saccade cost off (the default). Stage 1 (fixed beta): --rl.saccade-cost {log,linear}
        # --rl.saccade-cost-weight {0,0.1,0.3,1.0} (combine 'standardized'). Stage 2 (one
        # beta-conditioned policy per form): --rl.saccade-cost-beta-mode conditioned
        # --rl.saccade-cost-weight 1.0 (beta ~ U[0, 1]) [--rl.saccade-cost-beta-eval 0.3].
        saccade_cost="none",
    )
