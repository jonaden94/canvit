"""ADE20K probe-training configuration (port of canvit_specialize's, trimmed per the
unification master plan: canvas_hidden only — recon_normalized dropped (D3); the
legacy full-finetune branch is not ported (full-FT arrives via the harness, P4)."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from canvit.core import resolve_canvit_repo
from canvit.core.data.ade20k import ResizeMode  # noqa: F401  (re-exported for this repo's consumers)

from ..harness.config import FoveatedScaleConfig
from ..harness.rollout.eval_viewpoints import EvalPolicy


def _default_wandb_project() -> str | None:
    return os.environ.get("WANDB_PROJECT")


def _default_wandb_entity() -> str | None:
    return os.environ.get("WANDB_ENTITY") or None


def _default_wandb_dir() -> Path | None:
    if d := os.environ.get("WANDB_DIR"):
        return Path(d)
    return None


def _default_ade20k_root() -> Path:
    if root := os.environ.get("ADE20K_ROOT"):
        return Path(root)
    if tmpdir := os.environ.get("SLURM_TMPDIR"):
        return Path(tmpdir) / "ADEChallengeData2016"
    return Path("/datasets/ADE20k/ADEChallengeData2016")


@dataclass
class Ade20kConfig:
    """Frozen-backbone ADE20K probe training."""

    model_repo: str = resolve_canvit_repo("canvitb16-add-vpe-pretrain-g128px-s512px-in21k-dv3b16-2026-02-02")
    retina: str | None = None
    """A MULTI-RETINA backbone is used through this one retina, by name as the backbone's patcher
    lists it (e.g. ``cmf0.75_res65_cps6``), for training and eval. Required for such a backbone,
    refused for a single-retina one (``core.patcher.foveated.fix_backbone_retina``)."""
    ade20k_root: Path = field(default_factory=_default_ade20k_root)
    scene_size: int = 512

    mode: Literal["frozen", "finetune"] = "frozen"
    """``frozen`` (default, the historical probe regime): freeze the CanViT backbone and
    train only the segmentation head. ``finetune``: train the whole model end to end.
    Mirrors ``In1kConfig.mode`` so both downstream tasks name the same thing the same way
    — before this, ade20k finetune was reachable ONLY via the generic ``--preset
    finetune``, which is a different entry point with different LR-schedule behaviour.
    Honored by the harness (``tasks/ade20k/task.py::default_spec``); the standalone
    entry point trains the probe only and ignores it."""

    probe_repo: str | None = None
    """A published segmentation probe to initialise the head from (core
    ``from_pretrained_with_probe``) instead of a fresh random one — the ADE20K analogue
    of ``In1kConfig.probe_repo`` and of specialize's ``init_probe_repo``. ``None`` =>
    fresh head.

    Honoured in BOTH modes (it was finetune-only until 2026-07-29):
    * ``finetune`` — starting from a RANDOM head at the small finetune LR trains far too
      slowly; that exact bug cost the unified in1k finetune a whole run (8f780ba).
    * ``frozen`` — **REQUIRED for policy training.** The scorer's reward is the fraction
      of the PROBE's cross-entropy a glimpse removes, so a random head makes the reward
      noise. `--preset policy_only` runs in frozen mode, where this used to be ignored and
      the run silently trained against an untrained probe. `rl_train.py` always loads one
      (its default is ``probe-ade20k-40k-s512-c{canvas_grid}-in21k``).

    Probe TRAINING leaves this unset and is unaffected."""

    # Rollout
    n_timesteps: int = 10
    bptt_chunk_size: int = 0
    """Truncated-BPTT chunk size for ``mode="finetune"``. 0 (default) = one graph over the
    whole rollout; n>0 = backward + detach every n glimpses, capping activation memory at
    n steps instead of ``n_timesteps``.

    ``n_timesteps`` need NOT be divisible by this: the rollout flushes the trailing
    partial chunk and every chunk normalises by ``n_timesteps``, so e.g. 7 glimpses at
    chunk 3 runs [0,1,2][3,4,5][6] with the same total gradient. ``n >= n_timesteps``
    collapses to one graph.

    IGNORED in ``mode="frozen"``: the backbone runs under ``no_grad`` there, and ``bptt``
    only ever moves the backbone — head gradients are bit-identical between ``none`` and
    ``full`` (measured 2026-07-28). Chunking a frozen backbone costs memory and changes
    nothing. See ``harness.spec.fixed_horizon_bptt``."""
    glimpse_px: int | None = None
    """Uniform-patcher glimpse crop size in px. None = derive from the model's
    glimpse_grid_size × patch size/stride, so the patch-embed conv yields exactly
    glimpse_grid_size tokens per side — see
    ``harness/rollout/episode.py::derive_glimpse_px``, which enforces it with a hard
    token-count guard. Ignored for foveated/square models (they consume the full image)."""
    canvas_grid: int | None = None
    """None = scene_size // patch_size."""

    # Viewpoint policy for TRAINING: pure IID random by default
    min_vp_scale: float = 0.05
    max_vp_scale: float = 1.0
    train_start_full: bool = False
    foveated_scale: FoveatedScaleConfig = field(default_factory=FoveatedScaleConfig)
    """Foveated/square patcher only: the view-scale law for the probe rollout,
    which MUST match how the backbone was pretrained — the foveated patcher
    derives its fixation window as ``fix_size = scale * H``, so glimpses at an
    unseen scale are out of distribution and actively degrade the canvas (this
    is not a soft mismatch: it shows up as mIoU *falling* with more glimpses).
    Pass the pretrain run's value, e.g. ``--foveated-scale.fixed-scale 2.0`` for
    exp22-fovi. Ignored for uniform models, which use ``min/max_vp_scale``."""

    # EVAL viewpoint policy — the SHARED option set (harness/eval_viewpoints.py), the
    # same one distill and in1k take.
    reward_score_res: int | None = 128
    """Resolution at which the POLICY REWARD's probe CE is scored (`per_image_loss`).
    Reward-only — nothing else reads it (`harness/rollout.py:268/310/327`).

    128 is CanViT-PyTorch-RL's value, validated there against full 512 at Spearman 0.999
    for candidate ranking. The harness previously scored at the probe's native 64 — cheaper
    but never validated, and it was the last known config difference from the reference
    (doc 15 §A gap #5, closed 2026-07-30). None = the masks' own resolution.
    """
    eval_override_scale: float | None = None
    """Pin every eval glimpse to this scale while keeping the policy's CENTERS. Mirrors
    ``canvit_eval``'s ``EpisodeConfig.override_scale``. ``None`` (default) = off.

    For a FIXED-SCALE FOVEATED backbone under ``coarse_to_fine``: the quadtree's scales
    ({1.0, 0.5, 0.25}) are ones the model never trained on, and ``fix_size = scale * H``
    makes every such glimpse out of distribution, so mIoU decays as glimpses accumulate.
    Set this to the pretraining view scale to evaluate "C2F's fixation sequence at the
    model's own scale" instead. Leave unset for uniform backbones, where the quadtree's
    scales are the point."""
    per_row_iou_out: Path | None = None
    """Also write per-(image, class, timestep) IoU COUNTS to this parquet path.

    A dataset mIoU says how good the model is; these rows say on WHICH images and classes,
    and how that moves as glimpses accumulate — the question a glimpse policy is actually
    about, and one an aggregate cannot answer. Costs one scatter-add per batch and no extra
    forward. ``None`` (default) = off, and the eval is unchanged."""
    eval_policy: EvalPolicy = "auto"
    """Validation trajectory. ``"auto"`` = this task's historical one, IID random from a
    full-scene anchor — inherited from the specialize probe, which TRAINED on random
    viewpoints so validating on random was consistent. That reasoning stops holding the
    moment a policy is in the loop: under ``"policy"`` the scorer is deployed by argmax
    and validation measures the thing that was actually trained. Other options:
    ``coarse_to_fine`` (quadtree, the canvit_eval deploy convention), ``full`` (repeated
    full scene), ``fixation_grid`` (deterministic centre + 3x3 at the training scale —
    the scale-safe choice for a fixed-scale foveated model).

    Left at ``"auto"`` for comparability: every specialize probe number and every exp24
    run was measured under random, so changing the default would silently break them."""

    # Training
    batch_size: int = 16
    eval_batch_size: int = 32
    num_workers: int = 4
    peak_lr: float = 3e-4
    weight_decay: float = 1e-3
    warmup_steps: int = 1500
    warmup_lr_ratio: float = 1e-6
    max_steps: int = 40000
    grad_clip: float = float("inf")
    dropout: float = 0.1

    # Data augmentation
    augment: bool = True
    """Train-split augmentation. ``False`` makes the TRAIN split use the val transform —
    the CanViT-PyTorch-RL protocol for policy training (`rl_train.py`: "NO augmentation,
    both splits use the val transform"). Mirrors ``In1kConfig.augment``.

    This has to be its own flag: neutralising ``aug_scale_range``/``aug_flip_prob`` does
    NOT disable augmentation, because `make_segmentation_train_transforms` still applies
    `RandomCropWithLabel` and an unconditional `PhotoMetricDistortion` (colour jitter),
    neither of which has a knob. Setting the two to identity gets you a DIFFERENTLY
    augmented pipeline, not an unaugmented one.

    Default ``True``: every ADE20K probe/finetune number, including the specialize
    reference the harness was gated against, was measured with augmentation on."""
    aug_scale_range: tuple[float, float] = (0.5, 2.0)
    aug_flip_prob: float = 0.5
    """Both ignored when ``augment=False``."""

    # Validation resize
    resize_mode: ResizeMode = "center_crop"
    """How val images/masks are fitted to the square ``scene_size``. Both modes are
    supported for every patcher; neither is "the correct one" — each trades away something
    different, so pick per experiment and say which one a number came from.

    ``center_crop`` (default): resize the short side, crop the central square. Preserves
    geometry (a circle stays a circle), matching how the backbone was pretrained and how
    the probe trains; the cost is that long-side content is thrown away, so the mIoU is over
    the central crop rather than the whole image.

    ``squish``: resize to (size, size), distorting aspect ratio. Keeps the whole field of
    view, and it is what the CanViT-PyTorch-RL numbers (qband band / EG-C2F) and the
    specialize reference were measured under — so it is the mode to use for comparability
    with those.

    Which to prefer for a FOVEATED/square model: for comparison against **human** viewing,
    aspect-preserving is the better match, since the cortical-magnification geometry is
    defined on the undistorted scene — a human looking at the same picture does not see it
    anisotropically stretched. For comparability with the **original CanViT** results,
    squish is fine, and the distortion is a mild domain shift, not a broken input (unlike a
    view-scale mismatch, which really does degrade the canvas — see ``foveated_scale``).
    NB the truest "what a human sees" option would be padding/letterbox (preserve aspect
    AND the full frame), which is not implemented: ``ResizeMode`` is center_crop | squish."""

    # Debug / smoke: cap batches per eval (None = full), mirroring In1kConfig. Honored by
    # both entry points (the standalone val loop and the harness task's `evaluate`).
    limit_val_batches: int | None = None

    # Run identity — the same trio Config/In1kConfig carry, so every task names its runs
    # the same way. The harness derives BOTH the tracker run name and
    # `logs_dir/run_group/run_name/` (checkpoints + visualization) from these.
    # `run_group` is REQUIRED: a run without one is refused, so no artifact can land
    # outside `logs_dir`. `run_name` is derived when unset.
    run_group: str | None = None
    run_name: str | None = None
    """None => auto: the descriptive `ade20k_{model}_{T}t_s{scene}_c{grid}_{ts}` name in
    the standalone, `ade20k_{timestamp}` in the harness."""
    logs_dir: Path = Path("logs")

    # Logging / checkpoints
    log_every: int = 20
    val_every: int = 500
    viz_every: int = 500
    """Render the segmentation overlay figure every N steps (0 = off), for the training
    batch and the first val batch. Specialize's default, restored — but the figures go to
    ``{run_dir}/visualization/seg_{train,val}/`` on disk instead of the wandb Media tab."""
    viz_samples: int = 4
    """Images per figure (one row each)."""
    device: str = "cuda"
    amp: bool = True
    seed: int = 0
    """`torch.manual_seed(seed + rank)`. Historically the probe had NO seed at all (both
    entry points), which made A/B gates against it impossible — same config, different
    curve. Both honor it now; 0 keeps the value the harness was already passing."""
    tracker: Literal["comet", "wandb", "none"] = "wandb"
    wandb_project: str | None = field(default_factory=_default_wandb_project)
    wandb_entity: str | None = field(default_factory=_default_wandb_entity)
    wandb_dir: Path | None = field(default_factory=_default_wandb_dir)
