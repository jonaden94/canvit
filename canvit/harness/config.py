"""Config shared by every task: the foveated view-scale law and the joint task+policy
(RL) recipe. Both are referenced by all three task configs, so they live with the
harness rather than in any one task's folder.

Each task's OWN config sits in its own folder: ``distill/config.py::Config``,
``ade20k/config.py::Ade20kConfig``, ``in1k/config.py::In1kConfig``.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class FoveatedScaleConfig:
    """How the per-glimpse view scale is sampled for the foveated/square patchers.

    The foveation window is ``fix_size = scale * H``. The FULL start glimpse is
    always centered; its scale is ``fixed_scale`` in ``mode='fixed'`` (so it
    matches every other glimpse) and ``1`` in ``per_rollout``/``per_glimpse`` (a
    full-image anchor). RANDOM glimpses draw their scale per ``mode``: a single
    ``fixed_scale`` (``fixed``), one scale frozen across the rollout
    (``per_rollout``), or a fresh scale each glimpse (``per_glimpse``). The
    uniform patcher path is unaffected by this config.
    """

    mode: Literal["fixed", "per_rollout", "per_glimpse"] = "fixed"
    """``fixed``: one constant scale everywhere (= current full-image training).
    ``per_rollout``: one scale per rollout (per image), held across its glimpses.
    ``per_glimpse``: a fresh scale every glimpse."""
    distribution: Literal["uniform", "safebox"] = "uniform"
    """Sampled-scale distribution (ignored when ``mode='fixed'``). ``uniform``:
    ``scale ~ U(min_scale, max_scale)`` with centers uniform over ``[-1,1]^2``
    (``max_scale > 1`` allows zoom-out). ``safebox``: reuse the uniform-patcher
    safe-box joint sampler (center coupled to scale, no overshoot, ``scale<=1``)."""
    fixed_scale: float = 1.0
    """Scale used when ``mode='fixed'`` (1.0 = full-image foveation)."""
    min_scale: float = 0.5
    max_scale: float = 1.0
    """Sampled-scale range (scale units = fraction of the image side)."""


@dataclass
class JointPolicyConfig:
    """Joint task+policy training (unification P4b): learn a ViewpointScorer *while*
    the distill task trains, driving the glimpses from the policy's candidate grid.

    A policy is built iff the run's ``TrainSpec`` asks for one (``train_policy`` or
    ``policy_weight > 0``); with no policy the loop builds the historical RandomSelector
    and behaves byte-for-byte like pre-P4b pretraining (the parity gate). When a policy
    IS trained, the exploratory glimpses (t>=1) of the policy branches come from
    the scorer's discrete candidate grid via ε-greedy (QReg) or on-policy sampling
    (PG); the per-glimpse reward is the fractional reduction in per-image distill MSE
    (master plan §3), standardized per depth. The action space is the fixation grid
    for foveated/square models and the safe-box grid for uniform.
    """

    rl_weight: float = 1.0
    """Scale on the policy loss added to the per-glimpse distill loss."""
    feats_detached: bool = True
    """Detach the canvas state feeding the scorer, so the policy gradient reaches
    ONLY the scorer (backbone/head shaped purely by the distill loss). False couples
    them: the policy loss also reshapes the backbone ('glimpse-plannable' features),
    more ambitious and more memory (backbone activations enter the policy graph)."""
    select_bn_eval: bool = True
    """Pick the glimpse with a separate EVAL-mode forward of the scorer ("BN mode (b)").

    **DEFAULT since 2026-07-30 (owner-approved): this is the configuration that reproduces
    the published qband checkpoints.** Measured on full ADE20K val at the last step, scored
    by the eval validated against all 8 published policies:

        harness mode (b) : CE 0.6876 / 0.6865   mIoU t4 44.80 / 44.90
        harness mode (a) : CE 0.6865 / 0.6869   mIoU t4 44.81 / 44.61
        band, last step  : CE 0.6863             mIoU t4 44.91

    The scorer's one BatchNorm makes train-mode selection normalize on batch statistics
    where the reference uses running statistics — 45.7% of chosen glimpses differ. It is
    also the train/deploy-consistent choice: deployment always selects under eval-mode BN,
    so mode (a) trained on a state distribution the deployed policy never visits.

    Costs one extra scorer forward per glimpse (~9% step time, backbone path untouched).
    `--rl.no-select-bn-eval` restores mode (a). The `run_rollout` parity digest is
    unaffected either way: it is measured with no policy attached."""
    keep_random_branch: bool = False
    """False (default): every branch is a policy branch (all t>=1 glimpses are the
    policy's grid picks; the distill loss trains on exactly those states). True:
    the FULL-start branches become policy branches while the RANDOM-start branches
    stay pure continuous-random (distill-only, no policy loss) for broad view
    coverage — needs n_full_start_branches>=1 and n_random_start_branches>=1."""

    # Objective
    objective: Literal["qreg", "pg", "vpg"] = "qreg"
    """``qreg``/``pg``: the CanViT-PyTorch-RL recipes (per-glimpse fractional reward,
    EMA-standardized). ``vpg``: vanilla policy gradient with a learned V(s) baseline and a
    TERMINAL reward — the ``autoreg_tryout`` recipe (see :class:`~canvit.train.rl.VPG`).
    ``vpg`` targets the FOVEATED/SQUARE patcher (the only kind autoreg ever ran: a pure
    fixation heatmap, no scale dimension, centred t0), forces ``dueling=True`` (that is
    where V(s) lives), and is incompatible with ``bptt.mode='chunked'`` +
    ``policy_grad_to_backbone=True``."""
    prime_on_policy: float = 0.5
    """QReg ε-greedy: fraction of glimpses taken by the net's argmax (rest = a random
    candidate). Ramped 0 -> this over ``policy_warmup_steps`` (the ε-curriculum)."""
    dueling: bool = True
    entropy_bonus: float = 0.01
    entropy_target: float | None = 1.0
    alpha_lr: float = 0.05
    # PG entropy dual ascent: log(alpha) += alpha_lr * (entropy_target - mean entropy), clipped.
    # The same update as FoAViS Appendix J (lambda <- lambda * exp(eta (H* - H))), in log space.
    # None = the historical behaviour, so no existing run changes.
    entropy_alpha_init: float | None = None
    """Initial alpha. None = ``entropy_bonus`` (historical). FoAViS: 1e-3."""
    entropy_alpha_min: float | None = None
    """Lower clip of alpha. None = ``entropy_bonus`` (historical: init and floor coincide). FoAViS: 1e-6."""
    entropy_alpha_max: float | None = None
    """Upper clip of alpha. None = ``rl.ALPHA_MAX`` = 10 (historical). FoAViS: 0.1."""
    entropy_update: Literal["per_glimpse", "per_rollout"] = "per_glimpse"
    """``per_glimpse`` (historical): one dual step per policy glimpse, i.e. T-1 per training
    step, each on that glimpse's batch-mean entropy. ``per_rollout`` (FoAViS): one step per
    rollout on the entropy averaged over its decision steps."""
    policy_warmup_steps: int = 0
    """Steps to ramp prime_on_policy 0 -> its target (0 = constant target from step 0)."""

    # Reward, and how it is centred and scaled before it multiplies log pi (or is regressed,
    # QReg). "default" resolves to each objective's historical behaviour (see
    # `canvit.harness.policy.rl.resolve_reward`), so no existing run changes. Combinations an
    # objective cannot honour are refused when the policy is built.
    reward: Literal["default", "fractional", "difference", "terminal"] = "default"
    """``fractional``: (L_{t-1} - L_t) / L_{t-1}, credited to the action that chose glimpse t
    (qreg/pg default). ``difference``: L_{t-1} - L_t, same credit (FoAViS's per-step reward).
    ``terminal``: -L_T of the last glimpse, given to every action (vpg default; Haoze's
    search default). L is the task's ``per_image_loss``."""
    reward_center: Literal["default", "none", "running_mean", "batch_mean"] = "default"
    """``running_mean``: subtract an EMA of the reward, one per depth (qreg/pg default, the
    mean half of ``RunningNorm``). ``batch_mean``: subtract the batch mean at each step
    (FoAViS ``per_step``). ``none``: no centring."""
    reward_scale: Literal["default", "none", "running_std", "batch_std", "ema_abs"] = "default"
    """``running_std``: divide by the EMA std, one per depth (qreg/pg default, the std half of
    ``RunningNorm``). ``batch_std``: divide by the batch std at each step (FoAViS
    ``per_step``). ``ema_abs``: divide by an EMA of the mean absolute reward and clip to +-10
    (Haoze). ``none``: no scaling."""

    # VPG only (objective="vpg"); ignored by qreg/pg. Defaults = autoreg exp12's values.
    vpg_entropy_bonus: float = 5e-3
    """FIXED entropy weight (autoreg ``rl_entropy_weight``). Separate from ``entropy_bonus``
    because that one doubles as PG's dual-ascent alpha init/floor and VPG has no dual."""
    vpg_reinforce_weight: float = 1.0
    vpg_baseline_weight: float = 1.0
    vpg_normalize_advantage: bool = True
    """Normalize the advantage per timestep across the batch (autoreg exp12: True)."""
    vpg_value_bias_init: float | None = None
    """Warm-start for V(s)'s output bias. autoreg uses -log(num_classes); that is wrong
    here (a policy run starts from a pretrained probe at CE ~0.8, not chance ~5.0), so the
    default leaves torch's init."""

    # Action space / scorer net
    centers_per_axis: int = 16
    scales: tuple[float, ...] = (0.5, 0.25)
    """Uniform-patcher candidate scales; ignored for foveated/square (fixation grid)."""
    width: int = 128
    block_layers: int = 3
    policy_readout: Literal["unet", "local"] = "unet"
    """Scorer body — how much of the scene ONE candidate's score sees.

    ``unet`` (default, historical): a ConvNeXt-V2 U-Net pooling 32->1, so every score is
    conditioned on a global bottleneck. ``local``: score each candidate straight off the
    (purely per-token) Frontend map with a 1x1 conv, so a candidate's score depends only on
    its own canvas cell — the ``autoreg_tryout`` policy head (``Linear(D,1)`` per state
    token). ``block_layers`` is unused under ``local``.

    Exact one-cell-per-score alignment needs ``centers_per_axis=32`` (the POLICY_GRID the
    features arrive on); coarser grids interpolate a 2x2 neighbourhood. autoreg's heatmap
    was defined ON its state grid, so 32 is the faithful setting."""
    feature_groups: tuple[str, ...] | None = None
    """What the scorer LOOKS AT, overriding the task's own set. ``None`` (default) = the
    task's default: the full probe-entropy set for ade20k, ``INTRINSIC_GROUPS`` (probe-free)
    for in1k/distill. This field used to default to ``INTRINSIC_GROUPS`` and was then
    SILENTLY IGNORED — every task passed its own module constant to ``build_policy`` — so
    the knob existed and did nothing. ``None`` keeps each task's default intact (no run
    changes) while making the override real.

    ``("ln_feat",)`` is the "just the raw canvas" setting: ``ln_feat`` is the canvas spatial
    features with a per-token channel LayerNorm and no other derivation. The remaining
    groups (``cos_prev``/``cos_init``/``feat_delta``/``ent``/``ent_delta``) are the engineered
    ones. The LayerNorm is load-bearing — raw backbone features have std ~186, which is why
    the scorer body uses LayerNorm2d rather than BatchNorm."""

    # Target standardizer + policy optimizer. These are the CanViT-PyTorch-RL canonical
    # values and they apply to a policy group on ANY task (distill / ade20k / in1k) —
    # the scorer's recipe belongs to the scorer, not to whatever task it rides on.
    target_momentum: float = 0.997
    policy_lr: float = 2e-4
    policy_weight_decay: float = 1e-2
    policy_betas: tuple[float, float] = (0.9, 0.95)
    """AdamW betas for the scorer. NOT torch's default: the RL recipe uses beta2=0.95,
    and the harness silently used 0.999 until 2026-07-29 (doc 15 §A, gap #1)."""
    policy_warmup_frac: float = 0.125
    """Fraction of the run spent linearly ramping the scorer's LR, then HOLD
    (``warmup_constant``) — `rl_train.py`'s `warmup_frac`. The harness previously left
    the policy group on ``ScheduleSpec()`` = warmup_steps 0, i.e. no ramp at all
    (doc 15 §A, gap #2). 0.0 disables the ramp.

    Needs a run length to resolve against. ade20k/in1k have ``cfg.max_steps``; DISTILL
    DOES NOT — it is SLURM-array-shaped (``steps_per_job``) and its total is not known at
    config time. On such a task use ``policy_warmup_steps`` instead; ``resolve_spec``
    warns rather than silently ramping over 0 steps."""
    policy_warmup_steps: int = 0
    """Absolute scorer warmup, in steps. Wins over ``policy_warmup_frac`` when > 0. This
    is the escape hatch for tasks with no config-time total (distill). The RL recipe is
    0.125 * 8000 = 1000 steps if you want to set it explicitly."""

