"""Generic RL training machinery for viewing policies (unification P3).

Ported from canvit_pytorch_rl.training.{config,stats,train}: the objective sum
type (QReg | PG), the online target standardizer (RunningNorm), the SAC-style
entropy-floor dual step, and the per-objective loss composition. Task-agnostic:
the reward is whatever per-glimpse fractional loss reduction the caller measured
(seg CE here, distill MSE later — master plan §3)."""

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor

ALPHA_MAX = 10.0  # entropy-floor dual cap: a TIGHT cap binds and defeats the floor


@dataclass(frozen=True)
class QReg:
    """Value regression — the recipe: MSE at the taken cell, ε-greedy rollout, argmax deploy."""

    prime_on_policy: float = 0.5  # fraction of state-advancing glimpses taken by the net's argmax (DAgger)
    dueling: bool = True  # Q(s,a) = V(s) + mean-zero A(s,a); argmax (deploy) unchanged


@dataclass(frozen=True)
class PG:
    """Actor — the SAME net trained by on-policy score-function credit: softmax sampling
    over the candidate readout, loss -(z * log pi(a|s)) - alpha * H(pi) with alpha held
    by the entropy floor (dual ascent to entropy_target), argmax deploy. Dueling is
    structurally absent (softmax is shift-invariant: V(s) cancels)."""

    entropy_bonus: float = 0.01  # alpha init AND floor (alpha_min); fixed alpha when entropy_target=None
    entropy_target: float | None = 1.0  # nats; dual ascent holds mean policy entropy AT the target
    alpha_lr: float = 0.05  # dual step size on log(alpha)
    alpha_init: float | None = None  # None = entropy_bonus (historical)
    alpha_min: float | None = None  # None = entropy_bonus (historical)
    alpha_max: float | None = None  # None = ALPHA_MAX (historical)
    entropy_update: Literal["per_glimpse", "per_rollout"] = "per_glimpse"  # FoAViS: per_rollout
    qprop: bool = False  # exact-expectation discrete Q-Prop: second net as control-variate critic
    z_subtract_only: bool = False  # advantage = reward - running mean, NO std division
    credit: Literal["immediate", "return"] = "immediate"


@dataclass(frozen=True)
class VPG:
    """Vanilla policy gradient with a LEARNED state-value baseline — the port of the
    ``autoreg_tryout`` REINFORCE recipe (``jon_exp12_imagenet_rvm_fov_plus_rl_xareadout_nooverrides_bl``:
    ``rl_loss_stepwise=False``, ``rl_base_reward=neg_ce``, ``rl_normalize_reward=False``,
    ``rl_normalize_advantage=True``, ``rl_grad_mode=heads_only``).

    **Targets the FOVEATED/SQUARE patcher.** ``autoreg_tryout`` only ever ran a foveated
    model, and the foveated action space is the faithful one: ``fixation_candidates`` gives
    ``centers_per_axis**2`` fixation centres over the full field with NO scale dimension,
    and (per ``policy/net.py``) those cell centres align with the scorer's score-map pixels,
    so the readout degenerates to reading a heatmap over grid cells — exactly autoreg's
    ``heatmap_head``. The forced t0 anchor is likewise a CENTRED fixation, as it is there.
    ``centers_per_axis=14`` reproduces autoreg's 14x14 splatting grid exactly (default 16).

    VPG runs on the uniform patcher too and is internally consistent there, but the action
    space becomes ``n_scale * centers_per_axis**2`` centre-AND-scale pairs and t0 becomes a
    full-scene view — a different problem from the one this recipe was tuned on, so
    ``build_policy`` warns.

    Differs from :class:`PG` in all four of the things that make it "the textbook
    algorithm" rather than the CanViT-PyTorch-RL recipe:

    ==================  ==========================  ==============================
    aspect              PG (the RL-repo recipe)     VPG (this)
    ==================  ==========================  ==============================
    reward              per-glimpse FRACTIONAL      TERMINAL ``-per_image_loss`` of
                        loss reduction at t         the LAST glimpse, broadcast to
                                                    every action step
    baseline            ``RunningNorm`` EMA of      LEARNED ``V(s_t)`` (the scorer's
                        the reward stream           dueling ``vhead``), MSE-regressed
                                                    on the return
    advantage scale     EMA std of the reward       per-timestep mean/std over the
                                                    BATCH
    entropy             dual-ascent alpha holding   FIXED weight (no floor, no dual)
                        a mean-entropy floor
    ==================  ==========================  ==============================

    Because the reward is terminal, credit cannot be assigned inside the glimpse loop:
    ``run_rollout`` buffers ``(scores, flat_idx, value)`` per step and applies
    :func:`vpg_loss` ONCE at rollout end. See ``JointPolicy.defers_credit``.

    Requires ``dueling=True`` on the scorer: that is what gives ``V(s)`` its own
    parameters (``ViewpointScorer.vhead``). Softmax is shift-invariant, so the ``+V(s)``
    term provably cannot perturb the action distribution and the score-function term
    provably cannot reach ``vhead`` — the two losses stay on disjoint parameters.
    """

    entropy_bonus: float = 5e-3  # FIXED entropy weight (autoreg's rl_entropy_weight)
    reinforce_weight: float = 1.0  # autoreg's rl_reinforce_weight
    baseline_weight: float = 1.0  # autoreg's rl_baseline_weight
    normalize_advantage: bool = True  # autoreg's rl_normalize_advantage (exp12: True)
    value_bias_init: float | None = None
    """Warm-start for ``vhead``'s output bias = the expected return at init, so the
    baseline MSE does not open with a large spike. autoreg uses ``-log(num_classes)``
    (chance CE on ImageNet, ~-6.9); that is WRONG here — a CanViT policy run starts from
    a PRETRAINED probe whose CE is ~0.8, not ~5.0. None (default) leaves torch's init and
    lets advantage normalization absorb the level."""


Objective = QReg | PG | VPG

RewardKind = Literal["fractional", "difference", "terminal"]
RewardCenter = Literal["none", "running_mean", "batch_mean"]
RewardScale = Literal["none", "running_std", "batch_std", "ema_abs"]
EMA_ABS_MOMENTUM = 0.99  # Haoze's `_normalize_reward` alpha
EMA_ABS_CLIP = 10.0  # Haoze's `_reward_clip_value`


@dataclass(frozen=True)
class RewardSpec:
    """The resolved reward: what is measured per glimpse, then how it is centred and
    scaled before it becomes the policy-gradient weight (PG) or regression target (QReg).
    Unused fields for VPG, which owns its learned baseline and advantage normalisation."""

    kind: RewardKind
    center: RewardCenter
    scale: RewardScale


def resolve_reward(*, objective: Objective, reward: str = "default", center: str = "default",
                   scale: str = "default") -> RewardSpec:
    """Resolve ``default`` to the objective's historical behaviour and refuse combinations
    it cannot honour. QReg/PG default: fractional reward standardised by ``RunningNorm``
    (``running_mean`` + ``running_std``; PG ``z_subtract_only`` = ``running_mean`` +
    ``none``). VPG: terminal reward and nothing else."""
    if isinstance(objective, VPG):
        if reward not in ("default", "terminal"):
            raise ValueError(f"objective='vpg' is defined by its TERMINAL reward; reward={reward!r} "
                             "is not supported (use objective='pg' for per-glimpse rewards).")
        if center != "default" or scale != "default":
            raise ValueError("objective='vpg' owns its baseline (learned V(s)) and its advantage "
                             "normalisation; reward_center / reward_scale must stay 'default'.")
        return RewardSpec(kind="terminal", center="none", scale="none")
    kind = "fractional" if reward == "default" else reward
    c = "running_mean" if center == "default" else center
    if scale != "default":
        s = scale
    elif isinstance(objective, PG) and objective.z_subtract_only:
        s = "none"
    else:
        s = "running_std"
    if isinstance(objective, QReg) and kind == "terminal":
        raise ValueError("objective='qreg' regresses the IMMEDIATE reward at the taken cell; a "
                         "terminal reward (known only after the last glimpse) is not defined for it.")
    if s == "running_std" and c != "running_mean":
        raise ValueError(f"reward_scale='running_std' is the std around the running mean, so it "
                         f"needs reward_center='running_mean' (got {c!r}).")
    return RewardSpec(kind=kind, center=c, scale=s)  # type: ignore[arg-type]


@torch.no_grad()
def entropy_floor_step(*, log_alpha: Tensor, entropy: Tensor, target: float, alpha_lr: float,
                       alpha_min: float, alpha_max: float = ALPHA_MAX) -> None:
    """One dual-ascent step on log(alpha), in place: alpha grows while mean policy
    entropy sits below the target, decays toward alpha_min above it. On-device, sync-free."""
    log_alpha += alpha_lr * (target - entropy)
    log_alpha.clamp_(min=math.log(alpha_min), max=math.log(alpha_max))


class RunningNorm:
    """Online global mean/std (EMA) of a scalar stream — standardizes the fractional
    reward across images/steps without per-scene statistics, so ONE sampled cell per
    scene is a valid SGD step. Adam-style bias correction -> unbiased from step 1.
    On-device, sync-free (under DDP each rank keeps its own EMA)."""

    def __init__(self, *, momentum: float, device: torch.device):
        self.m = momentum
        self.mean = torch.zeros((), device=device)
        self.sq = torch.zeros((), device=device)
        self.count = 0

    @torch.no_grad()
    def normalize(self, x: Tensor, *, subtract_only: bool = False) -> Tensor:
        self.count += 1
        self.mean = self.m * self.mean + (1 - self.m) * x.mean()
        self.sq = self.m * self.sq + (1 - self.m) * (x * x).mean()
        bc = 1 - self.m**self.count  # bias correction
        mean, sq = self.mean / bc, self.sq / bc
        if subtract_only:
            return x - mean
        return (x - mean) / (sq - mean**2).clamp_min(1e-4).sqrt()


@torch.no_grad()
def standardize_reward(
    r: Tensor, *, spec: RewardSpec, running: RunningNorm, ema_abs: Tensor | None,
) -> tuple[Tensor, Tensor | None]:
    """Centre then scale one depth's raw reward ``r`` [N]. Returns the result and the
    updated ``ema_abs`` state (only ``scale='ema_abs'`` uses it).

    ``running_mean`` + ``running_std`` calls ``running.normalize(r)`` and nothing else, and
    ``running_mean`` + ``none`` calls ``normalize(r, subtract_only=True)``: exactly the
    historical QReg/PG paths, so their numbers cannot move. ``batch_std`` is the population
    std of this step's batch (FoAViS); a batch whose rewards are all equal is left unscaled
    rather than divided by ~0 (FoAViS falls back to the pooled std over steps instead, which
    this per-step path does not have). ``ema_abs`` divides by an EMA of the mean absolute raw
    reward and clips to +-10 (Haoze's ``_normalize_reward``; he applies it before his
    baseline, here it follows the centring — the same up to where the clip bites)."""
    c, s = spec.center, spec.scale
    if c == "running_mean":
        if s == "running_std":
            return running.normalize(r), ema_abs
        x = running.normalize(r, subtract_only=True)
    elif c == "batch_mean":
        x = r - r.mean()
    else:
        x = r
    if s == "batch_std":
        std = r.std(unbiased=False)
        x = torch.where(std > 1e-8, x / std.clamp_min(1e-8), x)
    elif s == "ema_abs":
        batch = r.abs().mean()
        ema_abs = batch if ema_abs is None else EMA_ABS_MOMENTUM * ema_abs + (1 - EMA_ABS_MOMENTUM) * batch
        x = (x / (ema_abs + 1e-8)).clamp(-EMA_ABS_CLIP, EMA_ABS_CLIP)
    return x, ema_abs


def qreg_loss(pred_all: Tensor, flat_idx: Tensor, target: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
    """MSE between the predicted Q at the taken cell and the standardized reward.
    pred_all [N, A] (grad), flat_idx [N], target [N] (detached)."""
    pred_sel = pred_all.gather(1, flat_idx[:, None]).squeeze(1)
    loss = F.mse_loss(pred_sel, target)
    return loss, {"train_loss": loss, "q_sel": pred_sel.mean()}


def pg_loss(
    pred_all: Tensor,
    flat_idx: Tensor,
    target: Tensor,
    *,
    alpha: Tensor | float,
    crit_all: Tensor | None = None,
) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    """Score-function credit on the on-policy taken cell (+ optional exact Q-Prop).
    Returns (loss, entropy, metrics). crit_all [N, A] (grad — its MSE is included)."""
    logp_all = F.log_softmax(pred_all, dim=1)
    pred_sel = logp_all.gather(1, flat_idx[:, None]).squeeze(1)
    entropy = -(logp_all.exp() * logp_all).sum(dim=1).mean()
    metrics: dict[str, Tensor] = {"policy_entropy": entropy, "taken_logp": pred_sel.mean(), "adv_std": target.std()}
    if crit_all is not None:  # Q-Prop: score-function on the residual + exact grad of E_pi[Q]
        crit_sel = crit_all.gather(1, flat_idx[:, None]).squeeze(1)
        critic_loss = F.mse_loss(crit_sel, target)
        analytic = (logp_all.exp() * crit_all.detach()).sum(dim=1).mean()
        loss = -((target - crit_sel.detach()) * pred_sel).mean() - analytic - alpha * entropy + critic_loss
        metrics |= {"critic_loss": critic_loss, "resid_std": (target - crit_sel.detach()).std()}
    else:
        loss = -(target * pred_sel).mean() - alpha * entropy
    return loss, entropy, metrics


def vpg_loss(
    scores: Tensor,
    flat_idx: Tensor,
    values: Tensor,
    reward: Tensor,
    *,
    entropy_bonus: float,
    reinforce_weight: float,
    baseline_weight: float,
    normalize_advantage: bool,
) -> tuple[Tensor, dict[str, Tensor]]:
    """REINFORCE + baseline MSE + entropy, over a WHOLE trajectory.

    A transcription of ``autoreg_tryout/utils/train.py::_reinforce_loss`` (the
    ``rl_loss_stepwise=False`` branch) onto the harness's tensors, with the same
    reductions: MSE meaned over (B, T), the score-function term SUMMED over T and
    meaned over B, entropy meaned over (B, T).

    ``scores`` [B, T, A] and ``values`` [B, T] carry grad; ``flat_idx`` [B, T] is the
    sampled candidate; ``reward`` [B] is the detached TERMINAL reward, broadcast over T.
    T is the number of *sampled* glimpses (t0 is the forced anchor and takes no action).
    """
    logp_all = F.log_softmax(scores, dim=2)
    logp = logp_all.gather(2, flat_idx[..., None]).squeeze(2)  # [B, T]
    entropy = -(logp_all.exp() * logp_all).sum(dim=2).mean()

    ret = reward[:, None].expand_as(values)  # [B, T] — terminal reward, broadcast
    loss_baseline = F.mse_loss(values, ret) * baseline_weight
    adv = ret - values.detach()
    if normalize_advantage:  # per timestep (column) across the batch
        adv = (adv - adv.mean(dim=0, keepdim=True)) / (adv.std(dim=0, keepdim=True) + 1e-8)
    loss_reinforce = -(logp * adv).sum(dim=1).mean() * reinforce_weight
    loss_entropy = -entropy_bonus * entropy

    loss = loss_baseline + loss_reinforce + loss_entropy
    return loss, {
        "policy_entropy": entropy.detach(),
        "loss_reinforce": loss_reinforce.detach(),
        "loss_baseline": loss_baseline.detach(),
        "value_mean": values.detach().mean(),
        "adv_std": adv.detach().std(),
        "taken_logp": logp.detach().mean(),
    }
