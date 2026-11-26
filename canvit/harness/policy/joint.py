"""Joint task+policy training orchestrator (unification P4b).

Ties the P4a selector seam to the P3 RL objectives so a ViewpointScorer learns
*inside* the pretraining rollout: the distill task keeps training while the
policy drives the glimpses from its candidate grid (in-graph, master plan §4.3;
BatchNorm mode (a) — the train-mode scorer forward that picks the action is the
one the loss reads). ``run.py`` builds a policy iff the spec asks for one
(``spec.train_policy or spec.policy_loss_active``); otherwise ``training_step``
sees ``joint=None`` and runs the byte-identical historical path (parity gate).

This module now holds :class:`JointPolicy` only. Its ``build_joint_policy``
factory — the distill-hardwired predecessor of
:func:`canvit.harness.policy.build.build_policy` — was removed on 2026-09-07: no
production path called it, and its sole remaining caller was a test asserting it
refused ``objective='vpg'``. The live equivalent of that guard is
:func:`canvit.harness.policy.check_credit_regime`, which is tested directly.

Reward (master plan §3): per glimpse, the fractional reduction in per-image
distill MSE ``r_t = (L_{t-1} - L_t) / L_{t-1}`` (measured detached by the caller),
standardized per depth by an online ``RunningNorm`` — the QReg regression target /
the PG advantage. The scorer + encoder are probe-free here (INTRINSIC feature
groups): distillation has no task probe, unlike the ADE20K policy trainer.
"""

from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn

from canvit.core.policy import (
    ViewpointScorer,
)
from canvit.harness.policy.rl import (
    ALPHA_MAX,
    PG,
    SAC,
    SAC_ALPHA_MIN,
    VPG,
    Objective,
    QReg,
    RewardSpec,
    RunningNorm,
    entropy_floor_step,
    pg_loss,
    qreg_loss,
    sac_actor_loss,
    standardize_reward,
    vpg_loss,
)
from canvit.harness.rollout.selector import PolicySelector, RandomSelector
from canvit.harness.rollout.viewpoint import ViewpointType


@dataclass(frozen=True)
class SaccadeCost:
    """Optional cost of each policy saccade on the terminal-reward PG target
    (``JointPolicyConfig.saccade_cost*``). ``d`` is the Euclidean distance between
    consecutive glimpse centres in the [-1, 1] centre units (scene width = 2).

    ``log``: c(d) = log(1 + d / r), r = half a policy-grid cell (= 1 / centers_per_axis).
    ``linear``: c(d) = d / 2, the fraction of the scene width. ``none``: no cost (default).

    ``combine='standardized'``: target_t = std(R) - weight * z_batch(c_t), where std(R) is
    the reward as the resolved ``RewardSpec`` centres/scales it; unit-free, so ``weight``
    keeps its meaning while R shrinks over training. ``combine='raw'``: the cost enters in
    reward units, target_t = std(R - weight * sum_t c_t)."""

    form: Literal["none", "log", "linear"] = "none"
    weight: float = 0.0
    combine: Literal["standardized", "raw"] = "standardized"
    r: float = 1.0 / 32
    std_eps: float = 1e-2
    """Floor on the batch std of c_t in the ``standardized`` combination (cost units; the log
    cost's batch std is ~1, the linear one's ~0.2)."""
    conditioned: bool = False
    """Stage 2 (``saccade_cost_beta_mode='conditioned'``): each trial draws its own
    beta ~ U[0, weight], the policy sees :meth:`features` (beta) next to its FiLM cue, and
    action t's target is lowered by beta_b * z_batch(c_t). False: one fixed beta = ``weight``."""
    beta_eval: float = 0.0
    """``conditioned`` only: the beta every trial gets at validation / deploy."""

    @property
    def active(self) -> bool:
        """Saccade lengths are measured (and logged). ``weight == 0`` leaves the target unchanged."""
        return self.form != "none"

    def sample_beta(self, batch_size: int, device: torch.device) -> Tensor:
        """``conditioned``: one beta per trial, U[0, weight], held for the whole trajectory."""
        assert self.conditioned
        return self.weight * torch.rand(batch_size, device=device)

    @staticmethod
    def features(beta: Tensor) -> Tensor:
        """[B] -> [B, BETA_FEATURES], appended to the policy's FiLM cue. beta = 0 is a real point."""
        beta = beta.float()
        return torch.stack([beta, torch.log1p(beta / 0.1)], dim=-1)

    def cost(self, d: Tensor) -> Tensor:
        if self.form == "log":
            return torch.log1p(d / self.r)
        assert self.form == "linear", f"no cost for form={self.form!r}"
        return d / 2.0


BETA_FEATURES = 2
"""Width of :meth:`SaccadeCost.features`: the FiLM cue of a beta-conditioned policy grows by this."""


def _batch_z(x: Tensor, mask: Tensor | None, *, eps: float) -> Tensor:
    """Population z-score over the batch (only masked-in rows; the others are 0), the std
    floored at ``eps``: a batch whose saccades are (nearly) equal long gets a (nearly) zero
    cost term instead of its rounding noise blown up to unit scale."""
    if mask is not None:
        out = torch.zeros_like(x)
        if bool(mask.any()):
            out[mask] = _batch_z(x[mask], None, eps=eps)
        return out
    return (x - x.mean()) / x.std(unbiased=False).clamp_min(eps)
class TwinCritic(nn.Module):
    """SAC's two soft-Q critics as ONE module (policy param group, sidecar, DDP sync)."""

    def __init__(self, q1: nn.Module, q2: nn.Module):
        super().__init__()
        self.q1, self.q2 = q1, q2

    def forward(self, feats: Tensor) -> tuple[Tensor, Tensor]:
        b = feats.shape[0]
        return self.q1(feats).reshape(b, -1), self.q2(feats).reshape(b, -1)


@dataclass
class JointPolicy:
    """Owns the policy machinery consulted by ``training_step`` when RL is on: the
    two selectors (policy / pure-random), the scorer module (its params go into the
    optimizer + checkpoint), the objective, the per-depth reward standardizers, and
    the PG entropy-floor dual variable."""

    policy_selector: PolicySelector
    random_selector: RandomSelector
    scorer: ViewpointScorer
    objective: Objective
    rl_weight: float
    keep_random_branch: bool
    target_momentum: float
    device: torch.device
    second_branch: str | None = None  # JointPolicyConfig.second_branch (None = the task's branches)
    prime_target: float = 1.0  # ε-curriculum: prime_on_policy ramps 0 -> this...
    prime_warmup: int = 0  # ...over this many steps (0 = constant target)
    running: dict[int, RunningNorm] = field(default_factory=dict)
    log_alpha: Tensor | None = None
    last_step: dict = field(default_factory=dict)  # detached per-step policy metrics for logging
    reward: RewardSpec = field(default_factory=lambda: RewardSpec("fractional", "running_mean", "running_std"))
    """The resolved reward (``rl.resolve_reward``). The default is the historical QReg/PG one."""
    ema_abs: dict[int, Tensor] = field(default_factory=dict)  # per-depth state of reward_scale='ema_abs'
    saccade_cost: SaccadeCost = field(default_factory=SaccadeCost)
    """Optional per-saccade cost (terminal-reward PG only). The default is off."""
    _rollout_entropies: list = field(default_factory=list)  # entropy_update='per_rollout' accumulator
    _logged_entropies: list = field(default_factory=list)  # inline PG: this rollout's entropies, for logging
    critic: TwinCritic | None = None  # SAC only: twin soft-Q critics (canvas U-Nets)
    target_net: TwinCritic | None = None  # SAC only: their Polyak-averaged copy (no grad)
    critic_encoder: object | None = None  # SAC only: StateEncoder the critics read the canvas through
    _sac_actor: list = field(default_factory=list)  # SAC: this rollout's per-step actor losses (logging)
    _sac_critic: list = field(default_factory=list)  # SAC: per-step critic losses (logging)
    _sac_td: list = field(default_factory=list)  # SAC: per-step mean TD targets (logging)

    def policy_parameters(self) -> list[nn.Parameter]:
        """Every parameter the policy optimizer group trains: the actor (scorer or policy
        tokens) and, for SAC, the critics. The target critics are not trained."""
        ps = list(self.scorer.parameters())
        if self.critic is not None:
            ps += list(self.critic.parameters())
        return ps

    def set_trainable(self, flag: bool) -> None:
        self.scorer.requires_grad_(flag)
        if self.critic is not None:
            self.critic.requires_grad_(flag)

    def clip_grads(self, max_norm: float) -> None:
        """Actor and critics clipped SEPARATELY (two losses on disjoint parameters; one
        joint norm would let a large critic gradient shrink the actor's step)."""
        torch.nn.utils.clip_grad_norm_(self.scorer.parameters(), max_norm)
        if self.critic is not None:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm)

    @property
    def extra_candidates(self) -> int:
        """Dense QReg: random extra cells glimpsed per policy state (0 = off)."""
        return self.objective.extra_candidates if isinstance(self.objective, QReg) else 0

    @property
    def needs_step_rewards(self) -> bool:
        """SAC: deferred to the episode end, but with the per-glimpse reward, the state
        features and the taken cell buffered per glimpse (``run_rollout``)."""
        return isinstance(self.objective, SAC)

    @torch.no_grad()
    def critic_feats(self, state: object) -> Tensor:
        """The critics' (detached) input at a policy state."""
        assert self.critic_encoder is not None
        return self.critic_encoder(state).float()  # type: ignore[operator]

    @torch.no_grad()
    def after_optimizer_step(self) -> None:
        """Polyak-update SAC's target critics (the loop calls this after every optimizer
        step). A no-op for every other objective."""
        if self.target_net is None:
            return
        assert self.critic is not None and isinstance(self.objective, SAC)
        for pt, p in zip(self.target_net.parameters(), self.critic.parameters()):
            pt.lerp_(p, self.objective.target_tau)
        for bt, b in zip(self.target_net.buffers(), self.critic.buffers()):
            bt.copy_(b)

    def sac_actor_step(self, *, scores: Tensor, feats: Tensor) -> Tensor:
        """SAC's weighted actor loss at ONE policy state (eq. 12), meaned over the batch, with
        the critics at that state (no grad) and the temperature fixed for the rollout. Built per
        glimpse so it leaves with the glimpse's TBPTT chunk; the entropy is kept for the
        temperature step in :meth:`sac_critic_step`."""
        assert isinstance(self.objective, SAC) and self.critic is not None and self.log_alpha is not None
        with torch.no_grad():
            q1, q2 = self.critic(feats)
            qmin = torch.min(q1, q2)
        loss, entropy = sac_actor_loss(scores, qmin, alpha=self.log_alpha.exp())
        self._rollout_entropies.append(entropy.detach())
        self._sac_actor.append(loss.detach())
        return self.rl_weight * loss

    def sac_reward(self, reward: Tensor) -> Tensor:
        """SAC's reward scale: divided by a running std, no mean shift (one EMA for all steps)."""
        return self._norm(-1).rescale(reward)

    def sac_critic_term(self, *, feats: Tensor, flat_idx: Tensor, r: Tensor,
                        next_feats: Tensor | None = None, next_logp: Tensor | None = None) -> Tensor:
        """SAC's weighted twin-critic loss (eq. 10) for ONE policy step, meaned over the batch:
        1/2 sum_i (Q_i(s_t, a_t) - y_t)^2 with y_t = r_t + gamma pi(s_{t+1})^T [min_i Qbar_i(s_{t+1})
        - alpha log pi(s_{t+1})], or y_t = r_t at the last glimpse (``next_feats=None``) and for
        gamma = 0. Built one glimpse late (when s_{t+1} is known), so memory does not grow with
        the trajectory; the mean over steps equals ``rl.sac_critic_loss`` on the episode."""
        obj = self.objective
        assert isinstance(obj, SAC) and self.critic is not None and self.target_net is not None
        assert self.log_alpha is not None
        q1, q2 = self.critic(feats)
        with torch.no_grad():
            y = r.clone()
            if next_feats is not None and obj.gamma != 0:
                assert next_logp is not None
                t1, t2 = self.target_net(next_feats)
                y = y + obj.gamma * (next_logp.exp() * (torch.min(t1, t2)
                                                        - self.log_alpha.exp() * next_logp)).sum(dim=1)
        qa1 = q1.gather(1, flat_idx[:, None]).squeeze(1)
        qa2 = q2.gather(1, flat_idx[:, None]).squeeze(1)
        loss = 0.5 * (F.mse_loss(qa1, y) + F.mse_loss(qa2, y))
        self._sac_critic.append(loss.detach())
        self._sac_td.append(y.mean())
        return self.rl_weight * loss

    def sac_end_rollout(self) -> None:
        """After a policy branch: the temperature's dual step on the rollout's mean entropy
        (eq. 11; alpha was fixed during the rollout) and the logged SAC metrics."""
        obj = self.objective
        assert isinstance(obj, SAC) and self.log_alpha is not None
        alpha = self.log_alpha.exp().detach().float()
        entropy = torch.stack(self._rollout_entropies).mean()
        self.last_step = {"policy_entropy": entropy.float(), "alpha": alpha,
                          "actor_loss": torch.stack(self._sac_actor).mean(),
                          "critic_loss": torch.stack(self._sac_critic).mean(),
                          "td_target": torch.stack(self._sac_td).mean()}
        self._rollout_entropies, self._sac_actor, self._sac_critic, self._sac_td = [], [], [], []
        entropy_floor_step(log_alpha=self.log_alpha, entropy=entropy, target=obj.target_entropy,
                           alpha_lr=obj.alpha_lr, alpha_min=SAC_ALPHA_MIN)

    def branches(self, task_branches: list[ViewpointType]) -> list[ViewpointType]:
        """The rollout's branch t0 types: the task's, or with ``second_branch`` set one policy
        branch (FULL) plus the second branch (none / RANDOM start / FULL start)."""
        if self.second_branch is None:
            return task_branches
        second = {"none": [], "random_start": [ViewpointType.RANDOM],
                  "center_start": [ViewpointType.FULL]}[self.second_branch]
        return [ViewpointType.FULL] + second

    def branch_selector(self, t0_type: ViewpointType, branch_idx: int = 0) -> tuple[object, bool]:
        """(selector, is_policy_branch) for a branch with this t0 type. Policy
        branches are FULL-anchored (the caller forces t0=FULL and rewards t>=1);
        with ``keep_random_branch`` the RANDOM-start branches stay pure-random
        (distill-only, no policy loss). Off => every branch is a policy branch.
        With ``second_branch`` set, branch 0 is the policy branch and the other is random."""
        if self.second_branch is not None:
            is_policy = branch_idx == 0
        else:
            is_policy = (not self.keep_random_branch) or (t0_type == ViewpointType.FULL)
        return (self.policy_selector, True) if is_policy else (self.random_selector, False)

    def set_prime_for_step(self, step: int) -> None:
        """ε-curriculum: ramp prime_on_policy 0 -> prime_target over prime_warmup
        steps, then hold (QReg only; PG always samples on-policy, prime ignored)."""
        if not isinstance(self.objective, QReg):
            return
        frac = 1.0 if self.prime_warmup <= 0 else min(1.0, step / self.prime_warmup)
        self.policy_selector.prime_on_policy = self.prime_target * frac

    # --- DDP: the scorer is NOT DDP-wrapped (its forward is deep inside the rollout),
    # so sync it manually — broadcast once so all ranks start identical, AllReduce its
    # grads each step (else per-rank scorers drift + see √N grad noise, the head bug
    # task.py documents). RunningNorm stays per-rank (rl.py) — standardization is local.
    def broadcast(self, src: int = 0) -> None:
        mods = [m for m in (self.scorer, self.critic, self.target_net) if m is not None]
        for t in [t for m in mods for t in list(m.parameters()) + list(m.buffers())]:
            dist.broadcast(t.data, src=src)

    def allreduce_grads(self) -> None:
        for p in self.policy_parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

    def state_dict(self) -> dict:
        """Policy training state for the checkpoint sidecar (scorer + reward
        standardizers + PG dual variable). The optimizer/scheduler state — including
        the policy param group — rides in the main checkpoint."""
        sd = {
            "scorer": self.scorer.state_dict(),
            "running": {d: (rn.mean, rn.sq, rn.count) for d, rn in self.running.items()},
            "log_alpha": None if self.log_alpha is None else self.log_alpha.detach().cpu(),
        }
        if self.ema_abs:  # only reward_scale='ema_abs' has this state; absent key = historical sidecar
            sd["ema_abs"] = {d: v.detach().cpu() for d, v in self.ema_abs.items()}
        if self.critic is not None:  # SAC only; absent keys = historical sidecar
            assert self.target_net is not None
            sd["critic"] = self.critic.state_dict()
            sd["target_net"] = self.target_net.state_dict()
        return sd

    def load_state_dict(self, sd: dict) -> None:
        self.scorer.load_state_dict(sd["scorer"])
        for d, (mean, sq, count) in sd.get("running", {}).items():
            rn = self._norm(int(d))
            rn.mean, rn.sq, rn.count = mean.to(self.device), sq.to(self.device), count
        if sd.get("log_alpha") is not None and self.log_alpha is not None:
            self.log_alpha = sd["log_alpha"].to(self.device)
        self.ema_abs = {int(d): v.to(self.device) for d, v in sd.get("ema_abs", {}).items()}
        if self.critic is not None:
            assert self.target_net is not None and "critic" in sd, "SAC resume: sidecar has no critics"
            self.critic.load_state_dict(sd["critic"])
            self.target_net.load_state_dict(sd["target_net"])

    def _norm(self, depth: int) -> RunningNorm:
        rn = self.running.get(depth)
        if rn is None:  # created on first use — horizon varies per step (continue_prob)
            rn = RunningNorm(momentum=self.target_momentum, device=self.device)
            self.running[depth] = rn
        return rn

    @property
    def defers_credit(self) -> bool:
        """True => the reward is TERMINAL, so ``run_rollout`` must buffer the per-step
        policy aux and call :meth:`trajectory_loss` once at rollout end instead of
        :meth:`glimpse_loss` inside the loop (VPG, or PG with ``reward='terminal'``).
        False => the historical inline path."""
        return isinstance(self.objective, (VPG, SAC)) or self.reward.kind == "terminal"

    @property
    def needs_value(self) -> bool:
        """Whether the deferred path needs ``V(s_t)`` per step (VPG's learned baseline)."""
        return isinstance(self.objective, VPG)

    def inline_reward(self, prev_loss: Tensor, cur_loss: Tensor) -> Tensor:
        """The raw per-glimpse reward [B] for the glimpse that moved the per-image task loss
        from ``prev_loss`` to ``cur_loss`` (both detached). ``fractional`` is the historical
        expression, unchanged."""
        if self.reward.kind == "fractional":
            return (prev_loss - cur_loss) / prev_loss.clamp_min(1e-4)
        assert self.reward.kind == "difference", f"no inline reward for kind={self.reward.kind!r}"
        return prev_loss - cur_loss

    def _standardize(self, depth: int, reward: Tensor, mask: Tensor | None = None) -> Tensor:
        """Centre/scale one depth's reward. With ``mask``, only the masked-in rows enter the
        statistics (batch or running); the other rows come back as 0."""
        if mask is not None:
            out = torch.zeros_like(reward)
            if bool(mask.any()):
                out[mask] = self._standardize(depth, reward[mask])
            return out
        out, self_ema = standardize_reward(
            reward, spec=self.reward, running=self._norm(depth), ema_abs=self.ema_abs.get(depth))
        if self_ema is not None:
            self.ema_abs[depth] = self_ema
        return out.detach()

    def _require_pg_for_mask(self, mask: Tensor | None) -> None:
        if mask is not None and not isinstance(self.objective, PG):
            raise NotImplementedError(
                f"a per-trial reward mask is implemented for objective='pg' only, not "
                f"{type(self.objective).__name__}")

    def _alpha(self, obj: PG) -> Tensor | float:
        return obj.entropy_bonus if self.log_alpha is None else self.log_alpha.exp()

    def _entropy_step(self, obj: PG, entropy: Tensor) -> None:
        assert self.log_alpha is not None and obj.entropy_target is not None
        entropy_floor_step(
            log_alpha=self.log_alpha, entropy=entropy.detach(),
            target=obj.entropy_target, alpha_lr=obj.alpha_lr,
            alpha_min=obj.entropy_bonus if obj.alpha_min is None else obj.alpha_min,
            alpha_max=ALPHA_MAX if obj.alpha_max is None else obj.alpha_max,
        )

    def end_rollout(self) -> None:
        """Called by ``run_rollout`` after each policy branch. Takes the ``per_rollout``
        entropy dual step on the inline path; a no-op otherwise. Inline PG: also logs the
        policy branch's mean entropy and the entropy weight alpha (after the dual step)."""
        if self._rollout_entropies and not isinstance(self.objective, SAC):
            obj = self.objective
            assert isinstance(obj, PG)
            self._entropy_step(obj, torch.stack(self._rollout_entropies).mean())
            self._rollout_entropies = []
        if self._logged_entropies:
            self.last_step = {"policy_entropy": torch.stack(self._logged_entropies).mean().float(),
                              "alpha": torch.as_tensor(self._alpha(self.objective)).detach().float()}
            self._logged_entropies = []

    def state_value(self, feats: Tensor) -> Tensor:
        """``V(s)`` [B] for the VPG baseline, read from the scorer's dueling value head.

        Deliberately a SECOND (very cheap: pool -> 64 -> 1) forward of ``vhead`` rather
        than recovering V from the score map's row mean: with ``dueling=True`` the map is
        ``V(s) + mean-zero A(s,a)`` so the row mean IS V analytically, but only to float
        error, and reading it that way would silently break if the dueling algebra ever
        changed. This keeps the baseline exact and the coupling explicit."""
        assert self.scorer.vhead is not None, (
            "VPG needs a learned V(s): build the ViewpointScorer with dueling=True"
        )
        return self.scorer.vhead(feats.float().mean(dim=(2, 3))).squeeze(-1)

    def trajectory_loss(self, *, scores: Tensor, flat_idx: Tensor, values: Tensor | None,
                        reward: Tensor, mask: Tensor | None = None,
                        saccade_len: Tensor | None = None, saccade_beta: Tensor | None = None) -> Tensor:
        """The weighted policy loss for a WHOLE trajectory (the deferred, terminal-reward
        path). ``scores`` [B, T, A] carries grad; ``reward`` [B] is the detached terminal
        reward, given to every one of the T actions. ``mask`` [B] (PG only): the trials that
        carry a reward; the others get only the entropy term (see ``pg_loss``).

        VPG (``values`` [B, T], with grad): unchanged — not per-depth, no ``RunningNorm``;
        VPG standardizes the ADVANTAGE per timestep over the batch (see :class:`VPG`).

        PG (``values=None``): per action step t, the reward is centred/scaled under the
        resolved ``RewardSpec`` (depth t+1, the depth the inline path would use) and fed to
        :func:`pg_loss`; the result is the MEAN over the T steps, the same scale as the
        inline path's mean over depths.

        ``saccade_len`` [B, T] (detached; only with an active ``saccade_cost``, PG only): the
        length of each action's saccade. Action t's target is lowered by its OWN cost c_t
        (``standardized``) or the reward by the trajectory's summed cost (``raw``); see
        :class:`SaccadeCost`. Absent => the historical computation, unchanged.
        ``saccade_beta`` [B] (``conditioned`` cost only): each trial's own beta, in place of
        the scalar ``weight``; a constant beta gives the fixed-beta target exactly."""
        obj = self.objective
        self._require_pg_for_mask(mask)
        sc = self.saccade_cost
        cost = None
        if saccade_len is not None:
            if not isinstance(obj, PG):
                raise NotImplementedError("the saccade cost is implemented for objective='pg' only")
            cost = sc.cost(saccade_len.float())
            assert (saccade_beta is not None) == sc.conditioned, "a conditioned cost needs the per-trial beta"
            if sc.combine == "raw" and sc.weight != 0.0:
                reward = reward - sc.weight * cost.sum(dim=1)
        if isinstance(obj, VPG):
            assert values is not None
            loss, metrics = vpg_loss(
                scores, flat_idx, values, reward,
                entropy_bonus=obj.entropy_bonus, reinforce_weight=obj.reinforce_weight,
                baseline_weight=obj.baseline_weight, normalize_advantage=obj.normalize_advantage,
            )
            self.last_step = metrics
            return self.rl_weight * loss
        assert isinstance(obj, PG), f"trajectory_loss: unsupported objective {type(obj).__name__}"
        n_steps = scores.shape[1]
        losses, entropies = [], []
        for t in range(n_steps):
            target = self._standardize(t + 1, reward, mask)
            if cost is not None and sc.combine == "standardized" and sc.weight != 0.0:
                beta = sc.weight if saccade_beta is None else saccade_beta
                target = target - beta * _batch_z(cost[:, t], mask, eps=sc.std_eps)
            if mask is None:
                loss_t, ent_t, _ = pg_loss(scores[:, t], flat_idx[:, t], target, alpha=self._alpha(obj),
                                           crit_all=None)
            else:
                loss_t, ent_t, _ = pg_loss(scores[:, t], flat_idx[:, t], target, alpha=self._alpha(obj),
                                           crit_all=None, mask=mask)
            losses.append(loss_t)
            entropies.append(ent_t.detach())
            if self.log_alpha is not None and obj.entropy_update == "per_glimpse":
                self._entropy_step(obj, ent_t)
        if self.log_alpha is not None and obj.entropy_update == "per_rollout":
            self._entropy_step(obj, torch.stack(entropies).mean())
        ent_mean = torch.stack(entropies).mean()
        self.last_step = {"policy_entropy": ent_mean,
                          "alpha": torch.as_tensor(self._alpha(obj)).detach().float()}
        if cost is not None:  # logged as policy/saccade_len (centre units, scene width 2), policy/saccade_cost
            assert saccade_len is not None
            self.last_step["saccade_len"] = saccade_len.float().mean()
            self.last_step["saccade_cost"] = cost.mean()
        return self.rl_weight * torch.stack(losses).mean()

    def glimpse_loss(self, *, depth: int, scores: Tensor, flat_idx: Tensor, reward: Tensor,
                     mask: Tensor | None = None, extra_idx: Tensor | None = None,
                     extra_reward: Tensor | None = None) -> Tensor:
        """The weighted policy loss for ONE glimpse. ``scores`` [B, A] carries grad
        (the train-mode scorer forward); ``flat_idx`` [B] is the taken candidate;
        ``reward`` [B] is the raw fractional distill-MSE reduction (detached). ``mask``: as
        in :meth:`trajectory_loss` (PG, and plain QReg: its MSE over the masked-in rows only)."""
        obj = self.objective
        if not (isinstance(obj, QReg) and extra_idx is None):
            self._require_pg_for_mask(mask)
        if isinstance(obj, PG):
            # Historical default (fractional, running_mean + running_std / z_subtract_only)
            # resolves to exactly `self._norm(depth).normalize(reward, subtract_only=...)`.
            target = self._standardize(depth, reward, mask)
            if mask is None:
                loss, entropy, _ = pg_loss(scores, flat_idx, target, alpha=self._alpha(obj), crit_all=None)
            else:
                loss, entropy, _ = pg_loss(scores, flat_idx, target, alpha=self._alpha(obj), crit_all=None,
                                           mask=mask)
            self._logged_entropies.append(entropy.detach())
            if self.log_alpha is not None:
                if obj.entropy_update == "per_glimpse":
                    self._entropy_step(obj, entropy)
                else:  # per_rollout: one step on the rollout's mean entropy, in end_rollout()
                    self._rollout_entropies.append(entropy.detach())
        elif extra_idx is not None:  # dense QReg: the taken cell + K random cells, one MSE
            assert extra_reward is not None and isinstance(obj, QReg)
            B, K = extra_idx.shape
            target = self._standardize(depth, torch.cat([reward, extra_reward.reshape(-1)]))
            idx = torch.cat([flat_idx, extra_idx.reshape(-1)])
            rows = torch.cat([torch.arange(B, device=scores.device),
                              torch.arange(B, device=scores.device).repeat_interleave(K)])
            loss, _ = qreg_loss(scores[rows], idx, target)
        elif mask is None:
            target = self._standardize(depth, reward)
            loss, _ = qreg_loss(scores, flat_idx, target)
        else:  # visual search: only the target-present trials carry a reward
            target = self._standardize(depth, reward, mask)
            loss = (qreg_loss(scores[mask], flat_idx[mask], target[mask])[0] if bool(mask.any())
                    else scores.sum() * 0.0)
        return self.rl_weight * loss
