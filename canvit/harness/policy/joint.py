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

import torch
import torch.distributed as dist
from torch import Tensor

from canvit.core.policy import (
    ViewpointScorer,
)
from canvit.harness.policy.rl import (
    ALPHA_MAX,
    PG,
    VPG,
    Objective,
    QReg,
    RewardSpec,
    RunningNorm,
    entropy_floor_step,
    pg_loss,
    qreg_loss,
    standardize_reward,
    vpg_loss,
)
from canvit.harness.rollout.selector import PolicySelector, RandomSelector
from canvit.harness.rollout.viewpoint import ViewpointType


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
    prime_target: float = 1.0  # ε-curriculum: prime_on_policy ramps 0 -> this...
    prime_warmup: int = 0  # ...over this many steps (0 = constant target)
    running: dict[int, RunningNorm] = field(default_factory=dict)
    log_alpha: Tensor | None = None
    last_step: dict = field(default_factory=dict)  # detached per-step policy metrics for logging
    reward: RewardSpec = field(default_factory=lambda: RewardSpec("fractional", "running_mean", "running_std"))
    """The resolved reward (``rl.resolve_reward``). The default is the historical QReg/PG one."""
    ema_abs: dict[int, Tensor] = field(default_factory=dict)  # per-depth state of reward_scale='ema_abs'
    _rollout_entropies: list = field(default_factory=list)  # entropy_update='per_rollout' accumulator

    def branch_selector(self, t0_type: ViewpointType) -> tuple[object, bool]:
        """(selector, is_policy_branch) for a branch with this t0 type. Policy
        branches are FULL-anchored (the caller forces t0=FULL and rewards t>=1);
        with ``keep_random_branch`` the RANDOM-start branches stay pure-random
        (distill-only, no policy loss). Off => every branch is a policy branch."""
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
        for t in list(self.scorer.parameters()) + list(self.scorer.buffers()):
            dist.broadcast(t.data, src=src)

    def allreduce_grads(self) -> None:
        for p in self.scorer.parameters():
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
        return sd

    def load_state_dict(self, sd: dict) -> None:
        self.scorer.load_state_dict(sd["scorer"])
        for d, (mean, sq, count) in sd.get("running", {}).items():
            rn = self._norm(int(d))
            rn.mean, rn.sq, rn.count = mean.to(self.device), sq.to(self.device), count
        if sd.get("log_alpha") is not None and self.log_alpha is not None:
            self.log_alpha = sd["log_alpha"].to(self.device)
        self.ema_abs = {int(d): v.to(self.device) for d, v in sd.get("ema_abs", {}).items()}

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
        return isinstance(self.objective, VPG) or self.reward.kind == "terminal"

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

    def _standardize(self, depth: int, reward: Tensor) -> Tensor:
        out, self_ema = standardize_reward(
            reward, spec=self.reward, running=self._norm(depth), ema_abs=self.ema_abs.get(depth))
        if self_ema is not None:
            self.ema_abs[depth] = self_ema
        return out.detach()

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
        entropy dual step on the inline path; a no-op otherwise."""
        if self._rollout_entropies:
            obj = self.objective
            assert isinstance(obj, PG)
            self._entropy_step(obj, torch.stack(self._rollout_entropies).mean())
            self._rollout_entropies = []

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
                        reward: Tensor) -> Tensor:
        """The weighted policy loss for a WHOLE trajectory (the deferred, terminal-reward
        path). ``scores`` [B, T, A] carries grad; ``reward`` [B] is the detached terminal
        reward, given to every one of the T actions.

        VPG (``values`` [B, T], with grad): unchanged — not per-depth, no ``RunningNorm``;
        VPG standardizes the ADVANTAGE per timestep over the batch (see :class:`VPG`).

        PG (``values=None``): per action step t, the reward is centred/scaled under the
        resolved ``RewardSpec`` (depth t+1, the depth the inline path would use) and fed to
        :func:`pg_loss`; the result is the MEAN over the T steps, the same scale as the
        inline path's mean over depths."""
        obj = self.objective
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
            target = self._standardize(t + 1, reward)
            loss_t, ent_t, _ = pg_loss(scores[:, t], flat_idx[:, t], target, alpha=self._alpha(obj), crit_all=None)
            losses.append(loss_t)
            entropies.append(ent_t.detach())
            if self.log_alpha is not None and obj.entropy_update == "per_glimpse":
                self._entropy_step(obj, ent_t)
        if self.log_alpha is not None and obj.entropy_update == "per_rollout":
            self._entropy_step(obj, torch.stack(entropies).mean())
        ent_mean = torch.stack(entropies).mean()
        self.last_step = {"policy_entropy": ent_mean,
                          "alpha": torch.as_tensor(self._alpha(obj)).detach().float()}
        return self.rl_weight * torch.stack(losses).mean()

    def glimpse_loss(self, *, depth: int, scores: Tensor, flat_idx: Tensor, reward: Tensor) -> Tensor:
        """The weighted policy loss for ONE glimpse. ``scores`` [B, A] carries grad
        (the train-mode scorer forward); ``flat_idx`` [B] is the taken candidate;
        ``reward`` [B] is the raw fractional distill-MSE reduction (detached)."""
        obj = self.objective
        if isinstance(obj, PG):
            # Historical default (fractional, running_mean + running_std / z_subtract_only)
            # resolves to exactly `self._norm(depth).normalize(reward, subtract_only=...)`.
            target = self._standardize(depth, reward)
            loss, entropy, _ = pg_loss(scores, flat_idx, target, alpha=self._alpha(obj), crit_all=None)
            if self.log_alpha is not None:
                if obj.entropy_update == "per_glimpse":
                    self._entropy_step(obj, entropy)
                else:  # per_rollout: one step on the rollout's mean entropy, in end_rollout()
                    self._rollout_entropies.append(entropy.detach())
        else:
            target = self._standardize(depth, reward)
            loss, _ = qreg_loss(scores, flat_idx, target)
        return self.rl_weight * loss
