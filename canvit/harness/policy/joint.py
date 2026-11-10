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
    PG,
    VPG,
    Objective,
    QReg,
    RunningNorm,
    entropy_floor_step,
    pg_loss,
    qreg_loss,
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
    critic: ViewpointScorer | None = None
    """Q-Prop only (PG objective, obj.qprop=True); None for every other objective/config
    (qreg, vpg, plain pg) -- every method below that branches on `self.critic is not None`
    is therefore an exact no-op for anything that isn't a qprop run."""

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
        if self.critic is not None:
            for t in list(self.critic.parameters()) + list(self.critic.buffers()):
                dist.broadcast(t.data, src=src)

    def allreduce_grads(self) -> None:
        for p in self.scorer.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
        if self.critic is not None:
            for p in self.critic.parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

    def state_dict(self) -> dict:
        """Policy training state for the checkpoint sidecar (scorer + reward
        standardizers + PG dual variable). The optimizer/scheduler state — including
        the policy param group — rides in the main checkpoint."""
        return {
            "scorer": self.scorer.state_dict(),
            "running": {d: (rn.mean, rn.sq, rn.count) for d, rn in self.running.items()},
            "log_alpha": None if self.log_alpha is None else self.log_alpha.detach().cpu(),
            "critic": None if self.critic is None else self.critic.state_dict(),
        }

    def load_state_dict(self, sd: dict) -> None:
        self.scorer.load_state_dict(sd["scorer"])
        for d, (mean, sq, count) in sd.get("running", {}).items():
            rn = self._norm(int(d))
            rn.mean, rn.sq, rn.count = mean.to(self.device), sq.to(self.device), count
        if sd.get("log_alpha") is not None and self.log_alpha is not None:
            self.log_alpha = sd["log_alpha"].to(self.device)
        if sd.get("critic") is not None and self.critic is not None:
            self.critic.load_state_dict(sd["critic"])

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
        :meth:`glimpse_loss` inside the loop (VPG). False => the historical inline path."""
        return isinstance(self.objective, VPG)

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

    def trajectory_loss(self, *, scores: Tensor, flat_idx: Tensor, values: Tensor, reward: Tensor) -> Tensor:
        """The weighted policy loss for a WHOLE trajectory (VPG only). ``scores``
        [B, T, A] / ``values`` [B, T] carry grad; ``reward`` [B] is the detached terminal
        reward. Unlike :meth:`glimpse_loss` this is NOT per-depth, so there is no
        ``RunningNorm`` — VPG standardizes the ADVANTAGE per timestep over the batch
        instead of the reward stream over time (see :class:`VPG`)."""
        obj = self.objective
        assert isinstance(obj, VPG)
        loss, metrics = vpg_loss(
            scores, flat_idx, values, reward,
            entropy_bonus=obj.entropy_bonus, reinforce_weight=obj.reinforce_weight,
            baseline_weight=obj.baseline_weight, normalize_advantage=obj.normalize_advantage,
        )
        self.last_step = metrics
        return self.rl_weight * loss

    def glimpse_loss(self, *, depth: int, scores: Tensor, flat_idx: Tensor, reward: Tensor) -> Tensor:
        """The weighted policy loss for ONE glimpse. ``scores`` [B, A] carries grad
        (the train-mode scorer forward); ``flat_idx`` [B] is the taken candidate;
        ``reward`` [B] is the raw fractional distill-MSE reduction (detached)."""
        obj = self.objective
        if isinstance(obj, PG):
            target = self._norm(depth).normalize(reward, subtract_only=obj.z_subtract_only).detach()
            alpha = obj.entropy_bonus if self.log_alpha is None else self.log_alpha.exp()
            crit_all = None
            if self.critic is not None:
                # Q-Prop: the critic scores the SAME state features the actor's
                # PolicySelector just featurized for this glimpse (stashed on
                # `last_aux` by `select()`, same read pattern as `state_value` uses
                # for VPG's baseline above). A live (grad-enabled) forward, since
                # `critic_loss` inside `pg_loss` must backprop into the critic's own
                # parameters.
                aux = self.policy_selector.last_aux
                assert aux is not None, "qprop critic needs this glimpse's `feats`, but last_aux is empty"
                # ViewpointScorer's raw forward is [B, n_scale, cpa, cpa], not the flat
                # [B, A] `pg_loss` expects -- match PolicySelector.select()'s own
                # `.reshape(batch_size, -1)` (selector.py:114) exactly, or `.gather`
                # below raises on a dim mismatch (caught by the exp39 smoke test).
                crit_all = self.critic(aux["feats"].float()).reshape(scores.shape[0], -1)
            loss, entropy, metrics = pg_loss(scores, flat_idx, target, alpha=alpha, crit_all=crit_all)
            if self.critic is not None:  # surface critic_loss/resid_std for logging; plain PG is unchanged
                # pg_loss's metrics dict carries live (non-detached) tensors -- e.g.
                # critic_loss, which must stay attached for its own .backward() to reach
                # the critic. Detach the copy that goes into `last_step` (log-only, read
                # after the step's backward/optimizer.step() already ran) the same way
                # vpg_loss's own returned metrics already are, so `float(v)` logging
                # doesn't warn about converting a grad-carrying tensor and doesn't keep
                # this step's graph referenced any longer than necessary.
                self.last_step = {k: v.detach() if isinstance(v, Tensor) else v for k, v in metrics.items()}
            if self.log_alpha is not None:
                assert obj.entropy_target is not None
                entropy_floor_step(
                    log_alpha=self.log_alpha, entropy=entropy.detach(),
                    target=obj.entropy_target, alpha_lr=obj.alpha_lr, alpha_min=obj.entropy_bonus,
                )
        else:
            target = self._norm(depth).normalize(reward).detach()
            loss, _ = qreg_loss(scores, flat_idx, target)
        return self.rl_weight * loss
