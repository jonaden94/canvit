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
from torch import Tensor, nn

from canvit.core.policy import (
    ViewpointScorer,
)
from canvit.harness.policy.rl import (
    AC,
    PG,
    QTD,
    SAC,
    SAC_ALPHA_MIN,
    VPG,
    Objective,
    QReg,
    RunningNorm,
    ac_loss,
    entropy_floor_step,
    gae,
    pg_loss,
    qreg_loss,
    sac_loss,
    vpg_loss,
)
from canvit.harness.rollout.selector import PolicySelector, RandomSelector
from canvit.harness.rollout.viewpoint import ViewpointType


class TwinCritic(nn.Module):
    """SAC's two soft-Q critics as ONE module, so they ride ``JointPolicy.critic`` -- the
    policy param group, the sidecar, DDP sync -- exactly like the Q-Prop / AC critic.
    forward(feats [N, C, H, W]) -> (Q1, Q2), each [N, A]."""

    def __init__(self, q1: ViewpointScorer, q2: ViewpointScorer):
        super().__init__()
        self.q1, self.q2 = q1, q2

    def forward(self, feats: Tensor) -> tuple[Tensor, Tensor]:
        n = feats.shape[0]
        return self.q1(feats).reshape(n, -1), self.q2(feats).reshape(n, -1)


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
    target_net: nn.Module | None = None
    """QTD: the Polyak-averaged target copy of ``scorer``. SAC: of ``critic`` (the twin)."""
    optimizer: object | None = None
    """Set by ``run()`` after the optimizer exists. Only AC with ppo_epochs > 1 uses it,
    for its inner PPO passes; every other objective ignores it."""
    grad_clip: float = 1.0
    critic: nn.Module | None = None
    """Q-Prop (PG objective, obj.qprop=True) and AC: a ViewpointScorer; SAC: a
    :class:`TwinCritic`. None for every other objective/config (qreg, qtd, vpg, plain pg)
    -- every method below that branches on `self.critic is not None` is therefore an exact
    no-op for those."""

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
        if not isinstance(self.objective, (QReg, QTD)):
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
            **({"target_net": self.target_net.state_dict()} if self.target_net is not None else {}),
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
        if sd.get("target_net") is not None and self.target_net is not None:
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
        :meth:`glimpse_loss` inside the loop (VPG). False => the historical inline path."""
        return isinstance(self.objective, (VPG, AC, QTD, SAC))

    @property
    def needs_step_rewards(self) -> bool:
        """Deferred objectives that ALSO need the per-glimpse fractional reward (AC, QTD):
        ``run_rollout`` then measures it every glimpse exactly as the inline path does,
        buffers it with the state features, and calls :meth:`episode_loss`. VPG (terminal
        reward only) keeps its own historical path."""
        return isinstance(self.objective, (AC, QTD, SAC))

    @property
    def extra_candidates(self) -> int:
        """Dense QReg: random extra cells glimpsed per state for supervision (0 = off)."""
        return self.objective.extra_candidates if isinstance(self.objective, QReg) else 0

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

    def glimpse_loss(self, *, depth: int, scores: Tensor, flat_idx: Tensor, reward: Tensor,
                     extra_idx: Tensor | None = None, extra_reward: Tensor | None = None) -> Tensor:
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
        elif extra_idx is not None:  # dense QReg: taken cell + K random cells, one MSE
            assert extra_reward is not None
            B, K = extra_idx.shape
            target = self._norm(depth).normalize(torch.cat([reward, extra_reward.reshape(-1)])).detach()
            idx = torch.cat([flat_idx, extra_idx.reshape(-1)])
            rows = torch.cat([torch.arange(B, device=scores.device),
                              torch.arange(B, device=scores.device).repeat_interleave(K)])
            loss, _ = qreg_loss(scores[rows], idx, target)
        else:
            target = self._norm(depth).normalize(reward).detach()
            loss, _ = qreg_loss(scores, flat_idx, target)
        return self.rl_weight * loss

    # --- policy-algos branch: whole-episode credit with per-glimpse rewards (AC, QTD) ---

    def _value(self, feats: Tensor) -> Tensor:
        """AC state value V(s) [B]: the critic ViewpointScorer's candidate map, averaged.
        The whole critic U-Net is the value function (capacity of Q-Prop's critic)."""
        assert self.critic is not None
        return self.critic(feats.float()).reshape(feats.shape[0], -1).mean(dim=1)

    def _clip_and_step(self) -> None:
        params = list(self.scorer.parameters()) + (list(self.critic.parameters()) if self.critic is not None else [])
        torch.nn.utils.clip_grad_norm_(list(self.scorer.parameters()), self.grad_clip)
        if self.critic is not None:
            torch.nn.utils.clip_grad_norm_(list(self.critic.parameters()), self.grad_clip)
        assert self.optimizer is not None, "AC with ppo_epochs > 1 needs joint.optimizer (set in run())"
        self.optimizer.step()  # type: ignore[attr-defined]
        for p in params:
            p.grad = None

    def episode_loss(self, *, scores: Tensor, flat_idx: Tensor, feats: list[Tensor], rewards: Tensor) -> Tensor:
        """The weighted policy loss for a WHOLE episode (AC / QTD), meaned over (B, T).
        ``scores`` [B, T, A] (grad, the selection-time train-mode forward), ``flat_idx``
        [B, T], ``feats`` the T detached state features, ``rewards`` [B, T] the detached
        fractional CE reductions."""
        obj = self.objective
        B, T, A = scores.shape
        r = self._norm(-1).rescale(rewards.reshape(-1)).reshape(B, T)  # global scale, no shift
        if isinstance(obj, QTD):
            return self.rl_weight * self._qtd_loss(scores=scores, flat_idx=flat_idx, feats=feats, r=r)
        if isinstance(obj, SAC):
            return self.rl_weight * self._sac_loss(scores=scores, flat_idx=flat_idx, feats=feats, r=r)
        assert isinstance(obj, AC)
        alpha = obj.entropy_bonus if self.log_alpha is None else self.log_alpha.exp()
        feats_all = torch.cat(feats, dim=0)  # [T*B, ...], glimpse-major

        def values_now() -> Tensor:  # ONE critic forward over the whole episode -> [B, T]
            return self._value(feats_all).reshape(T, B).t()

        values = values_now()  # with grad; the critic is unchanged until the first step below
        with torch.no_grad():
            v_old = values.detach()
            adv, ret = gae(r, v_old, gamma=obj.gamma, lam=obj.gae_lambda)
            if obj.normalize_advantage:
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            old_logp = torch.log_softmax(scores.detach(), dim=2).gather(2, flat_idx[..., None]).squeeze(2)

        def one_pass(sc: Tensor, vals: Tensor) -> tuple[Tensor, Tensor, dict]:
            return ac_loss(sc.reshape(B * T, A), flat_idx.reshape(-1), old_logp.reshape(-1),
                           adv.reshape(-1), vals.reshape(-1), ret.reshape(-1),
                           clip_eps=obj.clip_eps, alpha=alpha, value_weight=obj.value_weight)

        def fresh_scores() -> Tensor:  # per glimpse, so BatchNorm sees the rollout's own batches
            return torch.stack([self.scorer(f.float()).reshape(B, -1) for f in feats], dim=1)

        # Pass 1 reuses the rollout's own grad-carrying scores and the values above (nothing
        # has stepped yet, so they ARE the current net's); later passes recompute both.
        sc, vals = scores, values
        for _ in range(obj.ppo_epochs - 1):  # PPO inner passes: own optimizer steps
            loss, _, _ = one_pass(sc, vals)
            (self.rl_weight * loss).backward()
            self._clip_and_step()
            sc, vals = fresh_scores(), values_now()
        loss, entropy, metrics = one_pass(sc, vals)
        if self.log_alpha is not None:
            assert obj.entropy_target is not None
            entropy_floor_step(log_alpha=self.log_alpha, entropy=entropy.detach(), target=obj.entropy_target,
                               alpha_lr=obj.alpha_lr, alpha_min=obj.entropy_bonus)
        self.last_step = metrics | {"adv_raw_std": (ret - v_old).std()}
        return self.rl_weight * loss

    def _qtd_loss(self, *, scores: Tensor, flat_idx: Tensor, feats: list[Tensor], r: Tensor) -> Tensor:
        obj = self.objective
        assert isinstance(obj, QTD) and self.target_net is not None
        B, T, A = scores.shape
        with torch.no_grad():
            boot = torch.zeros_like(r)
            for t in range(T - 1):  # double-Q: online argmax at s_{t+1}, target-net value
                a_star = scores[:, t + 1].detach().argmax(dim=1)
                q_tgt = self.target_net(feats[t + 1].float()).reshape(B, -1)
                boot[:, t] = q_tgt.gather(1, a_star[:, None]).squeeze(1)
            y = r + obj.gamma * boot  # last glimpse: boot = 0 (the episode ends)
        loss, m = qreg_loss(scores.reshape(B * T, A), flat_idx.reshape(-1), y.reshape(-1))
        self.last_step = {"q_sel": m["q_sel"].detach(), "td_target": y.mean(), "boot_mean": boot[:, :-1].mean()}
        return loss

    def _sac_loss(self, *, scores: Tensor, flat_idx: Tensor, feats: list[Tensor], r: Tensor) -> Tensor:
        obj = self.objective
        assert isinstance(obj, SAC) and self.critic is not None and self.target_net is not None
        assert self.log_alpha is not None
        B, T, A = scores.shape

        def bta(q: Tensor, t: int) -> Tensor:  # [t*B, A] glimpse-major -> [B, t, A]
            return q.reshape(t, B, A).transpose(0, 1)

        q1, q2 = self.critic(torch.cat(feats, dim=0).float())  # ONE forward per critic over the episode
        with torch.no_grad():
            if T > 1:
                t1, t2 = self.target_net(torch.cat(feats[1:], dim=0).float())
                t1, t2 = bta(t1, T - 1), bta(t2, T - 1)
            else:
                t1 = t2 = scores.new_zeros(B, 0, A)
        alpha = self.log_alpha.exp()
        loss, entropy, metrics = sac_loss(scores, bta(q1, T), bta(q2, T), t1, t2, flat_idx, r,
                                          alpha=alpha, gamma=obj.gamma)
        entropy_floor_step(log_alpha=self.log_alpha, entropy=entropy.detach(), target=obj.target_entropy,
                           alpha_lr=obj.alpha_lr, alpha_min=SAC_ALPHA_MIN)
        self.last_step = metrics | {"alpha": alpha}
        return loss

    @torch.no_grad()
    def after_optimizer_step(self) -> None:
        """Polyak-update the target net (called by the loop after every optimizer step):
        QTD's tracks the scorer, SAC's the twin critic. A no-op for every other objective."""
        if self.target_net is None:
            return
        src = self.critic if isinstance(self.objective, SAC) else self.scorer
        assert src is not None
        tau = self.objective.target_tau  # type: ignore[union-attr]
        for pt, p in zip(self.target_net.parameters(), src.parameters()):
            pt.lerp_(p, tau)
        for bt, b in zip(self.target_net.buffers(), src.buffers()):
            bt.copy_(b)

