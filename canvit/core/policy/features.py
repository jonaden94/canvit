"""Canvas state -> scorer input: the derived, scale-equalized feature groups and the
encoder that builds them. Ported from canvit_pytorch_rl.policy.features, decoupled
from the RL TrainConfig (explicit canvas_grid / feature_groups arguments).

FEATURE_GROUPS names the groups in channel order; ent/ent_delta/cos_* are 1 channel,
ln_feat/feat_delta are canvas_dim wide. The probe-entropy groups (ent, ent_delta)
require a task probe — tasks without one (e.g. distillation pretraining) select the
four intrinsic groups (master plan §3: feature groups are task-configurable).

"rec_cls" is a fifth, OPT-IN group — absent from both FEATURE_GROUPS and
INTRINSIC_GROUPS, reachable only via `--rl.feature-groups`, so selecting it is a
no-op for every existing config/checkpoint. It is the backbone's `recurrent_cls`
summary token (cross-glimpse memory, separate from the canvas), broadcast to every
grid cell. Its width is `local_dim` — a backbone config INDEPENDENT of canvas_dim
(the model sizes them separately) — so callers that use it must supply `local_dim`
to `group_sizes`/`feature_channels`. Unlike ln_feat/feat_delta it has no init-
reference or delta variant: it is already an absolute, cross-glimpse signal.

"local_feat" is a sixth, likewise OPT-IN group: the backbone's RAW local token
output for the glimpse just taken (`CanViTOutput.local_patches`, [B, n_patch,
local_dim]) -- the glimpse's own processed features, BEFORE/AS they are written
into the canvas, as opposed to ln_feat which is the canvas's state AFTER writing
(accumulated across every glimpse so far). Mean-pooled over the patch axis to one
vector per sample (the engine does not thread per-patch SCENE POSITION to this
seam, so there is no cheap way to place it spatially within the grid the way the
canvas's own cells are placed) and broadcast like rec_cls. Also local_dim wide.
None at t0 (no glimpse has been taken yet when the t0 anchor is selected).

"depth_feat" is a seventh, likewise OPT-IN group: HOW MUCH the canvas still
changed WITHIN the glimpse just taken, between its first and its last write
(`CanViTOutput.write_snapshots`) -- as opposed to feat_delta, which measures
change ACROSS glimpses. canvas_dim wide (derived from the canvas itself, not a
local-token quantity, so unlike rec_cls/local_feat it needs no local_dim). Zero-
filled at t0 or whenever fewer than two writes happened in that glimpse (nothing
to take a within-glimpse delta over)."""

import torch
import torch.nn.functional as F
from torch import Tensor

from canvit.core.model.segmentation import CanViTForSemanticSegmentation

from .scoring import entropy_from_logits, head_logits, probe_entropy

FEATURE_GROUPS = ("ent", "ent_delta", "cos_prev", "cos_init", "ln_feat", "feat_delta")
INTRINSIC_GROUPS = ("cos_prev", "cos_init", "ln_feat", "feat_delta")  # no probe needed

POLICY_GRID = 32  # the scorer's working resolution; larger canvases are pooled to it

_SCALAR_GROUPS = {"ent", "ent_delta", "cos_prev", "cos_init"}


def group_sizes(canvas_dim: int, groups: tuple[str, ...], local_dim: int | None = None) -> list[int]:
    sizes = []
    for g in groups:
        if g in ("rec_cls", "local_feat"):
            assert local_dim is not None, f"the '{g}' group needs local_dim (the backbone's local token width)"
            sizes.append(local_dim)
        elif g in _SCALAR_GROUPS:
            sizes.append(1)
        else:
            sizes.append(canvas_dim)
    return sizes


def feature_channels(
    canvas_dim: int, groups: tuple[str, ...] = FEATURE_GROUPS, local_dim: int | None = None
) -> int:
    return sum(group_sizes(canvas_dim, groups, local_dim))


def _lnc(x: Tensor) -> Tensor:  # channel-wise LayerNorm per token
    return F.layer_norm(x.permute(0, 2, 3, 1), (x.shape[1],)).permute(0, 3, 1, 2)


def _canvas_spatial(seg: CanViTForSemanticSegmentation, canvas: Tensor, canvas_grid: int) -> Tensor:
    s = seg.canvit.get_spatial(canvas).float()  # [N,T,D] tokens -> [N,D,g,g] map
    b, _, d = s.shape
    return s.permute(0, 2, 1).reshape(b, d, canvas_grid, canvas_grid)


def init_reference(
    seg: CanViTForSemanticSegmentation, *, canvas_grid: int, with_entropy: bool
) -> tuple[Tensor, Tensor | None]:
    """The init (blank) canvas's (spatial feats, probe entropy): an image-independent
    template, so the t0 delta/cos features carry deviation-from-template instead of
    dead zeros. with_entropy=False for probe-free tasks (intrinsic groups only)."""
    canvas = seg.canvit.init_state(batch_size=1, canvas_grid_size=canvas_grid).canvas
    ent = None
    if with_entropy:
        # Force EVAL mode for the probe forward. The template must be a property of the
        # WEIGHTS, not of whatever mode the caller happened to be in when it constructed
        # the StateEncoder: the segmentation head carries a BatchNorm, so under train-mode
        # BN this normalizes a batch of ONE synthetic blank canvas by its own statistics
        # and returns a different template entirely.
        #
        # Measured 2026-07-30: train- vs eval-mode construction moved the entropy template
        # by 1.621288, which propagated verbatim into every `ent_delta`/`cos_init` feature
        # and shifted ~14/32 of a trained policy's chosen glimpses, costing ~0.1 mIoU at
        # each policy timestep. It bit CanViT-pretrain's harness, which builds the policy
        # BEFORE freezing the model (`harness/run.py` build_policy at 277, freeze at 280),
        # while `ade20k/rl_train.py` happens to call seg.eval() first and was unaffected.
        # Fixing it here rather than at one call site: no caller should have to know that
        # constructing a feature encoder depends on module mode.
        head = getattr(seg, "head", None)
        was_training = head is not None and head.training
        if head is not None:
            head.eval()
        try:
            ent = probe_entropy(seg, canvas, canvas_grid=canvas_grid).float()
        finally:
            if was_training:
                head.train()
    return _canvas_spatial(seg, canvas, canvas_grid), ent


def assemble_features(
    cur: Tensor,
    prev: Tensor,
    cur_ent: Tensor | None,
    prev_ent: Tensor | None,
    init_ln: Tensor,
    groups: tuple[str, ...],
    rec_cls: Tensor | None = None,
    local_feat: Tensor | None = None,
    depth_feat: Tensor | None = None,
) -> Tensor:
    """The selected feature groups -> [B, sum(sizes), POLICY_GRID, POLICY_GRID].
    `prev` is the previous state's spatial feats (the init template at t0), so
    deltas read deviation-from-prev. Entropy tensors may be None iff no entropy
    group is selected. `rec_cls` ([B,1,local_dim], the backbone's recurrent_cls)
    may be None iff "rec_cls" is not in `groups`. `local_feat` ([B,local_dim],
    ALREADY pooled over the patch axis and t0-zero-filled by the caller) may be
    None iff "local_feat" is not in `groups`. `depth_feat` ([B,canvas_dim,g,g],
    ALREADY LayerNorm-diffed and zero-filled by the caller) may be None iff
    "depth_feat" is not in `groups`."""
    ln, ln_prev = _lnc(cur), _lnc(prev)
    avail: dict[str, Tensor] = {
        "cos_prev": (1 - F.cosine_similarity(ln, ln_prev, dim=1)).unsqueeze(1),
        "cos_init": (1 - F.cosine_similarity(ln, init_ln, dim=1)).unsqueeze(1),
        "ln_feat": ln,
        "feat_delta": ln - ln_prev,
    }
    if cur_ent is not None:
        assert prev_ent is not None
        avail["ent"] = cur_ent.unsqueeze(1)
        avail["ent_delta"] = (cur_ent - prev_ent).unsqueeze(1)
    if rec_cls is not None:
        # Channel-wise LayerNorm per sample (same scale-equalizing treatment as
        # ln_feat), then broadcast the one global vector to every grid cell.
        rc = F.layer_norm(rec_cls.squeeze(1), (rec_cls.shape[-1],))
        avail["rec_cls"] = rc[:, :, None, None].expand(-1, -1, cur.shape[-2], cur.shape[-1])
    if local_feat is not None:
        lf = F.layer_norm(local_feat, (local_feat.shape[-1],))
        avail["local_feat"] = lf[:, :, None, None].expand(-1, -1, cur.shape[-2], cur.shape[-1])
    if depth_feat is not None:
        avail["depth_feat"] = depth_feat
    assert all(g in avail for g in groups), (
        f"groups {groups} include a probe-entropy, rec_cls, local_feat, or depth_feat "
        f"group but the matching tensor was not provided (probe-free task? use INTRINSIC_GROUPS)"
    )
    out = torch.cat([avail[g] for g in groups], dim=1)
    return F.adaptive_avg_pool2d(out, POLICY_GRID) if out.shape[-1] != POLICY_GRID else out


class StateEncoder:
    """Canvas state -> scorer input, holding the rolling previous-state reference the
    delta features need. Built once from (seg, canvas_grid, feature_groups);
    `reset()` at t0, then call per step. The ONE place featurization is done —
    trainer, rollout, policy and eval all share it."""

    def __init__(
        self,
        seg: CanViTForSemanticSegmentation,
        *,
        canvas_grid: int,
        feature_groups: tuple[str, ...] = FEATURE_GROUPS,
    ):
        self.seg = seg
        self.canvas_grid = canvas_grid
        self.feature_groups = feature_groups
        self.needs_entropy = any(g in ("ent", "ent_delta") for g in feature_groups)
        self.needs_rec_cls = "rec_cls" in feature_groups
        self.needs_local_feat = "local_feat" in feature_groups
        if self.needs_local_feat:
            self.local_dim = seg.canvit.local_dim
        self.needs_depth_feat = "depth_feat" in feature_groups
        # prev/init are REFERENCES the delta features read, never backprop targets, so
        # keep them detached: otherwise (joint task+policy training, CanViT-pretrain
        # P4b) the retained graph is re-entered on the next TBPTT chunk / step and
        # autograd raises "backward through the graph a second time". Under the frozen
        # no_grad callers (rl_train, eval) this detach is a harmless no-op.
        init_sp, init_ent = init_reference(seg, canvas_grid=canvas_grid, with_entropy=self.needs_entropy)
        self.init = (init_sp.detach(), init_ent.detach() if init_ent is not None else None)
        self.init_ln = _lnc(self.init[0])
        self.prev: tuple[Tensor, Tensor | None] = self.init  # rolling (spatial, entropy)

    def reset(self) -> None:
        self.prev = self.init

    def __call__(
        self,
        state,
        logits: Tensor | None = None,
        local_patches: Tensor | None = None,
        write_snapshots: list[Tensor] | None = None,
    ) -> Tensor:
        """`logits` = probe logits of state.canvas when the caller already has them
        (a rollout computing them for the CE reward shares that one head_logits
        call). None -> computed here when an entropy group needs them. `local_patches`
        ([B,n_patch,local_dim], the backbone's RAW output for the glimpse just taken)
        is only read when "local_feat" is selected; None at t0 (no glimpse yet) ->
        zero-filled rather than left out, so every sample gets the same channel count.
        `write_snapshots` (the glimpse's canvas right after each of its writes) is only
        read when "depth_feat" is selected; None/fewer-than-2-writes -> zero-filled."""
        cur = _canvas_spatial(self.seg, state.canvas, self.canvas_grid)
        cur_ent: Tensor | None = None
        if self.needs_entropy:
            if logits is None:
                logits = head_logits(self.seg, state.canvas, canvas_grid=self.canvas_grid)
            cur_ent = entropy_from_logits(logits).float()
        rec_cls = state.recurrent_cls.float() if self.needs_rec_cls else None
        local_feat: Tensor | None = None
        if self.needs_local_feat:
            local_feat = (
                local_patches.float().mean(dim=1) if local_patches is not None
                else cur.new_zeros(cur.shape[0], self.local_dim)
            )
        depth_feat: Tensor | None = None
        if self.needs_depth_feat:
            if write_snapshots is not None and len(write_snapshots) >= 2:
                s_first = _canvas_spatial(self.seg, write_snapshots[0], self.canvas_grid)
                s_last = _canvas_spatial(self.seg, write_snapshots[-1], self.canvas_grid)
                depth_feat = _lnc(s_last) - _lnc(s_first)
            else:
                depth_feat = torch.zeros_like(cur)
        feats = assemble_features(
            cur, self.prev[0], cur_ent, self.prev[1], self.init_ln, self.feature_groups,
            rec_cls, local_feat, depth_feat,
        )
        # Store the reference DETACHED (see __init__): the current feats keep grad
        # through `cur`, but next step's delta must not backprop into this step's state.
        self.prev = (cur.detach(), cur_ent.detach() if cur_ent is not None else None)
        return feats
