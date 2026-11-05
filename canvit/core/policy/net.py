"""The candidate-scorer network over viewpoints: action geometry and the net itself
(state featurization lives in policy.features). One architecture, two readings of its
output map (chosen by the TRAINING objective, which lives in CanViT-pretrain):
Q-values regressed on rewards (QReg) or policy logits (PG).

Ported from canvit_pytorch_rl.policy.net unchanged except: (1) decoupled from the RL
TrainConfig (explicit kwargs only — build_scorer/objective_from_ckpt stay with the
trainer); (2) the NEW "fixation" action space for foveated/square models (user
decision 2026-07-22: actions = fixation centers only, one per grid cell over the
full field, no scale dimension). config.json schema is backward-compatible — old
checkpoints lack action_space and get the historical "safebox".

Arch [2026-06-14: import a battle-tested block, don't hand-roll init]: timm
ConvNeXt-V2 blocks. LayerNorm2d in the body (BatchNorm's running stats are corrupted
by the raw std-186 backbone features -> eval instability; the Frontend's input
BatchNorm is safe because its inputs are pre-normalized). Readout: encoder pools the
32x32 canvas to a 1x1 global bottleneck (so 'where to look' sees the whole scene);
the decoder rebuilds the full 32x32 map; scores are read at the candidate centres
through a shared head, so output cell (i,j) reads the features AT viewpoint (i,j)'s
centre — aligned by construction.
"""

from typing import Literal

import torch
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from timm.layers.norm import LayerNorm2d
from timm.models.convnext import ConvNeXtBlock
from torch import Tensor, nn

from canvit.core.checkpoints import resolve_canvit_repo
from canvit.core.model.hub_mixin import SafeHubMixin

from .features import FEATURE_GROUPS, feature_channels, group_sizes

# The flagship published policy — ViewpointScorer.from_pretrained(DEFAULT_POLICY_REPO)
# just works. = the 8-seed qband's best seed by mean(t1-t4) val CE; all 8 seeds are
# published as ...-qband-2026-07-04-s{0..7} for reproduction.
DEFAULT_POLICY_REPO = resolve_canvit_repo("qpolicy-ade20k-c64-t5-qband-2026-07-04-s2")

ActionSpace = Literal["safebox", "fixation"]
Readout = Literal["unet", "local"]


def candidate_viewpoints(scales: tuple[float, ...], centers_per_axis: int) -> Tensor:
    """[n_scale, centers_per_axis, centers_per_axis, 3] (cy, cx, scale): a grid of
    centres spanning the safe box [-(1-s), 1-s] at each scale s (overlapping
    glimpses; placement resolution decoupled from scale, so the glimpse box never
    leaves the image). The UNIFORM-patcher action space."""
    vps = []
    for s in scales:
        c = torch.linspace(-(1 - s), 1 - s, centers_per_axis)
        cy, cx = torch.meshgrid(c, c, indexing="ij")
        vps.append(torch.stack([cy, cx, torch.full_like(cy, s)], dim=-1))
    return torch.stack(vps)


def fixation_candidates(centers_per_axis: int) -> Tensor:
    """[1, centers_per_axis, centers_per_axis, 3] (cy, cx, 1.0): fixation centres at
    the CELL CENTRES of a grid over the full field [-1, 1]² — the FOVEATED/SQUARE
    action space (no scale dimension; foveation always sees the whole image, the
    view scale comes from FoveatedScaleConfig at glimpse time, not from the policy).
    Cell centres align with score-map pixels under grid_sample align_corners=False,
    so the readout degenerates to reading the map directly."""
    step = 1.0 / centers_per_axis
    c = torch.linspace(-1 + step, 1 - step, centers_per_axis)
    cy, cx = torch.meshgrid(c, c, indexing="ij")
    return torch.stack([cy, cx, torch.ones_like(cy)], dim=-1).unsqueeze(0)


def _stage(ci: int, co: int, n: int) -> nn.Sequential:
    """A 1x1 channel projection (only if ci != co) then n ConvNeXt-V2 blocks at co
    channels. ConvNeXtBlock keeps channels (depthwise conv needs out%groups==0), so
    cross-channel changes (decoder 2w->w) are an explicit 1x1; the blocks do spatial
    mixing at fixed width."""
    layers: list[nn.Module] = [] if ci == co else [nn.Conv2d(ci, co, 1)]
    layers += [ConvNeXtBlock(co, co, use_grn=True) for _ in range(n)]
    return nn.Sequential(*layers)


class _TokenMLP(nn.Module):
    """Per-token (1x1) pre-norm residual MLP: x + mlp(LayerNorm(x))."""

    def __init__(self, ch: int, *, hidden_mult: int = 2):
        super().__init__()
        self.norm = LayerNorm2d(ch)
        self.net = nn.Sequential(nn.Conv2d(ch, hidden_mult * ch, 1), nn.GELU(), nn.Conv2d(hidden_mult * ch, ch, 1))

    def forward(self, x: Tensor) -> Tensor:  # [B, C, H, W]
        return x + self.net(self.norm(x))


class FiLMGen(nn.Module):
    """Target embedding ``[B, cond_dim]`` -> FiLM ``(gamma, beta)``, each ``[B, ch, 1, 1]``.

    Haoze's ``FiLMGenMLP`` (autoregressive_detection, haoze-wip): Linear -> ReLU -> Linear,
    hidden width ``max(128, 2 * cond_dim)``, the last layer ZERO-initialised so the
    modulation starts as the identity."""

    def __init__(self, cond_dim: int, ch: int):
        super().__init__()
        hidden = max(128, 2 * cond_dim)
        self.net = nn.Sequential(nn.Linear(cond_dim, hidden), nn.ReLU(), nn.Linear(hidden, 2 * ch))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, cond: Tensor) -> tuple[Tensor, Tensor]:
        gamma, beta = self.net(cond).chunk(2, dim=-1)
        return gamma[:, :, None, None], beta[:, :, None, None]


class FiLM(nn.Module):
    """One conditioned stage output: ``(1 + gamma) * GroupNorm(x) + beta`` — Haoze's
    ``apply_generic_film`` with his ``GroupNorm(min(8, ch), ch)``. GroupNorm is per sample,
    so this adds no train/eval difference."""

    def __init__(self, cond_dim: int, ch: int):
        super().__init__()
        self.norm = nn.GroupNorm(min(8, ch), ch)
        self.gen = FiLMGen(cond_dim, ch)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        gamma, beta = self.gen(cond)
        return (1 + gamma) * self.norm(x) + beta


N_UNET_ENC, N_UNET_DEC = 6, 5  # 32,16,8,4,2,1 down; ->2,4,8,16,32 up


def build_unet(width: int, block_layers: int) -> tuple[nn.ModuleList, nn.ModuleList]:
    """The scorer's ConvNeXt-V2 U-Net body: ``N_UNET_ENC`` encoder stages at ``width``
    (average-pooled 2x between them, 32 -> 1) and ``N_UNET_DEC`` decoder stages, each
    taking ``[upsampled, skip]`` (2*width) back to width."""
    enc = nn.ModuleList([_stage(width, width, block_layers) for _ in range(N_UNET_ENC)])
    dec = nn.ModuleList([_stage(2 * width, width, block_layers) for _ in range(N_UNET_DEC)])
    return enc, dec


def build_unet_film(cond_dim: int, width: int) -> tuple[nn.ModuleList, nn.ModuleList]:
    """One :class:`FiLM` per encoder and per decoder stage of :func:`build_unet`."""
    return (nn.ModuleList([FiLM(cond_dim, width) for _ in range(N_UNET_ENC)]),
            nn.ModuleList([FiLM(cond_dim, width) for _ in range(N_UNET_DEC)]))


def unet_forward(
    h: Tensor, enc: nn.ModuleList, dec: nn.ModuleList, *,
    film: tuple[nn.ModuleList, nn.ModuleList] | None = None, cond: Tensor | None = None,
) -> Tensor:
    """Run the U-Net body on ``h`` [B, width, 32, 32] -> [B, width, 32, 32].

    With ``film=None`` this is exactly the scorer's historical forward (same ops, same
    order). With ``film=(film_enc, film_dec)`` each stage output is FiLM-modulated by
    ``cond`` before it is pooled / skipped / upsampled."""
    assert (film is None) == (cond is None), "pass film and cond together"
    skips = []
    for i, stage in enumerate(enc):
        h = stage(h)
        if film is not None:
            h = film[0][i](h, cond)
        skips.append(h)
        if i < len(enc) - 1:
            h = F.avg_pool2d(h, 2)
    d = skips[-1]  # 1x1 global bottleneck
    for j, stage in enumerate(dec):
        d = stage(torch.cat([F.interpolate(d, scale_factor=2, mode="nearest"), skips[len(enc) - 2 - j]], dim=1))
        if film is not None:
            d = film[1][j](d, cond)
    return d


InputNorm = Literal["batch", "none"]


class Frontend(nn.Module):
    """Fuse the derived feature groups into [B, width, 32, 32]: per-channel BatchNorm
    (scale-equalize the heterogeneous inputs) -> each group its OWN 1x1 conv to width
    -> per-group LayerNorm2d (affine, so it reweights per-channel importance) -> x
    per-group gate -> sum -> pre-norm token-MLP -> 1x1. (A plain per-group-proj sum
    WITHOUT the LN equals one shared proj; the LN is what makes it non-trivial.)

    ``input_norm='none'`` drops the BatchNorm. It exists to equalise the scales of the
    handcrafted groups (cos/ent maps at ~1e-3 next to the 768 canvas channels); a scorer
    that reads only ``ln_feat`` (already LayerNorm'd per token) does not need it, and
    without it the scorer has no train/eval difference at all (visual search, 2026-11-05)."""

    sizes: list[int]
    groups: tuple[str, ...]

    def __init__(self, canvas_dim: int, width: int, *, groups: tuple[str, ...] = FEATURE_GROUPS,
                 input_norm: InputNorm = "batch"):
        super().__init__()
        self.groups = groups
        self.sizes = group_sizes(canvas_dim, groups)
        assert input_norm in ("batch", "none"), f"unknown input_norm {input_norm!r}"
        self.bn = nn.BatchNorm2d(sum(self.sizes)) if input_norm == "batch" else nn.Identity()
        self.proj = nn.ModuleList([nn.Conv2d(sz, width, 1) for sz in self.sizes])
        self.norm = nn.ModuleList([LayerNorm2d(width) for _ in self.sizes])
        self.gate = nn.Parameter(torch.ones(len(self.sizes)))
        self.mlp = _TokenMLP(width)
        self.out = nn.Conv2d(width, width, 1)

    def forward(self, feats: Tensor) -> Tensor:
        b = self.bn(feats)
        x, off = None, 0
        for i, (proj, norm, sz) in enumerate(zip(self.proj, self.norm, self.sizes)):
            xg = norm(proj(b[:, off : off + sz])) * self.gate[i]
            x, off = xg if x is None else x + xg, off + sz
        return self.out(self.mlp(x))

    @torch.no_grad()
    def log(self) -> dict[str, float]:  # per-group gate (favor/suppress importance readout)
        return {f"gate_{n}": float(self.gate[i]) for i, n in enumerate(self.groups)}


class ViewpointScorer(  # pyright: ignore[reportIncompatibleMethodOverride]  # SafeHubMixin types
    # _load_as_safetensor as nn.Module where the hub mixin uses a TypeVar — upstream clash, not ours
    nn.Module,
    SafeHubMixin,
    PyTorchModelHubMixin,
    library_name="canvit-pytorch",
    repo_url="https://github.com/m2b3/CanViT-PyTorch",
):
    """Frontend -> body -> readout at the candidate centres + a shared head -> one score
    per candidate [B, n_scale, cpa, cpa] (cpa=centers_per_axis).

    ``readout`` selects the body, i.e. how much of the scene one candidate's score sees:

    * ``"unet"`` (default, the historical architecture) — a ConvNeXt-V2 U-Net: the encoder
      pools 32->1, so every score is conditioned on a GLOBAL bottleneck, and the decoder
      rebuilds a full-image-registered 32x32 map. "Where to look" sees the whole scene.
    * ``"local"`` — no U-Net: score each candidate straight off the ``Frontend`` map with a
      1x1 conv. The ``Frontend`` is itself purely per-token (BatchNorm, per-group 1x1
      convs, LayerNorm2d, a 1x1 token-MLP, 1x1 out — no spatial mixing), so a candidate's
      score depends only on the canvas cell it sits on. This is the ``autoreg_tryout``
      policy head (``nn.Linear(D, 1)`` per state token -> a heatmap over grid cells);
      ``block_layers`` is unused in this mode.

      NB exact one-cell-per-score alignment needs ``centers_per_axis == 32`` (the
      ``POLICY_GRID`` the features arrive on): at a coarser grid the candidate centres fall
      between map pixels and ``grid_sample``'s bilinear interpolation mixes a 2x2
      neighbourhood. ``autoreg_tryout``'s heatmap was defined ON its state grid, so
      ``centers_per_axis=32`` is the faithful setting.

    Hub I/O follows the release-stack pattern (PyTorchModelHubMixin + the loud
    strict-load SafeHubMixin): the __init__ kwargs ARE config.json, so
    `ViewpointScorer.from_pretrained(repo_or_dir)` reconstructs the exact
    architecture, and `save_pretrained`/`push_to_hub` publish it. ``readout`` is
    backward-compatible exactly like ``action_space``: a published config.json without the
    key loads as ``"unet"``, so every existing checkpoint keeps working."""

    sb_grids: Tensor  # [n_scale, cpa, cpa, 2] per-scale candidate grid_sample grids (registered buffer)

    def __init__(
        self,
        *,
        canvas_dim: int,
        width: int,
        n_scale: int,
        scales: tuple[float, ...],
        centers_per_axis: int,
        block_layers: int,
        groups: tuple[str, ...] = FEATURE_GROUPS,
        dueling: bool = False,
        action_space: ActionSpace = "safebox",
        readout: Readout = "unet",
        input_norm: InputNorm = "batch",
        film_cond_dim: int | None = None,
    ):
        super().__init__()
        scales, groups = tuple(scales), tuple(groups)  # config.json round-trips tuples as lists
        assert len(scales) == n_scale, "need one entry in `scales` per scale channel"
        self.action_space: ActionSpace = action_space
        self.readout: Readout = readout
        self.frontend = Frontend(canvas_dim, width, groups=groups, input_norm=input_norm)
        # 'local' skips the U-Net entirely -- do not INSTANTIATE it either, or the unused
        # params would ride in the optimizer and the published state_dict.
        if readout == "unet":
            self.enc, self.dec = build_unet(width, block_layers)
        else:
            self.enc = self.dec = None
        if action_space == "fixation":
            assert n_scale == 1, "fixation action space has no scale dimension (n_scale must be 1)"
            cand = fixation_candidates(centers_per_axis)
        else:
            cand = candidate_viewpoints(scales, centers_per_axis)
        sb = cand[..., :2].flip(-1)  # (x,y) for grid_sample
        self.register_buffer("sb_grids", sb)
        self.scale_emb = nn.Parameter(torch.zeros(n_scale, width, 1, 1))  # per-scale conditioning of the shared head
        # 'unet': ConvNeXt blocks over the sampled candidate grid (spatial mixing AT the
        # action resolution). 'local': a bare 1x1, so one candidate = one cell.
        self.head = (
            nn.Sequential(_stage(width, width, block_layers), nn.Conv2d(width, 1, 1))
            if readout == "unet" else nn.Conv2d(width, 1, 1)
        )
        # dueling: V(s) from the mean-pooled INPUT features (a scalar is not location-resolved, so pooled
        # conditioning is fine); the map becomes a mean-zero advantage. Argmax — hence deploy — is unchanged.
        ch = feature_channels(canvas_dim, groups)
        self.vhead = nn.Sequential(nn.Linear(ch, 64), nn.GELU(), nn.Linear(64, 1)) if dueling else None
        # Target conditioning (visual search): FiLM after the Frontend and after every U-Net
        # stage. Built LAST and only when asked, so an unconditioned scorer consumes the same
        # init RNG, has the same state_dict and computes the same thing as before.
        self.film_cond_dim = film_cond_dim
        if film_cond_dim is not None:
            assert readout == "unet", "FiLM conditioning needs readout='unet' (it modulates the U-Net stages)"
            self.film_in = FiLM(film_cond_dim, width)
            self.film_enc, self.film_dec = build_unet_film(film_cond_dim, width)
        else:
            self.film_in = self.film_enc = self.film_dec = None

    def forward(self, feats: Tensor, cond: Tensor | None = None) -> Tensor:
        """``cond`` [B, film_cond_dim]: the target embedding; required iff the scorer was
        built with ``film_cond_dim``."""
        if (cond is None) != (self.film_cond_dim is None):
            raise ValueError(
                f"ViewpointScorer(film_cond_dim={self.film_cond_dim}) called with "
                f"cond={'None' if cond is None else tuple(cond.shape)}: a conditioned scorer "
                "needs the target embedding, an unconditioned one must not get one.")
        h = self.frontend(feats)
        if cond is not None:
            assert self.film_in is not None and self.film_enc is not None and self.film_dec is not None
            h = self.film_in(h, cond)
        if self.enc is None:  # readout='local': the Frontend map IS the score map
            d = h
        else:
            assert self.dec is not None
            film = None if cond is None else (self.film_enc, self.film_dec)
            d = unet_forward(h, self.enc, self.dec, film=film, cond=cond)  # type: ignore[arg-type]
        b = d.shape[0]
        vals = [
            self.head(F.grid_sample(d, self.sb_grids[k].expand(b, -1, -1, -1), align_corners=False) + self.scale_emb[k])
            for k in range(self.sb_grids.shape[0])
        ]
        q = torch.cat(vals, dim=1)  # [B, n_scale, cpa, cpa]
        if self.vhead is not None:  # Q = V(s) + mean-zero A(s,a)
            flat = q.reshape(b, -1)
            q = (flat - flat.mean(dim=1, keepdim=True) + self.vhead(feats.mean(dim=(2, 3)))).reshape(q.shape)
        return q

    def log(self) -> dict[str, float]:
        return self.frontend.log()
