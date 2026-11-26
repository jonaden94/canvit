"""Prompt conditioning (``canvit/search/prompt.py``): gates at 0 leave the pretrained model unchanged, the
visual-share rule, the prompt encoder's tokens."""

import numpy as np
import pytest
import torch

from canvit.search.config import PromptConfig, SearchConfig
from canvit.search.prompt import PromptConditioning, PromptEncoder, assign_prompt_types

from .test_search_task import _FOVEATED


def _tiny_canvit():
    from canvit.search.model import CanViTForVisualSearch
    torch.manual_seed(0)
    return CanViTForVisualSearch(backbone_name="vits16", model_config=_FOVEATED, n_classes=1, cond_dim=8,
                                 kind="linear", width=8).canvit.eval()


def _prompt(cond: PromptConditioning, b: int, *, visual_rows=(1, 3)):
    g = torch.Generator().manual_seed(1)
    visual = torch.zeros(b, dtype=torch.bool)
    visual[list(visual_rows)] = True
    cov = torch.rand(b, 14, generator=g)
    cov[:, 2] = 0  # an empty 2x2 cell
    return cond.encoder(visual=visual, text=torch.randn(b, 768, generator=g),
                        vp_feat=torch.randn(b, 14, 768, generator=g), vp_cov=cov)


def _rollout(canvit, images, n=3):
    from canvit.core import Viewpoint
    state = canvit.init_state(batch_size=images.shape[0], canvas_grid_size=8)
    outs = []
    for t in range(n):
        c = torch.full((images.shape[0], 2), -0.3 + 0.3 * t)
        vp = Viewpoint(centers=c, scales=torch.full((images.shape[0],), 0.5))
        out = canvit(image=images, state=state, viewpoint=vp)
        state = out.state
        outs += [out.state.canvas, out.state.recurrent_cls, out.local_patches]
    return outs


@pytest.mark.parametrize("tokens", ["none", "1x1", "2x2", "3x3"])
def test_gates_at_zero_leave_the_model_bit_identical(tokens):
    """AdaLN (zero-init) and gated prompt tokens (tanh(0)): the backbone's canvas, cls and local tokens after 3
    glimpses are bit-identical to the unconditioned model; opening the gates changes them."""
    canvit = _tiny_canvit()
    d, heads, n_blocks = canvit.backbone.embed_dim, canvit.backbone.num_heads, canvit.backbone.n_blocks
    cond = PromptConditioning(dim=d, num_heads=heads, n_blocks=n_blocks, adaln=True, tokens=tokens)
    images = torch.randn(4, 3, 224, 224, generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        plain = _rollout(canvit, images)
        cond.attach(canvit.backbone)
        prompt = _prompt(cond, 4)
        with cond.use(prompt):
            gated = _rollout(canvit, images)
        assert all(torch.equal(a, b) for a, b in zip(plain, gated))
        assert all(torch.equal(a, b) for a, b in zip(plain, _rollout(canvit, images)))  # no context: no effect
        for lin in cond.adaln:
            torch.nn.init.normal_(lin.weight, std=0.02)
        if cond.xattn is not None:
            for x in cond.xattn:
                x.alpha.fill_(0.5)
        with cond.use(prompt):
            opened = _rollout(canvit, images)
        assert not torch.equal(plain[0], opened[0])


def test_encoder_normalises_and_builds_the_tokens():
    enc = PromptEncoder(dim=16, tokens="2x2")
    seen = []
    enc.text.register_forward_hook(lambda m, i, o: seen.append(i[0].norm(dim=-1)))
    enc.visual.register_forward_hook(lambda m, i, o: seen.append(i[0].norm(dim=-1)))
    b = 4
    visual = torch.tensor([False, True, True, False])
    cov = torch.ones(b, 14)
    cov[1, 2] = 0          # row 1: one empty 2x2 cell
    cov[2, 1:5] = 0        # row 2: no covered 2x2 cell -> falls back to its 1x1 cell
    feat = torch.randn(b, 14, 768) * 7
    p = enc(visual=visual, text=torch.randn(b, 768) * 3, vp_feat=feat, vp_cov=cov)
    for n in seen:          # every vector enters its Linear with unit length
        assert torch.allclose(n, torch.ones_like(n), atol=1e-5)
    assert p.vec.shape == (b, 16) and p.tokens.shape == (b, 4, 16)
    assert p.token_mask.tolist() == [[True, False, False, False], [True, False, True, True],
                                     [True, False, False, False], [True, False, False, False]]
    assert torch.allclose(p.tokens[2, 0], p.vec[2]) and torch.allclose(p.tokens[0, 0], p.vec[0])
    assert torch.allclose(p.raw[1], torch.nn.functional.normalize(feat[1, 0], dim=0), atol=1e-6)
    p1 = PromptEncoder(dim=16, tokens="1x1")(visual=visual, text=torch.randn(b, 768), vp_feat=feat, vp_cov=cov)
    assert p1.tokens.shape == (b, 1, 16) and p1.token_mask.all()
    assert PromptEncoder(dim=16, tokens="none")(visual=visual, text=torch.randn(b, 768), vp_feat=feat,
                                                vp_cov=cov).tokens is None


def _has_vp(n=200_000, f=0.77, seed=3):
    return np.random.default_rng(seed).random(n) < f


@pytest.mark.parametrize("share", [0.3, 0.5, 0.77, 0.85, 0.95])
def test_visual_share_is_met(share):
    has_vp = _has_vp()
    f = has_vp.mean()
    r = assign_prompt_types(has_vp, share, seed=0, epoch=4)
    kept = r.keep.sum()
    realised = r.visual.sum() / kept
    sd = np.sqrt(share * (1 - share) / kept)
    assert abs(realised - share) < 5 * sd, (realised, share)
    assert not (r.visual & ~has_vp).any() and not (r.visual & ~r.keep).any()
    if share <= f:
        assert r.keep.all()            # nothing dropped while the epoch allows the share
    else:
        assert r.keep[has_vp].all()    # only text-only trials are dropped
        assert r.visual[has_vp].all()


def test_visual_share_extremes_and_determinism():
    has_vp = _has_vp(n=10_000)
    one = assign_prompt_types(has_vp, 1.0, seed=0, epoch=1)
    assert np.array_equal(one.keep, has_vp) and np.array_equal(one.visual, has_vp)
    zero = assign_prompt_types(has_vp, 0.0, seed=0, epoch=1)
    assert zero.keep.all() and not zero.visual.any()
    a, b = (assign_prompt_types(has_vp, 0.9, seed=5, epoch=2) for _ in range(2))
    assert np.array_equal(a.keep, b.keep) and np.array_equal(a.visual, b.visual)
    c = assign_prompt_types(has_vp, 0.9, seed=5, epoch=3)
    assert not np.array_equal(a.keep, c.keep)
    with pytest.raises(ValueError):
        assign_prompt_types(np.zeros(10, bool), 0.5, seed=0, epoch=0)


def test_share_zero_never_reads_visual_prompts():
    from canvit.search.release import uses_visual_prompts
    cfg = SearchConfig(dataset="release", prompt=PromptConfig(cue="prompt", visual_share=0.0))
    assert not uses_visual_prompts(cfg)
    assert uses_visual_prompts(SearchConfig(dataset="release", prompt=PromptConfig(cue="prompt")))
    assert cfg.n_timesteps == 10 and cfg.eval_policies == ("policy",)
    assert SearchConfig().n_timesteps == 4 and SearchConfig().eval_policies == ("random", "greedy", "policy")


def test_sdpa_guard_refuses_backbone_gradients_on_v100():
    from canvit.search.task import sdpa_guard
    for cap in (None, (8, 0), (9, 0)):  # CPU, A100, H100: never touched
        assert not sdpa_guard(cap, backprop_through_backbone=True, math_sdpa=False)
    assert not sdpa_guard((7, 0), backprop_through_backbone=False, math_sdpa=False)  # frozen class-cued run
    with pytest.raises(RuntimeError, match="v100-math-sdpa"):
        sdpa_guard((7, 0), backprop_through_backbone=True, math_sdpa=False)
    assert sdpa_guard((7, 0), backprop_through_backbone=True, math_sdpa=True)
