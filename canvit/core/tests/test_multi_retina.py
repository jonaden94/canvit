"""Multi-retina foveated patcher (``FoveatedPatcherConfig.retinas``).

The reference for every retina is a single-retina patcher built directly for it, on the
shared kernel grid (``ref_frame_side_length``) and with the same kernel weight: a mixed
batch through the multi-retina patcher must give each sample exactly that retina's
tokens and positions (FiLM is a no-op at init). Also: one kernel tensor, which every
retina's samples train; the retina vector of FiLM reaches only its own retina's samples;
and a forward without a retina choice fails.

Skipped when the optional ``fovi`` dependency is not installed.
"""

import importlib.util
from dataclasses import replace

import pytest
import torch

from canvit.core import FoveatedPatcherConfig, Viewpoint
from canvit.core.patcher.conditioning import PatchConditioningConfig
from canvit.core.patcher.foveated import FoveatedPatcher

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("fovi") is None,
    reason="fovi optional dependency not installed",
)

IMAGE_PX = 128
EMBED = 32
BASE = FoveatedPatcherConfig(fov=50.0, arch_flag="doubleres", rf_interp="bilinear",
                             conditioning=PatchConditioningConfig(mode="film"))

# Two retina lists with equal patch counts, kept small so they build in ~0.5 s each:
# three resolutions of one cmf_a (70 patches; native grids 8 / 10 / 12, so the shared
# grid is 12 and two retinas are moved to it), and two cmf_a at one resolution
# (115 patches; different patch positions per retina).
RETINA_SETS = {
    "resolutions": [(0.75, 24, 3, False), (0.75, 32, 4, False), (0.75, 40, 5, False)],
    "cmf_a": [(0.75, 32, 3, False), (2.2, 32, 3, False)],
}


def _multi(retinas):
    torch.manual_seed(0)
    return FoveatedPatcher(replace(BASE, retinas=retinas), embed_dim=EMBED)


def _single(retina, side, weight):
    """The single-retina reference: built directly on the shared grid, unconditioned."""
    a, r, c, f = retina
    p = FoveatedPatcher(replace(BASE, cmf_a=a, resolution=r, cart_patch_size=c,
                                force_patches_less_than_matched=f, arch_flag="",
                                ref_frame_side_length=side,
                                conditioning=PatchConditioningConfig(mode="none")),
                        embed_dim=EMBED)
    with torch.no_grad():
        p.kpe.weight.copy_(weight)
    return p


def _batch(B):
    g = torch.Generator().manual_seed(1)
    image = torch.randn(B, 3, IMAGE_PX, IMAGE_PX, generator=g)
    vp = Viewpoint(centers=torch.rand(B, 2, generator=g) * 1.6 - 0.8,
                   scales=torch.rand(B, generator=g) * 1.5 + 0.5)
    return image, vp


@pytest.mark.parametrize("name", list(RETINA_SETS))
def test_mixed_batch_matches_each_retina_alone(name):
    retinas = RETINA_SETS[name]
    multi = _multi(retinas)
    R = len(retinas)
    side = int(multi.kpe.ref_grid_size)
    assert multi.n_retinas == R
    for i in range(R):
        assert int(multi.geometry(i)[1].ref_grid_size) == side
    ids = torch.tensor([i % R for i in range(2 * R)]).flip(0)  # every retina, not sorted
    image, vp = _batch(len(ids))
    multi.set_retinas(ids)
    with torch.no_grad():
        patches, pos = multi(image, vp)
    assert patches.shape == (len(ids), multi.n_patches, EMBED)
    for i, retina in enumerate(retinas):
        ref = _single(retina, side, multi.kpe.weight)
        sel = ids == i
        with torch.no_grad():
            ref_patches, ref_pos = ref(image[sel], Viewpoint(centers=vp.centers[sel], scales=vp.scales[sel]))
        torch.testing.assert_close(patches[sel], ref_patches, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(pos[sel], ref_pos, rtol=0, atol=0)


def test_grid_is_largest_native_and_sizes():
    multi = _multi(RETINA_SETS["resolutions"])
    assert int(multi.kpe.ref_grid_size) == 12                      # native 8 / 10 / 12
    assert multi.kpe.weight.shape == (EMBED, 3 * 12 * 12)
    explicit = FoveatedPatcher(replace(BASE, arch_flag="", ref_frame_side_length=16,
                                       retinas=RETINA_SETS["resolutions"]), embed_dim=EMBED)
    assert int(explicit.kpe.ref_grid_size) == 16


def test_one_kernel_trained_by_every_retina():
    multi = _multi(RETINA_SETS["resolutions"])
    assert list(multi.extra_kpes.parameters()) == []               # geometry only
    kernel_keys = [k for k in multi.state_dict() if k.endswith("kpe.weight") or "kpes" in k]
    assert kernel_keys == ["kpe.weight"]
    image, vp = _batch(3)
    for i in range(multi.n_retinas):                               # each retina alone reaches the kernel
        multi.zero_grad()
        multi.set_retinas(i)
        multi(image, vp)[0].pow(2).sum().backward()
        assert multi.kpe.weight.grad is not None and multi.kpe.weight.grad.abs().sum() > 0


def test_retina_vector_reaches_only_its_retina():
    multi = _multi(RETINA_SETS["cmf_a"])
    film = multi.conditioner
    assert film.retina_embed.shape == (2, BASE.conditioning.film.mlp_hidden[0])
    with torch.no_grad():                                          # move FiLM off its no-op init
        film.mlp[-1].weight.normal_(0, 0.1)
    ids = torch.tensor([0, 1, 0, 1])
    image, vp = _batch(4)
    multi.set_retinas(ids)
    with torch.no_grad():
        before, _ = multi(image, vp)
        film.retina_embed[1] += 1.0
        after, _ = multi(image, vp)
    assert torch.equal(before[ids == 0], after[ids == 0])
    assert not torch.allclose(before[ids == 1], after[ids == 1])


def test_film_noop_at_init_and_retina_choice_required():
    multi = _multi(RETINA_SETS["cmf_a"])
    image, vp = _batch(2)
    with pytest.raises(AssertionError, match="set_retinas"):
        multi(image, vp)
    multi.set_retinas(torch.tensor([0]))
    with pytest.raises(AssertionError, match="batch of 2"):
        multi(image, vp)
    multi.set_retinas(1)                                           # an int applies to every sample
    with torch.no_grad():
        p, _ = multi(image, vp)
    gamma, beta = multi.conditioner._gamma_beta(None, torch.tensor([0, 1]))
    assert torch.equal(gamma, torch.ones_like(gamma)) and torch.equal(beta, torch.zeros_like(beta))
    assert multi.active_retina(0) == 1
    multi.set_retinas(None)
    with pytest.raises(AssertionError, match="set_retinas"):
        multi(image, vp)


def test_config_checks():
    with pytest.raises(AssertionError, match="at least two"):
        FoveatedPatcher(replace(BASE, retinas=RETINA_SETS["cmf_a"][:1]), embed_dim=EMBED)
    with pytest.raises(AssertionError, match="film"):
        FoveatedPatcher(replace(BASE, retinas=RETINA_SETS["cmf_a"],
                                conditioning=PatchConditioningConfig(mode="none")), embed_dim=EMBED)
    with pytest.raises(AssertionError, match="same number of patches"):
        FoveatedPatcher(replace(BASE, retinas=[(0.75, 24, 3, False), (0.75, 24, 3, True)]), embed_dim=EMBED)
    single = FoveatedPatcher(replace(BASE, conditioning=PatchConditioningConfig(mode="none"),
                                     resolution=24, cart_patch_size=3), embed_dim=EMBED)
    assert single.n_retinas == 1
    with pytest.raises(AssertionError, match="single retina"):
        single.set_retinas(0)

