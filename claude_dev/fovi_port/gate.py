"""Equality gate for the fovi port: run in a tree, save everything the patchers build and output.

    cd <tree> && python claude_dev/fovi_port/gate.py OUT.pt [--ckpt EXP22.pt] [--only NAME ...]
    python claude_dev/fovi_port/gate.py --compare A.pt B.pt

Run once in a tree that still imports fovi (before the port) and once in the ported
tree; ``--compare`` then demands BIT-IDENTICAL tensors under identical keys. The cwd's
``canvit/`` is imported, so the two runs differ only in the tree.

Per configuration it records, after ``torch.manual_seed(0)`` and ``np.random.seed(0)``
(fovi's ``isotropic_fixn`` style draws its geometry from the unseeded numpy RNG):
  * every parameter and buffer of the patcher (persistent or not),
  * every tensor attribute of the geometry objects (retina, sampler, its coords and
    pooler, the KNN embedding and its coords) -- the KNN routing, grids, coordinates,
  * the forward output (tokens, scene positions) on a fixed batch, and the gradient of
    every parameter after a backward of the token sum,
  * on CUDA (if present) the forward again after ``.to('cuda')``,
or the exception text if construction or forward raises (both trees must then raise
the same).
With ``--ckpt``, it also loads that pretraining checkpoint, checks the load is strict
(reported missing/unexpected keys) and records one glimpse's outputs.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from dataclasses import fields, is_dataclass, replace
from typing import Any

# A script's own directory, not the cwd, heads sys.path; put the cwd's tree first so the
# run measures that tree and not the editable install.
sys.path.insert(0, os.getcwd())

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import Tensor  # noqa: E402


def foveated_configs() -> dict[str, Any]:
    from canvit.core.patcher.conditioning import FiLMConfig, PatchConditioningConfig
    from canvit.core.patcher.foveated import FoveatedPatcherConfig as C

    # exp22 backbone geometry (retina fov 35, cmf_a 0.5, res 64), the other fields default
    exp22 = C(fov=35.0, cmf_a=0.5, resolution=64)
    small = C(fov=50.0, cmf_a=0.75, resolution=32, cart_patch_size=4)
    film = PatchConditioningConfig(mode="film")
    return {
        "exp22": exp22,
        "small": small,
        "sampler_grid_nn": replace(small, sampler="grid_nn"),
        "sampler_grid_bilinear": replace(small, sampler="grid_bilinear"),
        "sampler_gaussian_pooling": replace(small, sampler="gaussian_pooling"),
        "rf_bilinear": replace(small, rf_interp="bilinear"),
        "doubleres": replace(small, arch_flag="doubleres"),
        "ref_side_7": replace(small, ref_frame_side_length=7),
        "per_ring_kernel": replace(small, per_ring_kernel=True),
        "pad_learned": replace(small, padding="learned"),
        "pad_learned_shared": replace(small, padding="learned_shared"),
        "ring_pruning": replace(small, min_ring_new_pixels=4, pattern_reference_size=128),
        "cortex_false": replace(small, sample_cortex=False),
        "cortex_geodesic": replace(small, resolution=16, cart_patch_size=4, sample_cortex="geodesic"),
        "max_coord_1": replace(small, max_coord_val=1.0),
        "no_auto_match": replace(small, auto_match_cart_resources=False, resolution=12, cart_patch_size=2),
        "no_force_less": replace(small, force_patches_less_than_matched=False),
        "style_fixn": replace(small, style="isotropic_fixn", resolution=16, cart_patch_size=4),
        "style_logpolar": replace(small, style="logpolar", resolution=16, cart_patch_size=4),
        "mlp_embed": replace(small, hidden_dims_patch_embed=[48]),
        "film": replace(small, conditioning=film),
        "film_scale": replace(small, conditioning=PatchConditioningConfig(
            mode="film", film=FiLMConfig(encode_scale=True))),
        "coordconv": replace(small, conditioning=PatchConditioningConfig(mode="coordconv")),
        "coordconv_film": replace(small, conditioning=PatchConditioningConfig(mode="coordconv_film")),
        "multi_retina": replace(small, arch_flag="doubleres", rf_interp="bilinear", conditioning=film,
                                retinas=[(0.75, 24, 3, False), (0.75, 32, 4, False), (0.75, 40, 5, False)]),
    }


def square_configs() -> dict[str, Any]:
    from canvit.core.patcher.square import SquarePatcherConfig as S

    kw = dict(fov=50.0, cmf_a=0.75, resolution=32, cart_patch_size=4, pattern_reference_size=64)
    return {
        "sq_fovi": S(method="fovi", **kw),
        "sq_fovi_regularized": S(method="fovi_regularized", **kw),
        "sq_strided": S(method="strided", pattern_reference_size=64),
        "sq_strided_add": S(method="strided", pattern_reference_size=64, add_to_patch_size=1),
        "sq_fovi_pruned": S(method="fovi", min_ring_new_pixels=2, **kw),
        "sq_pad_learned": S(method="fovi", padding="learned", **kw),
    }


def _tensors(obj: Any, prefix: str, out: dict[str, Tensor]) -> None:
    """Tensor attributes of a geometry object (``__dict__`` + nn.Module registries)."""
    if obj is None:
        return
    d = dict(getattr(obj, "__dict__", {}))
    for reg in ("_parameters", "_buffers"):
        d.update({k: v for k, v in d.pop(reg, {}).items() if v is not None})
    for k, v in sorted(d.items()):
        if isinstance(v, Tensor):
            out[f"{prefix}.{k}"] = v.detach().clone()


def _geometry(patcher: Any, out: dict[str, Tensor]) -> None:
    if not hasattr(patcher, "geometry"):  # square patcher: everything is in its buffers
        return
    for i in range(patcher.n_retinas):
        retina, kpe = patcher.geometry(i)
        s = retina.sampler
        for name, obj in [("retina", retina), ("sampler", s), ("sampler.coords", getattr(s, "coords", None)),
                          ("sampler.highres_coords", getattr(s, "highres_coords", None)),
                          ("sampler.pooler", getattr(s, "pooler", None)), ("kpe", kpe),
                          ("kpe.in_coords", kpe.in_coords), ("kpe.out_coords", kpe.out_coords)]:
            _tensors(obj, f"geom{i}.{name}", out)
        for name in ("_k", "ref_grid_size", "resolution"):
            for obj, tag in ((kpe, "kpe"), (retina, "retina")):
                if hasattr(obj, name):
                    out[f"geom{i}.{tag}.{name}"] = torch.tensor(int(getattr(obj, name)))
        if hasattr(retina, "scatter_sizes"):
            out[f"geom{i}.retina.scatter_sizes"] = torch.as_tensor(retina.scatter_sizes)


def _batch(n_retinas: int):
    from canvit.core.viewpoint import Viewpoint

    g = torch.Generator().manual_seed(123)
    B, H = 4, 96
    img = torch.randn(B, 3, H, H, generator=g)
    centers = torch.rand(B, 2, generator=g) * 1.6 - 0.8
    scales = torch.rand(B, generator=g) * 1.2 + 0.3
    retinas = torch.arange(B) % n_retinas
    return img, Viewpoint(centers=centers, scales=scales), retinas


def run_patcher(kind: str, cfg: Any) -> dict[str, Tensor]:
    from canvit.core.patcher.foveated import FoveatedPatcher
    from canvit.core.patcher.square import SquarePatcher

    out: dict[str, Tensor] = {}
    torch.manual_seed(0)
    np.random.seed(0)  # style="isotropic_fixn" draws its geometry from the global numpy RNG
    cls = FoveatedPatcher if kind == "foveated" else SquarePatcher
    p = cls(copy.deepcopy(cfg), embed_dim=32, device="cpu")
    for k, v in p.named_parameters():
        out[f"param.{k}"] = v.detach().clone()
    for k, v in p.named_buffers():
        out[f"buffer.{k}"] = v.detach().clone()
    out["state_dict_keys"] = torch.tensor(list("\n".join(sorted(p.state_dict())).encode()), dtype=torch.uint8)
    _geometry(p, out)
    n_ret = getattr(p, "n_retinas", 1)
    img, vp, retinas = _batch(n_ret)
    if n_ret > 1:
        p.set_retinas(retinas)
    tokens, pos = p(img, vp)
    out["fwd.tokens"], out["fwd.scene_pos"] = tokens.detach().clone(), pos.detach().clone()
    tokens.sum().backward()
    for k, v in p.named_parameters():
        if v.grad is not None:
            out[f"grad.{k}"] = v.grad.detach().clone()
    if torch.cuda.is_available():
        p = p.to("cuda")
        if n_ret > 1:
            p.set_retinas(retinas)
        from canvit.core.viewpoint import Viewpoint
        with torch.no_grad():
            t, s = p(img.cuda(), Viewpoint(centers=vp.centers.cuda(), scales=vp.scales.cuda()))
        out["cuda.tokens"], out["cuda.scene_pos"] = t.cpu(), s.cpu()
    return out


def run_ckpt(path: str) -> dict[str, Tensor]:
    from canvit.core.model.pretraining.impl import CanViTForPretraining
    from canvit.core.viewpoint import Viewpoint

    raw = torch.load(path, map_location="cpu", weights_only=False)
    sd = raw.get("model", raw.get("state_dict", raw)) if isinstance(raw, dict) else raw
    torch.manual_seed(0)
    model = CanViTForPretraining.from_checkpoint(path, map_location="cpu").eval()
    out: dict[str, Tensor] = {}
    ck_keys = {k.removeprefix("_orig_mod.") for k in sd} if isinstance(sd, dict) else set()
    mk = set(model.state_dict())
    print(f"  ckpt: {len(mk & ck_keys)} shared keys, model-only {len(mk - ck_keys)}, ckpt-only {len(ck_keys - mk)}")
    for k, v in model.state_dict().items():
        if "patcher" in k:
            out[f"model.{k}"] = v.detach().clone()
    g = torch.Generator().manual_seed(7)
    B, H = 2, 128
    img = torch.randn(B, 3, H, H, generator=g)
    vp = Viewpoint(centers=torch.tensor([[0.0, 0.0], [0.3, -0.4]]), scales=torch.tensor([1.0, 0.6]))
    grid = int(getattr(model, "canvas_grid_size", 0) or getattr(getattr(model, "cfg", None), "canvas_grid_size", 0) or 8)
    with torch.no_grad():
        state = model.init_state(batch_size=B, canvas_grid_size=grid)
        o = model(image=img, state=state, viewpoint=vp, canvas_grid_size=grid)
    for f in fields(o) if is_dataclass(o) else []:
        v = getattr(o, f.name)
        if isinstance(v, Tensor):
            out[f"out.{f.name}"] = v.detach().clone()
        elif is_dataclass(v):
            for f2 in fields(v):
                v2 = getattr(v, f2.name)
                if isinstance(v2, Tensor):
                    out[f"out.{f.name}.{f2.name}"] = v2.detach().clone()
    return out


def compare(a_path: str, b_path: str) -> int:
    a, b = torch.load(a_path, weights_only=False), torch.load(b_path, weights_only=False)
    bad = 0
    for name in sorted(set(a) | set(b)):
        ra, rb = a.get(name), b.get(name)
        if ra is None or rb is None:
            print(f"FAIL {name}: only in {'A' if rb is None else 'B'}"); bad += 1; continue
        if isinstance(ra, str) or isinstance(rb, str):
            same = ra == rb
            print(f"{'ok  ' if same else 'FAIL'} {name}: raises {'the same' if same else 'differently'}: A={ra!r:.120} B={rb!r:.120}")
            bad += not same; continue
        keys = set(ra) | set(rb)
        diff = [k for k in sorted(keys) if k not in ra or k not in rb or ra[k].shape != rb[k].shape
                or ra[k].dtype != rb[k].dtype or not torch.equal(ra[k], rb[k])]
        nan_eq = [k for k in diff if k in ra and k in rb and ra[k].shape == rb[k].shape and ra[k].dtype == rb[k].dtype
                  and ra[k].is_floating_point() and torch.equal(torch.nan_to_num(ra[k], 7.0), torch.nan_to_num(rb[k], 7.0))
                  and torch.equal(torch.isnan(ra[k]), torch.isnan(rb[k]))]
        diff = [k for k in diff if k not in nan_eq]
        print(f"{'ok  ' if not diff else 'FAIL'} {name}: {len(keys)} tensors bit-identical" if not diff
              else f"FAIL {name}: {len(diff)}/{len(keys)} differ, e.g. {diff[:5]}")
        bad += bool(diff)
    print(f"{'PASS' if not bad else 'FAILED'}: {len(set(a) | set(b)) - bad}/{len(set(a) | set(b))} cases identical")
    return int(bool(bad))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?")
    ap.add_argument("--ckpt")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--compare", nargs=2)
    a = ap.parse_args()
    if a.compare:
        sys.exit(compare(*a.compare))
    import canvit
    print(f"canvit from {canvit.__file__}")
    cases = [("foveated", k, v) for k, v in foveated_configs().items()]
    cases += [("square", k, v) for k, v in square_configs().items()]
    res: dict[str, Any] = {}
    for kind, name, cfg in cases:
        if a.only and name not in a.only:
            continue
        print(f"[{name}]", flush=True)
        try:
            res[name] = run_patcher(kind, cfg)
        except Exception as e:  # recorded: both trees must fail the same way
            res[name] = f"{type(e).__name__}: {e}"
            print(f"  raised {res[name]!r:.200}")
    if a.ckpt and (not a.only or "ckpt" in a.only):
        print("[ckpt]", flush=True)
        res["ckpt"] = run_ckpt(a.ckpt)
    torch.save(res, a.out)
    print(f"saved {len(res)} cases -> {a.out}; fovi imported: {'fovi' in sys.modules}")


if __name__ == "__main__":
    main()
