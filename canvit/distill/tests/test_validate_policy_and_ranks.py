"""Distill validation: the policy pass (entropy series, policy-map figure) and the split of
the fixed val subset over DDP ranks (2 CPU processes, gloo).

The rank split is only worth having if it reports what one GPU would: each rank evaluates
whole chunks and the chunk-size-weighted sums are all-reduced, so the means must equal the
single-process result up to float summation order. And a failure on one rank must reach
the others through that same all-reduce instead of leaving them waiting forever.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

_B, _G, _IMG, _T, _D, _N_CHUNKS = 2, 8, 224, 3, 32, 4
_DEV = torch.device("cpu")


def _build():
    """Tiny pretraining model + untrained PG policy, identical in every process (seeded)."""
    from canvit import CanViTForPretraining, CanViTForPretrainingConfig
    from canvit.core import CLSStandardizer, PatchStandardizer, create_backbone
    from canvit.distill.task import POLICY_FEATURE_GROUPS
    from canvit.harness.config import FoveatedScaleConfig, JointPolicyConfig
    from canvit.harness.policy import build_policy

    torch.manual_seed(0)
    model = CanViTForPretraining(
        backbone=create_backbone("vits16"), cfg=CanViTForPretrainingConfig(teacher_dim=_D),
        glimpse_size_px=128, backbone_name="vits16", canvas_patch_grid_sizes=[_G],
    ).to(_DEV)
    gen = torch.Generator(device=_DEV)
    gen.manual_seed(0)
    joint = build_policy(
        canvit=model, rl=JointPolicyConfig(objective="pg", select_bn_eval=False),
        feature_groups=POLICY_FEATURE_GROUPS, device=_DEV, canvas_grid=_G,
        min_viewpoint_scale=0.05, foveated_scale=FoveatedScaleConfig(), generator=gen,
    )
    pnorm, cnorm = PatchStandardizer(grid_size=_G, embed_dim=_D), CLSStandardizer(embed_dim=_D)
    g = torch.Generator().manual_seed(7)
    pnorm.set_stats(torch.randn(16, _G * _G, _D, generator=g) * 2 + 1)
    cnorm.set_stats(torch.randn(16, 1, _D, generator=g) * 2 + 1)
    return model, joint, pnorm, cnorm


def _batches():
    """The fixed 'val subset': _N_CHUNKS chunks, each a deterministic function of its index."""
    for ci in range(_N_CHUNKS):
        g = torch.Generator().manual_seed(100 + ci)
        yield torch.randn(_B, 3, _IMG, _IMG, generator=g), torch.zeros(_B, dtype=torch.long)


def _targets(images, _sz, *, fail: bool = False):
    from canvit.core.backbone.vit import NormFeatures

    if fail:
        raise RuntimeError("injected failure")
    # Deterministic per image (seeded by its pixel sum), so a chunk's target does not
    # depend on which process evaluates it.
    g = torch.Generator().manual_seed(int(images.sum().abs().item() * 1000) % (2**31))
    return NormFeatures(patches=torch.randn(images.shape[0], _G * _G, _D, generator=g),
                        cls=torch.randn(images.shape[0], _D, generator=g))


def _run(eval_policy: str, *, rank: int = 0, world_size: int = 1, fail: bool = False,
         run_dir=None, log_policy_maps: bool = False) -> dict[str, float]:
    from canvit.distill.validate import validate

    # One CPU thread in EVERY process: CPU matmul rounding depends on the thread count, and
    # the 1-process and 2-process legs would otherwise get different counts (measured: 2e-6
    # apart on a 16-core node). This test checks the split logic, not thread numerics.
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model, joint, pnorm, cnorm = _build()
        return validate(
            exp=None, step=0, model=model,  # type: ignore[arg-type]
            compute_raw_targets=lambda im, sz: _targets(im, sz, fail=fail),
            scene_normalizer=pnorm, cls_normalizer=cnorm, val_batches=_batches(), device=_DEV,
            canvas_grid_size=_G, scene_size_px=_G * 16, glimpse_size_px=128,
            run_dir=run_dir, n_eval_viewpoints=_T, eval_policy=eval_policy, joint=joint,
            log_policy_maps=log_policy_maps, rank=rank, world_size=world_size,
        )
    finally:
        torch.set_num_threads(n_threads)


def test_policy_pass_reports_entropy_and_saves_maps(tmp_path):
    out = _run("policy", run_dir=tmp_path, log_policy_maps=True)
    n_actions = _build()[1].policy_selector.vp_flat.shape[0]  # uniform patcher: scales x 16^2
    ents = [out[f"policy_entropy_t{t}"] for t in range(1, _T)]
    assert len(ents) == _T - 1 and "policy_entropy_t0" not in out
    assert all(0.0 <= e <= math.log(n_actions) + 1e-4 for e in ents), ents
    assert list((tmp_path / "visualization" / "policy_maps_val").glob("step-0.png"))
    # the open-loop pass has no policy series
    assert not any(k.startswith("policy_entropy") for k in _run("fixation_grid"))


def _worker(rank, world_size, init_file, eval_policy, fail_rank, q):
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank,
                            world_size=world_size)
    try:
        q.put((rank, _run(eval_policy, rank=rank, world_size=world_size,
                          fail=(rank == fail_rank))))
    except Exception as e:  # noqa: BLE001 - reported to the parent
        q.put((rank, f"raised: {e}"))
    finally:
        dist.destroy_process_group()


def _spawn(tmp_path, eval_policy: str, fail_rank: int = -1) -> dict:
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    init_file = str(tmp_path / "pg_init")
    procs = [ctx.Process(target=_worker, args=(r, 2, init_file, eval_policy, fail_rank, q))
             for r in range(2)]
    for p in procs:
        p.start()
    results = dict(q.get(timeout=600) for _ in procs)
    for p in procs:
        p.join(timeout=60)
    return results


@pytest.mark.parametrize("eval_policy", ["fixation_grid", "policy"])
def test_rank_split_matches_single_process(tmp_path, eval_policy):
    single = _run(eval_policy)
    res = _spawn(tmp_path, eval_policy)
    for rank in (0, 1):
        got = res[rank]
        assert isinstance(got, dict), got
        assert set(got) == set(single)
        for k, v in single.items():
            assert abs(got[k] - v) < 1e-6, (rank, k, got[k], v)


def test_failure_on_one_rank_raises_on_all(tmp_path):
    res = _spawn(tmp_path, "fixation_grid", fail_rank=1)
    assert all(isinstance(v, str) and v.startswith("raised") for v in res.values()), res
