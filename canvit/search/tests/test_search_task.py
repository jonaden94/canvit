"""The search task end to end through ``harness.run`` on CPU (tiny foveated ViT-S, the
synthetic COCO-Search18 from conftest), plus the task-level refusals and hooks."""

from dataclasses import replace

import pytest
import torch

from canvit.harness.run import RunSettings, run
from canvit.search.config import SearchAugConfig, SearchConfig, search_policy_config
from canvit.search.model import CanViTForVisualSearch
from canvit.search.task import BoundSearchTask, SearchRunTask

_FOVEATED = {
    "patcher_name": "foveated",
    "foveated_patcher": {
        "fov": 35, "resolution": 64, "cmf_a": 0.5, "cart_patch_size": 5, "arch_flag": "doubleres",
        "conditioning": {"mode": "film", "film": {"fourier": {"num_features": 256, "sigma": 4}}},
    },
}


class _TinyTask(SearchRunTask):
    """build_model without a pretrained checkpoint: a random tiny foveated backbone."""

    def build_model(self, device, prior_model_config=None):
        torch.manual_seed(0)
        c = self.cfg
        model = CanViTForVisualSearch(backbone_name="vits16", model_config=_FOVEATED,
                                      n_classes=len(self._classes()), cond_dim=c.cond_dim, kind=c.head,
                                      width=c.head_width, block_layers=c.head_block_layers).to(device)
        return model, model.head


def _cfg(root, **kw) -> SearchConfig:
    return SearchConfig(
        data_root=root, scene_size=224, canvas_grid=32, n_timesteps=3, batch_size=2, eval_batch_size=2,
        num_workers=0, head_width=16, head_block_layers=1, cond_dim=8, max_steps=4, warmup_steps=1,
        limit_val_batches=1, device="cpu", amp=False, tracker="none",
        aug=SearchAugConfig(crop=True), **kw)


def _rl(**kw):
    return replace(search_policy_config(), width=16, block_layers=1, **kw)


def _run(task, n_steps=2):
    spec = task.default_spec()
    settings = RunSettings(n_steps=n_steps, eval_every=n_steps, log_every=1, grad_clip=1.0, amp=False,
                           seed=0, device="cpu", tracker="none")
    return run(task=task, spec=spec, settings=settings)


@pytest.mark.parametrize("embedding", ["separate", "shared"])
def test_joint_search_trains_end_to_end(fake_cs18, embedding):
    root, _ = fake_cs18
    task = _TinyTask(_cfg(root, embedding=embedding), rl=_rl())
    last = _run(task)
    assert torch.isfinite(torch.tensor(last["total_loss"]))
    assert task._joint is not None
    assert hasattr(task._joint.scorer, "target_embed") is (embedding == "separate")
    # the policy actually trained: its FiLM generators left their zero init
    gen = task._joint.scorer.film_in.gen.net[-1].weight
    assert gen.abs().sum() > 0
    assert task.best_metric == "hit_final_policy"


def test_evaluate_reports_all_policies(fake_cs18):
    root, _ = fake_cs18
    task = _TinyTask(_cfg(root), rl=_rl())
    model, head = task.build_model(torch.device("cpu"))
    joint = task.build_policy(model, device=torch.device("cpu"), canvas_grid=32,
                              generator=torch.Generator().manual_seed(0))
    model.canvit.eval()
    val = task.build_val_loader()
    a = task.evaluate(model=model, head=head, val_loader=val, device=torch.device("cpu"), step=0, joint=joint)
    for p in ("random", "greedy", "policy"):
        for t in range(3):
            assert 0.0 <= a[f"hit_t{t}_{p}"] <= 1.0 and a[f"loss_t{t}_{p}"] >= 0.0
    # t0 is the same centred fixation for every policy
    assert a["loss_t0_random"] == pytest.approx(a["loss_t0_greedy"]) == pytest.approx(a["loss_t0_policy"])
    # seeded: a second evaluation reproduces the random policy exactly
    b = task.evaluate(model=model, head=head, val_loader=val, device=torch.device("cpu"), step=0, joint=joint)
    assert a["loss_final_random"] == b["loss_final_random"]


def test_inline_reward_with_last_only_supervision_is_refused(fake_cs18):
    root, _ = fake_cs18
    task = _TinyTask(_cfg(root, supervise="last"), rl=_rl(reward="difference"))
    model, _ = task.build_model(torch.device("cpu"))
    with pytest.raises(ValueError, match="supervise='last'"):
        task.build_policy(model, device=torch.device("cpu"), canvas_grid=32, generator=torch.Generator())


def test_vpg_is_refused(fake_cs18):
    root, _ = fake_cs18
    task = _TinyTask(_cfg(root), rl=replace(search_policy_config(), objective="vpg", reward="default",
                                            reward_center="default", reward_scale="default",
                                            entropy_alpha_init=None, entropy_alpha_min=None,
                                            entropy_alpha_max=None, entropy_update="per_glimpse"))
    model, _ = task.build_model(torch.device("cpu"))
    with pytest.raises(ValueError, match="vpg"):
        task.build_policy(model, device=torch.device("cpu"), canvas_grid=32, generator=torch.Generator())


def test_bound_task_hooks_follow_the_config(fake_cs18):
    root, _ = fake_cs18
    task = _TinyTask(_cfg(root), rl=_rl())
    model, _ = task.build_model(torch.device("cpu"))
    batch = {"image": torch.zeros(2, 3, 224, 224), "target": torch.tensor([0, 1]),
             "boxes": torch.tensor([[[-0.1, -0.1, 0.1, 0.1]], [[0.2, 0.2, 0.4, 0.4]]])}
    every = BoundSearchTask(model=model, targets=batch["target"], boxes=batch["boxes"], canvas_grid=32,
                            cfg=_cfg(root, supervise="every"))
    assert not hasattr(every, "step_weight") and not hasattr(every, "policy_cond")
    last = BoundSearchTask(model=model, targets=batch["target"], boxes=batch["boxes"], canvas_grid=32,
                           cfg=_cfg(root, supervise="last"), policy_embed=lambda t: torch.zeros(len(t), 8))
    assert [last.step_weight(t, 4) for t in range(4)] == [0.0, 0.0, 0.0, 4.0]
    assert last.policy_cond().shape == (2, 8)


def test_cli_builds_search_with_its_own_policy_defaults(fake_cs18):
    import tyro

    from canvit.harness.cli import Command, resolve_spec
    root, _ = fake_cs18
    cmd = tyro.cli(Command, args=["search", "--cfg.data-root", str(root), "--cfg.run-group", "t",
                                   "--cfg.model-repo", "x", "--cfg.tracker", "none"])
    task, settings = cmd.build()
    assert task.name == "search" and task.rl.objective == "pg" and task.rl.reward == "terminal"
    spec = resolve_spec(task, cmd.preset, *cmd.lr_wd(), cmd.spec)
    assert spec.train_head and spec.train_policy and not spec.train_backbone
    assert set(spec.optim) == {"head", "policy"}
