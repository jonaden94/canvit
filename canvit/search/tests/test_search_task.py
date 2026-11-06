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
                                      width=c.head_width, block_layers=c.head_block_layers,
                                      presence=c.condition == "both").to(device)
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
    assert task.best_metric == "found_t2_policy"  # n_timesteps 3


def test_evaluate_reports_all_policies(fake_cs18):
    root, _ = fake_cs18
    task = _TinyTask(_cfg(root), rl=_rl())
    model, head = task.build_model(torch.device("cpu"))
    joint = task.build_policy(model, device=torch.device("cpu"), canvas_grid=32,
                              generator=torch.Generator().manual_seed(0))
    model.canvit.eval()
    val = task.build_val_loader()
    a = task.evaluate(model=model, head=head, val_loader=val, device=torch.device("cpu"), step=0, joint=joint)
    every = ("loss", "found", "hit", "correct_found", "presence_acc")  # logged at every glimpse
    last_only = ("found_10_percent", "hit_10_percent", "correct_found_10_percent", "correct_hit",
                 "correct_hit_10_percent", "recall", "specificity", "precision", "f1")
    for p in ("random", "greedy", "policy"):
        for t in range(3):
            for k in every:
                assert 0.0 <= a[f"{k}_t{t}_{p}"] <= (1.0 if k != "loss" else float("inf")), k
            for k in last_only:
                assert (f"{k}_t{t}_{p}" in a) is (t == 2), (k, t)
        for k in last_only:
            assert 0.0 <= a[f"{k}_t2_{p}"] <= 1.0, k
        assert a[f"loss_t0_{p}"] > 0  # averaged over the target-present trials only
    assert not any("_final" in k or "fixhit" in k for k in a)
    # t0 is the same centred fixation for every policy
    assert a["loss_t0_random"] == pytest.approx(a["loss_t0_greedy"]) == pytest.approx(a["loss_t0_policy"])
    # seeded: a second evaluation reproduces the random policy exactly
    b = task.evaluate(model=model, head=head, val_loader=val, device=torch.device("cpu"), step=0, joint=joint)
    assert a["loss_t2_random"] == b["loss_t2_random"]


def test_eval_writes_every_metric_to_jsonl_and_an_overview_to_the_tracker(fake_cs18, tmp_path):
    import json
    root, _ = fake_cs18
    task = _TinyTask(_cfg(root, viz_every=0), rl=_rl())
    model, head = task.build_model(torch.device("cpu"))
    joint = task.build_policy(model, device=torch.device("cpu"), canvas_grid=32,
                              generator=torch.Generator().manual_seed(0))
    model.canvit.eval()

    class _Tracker:
        logged: dict = {}

        def log_metrics(self, m, step=None):
            self.logged.update(m)

    tr = _Tracker()
    a = task.evaluate(model=model, head=head, val_loader=task.build_val_loader(), device=torch.device("cpu"),
                      step=7, joint=joint, tracker=tr, run_dir=tmp_path)
    rows = [json.loads(line) for line in (tmp_path / "eval.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["step"] == 7
    full = rows[0]
    assert all(full[k] == v for k, v in a.items())  # the logged subset is a subset of the file
    for p in ("random", "greedy", "policy"):
        for t in range(3):  # every metric, every glimpse
            for k in ("found_10_percent", "hit_10_percent", "correct_hit", "f1"):
                assert f"{k}_t{t}_{p}" in full
            # strict <= lenient; hit accumulates over the fixations (t0 included)
            assert full[f"found_t{t}_{p}"] <= full[f"found_10_percent_t{t}_{p}"]
            assert full[f"hit_t{t}_{p}"] <= full[f"hit_10_percent_t{t}_{p}"]
            if t:
                assert full[f"hit_t{t}_{p}"] >= full[f"hit_t{t - 1}_{p}"]
        assert full[f"hit_t0_{p}"] == full["hit_t0_random"]  # t0 is the same centre for all
    ov = {k.removeprefix("overview/"): v for k, v in tr.logged.items()}
    assert all(k.startswith("overview/") for k in tr.logged) and len(ov) == 20 + 2 * 3  # + t0, t1 of 3 series
    for t in range(2):
        for k in (f"found_t{t}_policy", f"found_t{t}_random", f"hit_t{t}_policy"):
            assert ov[k] == full[k], k
    assert ov["found_gain_t2_policy_vs_random"] == pytest.approx(full["found_t2_policy"] - full["found_t2_random"])
    for k in ("found_t2_greedy", "hit_10_percent_t2_policy", "correct_found_t2_random",
              "presence_acc_t2_policy", "loss_t2_policy"):
        assert ov[k] == full[k], k


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
    assert torch.equal(last.reward_mask, torch.tensor([True, True]))  # condition='both'
    tp_only = BoundSearchTask(model=model, targets=batch["target"], boxes=batch["boxes"], canvas_grid=32,
                              cfg=_cfg(root, condition="tp"))
    assert not hasattr(tp_only, "reward_mask")


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


def test_viz_figures_are_written_for_training_and_every_eval_policy(fake_cs18, tmp_path):
    root, _ = fake_cs18
    # viz_samples < batch size: the figure slices the batch (job 17122634 crashed on that)
    task = _TinyTask(replace(_cfg(root, viz_every=2, viz_samples=2), eval_batch_size=4), rl=_rl())
    settings = RunSettings(n_steps=2, eval_every=2, log_every=1, grad_clip=1.0, amp=False, seed=0,
                           device="cpu", tracker="none", viz_every=2, run_dir=tmp_path)
    run(task=task, spec=task.default_spec(), settings=settings)
    for sub in ("search_train", "search_val_random", "search_val_greedy", "search_val_policy"):
        assert list((tmp_path / "visualization" / sub).glob("step-*.png")), sub


def test_policy_panel_is_the_distribution_the_deployed_policy_acted_on(fake_cs18):
    """The figure's policy map for fixation t is recomputed from the canvas after glimpse
    t-1; its argmax must be the cell the deploy rollout actually fixated at t."""
    from contextlib import nullcontext
    root, _ = fake_cs18
    task = _TinyTask(_cfg(root), rl=_rl())
    model, _ = task.build_model(torch.device("cpu"))
    joint = task.build_policy(model, device=torch.device("cpu"), canvas_grid=32,
                              generator=torch.Generator().manual_seed(0))
    torch.manual_seed(1)
    with torch.no_grad():  # leave the zero init, so the argmax is not a tie
        for p in joint.scorer.parameters():
            p.add_(0.1 * torch.randn_like(p))
    model.canvit.eval()
    batch = next(iter(task.build_val_loader()))
    record: list = []
    with torch.no_grad():
        task._rollout_metrics(model=model, images=batch["image"], targets=batch["target"], boxes=batch["boxes"],
                              T=3, cg=32, amp=nullcontext(), joint=joint, record=record)
    assert len(record) == 3
    for t in range(1, 3):
        pl = record[t - 1]["policy_next"]  # [n, 32, 32]
        assert pl.flatten(1).std(-1).min() > 0
        idx = pl.flatten(1).argmax(-1)
        cy, cx = record[t]["center"][:, 0], record[t]["center"][:, 1]
        iy = ((cy + 1) / 2 * 32).floor().long()
        ix = ((cx + 1) / 2 * 32).floor().long()
        assert torch.equal(idx, iy * 32 + ix)
    ep = task._episode(record, images=batch["image"], boxes=batch["boxes"], targets=batch["target"],
                       present=batch["present"])
    assert torch.isnan(ep.policy[0]).all() and torch.equal(ep.policy[1], record[0]["policy_next"])


def test_step_loss_is_tp_heatmap_kl_plus_weighted_presence_loss(fake_cs18):
    """Haoze's composition: the heatmap KL over the target-present trials only, plus
    presence_weight x the presence loss over all trials."""
    from canvit.search.loss import per_image_loss as kl_fn
    from canvit.search.loss import presence_loss, target_map
    root, _ = fake_cs18
    cfg = _cfg(root)
    task = _TinyTask(cfg, rl=_rl())
    model, _ = task.build_model(torch.device("cpu"))
    boxes = torch.tensor([[[-0.1, -0.1, 0.1, 0.1]], [[float("nan")] * 4], [[0.2, 0.2, 0.4, 0.4]]])
    present = torch.tensor([True, False, True])
    targets = torch.tensor([0, 1, 2])
    bound = BoundSearchTask(model=model, targets=targets, boxes=boxes, canvas_grid=32, cfg=cfg, present=present)
    readout = torch.randn(3, 32, 32, model.canvit.canvas_dim)
    heat, pres = model.head(readout, targets)
    kl = kl_fn(heat, target_map(boxes, grid=32, kind=cfg.loss, sigma=cfg.target_sigma))
    expect = kl[present].mean() + 0.75 * presence_loss(pres, present, kind="focal", gamma=2.0).mean()
    assert torch.allclose(bound.step_loss(readout).combined, expect)
    assert kl[1] == 0 and torch.equal(bound.reward_mask, present)
    with pytest.raises(AssertionError, match="disagree"):
        BoundSearchTask(model=model, targets=targets, boxes=boxes, canvas_grid=32, cfg=cfg,
                        present=torch.tensor([True, True, True]))


def test_target_present_only_runs_without_a_presence_head(fake_cs18):
    root, _ = fake_cs18
    task = _TinyTask(_cfg(root, condition="tp"), rl=_rl())
    last = _run(task)
    assert torch.isfinite(torch.tensor(last["total_loss"]))
    assert task._joint is not None and task._bound is not None
    assert getattr(task._bound, "reward_mask", None) is None
    model, head = task.build_model(torch.device("cpu"))
    assert head.presence is None
    model.canvit.eval()
    a = task.evaluate(model=model, head=head, val_loader=task.build_val_loader(), device=torch.device("cpu"),
                      step=0, joint=None)
    assert "presence_acc_t2_random" not in a and a["correct_found_t2_random"] == a["found_t2_random"]
