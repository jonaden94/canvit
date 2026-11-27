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


_MULTI = {"patcher_name": "foveated", "foveated_patcher": {
    **_FOVEATED["foveated_patcher"], "fov": 50, "rf_interp": "bilinear",
    "retinas": [[0.75, 32, 3, False], [2.2, 32, 3, False]]}}


class _TinyMultiTask(SearchRunTask):
    """A random tiny MULTI-retina backbone, through the real retina check of build_model."""

    def build_model(self, device, prior_model_config=None):
        torch.manual_seed(0)
        c = self.cfg
        model = CanViTForVisualSearch(backbone_name="vits16", model_config=_MULTI,
                                      n_classes=len(self._classes()), cond_dim=c.cond_dim, kind=c.head,
                                      width=c.head_width, block_layers=c.head_block_layers,
                                      presence=c.condition == "both", levels=c.head_levels).to(device)
        self._apply_retina(model, source="tiny-multi")
        self.model = model
        return model, model.head


def test_multi_retina_backbone_trains_and_evaluates_through_the_fixed_retina(fake_cs18):
    root, _ = fake_cs18
    task = _TinyMultiTask(_cfg(root, retina="cmf2.2_res32_cps3"), rl=_rl())
    last = _run(task)                                            # 2 training steps + one eval
    assert torch.isfinite(torch.tensor(last["total_loss"]))
    patcher = task.model.canvit.patcher
    assert patcher.retina_names == ["cmf0.75_res32_cps3", "cmf2.2_res32_cps3"]
    assert patcher.fixed_retina == 1 and patcher.active_retina(0) == 1


def test_retina_choice_is_required_for_multi_and_refused_for_single(fake_cs18):
    root, _ = fake_cs18
    for bad in (None, "cmf9_res1_cps1"):
        with pytest.raises(ValueError, match="multi-retina backbone"):
            _TinyMultiTask(_cfg(root, retina=bad), rl=_rl()).build_model(torch.device("cpu"))
    from canvit.search.task import fix_backbone_retina
    model, _ = _TinyTask(_cfg(root), rl=_rl()).build_model(torch.device("cpu"))
    with pytest.raises(ValueError, match="needs a multi-retina backbone"):
        fix_backbone_retina(model, "cmf0.75_res32_cps3", "tiny")
    fix_backbone_retina(model, None, "tiny")                     # single retina, no choice: fine


def test_all_retinas_trains_on_drawn_retinas_and_validates_each(fake_cs18, tmp_path):
    """``--cfg.retina all``: no retina is fixed, the engine draws them in training, and
    validation runs every retina on the same images: the usual keys are the mean over
    retinas, each retina's values sit under ``<retina>/``, the choice is cleared after."""
    import json
    root, _ = fake_cs18
    task = _TinyMultiTask(_cfg(root, retina="all", supervise="every"), rl=_rl(reward="fractional",
                          reward_center="running_mean", reward_scale="running_std"))
    last = _run(task)
    assert torch.isfinite(torch.tensor(last["total_loss"]))
    model = task.model
    patcher = model.canvit.patcher
    assert patcher.fixed_retina is None and patcher._active_retinas is None
    names = patcher.retina_names
    model.canvit.eval()
    out = task.evaluate(model=model, head=model.head, val_loader=task.build_val_loader(),
                        device=torch.device("cpu"), step=7, joint=task._joint, run_dir=tmp_path)
    assert patcher._active_retinas is None
    for k in ("found_t2_policy", "hit_t2_random", "presence_acc_t2_greedy"):
        per = [out[f"{n}/{k}"] for n in names]
        assert out[k] == pytest.approx(sum(per) / len(per)), k
    assert "found_t0_policy" in out and f"{names[0]}/found_t0_policy" not in out  # per retina: last glimpse only
    row = [json.loads(line) for line in open(tmp_path / "eval.jsonl")][-1]
    assert row["step"] == 7 and f"{names[1]}/loss_t0_random" in row and "loss_t0_random" in row
    # each retina really is validated through its own retina: the t0 losses differ
    assert row[f"{names[0]}/loss_t0_random"] != row[f"{names[1]}/loss_t0_random"]


def test_all_retinas_is_refused_for_a_single_retina_backbone(fake_cs18):
    root, _ = fake_cs18
    task = _TinyMultiTask(_cfg(root, retina="all"), rl=_rl())
    model, _ = _TinyTask(_cfg(root), rl=_rl()).build_model(torch.device("cpu"))
    with pytest.raises(ValueError, match="needs a multi-retina backbone"):
        task._apply_retina(model, source="tiny")


def test_head_levels_sets_the_unet_depth_and_defaults_to_six():
    from canvit.core.policy.net import N_UNET_ENC
    from canvit.search.head import SearchHead
    kw = dict(canvas_dim=24, n_classes=3, cond_dim=8, width=16, block_layers=1, presence=True)
    deep, shallow = SearchHead(**kw), SearchHead(**kw, levels=3)
    assert len(deep.enc) == N_UNET_ENC == 6 and len(shallow.enc) == 3 and len(shallow.film_enc) == 3
    assert len(shallow.dec) == len(shallow.film_dec) == 2
    x, t = torch.randn(2, 32, 32, 24), torch.tensor([0, 2])
    heat, pres = shallow(x, t)
    assert heat.shape == (2, 32, 32) and pres.shape == (2,)
    # the default is the historical head: same parameters and the same init
    torch.manual_seed(0)
    a = SearchHead(**kw).state_dict()
    torch.manual_seed(0)
    b = SearchHead(**kw, levels=6).state_dict()
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    # a checkpoint's head_kw without "levels" (every run before the option) rebuilds 6 levels
    assert SearchConfig(data_root=".").head_levels == 6


# search's PG defaults configure the entropy dual ascent; QReg has none, so unset them
_QREG = dict(objective="qreg", entropy_alpha_init=None, entropy_alpha_min=None, entropy_alpha_max=None,
             entropy_update="per_glimpse")


def test_qreg_trains_on_search_with_the_target_present_mask(fake_cs18):
    root, _ = fake_cs18
    task = _TinyTask(_cfg(root, supervise="every"), rl=_rl(**_QREG, reward="fractional",
                     reward_center="running_mean", reward_scale="running_std"))
    last = _run(task)
    assert torch.isfinite(torch.tensor(last["total_loss"]))


def test_qreg_mask_regresses_only_the_masked_in_rows(fake_cs18):
    """QReg's per-glimpse loss with a reward mask = the plain QReg MSE over the masked-in
    rows (their reward standardised among themselves); masked-out rows get no gradient."""
    root, _ = fake_cs18
    kw = dict(_QREG, reward="fractional", reward_center="none", reward_scale="none")
    task = _TinyTask(_cfg(root, supervise="every"), rl=_rl(**kw))
    model, _ = task.build_model(torch.device("cpu"))
    joint = task.build_policy(model, device=torch.device("cpu"), canvas_grid=32,
                              generator=torch.Generator().manual_seed(0))
    torch.manual_seed(1)
    scores = torch.randn(4, 1024, requires_grad=True)
    idx, reward = torch.tensor([3, 7, 11, 5]), torch.tensor([0.5, -1.0, 2.0, 0.25])
    mask = torch.tensor([True, False, True, False])
    loss = joint.glimpse_loss(depth=1, scores=scores, flat_idx=idx, reward=reward, mask=mask)
    ref = joint.glimpse_loss(depth=1, scores=scores[mask], flat_idx=idx[mask], reward=reward[mask])
    assert torch.allclose(loss, ref)
    loss.backward()
    assert scores.grad[~mask].abs().sum() == 0 and scores.grad[mask].abs().sum() > 0
    none = joint.glimpse_loss(depth=1, scores=scores, flat_idx=idx, reward=reward,
                              mask=torch.zeros(4, dtype=torch.bool))
    assert none.item() == 0.0


def test_head_lr_schedule_cosine_by_default_constant_on_request(fake_cs18):
    root, _ = fake_cs18
    cos = _TinyTask(_cfg(root), rl=_rl()).default_spec().optim["head"].schedule
    assert cos.kind == "warmup_cosine" and cos.total_steps == 4 and cos.warmup_steps == 1
    const = _TinyTask(replace(_cfg(root, lr_schedule="constant"), warmup_steps=0), rl=_rl()).default_spec().optim["head"]
    assert const.schedule.kind == "warmup_constant" and const.schedule.warmup_steps == 0
    # the head then follows the same schedule shape as the policy group
    pol = _TinyTask(_cfg(root), rl=_rl(policy_warmup_steps=0, policy_warmup_frac=0.0)).default_spec().optim["policy"]
    assert pol.schedule.kind == const.schedule.kind


def test_resumed_search_run_keeps_the_runs_best_checkpoint(fake_cs18, tmp_path, monkeypatch):
    """Through harness.run: chunk 1 peaks, chunk 2 (resumed from latest.pt) validates worse,
    so best.pt must stay at chunk 1's peak (it used to be replaced: best was per job)."""
    from canvit.harness.infra.checkpoint import load_checkpoint
    root, _ = fake_cs18
    scores = {0: 0.2, 1: 0.8, 2: 0.5, 3: 0.4, 4: 0.3}   # chunk 1: 0, 1 + end 2 | chunk 2: 3 + end 4
    evals: list[int] = []

    def fake_eval(self, **kw):
        evals.append(kw["step"])
        return {"found_t2_policy": scores[kw["step"]]}
    monkeypatch.setattr(SearchRunTask, "evaluate", fake_eval)

    def chunk(n):
        task = _TinyTask(_cfg(root), rl=_rl())
        s = RunSettings(n_steps=n, eval_every=1, log_every=1, grad_clip=1.0, amp=False, seed=0,
                        device="cpu", tracker="none", ckpt_dir=tmp_path, resume=True)
        run(task=task, spec=task.default_spec(), settings=s)
    chunk(2)                                       # steps 0-1 + final weights (2): best 0.8 at step 1
    assert load_checkpoint(tmp_path / "best.pt", "cpu")["step"] == 1
    chunk(2)                                       # resumed at 2 (already validated): 3 + end 4
    assert load_checkpoint(tmp_path / "best.pt", "cpu")["step"] == 1
    assert load_checkpoint(tmp_path / "latest.pt", "cpu")["extra"]["best"]["value"] == 0.8
    assert evals == [0, 1, 2, 3, 4], "each step validated once; the run's final weights included"


def test_search_run_stops_at_max_steps_and_a_spare_array_task_does_nothing(fake_cs18, tmp_path, monkeypatch):
    """Search's defaults (checkpoint at every validation, keep 1) through harness.run, with
    a 4-step run in chunks of 3: chunk 2 trains only 1 step (to max_steps), chunk 3 starts
    at max_steps and exits without validating or training."""
    from canvit.harness.infra.checkpoint import load_checkpoint
    from canvit.harness.loop import read_last_val
    root, _ = fake_cs18
    evals: list[int] = []

    def fake_eval(self, **kw):
        evals.append(kw["step"])
        return {"found_t2_policy": 0.1 * kw["step"]}
    monkeypatch.setattr(SearchRunTask, "evaluate", fake_eval)

    def chunk():
        task = _TinyTask(_cfg(root), rl=_rl())
        s = RunSettings(n_steps=3, max_steps=4, eval_every=1, log_every=1, grad_clip=1.0, amp=False,
                        seed=0, device="cpu", tracker="none", ckpt_dir=tmp_path, resume=True,
                        ckpt_at_eval=True, keep_ckpts=1)
        return run(task=task, spec=task.default_spec(), settings=s)
    chunk()                                   # 0, 1, 2 + end 3
    chunk()                                   # resumed at 3 (validated): 1 step, end 4
    assert chunk() == {}                      # spare: start 4 >= max_steps
    assert evals == [0, 1, 2, 3, 4]
    assert sorted(p.name for p in tmp_path.glob("*.pt")) == ["best.pt", "latest.pt", "step-4.pt"]
    assert load_checkpoint(tmp_path / "latest.pt", "cpu")["step"] == 4 and read_last_val(tmp_path)["step"] == 4
    # Complete but its final weights not on record as validated (the final validation died,
    # or the run predates validate-at-end): the next job validates them once, trains nothing.
    (tmp_path / "last_val.json").unlink()
    mtime = (tmp_path / "step-4.pt").stat().st_mtime_ns
    chunk()
    assert evals == [0, 1, 2, 3, 4, 4]
    assert (tmp_path / "step-4.pt").stat().st_mtime_ns == mtime, "a validation-only job must not rewrite the checkpoint"
    assert read_last_val(tmp_path)["step"] == 4 and load_checkpoint(tmp_path / "best.pt", "cpu")["step"] == 4
    assert chunk() == {}                      # and now it is on record: nothing to do


def test_resume_from_continues_another_runs_state_at_this_runs_lr(fake_cs18, tmp_path, monkeypatch):
    """An LR decay of a finished run: a NEW run dir continues the source's full state (step,
    weights, AdamW moments, policy) with this run's LRs, validates the inherited weights at its
    start and keeps its own best; the source run's files are not touched."""
    from canvit.harness.infra.checkpoint import load_checkpoint
    root, _ = fake_cs18
    src, dst = tmp_path / "src", tmp_path / "dst"
    evals: list[tuple[str, int]] = []

    def fake_eval(self, **kw):
        evals.append((self.cfg.run_name, kw["step"]))
        return {"found_t2_policy": 0.1}
    monkeypatch.setattr(SearchRunTask, "evaluate", fake_eval)

    def go(ckpt_dir, *, lr, resume_from=None):
        task = _TinyTask(replace(_cfg(root, peak_lr=lr, lr_schedule="constant"), warmup_steps=0,
                                 run_name=ckpt_dir.name), rl=_rl(policy_lr=lr, policy_warmup_steps=0))
        s = RunSettings(n_steps=2, max_steps=4, eval_every=1, log_every=1, grad_clip=1.0, amp=False, seed=0,
                        device="cpu", tracker="none", ckpt_dir=ckpt_dir, resume=True, resume_from=resume_from)
        run(task=task, spec=task.default_spec(), settings=s)
    go(src, lr=1e-3)
    src_files = {p.name: p.stat().st_mtime_ns for p in src.iterdir()}
    go(dst, lr=1e-4, resume_from=src / "step-2.pt")
    assert evals == [("src", 0), ("src", 1), ("src", 2), ("dst", 2), ("dst", 3), ("dst", 4)]
    out = load_checkpoint(dst / "latest.pt", "cpu")
    assert out["step"] == 4 and sorted(p.name for p in dst.glob("step-*.pt")) == ["step-4.pt"]
    groups = out["optimizer_state"]["param_groups"]
    assert len(groups) == 2 and {g["lr"] for g in groups} == {1e-4}   # task head AND policy at the new LR
    assert {p.name: p.stat().st_mtime_ns for p in src.iterdir()} == src_files, "the source run was modified"



def test_cli_passes_search_resume_from(tmp_path):
    from canvit.harness.cli import SearchCmd
    p = tmp_path / "step-5000.pt"
    _, s = SearchCmd(cfg=SearchConfig(run_group="g", resume_from=p)).build()
    assert s.resume_from == p


def test_efference_copy_with_saccade_cost_trains_and_validation_reports_saccade_length(fake_cs18):
    """exp44's arm through harness.run: the exp43 recipe on all retinas, the efference copy as the scorer's
    last group, the per-glimpse log cost. Validation reports the deployed policy's saccade lengths."""
    root, _ = fake_cs18
    task = _TinyMultiTask(_cfg(root, retina="all", supervise="every"),
                          rl=_rl(reward="fractional", reward_center="running_mean", reward_scale="running_std",
                                 efference_copy=True, saccade_cost="log", saccade_cost_weight=0.3,
                                 saccade_cost_r=0.06))
    seen = {}
    orig = SearchRunTask.evaluate

    def spy(self, **kw):
        out = orig(self, **kw)
        seen[kw["step"]] = out
        return out
    task.evaluate = spy.__get__(task)  # type: ignore[method-assign]
    last = _run(task, n_steps=2)
    assert torch.isfinite(torch.tensor(last["total_loss"]))
    assert task._joint.scorer.frontend.groups[-1] == "efcopy" and task._joint.policy_selector.efference_copy
    out = seen[2]
    T = task.cfg.n_timesteps
    for p in ("policy", "greedy", "random"):
        lens = [out[f"saccade_len_t{t}_{p}"] for t in range(1, T)]
        assert all(0.0 <= v <= 2 * 2 ** 0.5 for v in lens)            # a distance inside the [-1, 1] square
        assert abs(out[f"saccade_len_{p}"] - sum(lens) / len(lens)) < 1e-9
    assert out["saccade_len_random"] > 0
