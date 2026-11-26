"""Prompt-cued search through the real task and ``harness.run`` on CPU: a tiny random foveated ViT-S, synthetic
release-format batches (both prompt types), the text encoder replaced by fixed random vectors (it has its own
parity test). LR groups, the distill switch, the val keys, COCO-Search18 as the zero-shot test."""

from dataclasses import replace

import pytest
import torch

from canvit.harness.run import RunSettings, run
from canvit.search.config import PromptConfig, SearchAugConfig, SearchConfig
from canvit.search.data import collate_search
from canvit.search.model import CanViTForVisualSearch
from canvit.search.task import SearchRunTask

from .test_release import fake_release, fake_vp_release  # noqa: F401  (fixtures)
from .test_search_task import _FOVEATED, _MULTI, _rl

S, G = 224, 32  # scene px, canvas grid (as the class-cued tests)
TEACHER_DIM = 768  # = the prompt width: the cosine baseline compares the two


def _batch(b: int, seed: int) -> dict:
    """Release-format samples: half present (one box), half absent; every other trial with a visual prompt."""
    g = torch.Generator().manual_seed(seed)
    samples = []
    for i in range(b):
        box = torch.tensor([[-0.2, -0.1, 0.1, 0.2]]) + 0.1 * (i % 3) if i % 2 == 0 else torch.zeros(0, 4)
        vp = i % 4 < 2
        samples.append({"image": torch.randn(3, S, S, generator=g), "boxes": box, "present": torch.tensor(i % 2 == 0),
                        "text": f"thing {i % 5}", "band": torch.tensor(i % 6),
                        "vp_feat": torch.randn(14, 768, generator=g) if vp else torch.zeros(14, 768),
                        "vp_cov": torch.rand(14, generator=g) if vp else torch.zeros(14),
                        "vp_valid": torch.tensor(vp)})
    return collate_search(samples)


class _FakeTeacher:
    calls = 0

    def forward_norm_features(self, images):
        _FakeTeacher.calls += 1
        from canvit.core.backbone.vit import NormFeatures
        b = images.shape[0]
        return NormFeatures(patches=images.mean() + torch.randn(b, G * G, TEACHER_DIM), cls=torch.zeros(b, TEACHER_DIM))


class _TinyPromptTask(SearchRunTask):
    """Builds a random tiny model, feeds synthetic batches, fakes the text encoder."""

    backbone_config = _FOVEATED

    def _build_prompt_model(self, device, prior_model_config):
        torch.manual_seed(0)
        p = self.cfg.prompt
        keep_scene = p.distill_weight > 0 or p.eval_cosine_baseline
        kw = {"adaln": p.adaln, "tokens": p.tokens, "token_mode": p.token_mode, "condition_heads": p.condition_heads,
              "scene_grid": G if keep_scene else None, "teacher_dim": TEACHER_DIM}
        model = CanViTForVisualSearch(backbone_name="vits16", model_config=self.backbone_config, n_classes=None,
                                      prompt_kw=kw, **self._head_kw())
        if keep_scene:
            model.scene["std"].set_stats(torch.randn(64, G * G, TEACHER_DIM))
        self.built = model
        return model.to(device)

    def _text_vectors(self, texts, device):
        return torch.stack([torch.randn(768, generator=torch.Generator().manual_seed(hash(t) % 1000))
                            for t in texts]).to(device)

    def build_loaders(self, *, world_size, rank):
        def endless():
            i = 0
            while True:
                yield _batch(4, seed=i)
                i += 1
        return endless(), [_batch(6, seed=100), _batch(6, seed=101)]

    def build_val_loader(self):
        return [_batch(6, seed=100), _batch(6, seed=101)]


def _cfg(**prompt) -> SearchConfig:
    return SearchConfig(dataset="release", prompt=PromptConfig(cue="prompt", **prompt), scene_size=S,
                        canvas_grid=G, n_timesteps=3, batch_size=4, eval_batch_size=6, num_workers=0,
                        head_width=16, head_block_layers=1, cond_dim=8, max_steps=4, warmup_steps=1,
                        device="cpu", amp=False, tracker="none", viz_every=0, model_repo="tiny",
                        aug=SearchAugConfig(crop=True), supervise="every")


def _run(task, n_steps=2):
    settings = RunSettings(n_steps=n_steps, eval_every=n_steps, log_every=1, grad_clip=1.0, amp=False, seed=0,
                           device="cpu", tracker="none")
    return run(task=task, spec=task.default_spec(), settings=settings)


@pytest.mark.parametrize("share", [0.0, 0.5, 1.0])
def test_prompt_search_trains_and_validates(share):
    task = _TinyPromptTask(_cfg(visual_share=share), rl=_rl(reward="fractional"))
    last = _run(task)
    assert torch.isfinite(torch.tensor(last["total_loss"]))
    types = ["text"] if share == 0 else ["visual"] if share == 1 else ["text", "visual"]
    assert task._eval_prompt_types() == types
    assert task.best_metric == "found_mean_policy"


def test_val_keys_per_prompt_type_and_band(tmp_path):
    import json
    task = _TinyPromptTask(_cfg(visual_share=0.5, eval_cosine_baseline=True), rl=_rl(reward="fractional"))
    model, head = task.build_model(torch.device("cpu"))
    joint = task.build_policy(model, device=torch.device("cpu"), canvas_grid=G,
                              generator=torch.Generator().manual_seed(0))
    out = task.evaluate(model=model, head=head, val_loader=task.build_val_loader(), device=torch.device("cpu"),
                        step=0, joint=joint, run_dir=tmp_path)
    full = json.loads((tmp_path / "eval.jsonl").read_text().splitlines()[0])
    for ty in ("text", "visual"):
        assert f"{ty}/found_mean_policy" in out and f"{ty}/found_t2_policy_band10-99" in out
        assert f"{ty}/found_mean_random" in out  # random is added at the first validation
        assert f"{ty}/found_cos_t2_policy" in full
    assert out["found_mean_policy"] == pytest.approx((out["text/found_mean_policy"]
                                                      + out["visual/found_mean_policy"]) / 2)
    # the visual pass sees only the trials with a visual prompt (i % 4 < 2: 8 of 12 here), the text pass all 12
    assert sum(v for k, v in full.items() if k.startswith("text/n_trials_band")) == 12
    assert sum(v for k, v in full.items() if k.startswith("visual/n_trials_band")) == 8
    out2 = task.evaluate(model=model, head=head, val_loader=task.build_val_loader(), device=torch.device("cpu"),
                         step=2, joint=joint)
    assert not any("random" in k for k in out2)  # not the first validation, not max_steps


def test_lr_groups():
    from canvit.harness.loop import apply_requires_grad
    from canvit.harness.optim import build_optimizer_and_scheduler
    for lrs, want in (({}, (1e-4, 1e-4, 1e-4)),
                      ({"backbone_lr": 1e-5, "head_lr": 3e-4, "policy_lr": 2e-4}, (1e-5, 3e-4, 2e-4))):
        task = _TinyPromptTask(_cfg(**lrs), rl=_rl())
        model, head = task.build_model(torch.device("cpu"))
        joint = task.build_policy(model, device=torch.device("cpu"), canvas_grid=G,
                                  generator=torch.Generator().manual_seed(0))
        spec = task.default_spec()
        assert spec.train_backbone and spec.task_grad_to_backbone and not spec.policy_grad_to_backbone
        apply_requires_grad(model=model, head=head, joint=joint, spec=spec)
        groups = task.trainable_param_groups(model=model, head=head, joint=joint, spec=spec)
        opt, _ = build_optimizer_and_scheduler(spec, groups)
        ids = [id(q) for g in groups.values() for q in g]
        assert len(ids) == len(set(ids)), "a parameter in two groups"
        trainable = {id(q) for q in list(model.parameters()) + list(joint.scorer.parameters()) if q.requires_grad}
        assert trainable == set(ids), "a trainable parameter in no group"
        prompt_ids = {id(q) for q in model.prompt.parameters()} | {id(q) for q in model.head_cond.parameters()}
        assert prompt_ids <= {id(q) for q in groups["head"]}
        assert {id(q) for q in model.canvit.parameters()} == {id(q) for q in groups["backbone"]}
        by_name = dict(zip(spec.trainable_modules(), opt.param_groups))
        got = tuple(by_name[m]["initial_lr"] for m in ("backbone", "head", "policy"))
        assert got == pytest.approx(want)


def test_distill_off_never_touches_the_teacher(monkeypatch):
    import canvit.core.teacher as teacher_mod

    def boom(*a, **k):
        raise AssertionError("teacher loaded with distill_weight 0")
    monkeypatch.setattr(teacher_mod, "load_teacher", boom)
    last = _run(_TinyPromptTask(_cfg(distill_weight=0.0), rl=_rl(reward="fractional")))
    assert torch.isfinite(torch.tensor(last["total_loss"]))


def test_distill_on_calls_the_teacher_once_per_batch(monkeypatch):
    import canvit.core.teacher as teacher_mod
    monkeypatch.setattr(teacher_mod, "load_teacher", lambda *a, **k: _FakeTeacher())
    _FakeTeacher.calls = 0
    task = _TinyPromptTask(_cfg(distill_weight=0.1), rl=_rl(reward="fractional"))
    out = _run(task, n_steps=2)
    assert torch.isfinite(torch.tensor(out["total_loss"]))
    # 2 training batches + 2 val batches x 2 prompt types (both passes run the teacher on their rows)
    assert _FakeTeacher.calls == 2 + 4


def test_refusals():
    with pytest.raises(ValueError, match="at least one"):
        task = SearchRunTask(_cfg(adaln=False, tokens="none", condition_heads=False), rl=_rl())
        task._build_prompt_model(torch.device("cpu"), None)
    assert SearchRunTask(_cfg(), rl=_rl()).caps().supports_ddp  # the release: DDP tested (job 17581598)
    coco = SearchRunTask(replace(_cfg(), dataset="coco_search18"), rl=_rl())
    assert not coco.caps().supports_ddp
    with pytest.raises(RuntimeError, match="DDP"):
        coco.build_loaders(world_size=2, rank=0)
    with pytest.raises(ValueError, match="release"):
        SearchRunTask(replace(_cfg(), dataset="coco_search18"), rl=_rl()).build_loaders(world_size=1, rank=0)


def test_coco_zero_shot_test_with_prompts(fake_cs18, fake_vp_release):  # noqa: F811
    """harness.evaluate's path on COCO-Search18 with cue='prompt': text = the category name, visual = the
    release's vp_test prompt of the category; keys per prompt type."""
    root, _ = fake_cs18
    cfg = replace(_cfg(visual_share=0.5), dataset="coco_search18", data_root=root, release_dir=fake_vp_release)
    task = _TinyPromptTask(cfg, rl=_rl())
    task.build_val_loader = SearchRunTask.build_val_loader.__get__(task)  # the real COCO val loader
    loader = task.build_val_loader()
    b = next(iter(loader))
    assert "vp_feat" in b and set(b["text"]) <= {"bottle", "bowl", "car", "chair", "clock", "cup", "fork",
                                                 "keyboard", "knife", "laptop"}
    assert b["vp_valid"].tolist() == [t in ("bottle", "bowl") for t in b["text"]]
    model, head = task.build_model(torch.device("cpu"))
    out = task.evaluate(model=model, head=head, val_loader=loader, device=torch.device("cpu"), step=0)
    assert "text/found_mean_random" in out and "visual/found_mean_random" in out


def test_prefix_token_mode_trains():
    """token_mode 'prefix' through the real task: trains, and the prefix modules are in the head group."""
    task = _TinyPromptTask(_cfg(visual_share=0.5, token_mode="prefix"), rl=_rl(reward="fractional"))
    last = _run(task)
    assert torch.isfinite(torch.tensor(last["total_loss"]))
    model, head = task.build_model(torch.device("cpu"))
    assert model.prompt.xattn is None and model.prompt.prefix is not None
    joint = task.build_policy(model, device=torch.device("cpu"), canvas_grid=G,
                              generator=torch.Generator().manual_seed(0))
    groups = task.trainable_param_groups(model=model, head=head, joint=joint, spec=task.default_spec())
    assert {id(q) for q in model.prompt.prefix.parameters()} <= {id(q) for q in groups["head"]}


def _frozen(spec):
    from canvit.harness.spec import BpttSpec
    return replace(spec, train_backbone=False, task_grad_to_backbone=False, bptt=BpttSpec(mode="none", horizon=3))


@pytest.mark.parametrize("frozen", [False, True])
def test_xattn_readout_trains(frozen):
    """--cfg.head xattn through the real task: trains with the backbone fine-tuned and frozen (the readout-only
    baseline: no AdaLN, no tokens -> the backbone stays bit-identical, the readout moves); its parameters are the
    head group; no FiLM head_cond is built."""
    prompt = dict(visual_share=0.5) if not frozen else dict(visual_share=0.5, adaln=False, tokens="none")
    task = _TinyPromptTask(replace(_cfg(**prompt), head="xattn", head_width=32), rl=_rl(reward="fractional"))
    spec = _frozen(task.default_spec()) if frozen else task.default_spec()
    settings = RunSettings(n_steps=2, eval_every=2, log_every=1, grad_clip=1.0, amp=False, seed=0, device="cpu",
                           tracker="none")
    last = run(task=task, spec=spec, settings=settings)
    assert torch.isfinite(torch.tensor(last["total_loss"]))
    trained = task.built
    fresh, _ = task.build_model(torch.device("cpu"))  # the same seeded init
    from canvit.search.head import PromptQueryHead
    assert isinstance(fresh.head, PromptQueryHead) and fresh.head_cond is None
    same_bb = all(torch.equal(a, b) for a, b in zip(trained.canvit.state_dict().values(),
                                                     fresh.canvit.state_dict().values()))
    assert same_bb == frozen
    assert not all(torch.equal(a, b) for a, b in zip(trained.head.state_dict().values(),
                                                     fresh.head.state_dict().values()))
    joint = task.build_policy(fresh, device=torch.device("cpu"), canvas_grid=G,
                              generator=torch.Generator().manual_seed(0))
    groups = task.trainable_param_groups(model=fresh, head=fresh.head, joint=joint, spec=spec)
    assert {id(q) for q in fresh.head.parameters()} <= {id(q) for q in groups["head"]}


def test_xattn_readout_refusals():
    with pytest.raises(ValueError, match="query"):
        _TinyPromptTask(replace(_cfg(condition_heads=False), head="xattn"), rl=_rl()).build_model(torch.device("cpu"))
    with pytest.raises(ValueError, match="Dropout2d"):
        _TinyPromptTask(replace(_cfg(), head="xattn", dropout=0.1), rl=_rl()).build_model(torch.device("cpu"))
    from canvit.search.model import CanViTForVisualSearch
    with pytest.raises(ValueError, match="prompt-cued"):
        CanViTForVisualSearch(backbone_name="vits16", model_config=_FOVEATED, n_classes=3, kind="xattn")


@pytest.mark.parametrize("rows", ["no_visual", "no_text"])
@pytest.mark.parametrize("ddp", [False, True])
@pytest.mark.parametrize("variant", ["xattn_tokens_cosine", "prefix_xattn_readout"])
def test_ddp_every_trainable_parameter_gets_a_grad(rows, ddp, variant):
    """Under DDP (``_ddp``) every trainable parameter of the model and the scorer has a grad tensor after the step's
    backward, also those this batch does not use (a batch without visual rows / without text rows; the kept scene
    head of the cosine baseline): the harness all-reduces only grads that are not None, rank by rank. Single GPU:
    unchanged, such parameters keep grad None."""
    cfg = (_cfg(visual_share=0.5, eval_cosine_baseline=True) if variant == "xattn_tokens_cosine"
           else replace(_cfg(visual_share=0.5, token_mode="prefix"), head="xattn", head_width=32))

    class _Task(_TinyPromptTask):
        def build_loaders(self, *, world_size, rank):
            self._ddp = ddp

            def endless():
                while True:
                    b = _batch(4, seed=0)
                    b["vp_valid"] = torch.full((4,), rows == "no_text")
                    if rows == "no_text":
                        b["vp_feat"], b["vp_cov"] = torch.randn(4, 14, 768), torch.rand(4, 14) + 0.1
                    yield b
            return endless(), []

    task = _Task(cfg, rl=_rl(reward="fractional"))
    settings = RunSettings(n_steps=1, eval_every=0, log_every=1, grad_clip=1.0, amp=False, seed=0, device="cpu",
                           tracker="none")
    run(task=task, spec=task.default_spec(), settings=settings)
    params = dict(task.built.named_parameters()) | {f"scorer.{k}": v for k, v in task._joint.scorer.named_parameters()}
    missing = [k for k, p in params.items() if p.requires_grad and p.grad is None]
    unused = "prompt.encoder.visual.weight" if rows == "no_visual" else "prompt.encoder.text.weight"
    if ddp:
        assert missing == []
    else:
        assert unused in missing


def test_prompt_mode_with_all_retinas(tmp_path):
    """--cfg.retina all in prompt mode: trains; validation runs every retina with the prompt types inside:
    eval.jsonl holds <retina>/<type>/<key>; the plain keys (and so best.pt's found_mean_policy) are the mean over
    the retinas of the mean over the types."""
    import json

    class _Multi(_TinyPromptTask):
        backbone_config = _MULTI

    task = _Multi(replace(_cfg(visual_share=0.5), retina="all"), rl=_rl(reward="fractional"))
    last = _run(task)
    assert torch.isfinite(torch.tensor(last["total_loss"]))
    model, joint = task.built, task._joint
    patcher = model.canvit.patcher
    names = patcher.retina_names
    assert len(names) == 2 and patcher.fixed_retina is None
    model.canvit.eval()
    out = task.evaluate(model=model, head=model.head, val_loader=task.build_val_loader(), device=torch.device("cpu"),
                        step=4, joint=joint, run_dir=tmp_path)  # max_steps: the random rule runs too
    assert patcher._active_retinas is None
    row = [json.loads(line) for line in open(tmp_path / "eval.jsonl")][-1]
    for k in ("found_mean_policy", "found_t2_policy", "text/found_mean_policy", "visual/found_t2_random"):
        per = [row[f"{n}/{k}"] for n in names]
        assert row[k] == pytest.approx(sum(per) / len(per)), k
    for n in names:
        assert row[f"{n}/found_mean_policy"] == pytest.approx(
            (row[f"{n}/text/found_mean_policy"] + row[f"{n}/visual/found_mean_policy"]) / 2)
    assert out["found_mean_policy"] == pytest.approx(row["found_mean_policy"])
    assert f"{names[0]}/found_mean_policy" in out and f"{names[1]}/found_t2_policy" in out
    assert row[f"{names[0]}/text/loss_t0_random"] != row[f"{names[1]}/text/loss_t0_random"]


_STAGE2 = dict(saccade_cost="log", saccade_cost_weight=1.0, saccade_cost_beta_mode="conditioned",
               saccade_cost_beta_eval=0.3, reward="terminal")


def test_prompt_mode_with_a_beta_conditioned_saccade_cost():
    """The saccade cost (stage 2) in prompt mode: each trial draws its beta, appended to the policy's FiLM input
    after the (detached) prompt vector's projection; trains and validates (eval beta). Without policy
    conditioning it is refused (beta has nowhere to go)."""
    from canvit.harness.policy.joint import BETA_FEATURES
    task = _TinyPromptTask(_cfg(visual_share=0.5), rl=_rl(**_STAGE2))
    last = _run(task)
    assert torch.isfinite(torch.tensor(last["total_loss"]))
    model, joint = task.built, task._joint
    assert joint.saccade_cost.conditioned and joint.scorer.film_cond_dim == 8 + BETA_FEATURES
    b = _batch(4, seed=3)
    bound = task.bind(b, torch.device("cpu"), model=model, head=model.head)
    cond = bound.policy_cond()
    assert bound.saccade_beta.shape == (4,) and cond.shape == (4, 8 + BETA_FEATURES)
    assert torch.equal(cond[:, :8], joint.scorer.prompt_cond(bound.cue().vec))
    eval_cond = task._policy_cond(model, bound.cue())
    assert torch.allclose(eval_cond[:, 8:], joint.saccade_cost.features(torch.full((4,), 0.3)))
    with pytest.raises(ValueError, match="FiLM-conditioned"):
        t2 = _TinyPromptTask(_cfg(visual_share=0.5, condition_policy=False), rl=_rl(**_STAGE2))
        m2, _ = t2.build_model(torch.device("cpu"))
        t2.build_policy(m2, device=torch.device("cpu"), canvas_grid=G, generator=torch.Generator().manual_seed(0))
