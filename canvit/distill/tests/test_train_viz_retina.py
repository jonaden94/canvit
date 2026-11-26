"""The training PCA figure (``pca_train``) of a multi-retina run names the retina of the sample
it shows: every training sample draws its own retina, so the figure is otherwise ambiguous.
Tiny 2-retina foveated ViT-S, the real engine rollout and ``DistillRunTask.render_viz``."""

import sys
from types import SimpleNamespace

import torch

from canvit import CanViTForPretraining, CanViTForPretrainingConfig
from canvit.core import create_backbone
from canvit.core.patcher.conditioning import PatchConditioningConfig
from canvit.core.patcher.foveated import FoveatedPatcherConfig
from canvit.distill.task import DistillRunTask
from canvit.harness.config import FoveatedScaleConfig
from canvit.harness.rollout import run_rollout
from canvit.harness.rollout.engine import GlimpseOut
from canvit.harness.rollout.selector import RandomSelector
from canvit.harness.rollout.viewpoint import ViewpointType
from canvit.harness.spec import BpttSpec

_G, _D = 8, 32


class _Task:  # the engine's per-batch task: the model's own forward, a dummy loss
    def forward_glimpse(self, *, model, images, state, viewpoint, backbone_no_grad):
        out = model(image=images, state=state, viewpoint=viewpoint)
        return GlimpseOut(readout=out, state=out.state, vpe=out.vpe)

    def step_loss(self, r):
        return SimpleNamespace(combined=r.scene_pred.float().pow(2).mean(), scene_pred=r.scene_pred)

    def per_image_loss(self, r):
        return r.scene_pred.float().pow(2).mean(dim=(1, 2))


def test_train_figure_names_the_retina_of_sample_0(tmp_path, monkeypatch):
    torch.manual_seed(0)
    fp = FoveatedPatcherConfig(fov=50, resolution=64, cmf_a=0.5, cart_patch_size=5, arch_flag="doubleres",
                               retinas=[(0.75, 32, 3, False), (2.2, 32, 3, False)],
                               conditioning=PatchConditioningConfig(mode="film"))
    model = CanViTForPretraining(
        backbone=create_backbone("vits16"),
        cfg=CanViTForPretrainingConfig(teacher_dim=_D, patcher_name="foveated", foveated_patcher=fp),
        glimpse_size_px=128, backbone_name="vits16", canvas_patch_grid_sizes=[_G])
    # render_viz / viz hooks only read these attributes of the run task
    run_task = SimpleNamespace(_glimpse_size_px=128, scene_norm=lambda x: x, _device=torch.device("cpu"),
                               cfg=SimpleNamespace(canvas_patch_grid_size=_G, glimpse_grid_size=8))
    hooks = SimpleNamespace(viz_init=lambda **kw: DistillRunTask.viz_init(run_task, **kw),
                            viz_frame=lambda **kw: DistillRunTask.viz_frame(run_task, **kw))
    sel = RandomSelector(is_foveated=True, foveated_scale=FoveatedScaleConfig(), min_viewpoint_scale=0.05)
    res = run_rollout(model=model, images=torch.randn(2, 3, 128, 128), task=_Task(), selector=sel,
                      bptt=BpttSpec(mode="none", horizon=2), branches=[ViewpointType.RANDOM],
                      canvas_grid_size=_G, amp_ctx=torch.enable_grad(), collect_viz=True, viz_task=hooks)
    retina = res.viz.initial["retina"]
    assert retina in model.patcher.retina_names

    plot_mod = sys.modules["canvit.distill.viz"]
    orig, titles = plot_mod.plot_multistep_pca, []

    def spy(**kw):
        fig = orig(**kw)
        titles.append(fig._suptitle.get_text() if fig._suptitle is not None else None)
        return fig

    monkeypatch.setattr(plot_mod, "plot_multistep_pca", spy)
    DistillRunTask.render_viz(run_task, res.viz, batch=(None, torch.randn(1, _G * _G, _D), None, None),
                              run_dir=tmp_path, step=0)
    assert titles == [f"training sample 0, retina {retina}"]  # the foveated layout used to drop it
    assert (tmp_path / "visualization" / "pca_train" / "step-0.png").exists()
