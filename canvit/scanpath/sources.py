"""Where a CanViT policy comes from: the task it was trained on, with the run's own flags (design §12).

    python -m canvit.harness.evaluate scanpath --cfg.model canvit_policy --cfg.ckpt <run>/checkpoints/best.pt \
        cfg.source:search --cfg.source.cfg.model-repo <backbone> --cfg.source.rl.<flag> ...

Each source holds the task config and the policy config of ONE task, with the defaults of that task's
training command (``harness/cli.py``: ``SearchCmd`` / ``DistillCmd`` / ``Ade20kCmd`` / ``In1kCmd``), and
builds the task exactly as that command's ``build()`` does -- minus the run identity (run group, output
dirs, tracker), which an evaluation has no use for and ``build()`` would require. Pass the run's training
flags under ``--cfg.source.cfg.*`` / ``--cfg.source.rl.*``: the policy is rebuilt from them and loaded
strictly, so a flag that changed the architecture and was not passed fails the load.

Notes per task (claude_dev/scanpath/00-design.md §12): ade20k's policy features read the trained seg probe
(``--cfg.source.cfg.probe-repo``); distill rebuilds its model from the checkpoint's own config and loads the
gated DINOv3 teacher only with ``init_backbone_from_teacher`` (default off).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Any

import tyro

from canvit.ade20k.config import Ade20kConfig
from canvit.distill.config import Config as DistillConfig
from canvit.harness.config import JointPolicyConfig
from canvit.in1k.config import In1kConfig
from canvit.search.config import SearchConfig, search_policy_config


@dataclass
class SearchSource:
    """A search run (``harness.run search``): policy and/or task heatmap; COCO-Search18 cues."""

    cfg: SearchConfig = field(default_factory=SearchConfig)
    rl: JointPolicyConfig = field(default_factory=search_policy_config)

    def task(self) -> Any:
        from canvit.search.task import SearchRunTask
        return SearchRunTask(self.cfg, rl=self.rl)


@dataclass
class DistillSource:
    """A distillation (reconstruction) run, e.g. ``--preset policy_only`` on a foveated backbone."""

    cfg: DistillConfig = field(default_factory=DistillConfig)

    def task(self) -> Any:
        from canvit.distill.task import DistillRunTask
        return DistillRunTask(self.cfg)


@dataclass
class Ade20kSource:
    """An ADE20K segmentation run with a policy."""

    cfg: Ade20kConfig = field(default_factory=Ade20kConfig)
    rl: JointPolicyConfig = field(default_factory=JointPolicyConfig)

    def task(self) -> Any:
        from canvit.ade20k.task import Ade20kRunTask
        return Ade20kRunTask(self.cfg, rl=self.rl)


@dataclass
class In1kSource:
    """An ImageNet-1k classification run with a policy."""

    cfg: In1kConfig = field(default_factory=In1kConfig)
    rl: JointPolicyConfig = field(default_factory=JointPolicyConfig)

    def task(self) -> Any:
        from canvit.in1k.task import In1kRunTask
        return In1kRunTask(self.cfg, rl=self.rl, total_steps=self.cfg.max_steps)


PolicySource = (Annotated[SearchSource, tyro.conf.subcommand("search")]
                | Annotated[DistillSource, tyro.conf.subcommand("distill")]
                | Annotated[Ade20kSource, tyro.conf.subcommand("ade20k")]
                | Annotated[In1kSource, tyro.conf.subcommand("in1k")])

__all__ = ["Ade20kSource", "DistillSource", "In1kSource", "PolicySource", "SearchSource"]
