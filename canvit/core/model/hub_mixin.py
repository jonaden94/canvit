"""Safe HF Hub loading mixin — checks for key mismatches that PyTorchModelHubMixin silently ignores."""

import logging

from torch import nn

log = logging.getLogger(__name__)


class SafeHubMixin:
    """Mixin that checks for key mismatches when loading HF Hub checkpoints.

    HuggingFace's PyTorchModelHubMixin loads with strict=False and ignores the
    (missing, unexpected) return value, silently leaving mismatched parameters
    at random init.  This mixin overrides _load_as_safetensor to check.

    Must appear BEFORE PyTorchModelHubMixin in the MRO so that this
    _load_as_safetensor is found first.

    Also pins ``canvit/*`` Hub repos to the revision in the format this code reads
    (see :func:`canvit.core.checkpoints.hub_revision`) unless the caller passes one.
    """

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *, revision: str | None = None, **kwargs):
        from canvit.core.checkpoints import hub_revision

        if revision is None:
            revision = hub_revision(str(pretrained_model_name_or_path))
        return super().from_pretrained(  # type: ignore[misc]  # resolves to PyTorchModelHubMixin via the MRO
            pretrained_model_name_or_path, revision=revision, **kwargs)

    @classmethod
    def _load_as_safetensor(cls, model: nn.Module, model_file: str, map_location: str, strict: bool) -> nn.Module:
        import safetensors.torch
        missing, unexpected = safetensors.torch.load_model(model, model_file, strict=False, device=map_location)
        if missing or unexpected:
            msg = (
                "\n"
                "╔══════════════════════════════════════════════════════════════╗\n"
                "║  WARNING: Checkpoint key mismatch during model loading!     ║\n"
                "╚══════════════════════════════════════════════════════════════╝\n"
            )
            if missing:
                msg += (
                    f"\n  {len(missing)} model keys not found in checkpoint:\n"
                    f"    {sorted(missing)[:5]}\n"
                    "  → These parameters are left at random initialization.\n"
                    "    The model will produce garbage outputs.\n"
                )
            if unexpected:
                msg += (
                    f"\n  {len(unexpected)} checkpoint keys not found in model:\n"
                    f"    {sorted(unexpected)[:5]}\n"
                    "  → These weights were silently dropped.\n"
                )
            msg += (
                "\n  This usually means your canvit version doesn't match\n"
                "  the checkpoint. Update with:\n"
                "\n"
                '    uv lock --upgrade-package canvit && uv sync\n'
                "\n"
                "  or:\n"
                "\n"
                '    uv add "canvit @ git+https://github.com/m2b3/CanViT-PyTorch.git"\n'
            )
            log.warning(msg)
        return model
