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
    def from_pretrained(cls, pretrained_model_name_or_path, *, revision: str | None = None,
                        strict: bool = True, **kwargs):
        """``strict`` defaults to True here (HF's default is False): mismatched weights
        raise. Pass ``strict=False`` to load a known-incomplete file with a warning."""
        from canvit.core.checkpoints import hub_revision

        if revision is None:
            revision = hub_revision(str(pretrained_model_name_or_path))
        return super().from_pretrained(  # type: ignore[misc]  # resolves to PyTorchModelHubMixin via the MRO
            pretrained_model_name_or_path, revision=revision, strict=strict, **kwargs)

    @classmethod
    def _load_as_safetensor(cls, model: nn.Module, model_file: str, map_location: str, strict: bool) -> nn.Module:
        import safetensors.torch
        missing, unexpected = safetensors.torch.load_model(model, model_file, strict=False, device=map_location)
        if missing or unexpected:
            msg = (
                f"Weights in {model_file} do not match {type(model).__name__}: "
                f"{len(missing)} missing, e.g. {sorted(missing)[:5]}; "
                f"{len(unexpected)} unexpected, e.g. {sorted(unexpected)[:5]}. "
                "For a canvit/* Hub repo this usually means the wrong revision: see "
                "canvit.core.checkpoints.hub_revision."
            )
            # An error by default: missing weights stay at random init and the model still
            # runs, so a warning lets a wrong checkpoint produce garbage silently.
            if strict:
                raise RuntimeError(msg)
            log.warning("%s Missing weights are left at random init.", msg)
        return model
