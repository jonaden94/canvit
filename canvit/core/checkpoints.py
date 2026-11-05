"""Repo-id construction for CanViT-owned models and probes.

Single source of truth for the CanViT checkpoint root. Every CanViT-owned
repo-id flows through :func:`resolve_canvit_repo`; third-party repos
(``facebook/...``, etc.) stay as bare string literals.

The default ``"canvit"`` resolves to canonical Hub IDs (e.g.
``canvit/canvitb16-...``). Override via ``$CANVIT_REPO_ROOT`` to redirect
every load — the value can be either an HF org prefix or a local path:

    # Local checkpoint bundle: ship checkpoints in a directory tree.
    export CANVIT_REPO_ROOT="$(pwd)/canvit_checkpoints"

Both shapes work transparently because ``PyTorchModelHubMixin.from_pretrained``
takes either an HF repo-id or a local directory path.
"""

import os

CANVIT_REPO_ROOT = os.environ.get("CANVIT_REPO_ROOT", "canvit").rstrip("/")


def resolve_canvit_repo(name: str) -> str:
    return f"{CANVIT_REPO_ROOT}/{name}"


# On 2026-09-26 upstream (m2b3/CanViT, canvit-pytorch 0.2) rewrote `main` of the published
# canvit/* Hub repos in a format this code cannot read: config.json keys changed, so the
# models fail to construct, and weight keys were renamed, which the hub mixin only WARNS
# about. The files this code reads stay at a tag, so every canvit/* Hub load pins it.
LEGACY_HUB_REVISION = "canvit-pytorch-0.1"
LEGACY_POLICY_HUB_REVISION = "original-format"  # the qpolicy-/pgpolicy- scorers' tag


def hub_revision(repo_id: str) -> str | None:
    """The Hub revision holding the format this code reads, for a ``canvit/*`` repo id.

    ``None`` (= ``main``) for anything else: a local directory, another org, and the
    DINOv3 IN1k probes (``dinov3-*``), which upstream left unchanged (card edits only).
    """
    org, _, name = str(repo_id).partition("/")
    if org != "canvit" or not name or os.path.isdir(str(repo_id)):
        return None
    if name.startswith("dinov3-"):
        return None
    if name.startswith(("qpolicy-", "pgpolicy-")):
        return LEGACY_POLICY_HUB_REVISION
    return LEGACY_HUB_REVISION
