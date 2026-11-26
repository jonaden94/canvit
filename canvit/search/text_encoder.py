"""Frozen text encoder for prompt-cued search: Talk2DINO for DINOv3 ViT-B (owner decision 2026-11-25).

A list of texts -> ``[B, 768]`` vectors in the DINOv3 ViT-B patch-feature space, computed at training time,
frozen, without gradients:

    text --CLIP ViT-B/16 text tower--> 512-d (OpenAI ``encode_text``: the final-LayerNorm'd EOT token
          times ``text_projection``) --Talk2DINO projection--> Linear(512, 768), tanh, Linear(768, 768)

- CLIP: OpenAI's ViT-B/16 weights through the transformers port ``openai/clip-vit-base-patch16`` (HF cache) and
  its ``CLIPTokenizer``, after OpenAI's own text cleaning (``clean_text``: ftfy, html unescape, whitespace). Texts
  longer than 77 tokens are cut, keeping the end token, as ``clip.tokenize(truncate=True)``, which Talk2DINO's
  feature extraction used. The ids equal OpenAI's for all 35,613 distinct texts the release can produce (epochs,
  val, synonyms, definitions; without the cleaning, 16 definitions with curly quotes differed).
- Projection: Talk2DINO's ``weights/dinov3_vitb_mlp_infonce.pth`` (``ProjectionLayer``, config
  ``dinov3_vitb_mlp_infonce.yaml``: act tanh, one hidden layer), read from the Talk2DINO clone by path and checked
  by sha256. Nothing is imported from Talk2DINO or OpenAI's ``clip`` package.
- Talk2DINO was trained against timm's ``vit_base_patch16_dinov3.lvd1689m``, the same weights as canvit's teacher
  ``facebook/dinov3-vitb16-pretrain-lvd1689m`` (all 162 tensors identical, checked 2026-11-25).
- The output is NOT normalised: Talk2DINO compares by cosine, so how the vector is scaled is the consumer's choice.
- Computed in fp32 with autocast off (Talk2DINO's own GPU path runs CLIP in fp16; the two agree to cos 0.999998).
  The parity test against Talk2DINO's own code: ``tests/test_text_encoder.py``.
"""

from __future__ import annotations

import hashlib
import html
import re
from pathlib import Path

import ftfy  # a dependency of dinov3 (canvit's); OpenAI's tokenizer cleans with it
import torch
from torch import Tensor, nn

TALK2DINO_PROJ = Path("/mnt/vast-nhr/projects/nib00021/jonathan/repos/dino_stuff/Talk2DINO/weights/"
                      "dinov3_vitb_mlp_infonce.pth")
TALK2DINO_PROJ_SHA256 = "6fc5ce7a05d8f1c829c174d87b861da05399cf0bfe50de3711ac940115d85fcc"
CLIP_REPO = "openai/clip-vit-base-patch16"
TEXT_DIM = 768


def clean_text(text: str) -> str:
    """OpenAI CLIP's ``basic_clean`` + ``whitespace_clean`` (``simple_tokenizer.py``); lower-casing is the
    tokenizer's."""
    text = html.unescape(html.unescape(ftfy.fix_text(text))).strip()
    return re.sub(r"\s+", " ", text).strip()


class Talk2DinoProjection(nn.Module):
    """Talk2DINO's ``ProjectionLayer.project_clip_txt`` for its tanh / one-hidden-layer configs (same parameter
    names, so its state dict loads strictly)."""

    def __init__(self, clip_dim: int = 512, dino_dim: int = TEXT_DIM):
        super().__init__()
        self.linear_layer = nn.Linear(clip_dim, dino_dim)
        self.hidden_layers = nn.ModuleList([nn.Linear(dino_dim, dino_dim)])

    def forward(self, x: Tensor) -> Tensor:
        return self.hidden_layers[0](torch.tanh(self.linear_layer(x.float())))


class TextEncoder(nn.Module):
    """``forward(texts) -> [B, 768]`` on the module's device. Frozen: no parameter requires grad, the module
    stays in eval mode (``train()`` is a no-op), and ``forward`` runs under ``no_grad`` with autocast off.

    ``cache=True`` keeps each distinct text's vector (on the device; ~3 KB per text, ~35k distinct texts in
    basic-v1 with every text option) and encodes only the texts not seen yet. A batch whose texts are all cached
    costs a lookup instead of the ~7 ms CLIP pass. A cached vector is the one computed with the batch the text
    first came in; other padding lengths change it by float rounding only (~1e-6). Moving the module clears it."""

    def __init__(self, proj_path: Path = TALK2DINO_PROJ, clip_repo: str = CLIP_REPO, *, cache: bool = False):
        super().__init__()
        from transformers import CLIPTextModelWithProjection, CLIPTokenizer

        data = Path(proj_path).read_bytes()
        if (h := hashlib.sha256(data).hexdigest()) != TALK2DINO_PROJ_SHA256:
            raise ValueError(f"{proj_path}: sha256 {h}, expected Talk2DINO's {TALK2DINO_PROJ_SHA256}")
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_repo)
        self.clip = CLIPTextModelWithProjection.from_pretrained(clip_repo)
        self.proj = Talk2DinoProjection()
        self.proj.load_state_dict(torch.load(proj_path, map_location="cpu", weights_only=True))
        self.requires_grad_(False)
        super().train(False)
        self._cache: dict[str, Tensor] | None = {} if cache else None

    def train(self, mode: bool = True) -> TextEncoder:
        return self  # frozen: always eval

    def _apply(self, fn, *args, **kwargs):
        if self._cache:
            self._cache = {}  # cached vectors stay on the old device / dtype otherwise
        return super()._apply(fn, *args, **kwargs)

    @torch.no_grad()
    def forward(self, texts: list[str]) -> Tensor:
        if self._cache is None:
            return self._encode(texts)
        new = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if new:
            self._cache.update(zip(new, self._encode(new)))
        return torch.stack([self._cache[t] for t in texts])

    def _encode(self, texts: list[str]) -> Tensor:
        dev = self.proj.linear_layer.weight.device
        tok = self.tokenizer([clean_text(t) for t in texts], padding=True, truncation=True, max_length=77,
                             return_tensors="pt")
        with torch.autocast(dev.type, enabled=False):
            x = self.clip(input_ids=tok.input_ids.to(dev, non_blocking=True),
                          attention_mask=tok.attention_mask.to(dev, non_blocking=True)).text_embeds
            return self.proj(x)


__all__ = ["CLIP_REPO", "TALK2DINO_PROJ", "TEXT_DIM", "Talk2DinoProjection", "TextEncoder", "clean_text"]
