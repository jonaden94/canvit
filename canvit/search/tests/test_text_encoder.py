"""The frozen Talk2DINO text encoder (``canvit/search/text_encoder.py``) and the text cue of COCO-Search18.

The parity test compares with vectors made by Talk2DINO's OWN code (OpenAI ``clip`` + its ``ProjectionLayer``) for
51 texts: ``test_data/talk2dino_text_reference.npz``, written by
``canvit_claude/canvit/search/text_encoder/make_reference.py`` (canvit never imports Talk2DINO or ``clip``). It
needs the CLIP weights in the HF cache and Talk2DINO's projection file, so it is ``slow`` and skips without them.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from canvit.search.text_encoder import CLIP_REPO, TALK2DINO_PROJ, TEXT_DIM, clean_text

REFERENCE = Path(__file__).resolve().parents[3] / "test_data" / "talk2dino_text_reference.npz"


def _clip_cached() -> bool:
    from huggingface_hub import try_to_load_from_cache
    return isinstance(try_to_load_from_cache(CLIP_REPO, "pytorch_model.bin"), str)


def test_clean_text_is_openais():
    assert clean_text("  a  “curly”\tquote’s &amp;amp; test ") == "a \"curly\" quote's & test"
    assert clean_text("slow_cooker") == "slow_cooker"


def test_coco_samples_carry_the_category_name(fake_cs18):
    from canvit.search.config import SearchConfig
    from canvit.search.data import SearchDataset, coco_search18_classes, collate_search, load_records

    root, pairs = fake_cs18
    classes = coco_search18_classes(root)
    recs = load_records(SearchConfig(data_root=root), "train", classes)
    assert all(r.text == classes[r.target] for r in recs)
    ds = SearchDataset(recs[:2], scene_size=32, aug=None)
    batch = collate_search([ds[0], ds[1]])
    assert batch["text"] == [recs[0].text, recs[1].text]
    assert batch["target"].tolist() == [recs[0].target, recs[1].target]  # the class-cued path is unchanged


@pytest.mark.slow
@pytest.mark.skipif(not TALK2DINO_PROJ.exists(), reason="Talk2DINO projection weights not on disk")
@pytest.mark.skipif(not _clip_cached(), reason=f"{CLIP_REPO} not in the HF cache")
def test_vectors_equal_talk2dinos_own_code():
    from canvit.search.text_encoder import TextEncoder

    z = np.load(REFERENCE, allow_pickle=False)
    ref = {"texts": z["texts"].tolist(), "fp32_cpu": torch.from_numpy(z["fp32_cpu"]),
           "fp16_cuda": torch.from_numpy(z["fp16_cuda"])}
    enc = TextEncoder()
    assert not any(p.requires_grad for p in enc.parameters())
    enc.train()
    assert not enc.training and not enc.clip.training  # frozen: train() is a no-op
    v = enc(ref["texts"])
    assert v.shape == (len(ref["texts"]), TEXT_DIM) and v.dtype == torch.float32 and not v.requires_grad
    for k in ("fp32_cpu", "fp16_cuda"):  # Talk2DINO's CPU (fp32) and GPU (fp16 CLIP) numerics
        cos = torch.nn.functional.cosine_similarity(v, ref[k].float())
        assert cos.min() > 0.9999, (k, cos.min().item())
    plain = enc(ref["texts"][:8])
    with torch.autocast("cpu", dtype=torch.bfloat16):  # a training step's autocast must not reach the encoder
        assert torch.equal(enc(ref["texts"][:8]), plain)
    # the optional cache: same vectors (up to the rounding of another padding length), repeated texts not re-encoded
    texts = ref["texts"][:6] + ref["texts"][:3]
    enc._cache = {}
    calls = []
    encode = enc._encode
    enc._encode = lambda t: (calls.append(list(t)), encode(t))[1]
    cached = enc(texts)
    assert calls == [ref["texts"][:6]]
    assert torch.allclose(cached, enc._encode(texts), atol=1e-5)
    enc(ref["texts"][2:5])
    assert len(calls) == 2  # only the explicit _encode above; the second batch came from the cache
