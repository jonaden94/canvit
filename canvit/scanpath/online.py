"""Human alignment measured DURING training (canvit_claude/canvit_working_guide.md §4): is the policy becoming more
human-like as it trains? A small fixed set of human scanpaths is scored teacher-forced against the LIVE task, model
and policy at the harness's eval steps (``--opts.human-eval-every N``), logged as ``val/human/*`` and never used to
select checkpoints: humans are measured, never optimised for.

Two halves:

- :func:`build` (offline, once; pysaliency): COCO-Search18 validation images, half target present and half target
  absent, with their COCO-Search18 scanpaths of that condition and their COCO-FreeView scanpaths (the FreeView
  validation images ARE COCO-Search18 validation images, pixel-identical, so the three sets share images). Per image:
  the letterboxed scene (uint8) and, on a fine 64 x 64 scene grid, the fraction of the image's pixels in each cell,
  the centre bias's mass (the full evaluation's leave-one-image-out KDE with the dataset's recorded fit) and a
  Gaussian KDE of the humans' fixations (1 deg). One ``.pt`` file; the training job needs nothing else.
- :class:`HumanEval` (training time; torch only): the policy's teacher-forced logits over the human scanpaths
  (``canvit_models.PolicyEpisode._episode``, the same code path as the standalone evaluation), scored on the policy's
  own grid G (the fine maps summed into G x G cells, 64 % G == 0):

  * ``ig_<set>``: IG over the centre bias, bits/fix, image-averaged (pysaliency's convention). Grid level: both
    densities are spread evenly over a cell's pixels, so the pixel terms cancel (earlier check: grid vs pixel
    level < 0.02 bits). At temperature 1, as trained; the standalone evaluation also reports the fitted tau.
  * ``ll_<set>``: log-likelihood over uniform, bits/fix, image-averaged.
  * ``entropy_<set>`` / ``entropy_human_<set>``: the policy map's entropy (nats over the image's cells, mean over
    scored fixations) and the entropy of the humans' KDE map on the same grid (mean over images).
  * ``sacc_len_<set>`` / ``sacc_len_human_<set>``: the mean saccade amplitude (deg) of the policy's OWN scanpaths
    (drawn at temperature 1 from the same density, ``sample_fixations`` long, one per human scanpath) and of the
    humans' first ``sample_fixations`` fixations.

  Sets: ``tp`` / ``ta`` with the scanpath's cue; ``fv`` without one (a search model: the equal-weight mixture over
  the 18 cues, as the standalone evaluation). The call leaves the training state untouched: RNG streams (global,
  CUDA, the policy's generator), train/eval modes.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from canvit.core.preprocess import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

log = logging.getLogger(__name__)

FINE = 64  # the cached maps' scene grid; the policy grid G must divide it
SETS = ("tp", "ta", "fv")
VERSION = 2


# --- offline: the cache -----------------------------------------------------------------------------------------

def _fine_maps(stimulus: np.ndarray, log_density: np.ndarray, *, scene: int, frame
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(fraction of the image's pixels, centre-bias mass) per fine scene cell [FINE*FINE] of one stimulus, and the
    fine cell of every pixel [h, w] (models._pixel_cells: the cell of the pixel's centre)."""
    from .models import _pixel_cells
    h, w = stimulus.shape[:2]
    cell, count = _pixel_cells(h, w, scene, FINE, frame)
    ld = np.asarray(log_density, np.float64)
    cb = np.bincount(np.asarray(cell).ravel(), weights=np.exp(ld - ld.max()).ravel(), minlength=FINE * FINE)
    return count / count.sum(), cb / cb.sum(), np.asarray(cell)


def _kde(points: np.ndarray, frac: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian KDE (sigma in scene units) of scene points [P, 2] as mass per fine cell, restricted to the image."""
    c = (np.arange(FINE) + 0.5) * (2.0 / FINE) - 1.0
    gx, gy = np.meshgrid(c, c)  # cell (row, col) = (y, x)
    d2 = (gx.ravel()[:, None] - points[:, 0]) ** 2 + (gy.ravel()[:, None] - points[:, 1]) ** 2
    m = np.exp(-0.5 * d2 / sigma ** 2).sum(1) * frac
    return m / m.sum()


def build(cs18, fv, *, fit_cs18: dict, fit_fv: dict, scene: int, frame, n_images: int = 12, seed: int = 0,
          ppd: float = 31.1, kde_deg: float = 1.0) -> dict:
    """The cache (see the module doc) from COCO-Search18 validation ``cs18`` and COCO-FreeView validation ``fv``,
    each ``(stimuli, fixations)`` as ``data.load_dataset`` returns them; ``fit_*`` the centre-bias fits
    (``evaluate.centre_bias_fit``) the full evaluation uses; ``n_images`` per condition, drawn with ``seed`` among
    the COCO-Search18 images that FreeView also shows. ``ppd``: pixels per degree of the displays (31.1 for both)."""
    from .data import ScanpathDataset, letterbox_geometry
    from .evaluate import centre_bias_model, select
    st_s, fx_s = cs18
    st_f, fx_f = fv
    stem = lambda p: Path(p).stem  # noqa: E731
    fv_index = {stem(p): i for i, p in enumerate(st_f.filenames)}
    rng = np.random.default_rng(seed)
    chosen: dict[str, np.ndarray] = {}
    taken: set[str] = set()  # a picture can be several COCO-Search18 stimuli (one per cue): draw each picture once
    for cond in ("tp", "ta"):
        n_of = np.unique(np.asarray(select(fx_s, condition=cond, limit=None).scanpaths.n))
        n_of = np.array([n for n in n_of if stem(st_s.filenames[n]) in fv_index and stem(st_s.filenames[n]) not in
                         taken])
        chosen[cond] = np.sort(rng.choice(n_of, min(n_images, len(n_of)), replace=False))
        taken |= {stem(st_s.filenames[n]) for n in chosen[cond]}
    images = np.concatenate([chosen["tp"], chosen["ta"]])  # cache image i = COCO-Search18 stimulus images[i]
    if len({stem(st_s.filenames[n]) for n in images}) != len(images):
        raise ValueError("a picture was drawn twice (two cues): its FreeView scanpaths would count twice")
    mean, std = (torch.tensor(v)[:, None, None] for v in (IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    out: dict[str, Any] = {"version": VERSION, "scene": scene, "frame": frame, "fine": FINE, "ppd": ppd,
                           "kde_deg": kde_deg, "seed": seed, "sets": {},
                           "stems": [stem(st_s.filenames[n]) for n in images],
                           "cs18_stimulus": [int(n) for n in images]}
    n_img = len(images)
    out["frac"] = torch.zeros(n_img, FINE * FINE)
    pix = torch.zeros(n_img, 3, scene, scene, dtype=torch.uint8)
    deg = torch.zeros(n_img)
    for cond, (st, fx, fit) in {"tp": (st_s, select(fx_s, condition="tp", limit=None), fit_cs18),
                                "ta": (st_s, select(fx_s, condition="ta", limit=None), fit_cs18),
                                "fv": (st_f, fx_f, fit_fv)}.items():
        cb_model = centre_bias_model(st, fx, fit)
        ds = ScanpathDataset(st, fx, scene_size=scene, frame=frame)
        sp_n = np.asarray(fx.scanpaths.n)
        rows: dict[str, list] = {"image": [], "cue": [], "fix": [], "length": [], "cell": []}
        cb, human = torch.zeros(n_img, FINE * FINE), torch.zeros(n_img, FINE * FINE)
        for i, n_s in enumerate(images):
            if cond != "fv" and n_s not in chosen[cond]:
                continue
            n = int(n_s) if cond != "fv" else fv_index[stem(st_s.filenames[n_s])]
            stim = st.stimuli[n]
            frac, cbm, pcell = _fine_maps(stim, cb_model.log_density(stim), scene=scene, frame=frame)
            out["frac"][i], cb[i] = torch.from_numpy(frac).float(), torch.from_numpy(cbm).float()
            h, w = stim.shape[:2]
            ch, cw, _, _ = letterbox_geometry(h, w, scene, frame)
            deg[i] = (w / cw) * (scene / 2) / ppd  # degrees per scene unit
            points = []
            for k in np.flatnonzero(sp_n == n):
                item = ds[int(k)]
                if not pix[i].any():  # the first scanpath of the image (the sets show the same pixels)
                    pix[i] = (item["image"] * std + mean).mul(255).round().clamp(0, 255).to(torch.uint8)
                rows["image"].append(i)
                rows["cue"].append(int(item["cue"]))
                rows["fix"].append(item["fix"])
                rows["length"].append(int(item["n_fix"]))
                L_k = int(item["n_fix"])  # each fixation's fine cell from its PIXEL, as pysaliency / models score it
                xs = np.clip(np.asarray(fx.scanpaths.xs[k][:L_k], float).astype(int), 0, w - 1)
                ys = np.clip(np.asarray(fx.scanpaths.ys[k][:L_k], float).astype(int), 0, h - 1)
                rows["cell"].append(torch.from_numpy(pcell[ys, xs]).long())
                points.append(item["fix"][1:].numpy())
            human[i] = torch.from_numpy(_kde(np.concatenate(points), frac, kde_deg / float(deg[i]))).float()
        L = max(rows["length"])
        fix = torch.full((len(rows["fix"]), L, 2), float("nan"))
        cells = torch.full((len(rows["fix"]), L), -1, dtype=torch.long)
        for j, (f, c) in enumerate(zip(rows["fix"], rows["cell"])):
            fix[j, :len(f)], cells[j, :len(c)] = f, c
        out["sets"][cond] = {"image": torch.tensor(rows["image"]), "cue": torch.tensor(rows["cue"]), "fix": fix,
                             "cell": cells,
                             "length": torch.tensor(rows["length"]), "cb": cb, "human": human,
                             "fit": dict(fit)}
        log.info("human subset %s: %d scanpaths on %d images", cond, len(rows["image"]),
                 len(set(rows["image"])))
    out["images"], out["deg_per_unit"] = pix, deg
    hw = [st_s.sizes[n] for n in images]
    out["hw"] = torch.tensor([[int(h), int(w)] for h, w in hw])
    out["content"] = torch.stack([_content(int(h), int(w), scene, frame) for h, w in hw])
    return out


def _content(h: int, w: int, scene: int, frame) -> Tensor:
    from .data import pixel_to_scene
    return pixel_to_scene(torch.tensor([[0.0, 0.0], [float(w), float(h)]]), hw=(h, w), scene=scene,
                          frame=frame).reshape(4)


# --- training time: the evaluation -------------------------------------------------------------------------------

def _pool(maps: Tensor, g: int) -> Tensor:
    """[I, FINE*FINE] fine maps summed into [I, g*g] cells (row-major, cell = row * g + col)."""
    if FINE % g:
        raise ValueError(f"the policy grid {g} does not divide the cached grid {FINE}")
    k = FINE // g
    return maps.reshape(-1, g, k, g, k).sum((2, 4)).reshape(-1, g * g)


def _coarse(fine_cell: Tensor, g: int) -> Tensor:
    """Fine cell index (row * FINE + col) -> the policy grid's cell (row * g + col) that contains it."""
    k = FINE // g
    return (fine_cell // FINE // k) * g + (fine_cell % FINE) // k


def _image_average(v: Tensor, image: Tensor) -> float:
    """Mean over images of the per-image means (pysaliency ``average='image'``)."""
    ids, inv = torch.unique(image, return_inverse=True)
    z = torch.zeros(len(ids), dtype=torch.float64, device=v.device)
    s = z.clone().index_add_(0, inv, v.double())
    c = z.clone().index_add_(0, inv, torch.ones_like(v, dtype=torch.float64))
    return float((s / c).mean())


@dataclass
class HumanEval:
    """The cached human subset, ready to score a live policy. ``max_fixations``: glimpses per teacher-forced episode
    (fixations 1..max_fixations of each human scanpath are scored); ``sample_fixations``: length of the policy's own
    scanpaths (the anchor + n - 1 chosen; default the search policy's trained horizon of 4 glimpses)."""

    data: dict
    device: torch.device
    max_fixations: int = 6
    sample_fixations: int = 4
    batch_size: int = 64
    seed: int = 0
    _maps: dict = field(default_factory=dict, repr=False)

    @classmethod
    def load(cls, path: Path, *, task: Any, device: torch.device, **kw) -> HumanEval:
        """The cache at ``path``, refused unless it was built for the task's scene size and display frame."""
        from .data import parse_display_frame
        data = torch.load(path, map_location="cpu", weights_only=False)
        if data.get("version") != VERSION:
            raise ValueError(f"{path}: cache version {data.get('version')}, this code reads {VERSION}; rebuild it")
        c = task.cfg
        scene = getattr(c, "scene_size", None) or getattr(c, "scene_resolution", None)
        frame = getattr(task, "display_frame", None)
        frame = parse_display_frame(frame) if isinstance(frame, str) else frame
        if scene is not None and scene != data["scene"]:
            raise ValueError(f"{path} was built for a {data['scene']} px scene; the {task.name} run's is {scene}")
        if frame != (tuple(data["frame"]) if data["frame"] is not None else None):
            raise ValueError(f"{path} was built for the display frame {data['frame']}; the {task.name} run trains "
                             f"on {frame}: rebuild it with that frame")
        return cls(data=data, device=device, **kw)

    def _grid_maps(self, g: int) -> dict:
        if g not in self._maps:
            d = self.data
            self._maps[g] = {"frac": _pool(d["frac"], g)} | {
                s: {"cb": _pool(d["sets"][s]["cb"], g), "human": _pool(d["sets"][s]["human"], g)} for s in SETS}
        return self._maps[g]

    def _batch(self, s: str, rows: Tensor) -> dict:
        d, S = self.data, self.data["sets"][s]
        img = S["image"][rows]
        mean, std = (torch.tensor(v)[:, None, None] for v in (IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
        image = (d["images"][img].float() / 255 - mean) / std
        dev = self.device
        return {"image": image.to(dev), "cue": S["cue"][rows].to(dev), "fix": S["fix"][rows].to(dev),
                "content": d["content"][img].to(dev), "hw": d["hw"][img], "img": img}

    @torch.no_grad()
    def __call__(self, *, task: Any, model: Any, joint: Any) -> dict[str, float]:
        """The ``val/human/*`` numbers (keys without the prefix) of the live ``task``/``model``/``joint``."""
        from .canvit_models import PolicyEpisode
        from .data import clamp_to_content
        if joint is None:
            raise ValueError("human alignment needs the learned policy (joint is None)")
        restore = _snapshot(model, joint, self.device)
        try:
            model.eval()
            joint.scorer.eval()
            ep = PolicyEpisode(task=task, model=model, joint=joint, device=self.device,
                               amp=getattr(task.cfg, "amp", True))
            gen = torch.Generator(device=self.device).manual_seed(self.seed)
            out = {}
            for s in SETS:
                out |= self._score_set(ep, s, gen, clamp_to_content)
            return out
        finally:
            restore()

    def _score_set(self, ep, s: str, gen: torch.Generator, clamp) -> dict[str, float]:
        S, deg_of = self.data["sets"][s], self.data["deg_per_unit"]
        K = 0
        if ep.is_search and s == "fv":
            from canvit.search.data import COCO_SEARCH18_CLASSES
            K = len(COCO_SEARCH18_CLASSES)
        n = min(self.max_fixations, S["fix"].shape[1] - 1)
        ig, ll, H, img_of, amp, amp_h = [], [], [], [], [], []
        maps = None
        # the cue mixture runs the scorer (and _episode the search head) for all K cues: a quarter of the batch
        for rows in torch.arange(len(S["image"])).split(max(1, self.batch_size // 4) if K else self.batch_size):
            b = self._batch(s, rows)
            human = b["fix"].nan_to_num(0.0)
            logits = ep._episode(b, n, lambda t, sc: clamp(human[:, t], b["content"]), all_cues=K)[0]["policy"]
            g = int(round(logits.shape[-1] ** 0.5))
            if maps is None:
                maps = self._grid_maps(g)
            frac = maps["frac"][b["img"]].to(self.device)
            logp = self._logp(logits, frac.log(), K)  # [B, n, G*G]
            cells = _coarse(S["cell"][rows][:, 1:n + 1].clamp_min(0), g).to(self.device)  # fixation t = 1..n
            lp = logp.gather(-1, cells[..., None])[..., 0]
            cb = maps[s]["cb"][b["img"]].to(self.device)
            valid = torch.arange(1, n + 1, device=self.device)[None] < S["length"][rows].to(self.device)[:, None]
            ig.append(((lp - cb.log().gather(-1, cells)) / math.log(2))[valid])
            ll.append(((lp - frac.log().gather(-1, cells)) / math.log(2))[valid])
            p = logp.exp()
            H.append(-(p * logp.nan_to_num(0.0, neginf=0.0)).sum(-1)[valid])
            img_of.append(b["img"].to(self.device)[:, None].expand_as(valid)[valid])
            # the policy's own scanpaths, from the same density at temperature 1
            m = self.sample_fixations

            def draw(t, sc):
                lq = self._logp(sc[:, None], frac.log(), K)[:, 0]
                idx = torch.multinomial(lq.exp(), 1, generator=gen).squeeze(1)
                c = (torch.arange(g, device=self.device, dtype=torch.float32) + 0.5) * (2.0 / g) - 1.0
                return clamp(torch.stack([c[idx % g], c[idx // g]], -1), b["content"])
            own = ep._episode(b, m, draw, all_cues=K)[1]
            d = deg_of[b["img"]].to(self.device)[:, None]
            amp.append(((own[:, 1:] - own[:, :-1]).norm(dim=-1) * d).flatten())
            hf = b["fix"][:, :m]
            ok = torch.arange(1, m, device=self.device)[None] < S["length"][rows].to(self.device)[:, None]
            amp_h.append(((hf[:, 1:] - hf[:, :-1]).norm(dim=-1) * d)[ok])
        ig, ll, H, img_of = (torch.cat(v) for v in (ig, ll, H, img_of))
        hum = maps[s]["human"]
        used = torch.unique(S["image"])
        Hh = -(hum[used] * hum[used].clamp_min(1e-30).log()).sum(-1)
        return {f"ig_{s}": _image_average(ig, img_of), f"ll_{s}": _image_average(ll, img_of),
                f"entropy_{s}": float(H.mean()), f"entropy_human_{s}": float(Hh.mean()),
                f"sacc_len_{s}": float(torch.cat(amp).mean()), f"sacc_len_human_{s}": float(torch.cat(amp_h).mean())}

    @staticmethod
    def _logp(scores: Tensor, logfrac: Tensor, K: int) -> Tensor:
        """log p(cell) = log_softmax(scores + log(pixels in cell)) [B, n, G*G]; ``K`` cues: scores [B, n, K, G*G],
        the equal-weight mixture of the per-cue p (cells off the image: -inf)."""
        if K:
            return torch.logsumexp(torch.log_softmax(scores.float() + logfrac[:, None, None], -1), 2) - math.log(K)
        return torch.log_softmax(scores.float() + logfrac[:, None], -1)


def _snapshot(model: Any, joint: Any, device: torch.device):
    """A function that restores the RNG streams and train/eval modes the evaluation may touch."""
    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
    gens = {}  # every generator the policy's selectors hold (the FULL anchor goes through the random fallback)
    for sel in (joint.policy_selector, getattr(joint.policy_selector, "fallback", None),
                getattr(joint, "random_selector", None)):
        g = getattr(sel, "generator", None)
        if isinstance(g, torch.Generator):
            gens[id(g)] = (g, g.get_state())
    modes = [(m, m.training) for m in (*model.modules(), *joint.scorer.modules())]

    def restore():
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)
        for g, st in gens.values():
            g.set_state(st)
        for m, t in modes:
            m.training = t
    return restore


# --- CLI: build the cache ----------------------------------------------------------------------------------------

@dataclass
class BuildConfig:
    out: Path
    """Where to write the cache (.pt)."""
    scene_size: int = 512
    """The training scene (search: ``SearchConfig.scene_size``)."""
    display_frame: str = "1680x1050"
    """The run task's ``display_frame`` (search: 1680x1050; distill/ade20k/in1k: scene)."""
    n_images: int = 12
    """Images per COCO-Search18 condition (each with ~10 COCO-Search18 and ~10 FreeView scanpaths)."""
    seed: int = 0
    data_location: Path | None = None
    """pysaliency's dataset location; default $SCANPATH_DATA_DIR."""
    fit_dir: Path | None = None
    """The centre-bias fits the full evaluation uses (centre_bias_fit_<dataset>.json); default
    $LOGS_DIR/scanpath_eval/baselines."""


def main(argv: list[str] | None = None) -> None:
    import json
    import os

    import tyro

    from .data import load_dataset, parse_display_frame
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    c = tyro.cli(BuildConfig, args=argv)
    loc = c.data_location or Path(os.environ["SCANPATH_DATA_DIR"])
    fits = c.fit_dir or Path(os.environ["LOGS_DIR"]) / "scanpath_eval" / "baselines"
    fit = {d: json.loads((fits / f"centre_bias_fit_{d}.json").read_text()) for d in ("coco_search18", "coco_freeview")}
    cache = build(load_dataset("coco_search18", "validation", location=loc),
                  load_dataset("coco_freeview", "validation", location=loc), fit_cs18=fit["coco_search18"],
                  fit_fv=fit["coco_freeview"], scene=c.scene_size, frame=parse_display_frame(c.display_frame),
                  n_images=c.n_images, seed=c.seed)
    c.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, c.out)
    log.info("wrote %s", c.out)


if __name__ == "__main__":
    main()

__all__ = ["BuildConfig", "HumanEval", "build", "main"]
