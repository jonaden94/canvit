"""VENDORED from get_datasets' reference sampler (``functionality/search_index/sampler.py``) as shipped in the search
releases: ``basic-v1/sampler.py`` (sha256 7054868f2b3f33acaa6e6406fc1cccbcccb75013caec72debe11a8536977eda6,
release commit d0cf175) and, with the visual-prompt extension, the 2026-11-25 partial visual-prompt release's
``sampler.py`` (sha256 44dc9e5b89e1f0070e70017766425b4eed1ac93328a1f143f1db7d373a9d01c3, uncommitted in
get_datasets), which is basic-v1's plus the ``vp_*`` parts. The releases' spec (README section 4) says: port it as
it is. Changes, all marked ``# canvit:``:
- each epoch row also carries ``image_idx`` (the row of ``images.parquet``, as the releases'
  ``reference/epoch_*.parquet`` do): present rows from ``examples.image_idx``, absent rows from
  ``neg_explicit.image_idx`` / ``implicit_images.image_idx``;
- ``visual_prompts`` takes that carried ``image_idx`` instead of looking (source, image) up in a MultiIndex of
  ``images.parquet`` (the same numbers; saves reading 3.2M string pairs).
Nothing else differs, so every random call is the reference's. ``canvit/search/tests/test_release.py`` checks the
reference epochs row for row (``vp_idx`` included on a visual-prompt release). Do not edit the logic here: a new
release brings its own sampler, to be vendored again.

Epoch sampler of the search training set (user decisions 2026-11-23; repo docs/search_dataset.md section 8).
Reads the index of build_index.py; only numpy / pandas, so the canvit loader can use it as it is.

Present trials:
- class: fixed cap K. A class with n examples gets min(n, K) draws per epoch (classes = nodes of the cut; finer
  prompts count under their class).
- position: inside a class, the 9 cells of the box centre as even as possible with at most r draws of one example
  per epoch (water-filling: a cell gets at most r x its examples, the rest goes evenly to the other cells).
- no size weight.
- an example's expected draws w are rounded at random to floor(w) or floor(w) + 1 each epoch (so never more than
  r), then all trials are shuffled.
- prompt: uniform among the example's prompts (its class and qualifying finer prompts); text: original or
  normalised label, 50 / 50.
Absent trials (share `absent` of all trials): prompt and text copied from a random present trial of the epoch whose
prompt has negatives; the image is drawn uniformly from all negatives of that prompt (explicit negatives at the
prompt or above it, implicit negatives of complete sources: no object at, below or above the prompt), never a val
absent trial.

Text options (all off by default; user 2026-11-24, for ablations): synonym_p = probability that a trial's text is
replaced by a random synonym of its prompt (the normalised labels of all sources on the node); definition_p =
probability that the prompt's definition is appended ("<text>, <definition>", as in DetCLIP), never one that names
a held-out category unless definition_heldout=True; template = format string around the final text, e.g.
"a photo of a {}". Absent trials copy the final text of their donor.

    s = SearchSampler(index_dir)            # K=300, r=2, absent=0.5
    trials = s.epoch(e, seed=0)             # DataFrame, one row per trial, deterministic in (seed, e)

Visual prompts (releases with vp_* tables, e.g. basic-v2): every trial also gets `vp_idx`, a row of
vp_boxes / vp_features: uniform among the prompt's candidates (vp_candidates) that are not in the target picture's
image group and not a near-duplicate of the target picture (vp_neardup); -1 if none is left. Drawn with a separate
random stream after everything else, so the other columns equal those of a release without visual prompts.
"""
# ruff: noqa: E702  (the reference's own one-line statements, kept verbatim)
from pathlib import Path

import numpy as np
import pandas as pd

BOX = ['box_x', 'box_y', 'box_w', 'box_h']


def fill(count, draws, r):
    """Water-filling over the cells of one class: cell i gets at most r * count[i]; as even as possible."""
    cap = r * count.astype(float)
    a = np.zeros(len(count))
    free = cap > 0
    left = float(draws)
    while left > 1e-9 and free.any():
        add = np.minimum(left / free.sum(), cap - a) * free
        a += add
        left -= add.sum()
        free &= (cap - a) > 1e-9
    return a


class SearchSampler:
    def __init__(self, index_dir, K=300, r=2, absent=0.5, synonym_p=0.0, definition_p=0.0, template=None,
                 definition_heldout=False):
        self.K, self.r, self.absent = K, r, absent
        self.synonym_p, self.definition_p, self.template = synonym_p, definition_p, template
        rd = lambda n, **kw: pd.read_parquet(f'{index_dir}/{n}.parquet', **kw)  # noqa: E731
        self.ex = rd('examples')
        ep = rd('example_prompts')
        assert (self.ex['example_id'].to_numpy() == np.arange(len(self.ex))).all()
        assert ep['example_id'].is_monotonic_increasing
        self.ep = ep.reset_index(drop=True)
        self.ep_off = np.searchsorted(ep['example_id'].to_numpy(), np.arange(len(self.ex)))
        self.n_prompts = self.ex['n_prompts'].to_numpy()
        self.w = self._expected_draws()
        # negatives
        pr = rd('prompts').set_index('prompt')
        self.srcs = [c[len('n_implicit_neg_'):] for c in pr.columns if c.startswith('n_implicit_neg_')]
        self.n_exp = pr['n_explicit_neg'].to_dict()
        self.n_imp = {x: v for x, v in zip(pr.index, pr[[f'n_implicit_neg_{s}' for s in self.srcs]].to_numpy())}
        neg = rd('neg_explicit')
        self.neg_src, self.neg_img = neg['source'].to_numpy(), neg['image'].to_numpy()
        self.neg_idx = neg['image_idx'].to_numpy()  # canvit: image_idx carried
        self.neg_by_node = {m: g.to_numpy() for m, g in pd.Series(np.arange(len(neg))).groupby(neg['node'].to_numpy())}
        imp = rd('implicit_images')
        self.imp_source, self.imp_image = imp['source'].to_numpy(), imp['image'].to_numpy()
        self.imp_idx = imp['image_idx'].to_numpy()  # canvit: image_idx carried
        src = np.array([x.split('/')[0] for x in self.imp_source])
        self.imp_by_src = {s: np.flatnonzero(src == s) for s in self.srcs}
        pres = rd('implicit_present')
        self.pres_off = np.searchsorted(pres['img'].to_numpy(), np.arange(len(imp) + 1))
        self.pres_code = pres['node_code'].to_numpy()
        self.imgs_by_code = {c: g.to_numpy() for c, g in pres['img'].groupby(pres['node_code'])}
        nodes = rd('nodes')
        self.code = dict(zip(nodes['node'], nodes['node_code']))
        self.node_of = nodes['node'].to_numpy()
        a = rd('ancestors')
        self.anc = a.groupby('node_code')['anc_code'].apply(list).to_dict()
        self.desc = a.groupby('anc_code')['node_code'].apply(list).to_dict()
        v = rd('val')
        v = v[~v['present']]
        self.val_absent = {x: set(zip(g['source'], g['image'])) for x, g in v.groupby('prompt')}
        self._block, self._exp, self._rare = {}, {}, {}
        pt = rd('prompt_texts')
        self.synonyms = {x: list(v) for x, v in zip(pt['prompt'], pt['synonyms']) if len(v)}
        ok = pt['definition'].ne('') & (definition_heldout | ~pt['definition_mentions_heldout'])
        self.definition = dict(zip(pt.loc[ok, 'prompt'], pt.loc[ok, 'definition']))
        self.vp = Path(f'{index_dir}/vp_candidates.parquet').exists()
        if self.vp:
            vc = rd('vp_candidates')
            self.vp_by_prompt = {x: g.to_numpy() for x, g in vc.groupby('prompt')['vp_idx']}
            self.vp_group = rd('vp_boxes', columns=['image_group'])['image_group'].to_numpy()
            nd = rd('vp_neardup')
            self.vp_block = set(zip(nd['target_image_idx'].tolist(), nd['prompt_image_group'].tolist()))
            self.im_group = rd('images', columns=['image_group'])['image_group'].to_numpy()  # canvit: no im_key

    def visual_prompts(self, prompts, image_idx, rng):
        """vp_idx per trial: uniform among the prompt's candidates outside the target's image group and not a
        near-duplicate of the target picture (rejection sampling, then the full list); -1 if none."""
        out = np.full(len(prompts), -1, dtype='int64')
        for t, (x, i) in enumerate(zip(prompts, image_idx)):
            lst = self.vp_by_prompt.get(x)
            if lst is None:
                continue
            g = self.im_group[i]
            for c in lst[rng.integers(len(lst), size=8)]:
                if self.vp_group[c] != g and (i, self.vp_group[c]) not in self.vp_block:
                    out[t] = c
                    break
            else:
                ok = [c for c in lst if self.vp_group[c] != g and (i, self.vp_group[c]) not in self.vp_block]
                if ok:
                    out[t] = ok[rng.integers(len(ok))]
        return out

    def _expected_draws(self):
        """w per example: class cap K, then water-filling over the 9 cells with at most r draws per example."""
        ex = self.ex
        cls, ci = np.unique(ex['class'].to_numpy(), return_inverse=True)
        cell = ex['cell'].to_numpy().astype(int)
        cnt = np.zeros((len(cls), 9))
        np.add.at(cnt, (ci, cell), 1)
        n = cnt.sum(1)
        alloc = np.array([fill(c, min(k, self.K), self.r) for c, k in zip(cnt, n)])
        per = np.divide(alloc, cnt, out=np.zeros_like(alloc), where=cnt > 0)
        return per[ci, cell]

    # --- negatives -------------------------------------------------------------------------------------------
    def has_negatives(self, x):
        return self.n_exp.get(x, 0) + self.n_imp[x].sum() > 0 if x in self.n_imp else False

    def _blocking(self, x):
        """codes of the nodes that make an implicit image no negative of x: x, below x, above x"""
        if x not in self._block:
            c = self.code[x]
            self._block[x] = np.array([c, *self.desc.get(c, []), *self.anc.get(c, [])])
        return self._block[x]

    def _ok(self, i, x):
        p = self.pres_code[self.pres_off[i]:self.pres_off[i + 1]]
        return not np.isin(p, self._blocking(x)).any() and \
            (self.imp_source[i], self.imp_image[i]) not in self.val_absent.get(x, ())

    def _explicit(self, x):
        """explicit negatives of x: at x or above it, one per picture, without val absent trials"""
        if x not in self._exp:
            nodes = [x, *self.node_of[self.anc.get(self.code[x], [])]]
            idx = np.concatenate([self.neg_by_node.get(m, np.empty(0, 'int64')) for m in nodes])
            keys = pd.Series(list(zip(self.neg_src[idx], self.neg_img[idx])))
            va = self.val_absent.get(x, set())
            keep = ~keys.duplicated().to_numpy() & np.array([k not in va for k in keys], bool)
            self._exp[x] = idx[keep]
        return self._exp[x]

    def _implicit(self, x, s, m, rng):
        """m implicit negatives of x from source s: rejection sampling, or from the full list when they are rare"""
        pool = self.imp_by_src[s]
        if self.n_imp[x][self.srcs.index(s)] < 0.05 * len(pool):
            if (x, s) not in self._rare:
                bad = np.zeros(len(self.imp_source), bool)
                for c in self._blocking(x):
                    if c in self.imgs_by_code:
                        bad[self.imgs_by_code[c]] = True
                lst = pool[~bad[pool]]
                va = self.val_absent.get(x, set())
                self._rare[(x, s)] = np.array([i for i in lst if (self.imp_source[i], self.imp_image[i]) not in va],
                                              dtype='int64')
            lst = self._rare[(x, s)]
            return lst[rng.integers(len(lst), size=m)] if len(lst) else np.empty(0, 'int64')
        out = []
        while len(out) < m:
            for i in pool[rng.integers(len(pool), size=2 * (m - len(out)) + 8)]:
                if self._ok(i, x):
                    out.append(i)
                    if len(out) == m:
                        break
        return np.array(out, dtype='int64')

    def negatives(self, x, m, rng):
        """m random negatives (source, image, image_idx) of prompt x, uniform over all its negatives"""
        cnt = np.r_[self.n_exp.get(x, 0), self.n_imp[x]].astype(float)
        ks = rng.multinomial(m, cnt / cnt.sum())
        src, img, idx = [], [], []  # canvit: image_idx carried
        if ks[0] and len(self._explicit(x)):  # (empty only if all were val absent trials)
            lst = self._explicit(x)
            j = lst[rng.integers(len(lst), size=ks[0])]
            src.append(self.neg_src[j]); img.append(self.neg_img[j]); idx.append(self.neg_idx[j])
        for s, k in zip(self.srcs, ks[1:]):
            if k:
                i = self._implicit(x, s, k, rng)
                src.append(self.imp_source[i]); img.append(self.imp_image[i]); idx.append(self.imp_idx[i])
        return np.concatenate(src), np.concatenate(img), np.concatenate(idx)

    def _texts(self, prompts, text, rng):
        """apply the text options to the drawn texts"""
        if not (self.synonym_p or self.definition_p or self.template):
            return text
        text = text.copy()
        u_syn, u_def = rng.random(len(text)), rng.random(len(text))
        for i, x in enumerate(prompts):
            if u_syn[i] < self.synonym_p and x in self.synonyms:
                v = self.synonyms[x]
                text[i] = v[rng.integers(len(v))]
            if u_def[i] < self.definition_p and x in self.definition:
                text[i] = f'{text[i]}, {self.definition[x]}'
            if self.template:
                text[i] = self.template.format(text[i])
        return text

    # --- epoch ------------------------------------------------------------------------------------------------
    def epoch(self, e, seed=0):
        rng = np.random.default_rng([seed, e])
        w = self.w
        k = np.floor(w).astype(int) + (rng.random(len(w)) < w - np.floor(w))
        eid = np.repeat(np.arange(len(w)), k)
        j = self.ep_off[eid] + (rng.random(len(eid)) * self.n_prompts[eid]).astype(int)
        ep = self.ep.iloc[j]
        text = np.where(rng.random(len(eid)) < 0.5, ep['text_orig'].to_numpy(), ep['text_norm'].to_numpy())
        text = self._texts(ep['prompt'].to_numpy(), text.astype(object), rng)
        ex = self.ex.iloc[eid]
        pres = pd.DataFrame({'present': True, 'example_id': eid, 'source': ex['source'].to_numpy(),
                             'image': ex['image'].to_numpy(), 'image_idx': ex['image_idx'].to_numpy(),  # canvit
                             'prompt': ep['prompt'].to_numpy(), 'text': text})
        for c in BOX:
            pres[c] = ex[c].to_numpy()
        n_abs = int(round(len(pres) * self.absent / (1 - self.absent)))
        ok = np.flatnonzero(pres['prompt'].map(self.has_negatives).to_numpy())
        pick = np.sort(rng.choice(ok, size=n_abs, replace=n_abs > len(ok)))
        donor = pres.iloc[pick][['prompt', 'text']].reset_index(drop=True)
        parts = []
        for x, g in donor.groupby('prompt', sort=True):
            s, im, ii = self.negatives(x, len(g), rng)  # canvit: image_idx carried
            parts.append(pd.DataFrame({'source': s, 'image': im, 'image_idx': ii, 'prompt': x,
                                       'text': g['text'].to_numpy()[:len(s)]}))
        ab = pd.concat(parts, ignore_index=True).assign(present=False, example_id=-1)
        out = pd.concat([pres, ab], ignore_index=True)
        out = out.iloc[rng.permutation(len(out))].reset_index(drop=True)
        if self.vp:
            idx = out['image_idx'].to_numpy()  # canvit: the carried image_idx (the reference looks it up)
            out['vp_idx'] = self.visual_prompts(out['prompt'].to_numpy(), idx, np.random.default_rng([seed, e, 1]))
        return out
