"""Design-matrix construction + scoring shared by train.py and submission.py (numpy only)."""
import numpy as np

import features as F
from common import parse_answer


def blocks(bank, E):
    return {'hand': F.hand_features(bank), 'cross': F.cross_emb_features(bank, E), 'E': E.astype(np.float32)}


def fit_prep(B, npca, seed=0):
    """Fit embedding standardiser + PCA on train candidates (B from blocks)."""
    E = B['E'].reshape(-1, B['E'].shape[-1]).astype(np.float64)
    mu, sd = E.mean(0), E.std(0) + 1e-6
    Z = (E - mu) / sd
    # PCA via covariance eigendecomposition (deterministic)
    C = np.cov(Z, rowvar=False)
    w, V = np.linalg.eigh(C)
    comp = V[:, ::-1][:, :npca].T.astype(np.float32)
    return {'mu': mu.astype(np.float32), 'sd': sd.astype(np.float32), 'comp': comp}


def design(B, cfg, prep):
    """Return (N,K,D) raw (unscaled) design tensor."""
    parts = []
    if 'hand' in cfg['blocks']:
        h = B['hand']
        parts.append(h)
        if cfg.get('rel'):
            parts.append(F.group_demean(h))
    if 'cross' in cfg['blocks']:
        parts.append(B['cross'])
    if 'emb' in cfg['blocks']:
        Z = ((B['E'] - prep['mu']) / prep['sd']) @ prep['comp'].T
        parts.append(Z)
        if cfg.get('rel'):
            parts.append(F.group_demean(Z))
    return np.concatenate(parts, -1).astype(np.float32)


def aggregate_choose(bank, p, agg=None):
    """p (N,K) candidate probabilities -> chosen index per group."""
    out = []
    for g, row in enumerate(bank):
        pg = p[g]
        if agg is None:
            out.append(int(np.argmax(pg)))
            continue
        ans = [parse_answer(c['text']) for c in row['candidates']]
        best, bi = -1.0, 0
        for a in set(x for x in ans if x is not None):
            mem = [j for j in range(len(ans)) if ans[j] == a]
            s = {'sum': pg[mem].sum(), 'max': pg[mem].max(), 'mean': pg[mem].mean()}[agg]
            if s > best:
                best, bi = s, max(mem, key=lambda j: pg[j])
        for j, a in enumerate(ans):
            if a is None and pg[j] > best:
                best, bi = pg[j], j
        out.append(int(bi))
    return out


def linear_proba(X, ck):
    Z = (X - ck['x_mean']) / ck['x_std']
    return 1 / (1 + np.exp(-(Z @ ck['coef'] + ck['intercept'])))
