"""5-fold CV over TRAIN groups (fold by group, seed 0). Score = accuracy on hard groups of the held-out fold.
Early stopping only inside training folds (train_one's internal holdout is restricted to the fold's training pool).
Run from code/:  .venv/bin/python method/cv.py [seeds]   -> writes method/cv_results.json
Per-seed held-out probabilities are cached in $CV_CACHE so ensembles/K-curves are free."""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import nn_train as T  # noqa: E402
import selector as S  # noqa: E402
import features as F  # noqa: E402
import features_v2 as V  # noqa: E402
from train import fit_lr, acc  # noqa: E402
from common import load_inputs  # noqa: E402

CACHE = Path(os.environ.get('CV_CACHE', '/tmp/claude-0/-root-DASE7506-MP2/24ec1cfd-5c48-4fdf-8a0d-54e05b389870/scratchpad/cv'))
CACHE.mkdir(parents=True, exist_ok=True)
NF = 5


def folds(n):
    return np.random.RandomState(0).randint(0, NF, n)


def score(tb, Y, hard, fold, P, agg='sum'):
    """per-fold hard acc of probs P (N,4) (valid on held-out entries) -> list of NF accs"""
    out = []
    for f in range(NF):
        te = np.where((fold == f) & hard)[0]
        ch = np.array(S.aggregate_choose([tb[i] for i in te], P[te], agg))
        out.append(acc(ch, Y[te]))
    return out


def nn_probs(name, cfg, d, fold, seed):
    fn = CACHE / f'{name}_s{seed}.npy'
    if fn.exists():
        return np.load(fn)
    P = np.zeros(d['Yt'].shape, np.float32)
    for f in range(NF):
        te = fold == f
        dd = dict(d, xv=d['xt'][te], ev=d['et'][te], mv=d['mt'][te])
        net, norm, p, ep = T.train_one(cfg, dd, seed * 100 + f, pool=~te)
        P[te] = p
    np.save(fn, P)
    return P


def lr_probs(tb, d, fold, v2):
    fn = CACHE / f'lr_v2{int(v2)}.npy'
    if fn.exists():
        return np.load(fn)
    B = S.blocks(tb, np.load(HERE.parent / 'data/candidates/gsm8k_train.npz')['embeddings'].reshape(len(tb), 4, -1))
    cfg = dict(blocks=['hand', 'cross', 'emb'], npca=64, rel=True)
    P = np.zeros(d['Yt'].shape, np.float32)
    x2 = V.extra_features(tb) if v2 else None
    for f in range(NF):
        te = fold == f
        prep = S.fit_prep({'E': B['E'][~te]}, 64)
        X = S.design(B, cfg, prep)
        if v2:
            X = np.concatenate([X, x2, F.group_demean(x2)], -1)
        sel = ~te & d['ht']
        ck = fit_lr(X[sel], d['Yt'][sel], 10.0)
        P[te] = S.linear_proba(X[te], ck)
    np.save(fn, P)
    return P


def main():
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    tb, vb, d1 = T.load_all(False); _, _, d2 = T.load_all(True)
    fold = folds(len(tb)); Y, hard = d1['Yt'], d1['ht']
    res = {}
    for v2 in (False, True):
        P = lr_probs(tb, d1, fold, v2); a = score(tb, Y, hard, fold, P)
        res[f'LR v1cfg{" +v2feat" if v2 else ""}'] = dict(single=a, ens5=None)
        print(f'LR v2={v2}: {np.mean(a):.2f}±{np.std(a):.2f}', flush=True)
    for inter in ('none', 'deepsets'):
        for v2 in (False, True):
            name = f'{inter}_v2{int(v2)}'
            cfg = dict(T.BASE, inter=inter, emb=True, v2=v2)
            d = d2 if v2 else d1
            t0 = time.time()
            Ps = [nn_probs(name, cfg, d, fold, s) for s in range(seeds)]
            r = {}
            r['per_seed'] = [float(np.mean(score(tb, Y, hard, fold, p))) for p in Ps]
            for K in (1, 3, 5, 10):
                if K <= seeds:
                    a = score(tb, Y, hard, fold, np.mean(Ps[:K], 0))
                    r[f'K{K}'] = dict(mean=float(np.mean(a)), std=float(np.std(a)), folds=a)
            res[name] = r
            print(name, 'seed-mean %.2f±%.2f' % (np.mean(r['per_seed']), np.std(r['per_seed'])),
                  {k: round(v['mean'], 2) for k, v in r.items() if k.startswith('K')}, f'{time.time()-t0:.0f}s', flush=True)
            json.dump(res, open(HERE / 'cv_results.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
