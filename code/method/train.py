"""Train selector variants on train split only; evaluate on validation; save best linear checkpoint.
Run from code/:  .venv/bin/python method/train.py"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import common  # noqa: E402
import features as F  # noqa: E402
import selector as S  # noqa: E402
from common import parse_answer, read_jsonl, DATA, load_inputs  # noqa: E402

SEED = 0
CK = HERE.parent / 'checkpoint_lr'


def labels(bank, split):
    gold = [parse_answer(g['answer']) for g in read_jsonl(DATA / f'gsm8k_{split}.jsonl')]
    return np.array([[a is not None and a == g for a in (parse_answer(c['text']) for c in r['candidates'])]
                     for r, g in zip(bank, gold)])


def acc(choice, Y, mask=None):
    c = Y[np.arange(len(Y)), choice]
    return 100 * (c.mean() if mask is None else c[mask].mean())


def fit_lr(X, Y, C, w=None):
    n, k, d = X.shape
    Xf = X.reshape(-1, d)
    mu, sd = Xf.mean(0), Xf.std(0) + 1e-6
    m = LogisticRegression(C=C, max_iter=2000, random_state=SEED)
    m.fit((Xf - mu) / sd, Y.reshape(-1), sample_weight=None if w is None else np.repeat(w, k))
    return {'x_mean': mu, 'x_std': sd, 'coef': m.coef_[0].astype(np.float32), 'intercept': np.float32(m.intercept_[0])}


def main():
    t0 = time.time()
    tb, ta = load_inputs('train'); vb, va = load_inputs('validation')
    Ytr, Yva = labels(tb, 'train'), labels(vb, 'validation')
    Btr, Bva = S.blocks(tb, ta['embeddings']), S.blocks(vb, va['embeddings'])
    htr, hva = F.hard_mask(tb), F.hard_mask(vb)
    print(f'feat time {time.time()-t0:.0f}s; train hard {htr.sum()}, val hard {hva.sum()}', flush=True)
    prep = S.fit_prep(Btr, 128)
    rng = np.random.RandomState(SEED)
    fold = rng.randint(0, 5, len(tb))

    def run(cfg, C, agg, hard_only=False, cv=True):
        npca = cfg.get('npca', 128)
        p2 = {**prep, 'comp': prep['comp'][:npca]}
        Xtr, Xva = S.design(Btr, cfg, p2), S.design(Bva, cfg, p2)
        sel = htr if hard_only else np.ones(len(tb), bool)
        ck = fit_lr(Xtr[sel], Ytr[sel], C)
        pv = S.linear_proba(Xva, ck)
        ch = S.aggregate_choose(vb, pv, agg)
        cvh = np.nan
        if cv:
            accs = []
            for f in range(5):
                tr, te = (fold != f) & sel, (fold == f) & htr
                c2 = fit_lr(Xtr[tr], Ytr[tr], C)
                cc = S.aggregate_choose([tb[i] for i in np.where(te)[0]], S.linear_proba(Xtr[te], c2), agg)
                accs.append(acc(np.array(cc), Ytr[te]))
            cvh = float(np.mean(accs))
        return ck, p2, acc(np.array(ch), Yva), acc(np.array(ch), Yva, hva), cvh

    rows, store = [], {}
    variants = [
        ('a hand', dict(blocks=['hand']), None),
        ('a2 hand+cross', dict(blocks=['hand', 'cross']), None),
        ('b hand+cross+emb64', dict(blocks=['hand', 'cross', 'emb'], npca=64), None),
        ('b2 hand+cross+emb128', dict(blocks=['hand', 'cross', 'emb'], npca=128), None),
        ('c emb64 only', dict(blocks=['emb'], npca=64), None),
        ('c2 emb128 only', dict(blocks=['emb'], npca=128), None),
        ('d1 hand+cross rel', dict(blocks=['hand', 'cross'], rel=True), None),
        ('d2 hand+cross rel, agg=sum', dict(blocks=['hand', 'cross'], rel=True), 'sum'),
        ('d3 hand+cross rel, agg=max', dict(blocks=['hand', 'cross'], rel=True), 'max'),
        ('d4 hand+cross+emb64 rel, agg=sum', dict(blocks=['hand', 'cross', 'emb'], npca=64, rel=True), 'sum'),
        ('d7 hand+cross+emb128 rel, agg=sum', dict(blocks=['hand', 'cross', 'emb'], npca=128, rel=True), 'sum'),
        ('d8 hand+cross+emb32 rel, agg=sum', dict(blocks=['hand', 'cross', 'emb'], npca=32, rel=True), 'sum'),
        ('d5 hand+cross, agg=sum', dict(blocks=['hand', 'cross']), 'sum'),
        ('d6 hand+cross rel, agg=mean', dict(blocks=['hand', 'cross'], rel=True), 'mean'),
    ]
    for name, cfg, agg in variants:
        best = None
        for C in (0.01, 0.1, 1.0, 10.0):
            r = run(cfg, C, agg)
            if best is None or r[4] > best[1][4]:   # choose C by train-CV hard acc
                best = (C, r)
        C, (ck, p2, va_all, va_hard, cvh) = best
        rows.append((name, 'LR', 'all', C, va_all, va_hard, cvh)); store[name] = (cfg, agg, C, ck, p2, va_hard)
        print(f'{name:36s} C={C:<5} val-all {va_all:.2f} val-hard {va_hard:.2f} cvhard {cvh:.2f}', flush=True)
    # f: hard-only training on the best-so-far configs
    top = sorted(store, key=lambda k: -store[k][5])[:3]
    for name in top:
        cfg, agg, C, *_ = store[name]
        ck, p2, a_, h_, cvh = run(cfg, C, agg, hard_only=True)
        rows.append((name, 'LR', 'hard-only', C, a_, h_, cvh)); store[name + ' [hard-only]'] = (cfg, agg, C, ck, p2, h_)
        print(f'{name+" [hard-only]":36s} C={C:<5} val-all {a_:.2f} val-hard {h_:.2f} cvhard {cvh:.2f}', flush=True)
    # e: HGB
    for name, cfg in [('e hgb hand+cross', dict(blocks=['hand', 'cross'])),
                      ('e2 hgb hand+cross+emb16', dict(blocks=['hand', 'cross', 'emb'], npca=16)),
                      ('e3 hgb hand+cross rel', dict(blocks=['hand', 'cross'], rel=True))]:
        p2 = {**prep, 'comp': prep['comp'][:cfg.get('npca', 128)]}
        Xtr, Xva = S.design(Btr, cfg, p2), S.design(Bva, cfg, p2)
        d = Xtr.shape[-1]
        for agg in (None, 'sum'):
            m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40,
                                               l2_regularization=1.0, random_state=SEED)
            m.fit(Xtr.reshape(-1, d), Ytr.reshape(-1))
            pv = m.predict_proba(Xva.reshape(-1, d))[:, 1].reshape(len(vb), -1)
            ch = np.array(S.aggregate_choose(vb, pv, agg))
            rows.append((name + (f' agg={agg}' if agg else ''), 'HGB', 'all', '-', acc(ch, Yva), acc(ch, Yva, hva), np.nan))
            print(f'{rows[-1][0]:36s} val-all {rows[-1][4]:.2f} val-hard {rows[-1][5]:.2f}', flush=True)
    # baselines for table
    # save best linear checkpoint (by val-hard)
    bname = max(store, key=lambda k: store[k][5])
    cfg, agg, C, ck, p2, vh = store[bname]
    CK.mkdir(exist_ok=True)
    np.savez(CK / 'selector.npz', x_mean=ck['x_mean'].astype(np.float32), x_std=ck['x_std'].astype(np.float32),
             coef=ck['coef'], intercept=np.float32(ck['intercept']), mu=p2['mu'], sd=p2['sd'], comp=p2['comp'])
    (CK / 'config.json').write_text(json.dumps({'cfg': cfg, 'agg': agg, 'C': C, 'name': bname, 'seed': SEED,
                                                'feature_names': F.HAND_NAMES, 'cross_names': F.CROSS_NAMES}, indent=1))
    print('BEST', bname, f'val-hard {vh:.2f}', flush=True)
    # feature importances (standardised |coef|) for best
    names = list(F.HAND_NAMES) + ([f'rel_{n}' for n in F.HAND_NAMES] if cfg.get('rel') else []) + F.CROSS_NAMES
    if 'hand' in cfg['blocks']:
        cf = ck['coef'][:len(names)] if 'emb' not in cfg['blocks'] else ck['coef'][:len(names)]
        idx = np.argsort(-np.abs(cf))[:12]
        print('top coefs:', ', '.join(f'{names[i]}:{cf[i]:+.2f}' for i in idx))
    with open(HERE / 'results_table.json', 'w') as f:
        json.dump([list(map(lambda x: x if not isinstance(x, (np.floating,)) else float(x), r)) for r in rows], f)
    print(f'total {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
