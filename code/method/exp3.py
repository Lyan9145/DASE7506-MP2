"""v3 experiments: DeepSets selector + cross-candidate text features (features_v3.py).
Same protocol as cv.py (5-fold train CV by group, RandomState(0) folds, hard-group score, cluster-sum decode)
plus validation (train on full train split). OOF / val probabilities cached in $CV_CACHE.
Run from code/:  .venv/bin/python method/exp3.py <variant> [seeds]   |   method/exp3.py export K=10 out=checkpoint_v3"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import nn_train as T  # noqa: E402
import cv as C  # noqa: E402
import features as F  # noqa: E402
import features_v3 as V3  # noqa: E402
import nn_rel as R  # noqa: E402

CACHE = Path(os.environ.get('CV_CACHE', '/tmp/claude-0/-root-DASE7506-MP2/2582e1a0-1533-4f79-bb74-33499067787b/scratchpad/cv3'))
CACHE.mkdir(parents=True, exist_ok=True)
FEAT = CACHE / 'v3feat.npz'


def v3_feats(tb, vb):
    if FEAT.exists():
        z = np.load(FEAT)
        return z['t'], z['v']
    t, v = V3.extra_features(tb), V3.extra_features(vb)
    np.savez(FEAT, t=t, v=v)
    return t, v


def rel_feats(tb, vb):
    fn = CACHE / 'relfeat.npz'
    if fn.exists():
        z = np.load(fn)
        return z['t'], z['v']
    t, v = V3.relation_features(tb), V3.relation_features(vb)
    np.savez(fn, t=t, v=v)
    return t, v


def with_v3(d, tb, vb, cols=None, rel=False):
    t, v = v3_feats(tb, vb)
    if cols is not None:
        t, v = t[..., cols], v[..., cols]
    if rel:
        rt, rv = rel_feats(tb, vb)
        t, v = np.concatenate([t, rt], -1), np.concatenate([v, rv], -1)
    d = dict(d)
    d['xt'] = np.concatenate([d['xt'], t, F.group_demean(t)], -1).astype(np.float32)
    d['xv'] = np.concatenate([d['xv'], v, F.group_demean(v)], -1).astype(np.float32)
    return d


def with_pairs(d, tb, vb):
    fn = CACHE / 'pairfeat.npz'
    if fn.exists():
        z = np.load(fn); pt, pv = z['t'], z['v']
    else:
        pt, pv = R.pair_inputs(tb, d['et']), R.pair_inputs(vb, d['ev'])
        np.savez(fn, t=pt, v=pv)
    return dict(d, pt=pt, pv=pv)


def with_pairs2(d, tb, vb):
    fn = CACHE / 'pairfeat2.npz'
    if fn.exists():
        z = np.load(fn); pt, pv = z['t'], z['v']
    else:
        pt, pv = V3.pair_features2(tb), V3.pair_features2(vb)
        np.savez(fn, t=pt, v=pv)
    return dict(d, pt=np.concatenate([d['pt'], pt], -1), pv=np.concatenate([d['pv'], pv], -1))


def train_one3(cfg, d, seed, pool=None):
    """nn_train.train_one for the relational model (same split/holdout/early-stopping protocol)."""
    import torch
    rng = np.random.RandomState(seed); torch.manual_seed(seed)
    n = len(d['xt'])
    hold = rng.rand(n) < 0.12
    use = d['ht']
    if pool is not None:
        use = use & pool; hold = hold & pool
    tr = use & ~hold & (d['Yt'].any(1))
    ho = d['ht'] & hold & (pool if pool is not None else True)
    dh, dp = d['xt'].shape[-1], d['pt'].shape[-1]
    xm, xs = d['xt'][tr].reshape(-1, dh).mean(0), d['xt'][tr].reshape(-1, dh).std(0) + 1e-6
    em, es = d['et'][tr].reshape(-1, 1024).mean(0), d['et'][tr].reshape(-1, 1024).std(0) + 1e-6
    pm, ps = d['pt'][tr].reshape(-1, dp).mean(0), d['pt'][tr].reshape(-1, dp).std(0) + 1e-6
    tt = lambda a: torch.tensor(a, dtype=torch.float32)
    Xt, Et, Mt, Pt = tt((d['xt'] - xm) / xs), tt((d['et'] - em) / es), tt(d['mt']), tt((d['pt'] - pm) / ps)
    Xv, Ev, Mv, Pv = tt((d['xv'] - xm) / xs), tt((d['ev'] - em) / es), tt(d['mv']), tt((d['pv'] - pm) / ps)
    Yt = torch.tensor(d['Yt'])
    net = R.build_torch(cfg, dh, 1024, dp)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    idx = np.where(tr)[0]; hb = torch.tensor(np.where(ho)[0])
    best, best_state, bad, best_ep = 1e9, None, 0, 0
    for ep in range(cfg['epochs']):
        net.train(); perm = rng.permutation(idx)
        for i in range(0, len(perm), 64):
            b = torch.tensor(perm[i:i+64])
            loss = T.listloss(net(Xt[b], Et[b], Mt[b], Pt[b]), Yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            hl = T.listloss(net(Xt[hb], Et[hb], Mt[hb], Pt[hb]), Yt[hb]).item()
        if hl < best - 1e-4:
            best, bad, best_ep = hl, 0, ep
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg['patience']:
                break
    net.load_state_dict(best_state)
    with torch.no_grad():
        pv = torch.softmax(net(Xv, Ev, Mv, Pv), -1).numpy()
    norm = dict(xm=xm, xs=xs, em=em, es=es, pm=pm, ps=ps)
    return net, norm, pv, best_ep


VARIANTS = {
    'base': dict(v3=False),
    'v3': dict(v3=True),
    'v3_H256': dict(v3=True, cfg=dict(H=256)),
    'v3_H64': dict(v3=True, cfg=dict(H=64)),
    'v3_p5': dict(v3=True, cfg=dict(p=0.5, p_emb=0.5)),
    'v3_pe6': dict(v3=True, cfg=dict(p_emb=0.6)),
    'v3_wd2': dict(v3=True, cfg=dict(wd=0.2)),
    'v3_lr3': dict(v3=True, cfg=dict(lr=3e-4, patience=12)),
    'v3_none': dict(v3=True, cfg=dict(inter='none')),
    'v34': dict(v3=True, relf=True),
    'v34_none': dict(v3=True, relf=True, cfg=dict(inter='none')),
    'rel': dict(v3=True, cfg=dict(rel=True)),
    'relp2': dict(v3=True, p2=True, cfg=dict(rel=True)),
    'rel34': dict(v3=True, relf=True, cfg=dict(rel=True)),
    'rel_ed': dict(v3=True, cfg=dict(rel=True, ed=True)),
    'rel_R16': dict(v3=True, cfg=dict(rel=True, R=16)),
    'rel_R64': dict(v3=True, cfg=dict(rel=True, R=64)),
    'rel_nov3': dict(v3=False, cfg=dict(rel=True)),
}


def run(variant, seeds, cfg_over=None, d=None, tb=None, vb=None, d0=None):
    if d is None:
        tb, vb, d0 = T.load_all(False)
        spec = VARIANTS[variant]
        d = with_v3(d0, tb, vb, spec.get('cols'), spec.get('relf', False)) if spec['v3'] else d0
        cfg_over = {**spec.get('cfg', {}), **(cfg_over or {})}
        if cfg_over.get('rel'):
            d = with_pairs(d, tb, vb)
            if spec.get('p2'):
                d = with_pairs2(d, tb, vb)
    cfg = dict(T.BASE, inter='deepsets', emb=True)
    cfg.update(cfg_over or {})
    fold = C.folds(len(tb)); Y, hard = d['Yt'], d['ht']
    tr1 = train_one3 if cfg.get('rel') else T.train_one
    Ps, Vs = [], []
    t0 = time.time()
    for s in range(seeds):
        fn = CACHE / f'{variant}_s{s}.npy'
        if fn.exists():
            P = np.load(fn)
        else:
            P = np.zeros(Y.shape, np.float32)
            for f in range(C.NF):
                te = fold == f
                dd = dict(d, xv=d['xt'][te], ev=d['et'][te], mv=d['mt'][te])
                if cfg.get('rel'):
                    dd['pv'] = d['pt'][te]
                P[te] = tr1(cfg, dd, s * 100 + f, pool=~te)[2]
            np.save(fn, P)
        Ps.append(P)
        fv = CACHE / f'{variant}_val_s{s}.npy'
        if fv.exists():
            pv = np.load(fv)
        else:
            pv = tr1(cfg, d, s)[2]
            np.save(fv, pv)
        Vs.append(pv)
    single = [np.mean(C.score(tb, Y, hard, fold, p)) for p in Ps]
    ens = C.score(tb, Y, hard, fold, np.mean(Ps, 0))
    vsingle = [T.evaluate(vb, d, p)[1] for p in Vs]
    vens = T.evaluate(vb, d, np.mean(Vs, 0))
    r = dict(variant=variant, cv_single=float(np.mean(single)), cv_single_sd=float(np.std(single)),
             cv_ens=float(np.mean(ens)), cv_ens_fold_sd=float(np.std(ens)), val_hard_single=float(np.mean(vsingle)),
             val_all_ens=vens[0], val_hard_ens=vens[1], seeds=seeds, sec=round(time.time() - t0))
    print(json.dumps(r), flush=True)
    with open(HERE / 'exp3_results.jsonl', 'a') as fh:
        fh.write(json.dumps(r) + '\n')
    return r


if __name__ == '__main__' and sys.argv[1] != 'export':
    run(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 3)


def export(spec, out, feat=None):
    """Train the members in spec [(cfg_over, seed), ...] on the full train split (hard groups) and write a
    numpy checkpoint (nn.npz + config.json) readable by nn3_infer; checks numpy vs torch on validation."""
    import torch
    import nn3_infer
    feat = feat or {'v3': True, 'relf': False}
    tb, vb, d0 = T.load_all(False)
    d = with_v3(d0, tb, vb, None, feat.get('relf', False))
    if any(c.get('rel') for c, _ in spec):
        d = with_pairs(d, tb, vb)
    arrs, members, tp, t0 = {}, [], [], time.time()
    for i, (over, seed) in enumerate(spec):
        cfg = dict(T.BASE, inter='deepsets', emb=True)
        cfg.update(over)
        net, norm, pv, ep = (train_one3 if cfg.get('rel') else T.train_one)(cfg, d, seed)
        tp.append(pv)
        for k, v in net.state_dict().items():
            arrs[f'm{i}/{k}'] = v.numpy().astype(np.float32)
        for k, v in norm.items():
            arrs[f'm{i}/{k}'] = np.asarray(v, np.float32)
        members.append({'cfg': cfg, 'seed': seed, 'best_epoch': int(ep)})
        print('member', i, cfg.get('inter'), cfg.get('rel', False), 'seed', seed, 'ep', ep, flush=True)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    np.savez(out / 'nn.npz', **arrs)
    (out / 'config.json').write_text(json.dumps({'feat': feat, 'agg': 'sum', 'members': members,
                                                 'train': 'hard train groups only (train split only)',
                                                 'torch': torch.__version__}, indent=1))
    E = np.load(HERE.parent / 'data/candidates/gsm8k_validation.npz')['embeddings'].reshape(len(vb), 4, -1)
    pn = nn3_infer.probs(vb, E, out)
    print('max abs diff numpy vs torch (val probs):', float(np.abs(np.mean(tp, 0) - pn).max()))
    print('val-all/val-hard (agg sum):', T.evaluate(vb, d, pn)[:2], 'train sec', round(time.time() - t0))


if __name__ == '__main__' and sys.argv[1] == 'export':
    # final checkpoint: .venv/bin/python method/exp3.py export K=10 out=checkpoint_v3
    kw = dict(a.split('=') for a in sys.argv[2:])
    export([({}, s) for s in range(int(kw.get('K', 10)))], kw.get('out', HERE.parent / 'checkpoint_v3'))
