"""Train listwise / per-candidate NN selectors (train split only), evaluate on validation.
Run from code/:  .venv/bin/python method/nn_train.py [grid|export]"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import nn_model as M  # noqa: E402
import selector as S  # noqa: E402
import features as F  # noqa: E402
from common import load_inputs  # noqa: E402
from train import labels, acc  # noqa: E402

CK = HERE.parent / 'checkpoint_nn'
CACHE = Path('/tmp/claude-0/-root-DASE7506-MP2/24ec1cfd-5c48-4fdf-8a0d-54e05b389870/scratchpad/nn_cache.npz')
torch.set_num_threads(2)


def load_all():
    tb, ta = load_inputs('train'); vb, va = load_inputs('validation')
    if CACHE.exists():
        z = np.load(CACHE)
        d = {k: z[k] for k in z.files}
    else:
        xt, et, mt = M.raw_inputs(tb, ta['embeddings']); xv, ev, mv = M.raw_inputs(vb, va['embeddings'])
        d = dict(xt=xt, et=et, mt=mt, xv=xv, ev=ev, mv=mv)
        np.savez(CACHE, **d)
    d.update(Yt=labels(tb, 'train'), Yv=labels(vb, 'validation'), ht=F.hard_mask(tb), hv=F.hard_mask(vb))
    return tb, vb, d


def listloss(s, Y):
    lp = torch.log_softmax(s, -1)
    return -(torch.logsumexp(lp.masked_fill(~Y, -1e9), -1)).mean()


def probs(net, x, e, m, bs=1024):
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            out.append(torch.softmax(net(x[i:i+bs], e[i:i+bs], m[i:i+bs]), -1))
    return torch.cat(out).numpy()


def train_one(cfg, d, seed, train_all=False, log=False):
    """Returns (net, norm, val probs). Early stopping on 12% train-internal holdout (hard groups' loss)."""
    rng = np.random.RandomState(seed); torch.manual_seed(seed)
    n = len(d['xt'])
    hold = rng.rand(n) < 0.12
    use = np.ones(n, bool) if train_all else d['ht']
    tr = use & ~hold & (d['Yt'].any(1))
    ho = d['ht'] & hold
    xm, xs = d['xt'][tr].reshape(-1, d['xt'].shape[-1]).mean(0), d['xt'][tr].reshape(-1, d['xt'].shape[-1]).std(0) + 1e-6
    em, es = d['et'][tr].reshape(-1, 1024).mean(0), d['et'][tr].reshape(-1, 1024).std(0) + 1e-6
    T = lambda x, e, m: (torch.tensor((x - xm) / xs, dtype=torch.float32), torch.tensor((e - em) / es, dtype=torch.float32), torch.tensor(m))
    Xt, Et, Mt = T(d['xt'], d['et'], d['mt']); Xv, Ev, Mv = T(d['xv'], d['ev'], d['mv'])
    Yt = torch.tensor(d['Yt'])
    net = M.build_torch(cfg, Xt.shape[-1], 1024)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    idx = np.where(tr)[0]; hidx = np.where(ho)[0]
    best, best_state, bad = 1e9, None, 0
    for ep in range(cfg['epochs']):
        net.train(); perm = rng.permutation(idx)
        for i in range(0, len(perm), 64):
            b = torch.tensor(perm[i:i+64])
            loss = listloss(net(Xt[b], Et[b], Mt[b]), Yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            hb = torch.tensor(hidx)
            hl = listloss(net(Xt[hb], Et[hb], Mt[hb]), Yt[hb]).item()
        if hl < best - 1e-4:
            best, bad = hl, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
            best_ep = ep
        else:
            bad += 1
            if bad >= cfg['patience']:
                break
    net.load_state_dict(best_state)
    norm = dict(xm=xm, xs=xs, em=em, es=es)
    return net, norm, probs(net, Xv, Ev, Mv), best_ep


def evaluate(vb, d, p):
    ch = np.array(S.aggregate_choose(vb, p, 'sum')); ch0 = p.argmax(1)
    return acc(ch, d['Yv']), acc(ch, d['Yv'], d['hv']), acc(ch0, d['Yv'], d['hv'])


BASE = dict(H=128, P=0, p=0.3, p_emb=0.3, lr=1e-3, wd=0.05, epochs=60, patience=8)


def grid():
    tb, vb, d = load_all()
    exps = []
    for inter in ('none', 'deepsets', 'attn'):
        for emb in (True, False):
            for ta in (False, True):
                exps.append((f'{inter}|emb={int(emb)}|train={"all" if ta else "hard"}', dict(BASE, inter=inter, emb=emb), ta))
    res = {}
    for name, cfg, ta in exps:
        t0 = time.time(); rows = []
        for seed in range(3):
            net, norm, p, ep = train_one(cfg, d, seed, ta)
            rows.append((*evaluate(vb, d, p), ep, p))
        pe = np.mean([r[4+1-1+1] if False else r[4] for r in rows], 0) if False else np.mean([r[-1] for r in rows], 0)
        ea = evaluate(vb, d, pe)
        a = np.array([r[:3] for r in rows])
        res[name] = dict(all_m=a[:, 0].mean(), all_s=a[:, 0].std(), hard_m=a[:, 1].mean(), hard_s=a[:, 1].std(),
                         hard_nocluster=a[:, 2].mean(), ens_all=ea[0], ens_hard=ea[1], eps=[r[3] for r in rows], sec=time.time()-t0)
        print(name, {k: (round(v, 2) if not isinstance(v, list) else v) for k, v in res[name].items()}, flush=True)
        json.dump(res, open(HERE / 'nn_grid.json', 'w'), indent=1)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'grid':
        grid()


def export(n_members=5, inter='deepsets'):
    """Train K seeds (hard train groups, full emb) and save numpy checkpoint; verify numpy vs torch on val."""
    import nn_infer
    tb, vb, d = load_all()
    cfg = dict(BASE, inter=inter, emb=True)
    arrs, torch_p, t0 = {}, [], time.time()
    for s in range(n_members):
        net, norm, p, ep = train_one(cfg, d, s)
        torch_p.append(p)
        for k, v in net.state_dict().items():
            arrs[f'm{s}/{k}'] = v.numpy().astype(np.float32)
        for k, v in norm.items():
            arrs[f'm{s}/{k}'] = v.astype(np.float32)
        print('seed', s, 'ep', ep, flush=True)
    CK.mkdir(exist_ok=True)
    np.savez(CK / 'nn.npz', **arrs)
    (CK / 'config.json').write_text(json.dumps({'cfg': cfg, 'agg': 'sum', 'n_members': n_members, 'seeds': list(range(n_members)),
                                                'train': 'hard train groups only', 'torch': torch.__version__}, indent=1))
    pt = np.mean(torch_p, 0)
    pn = nn_infer.probs(vb, np.load(Path(__file__).resolve().parent.parent / 'data/candidates/gsm8k_validation.npz')['embeddings'].reshape(len(vb), 4, -1), CK)
    print('max abs diff numpy vs torch (val probs):', float(np.abs(pt - pn).max()))
    print('val-all/val-hard (agg sum):', evaluate(vb, d, pn)[:2], 'train sec', round(time.time() - t0))


if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'export':
    export()
