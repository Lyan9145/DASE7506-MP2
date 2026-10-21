"""Numpy-only forward pass of the listwise NN selector (matches nn_model.build_torch in eval mode).
No torch import. Inputs are built with features.py/selector.py (numpy)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F  # noqa: E402
import selector as S  # noqa: E402


from nn_model import raw_inputs  # noqa: E402  (numpy-only at import; torch imported lazily in build_torch)


def _ln(x, g, b):
    mu = x.mean(-1, keepdims=True); v = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(v + 1e-5) * g + b


def _softmax(x):
    x = x - x.max(-1, keepdims=True); e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


def forward(W, cfg, xh, e, M):
    """W: dict of arrays for one member (torch Linear weights stored as (out,in)). Returns logits (N,K)."""
    lin = lambda x, n: x @ W[n + '.weight'].T + (W[n + '.bias'] if n + '.bias' in W else 0)
    xh = (xh - W['xm']) / W['xs']
    z = lin(xh, 'wh')
    if cfg['emb']:
        z = z + ((e - W['em']) / W['es']) @ W['we.weight'].T
    h = np.maximum(z, 0)
    h = np.maximum(lin(h, 'l2'), 0)
    if cfg['inter'] == 'deepsets':
        m = np.broadcast_to(h.mean(1, keepdims=True), h.shape)
        c = (M @ h) / M.sum(-1, keepdims=True)
        h = np.maximum(lin(np.concatenate([h, h - m, m, c], -1), 'mix'), 0)
    elif cfg['inter'] == 'attn':
        B, K, H = h.shape; nh = 4
        sp = lambda t: t.reshape(B, K, nh, H // nh).transpose(0, 2, 1, 3)
        q, k, v = sp(lin(h, 'q')), sp(lin(h, 'k')), sp(lin(h, 'v'))
        a = _softmax(q @ k.transpose(0, 1, 3, 2) / (H // nh) ** 0.5)
        h = _ln(h + lin((a @ v).transpose(0, 2, 1, 3).reshape(B, K, H), 'o'), W['ln.weight'], W['ln.bias'])
        h = _ln(h + lin(np.maximum(lin(h, 'f1'), 0), 'f2'), W['ln2.weight'], W['ln2.bias'])
    return lin(h, 'out')[..., 0]


def load(ckdir):
    ckdir = Path(ckdir)
    conf = json.loads((ckdir / 'config.json').read_text())
    with np.load(ckdir / 'nn.npz', allow_pickle=False) as z:
        allw = {k: z[k] for k in z.files}
    members = []
    for s in range(conf['n_members']):
        pre = f'm{s}/'
        members.append({k[len(pre):]: v for k, v in allw.items() if k.startswith(pre)})
    return conf, members


def probs(bank, E, ckdir, return_xin=False):
    conf, members = load(ckdir)
    xh, e, M = raw_inputs(bank, E, conf['cfg'].get('v2', False))
    ps = [_softmax(forward(W, conf['cfg'], xh, e, M)) for W in members]
    return np.mean(ps, 0)


def predict_nn(bank, E, ckdir):
    conf, _ = load(ckdir)
    return S.aggregate_choose(bank, probs(bank, E, ckdir), conf['agg'])
