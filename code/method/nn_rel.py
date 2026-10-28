"""Relational DeepSets selector (v3): the DeepSets model of nn_model.py plus a learned pair-message block.
For candidates i != j:  m_ij = relu(A h_i + B h_j + P p_ij + b),  r_i = mean_j m_ij,
mix layer sees [h, h - mean, mean, cluster-mean, r]. p_ij = features_v3.pair_features (answer-in-reasoning,
number overlap, embedding cosine, magnitude ratio, ...). Torch for training, numpy forward for inference."""
import numpy as np

import features_v3 as V3


def offdiag(K):
    return (1 - np.eye(K, dtype=np.float32))


def build_torch(cfg, dh, de, dp):
    import torch
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            H, R = cfg['H'], cfg.get('R', 32)
            s.wh = nn.Linear(dh, H)
            s.we = nn.Linear(de, H, bias=False)
            s.din = nn.Dropout(cfg['p_emb'])
            s.l2 = nn.Linear(H, H)
            s.drop = nn.Dropout(cfg['p'])
            s.pa = nn.Linear(H, R)
            s.pb = nn.Linear(H, R, bias=False)
            s.pp = nn.Linear(dp, R, bias=False)
            s.ed = cfg.get('ed', False)
            if s.ed:
                s.pe = nn.Linear(de, R, bias=False)
            s.mix = nn.Linear(4 * H + R, H)
            s.out = nn.Linear(H, 1)

        def forward(s, xh, e, M, Pp):
            h = torch.relu(s.wh(xh) + s.we(s.din(e)))
            h = s.drop(h)
            h = torch.relu(s.l2(h))
            h = s.drop(h)
            K = h.shape[1]
            pre = s.pa(h)[:, :, None, :] + s.pb(h)[:, None, :, :] + s.pp(Pp)
            if s.ed:
                pre = pre + s.pe(s.din((e[:, :, None, :] - e[:, None, :, :]).abs()))
            msg = torch.relu(pre)
            od = torch.tensor(offdiag(K))[None, :, :, None]
            r = (msg * od).sum(2) / (K - 1)
            m = h.mean(1, keepdim=True).expand_as(h)
            c = (M @ h) / M.sum(-1, keepdim=True)
            h = torch.relu(s.mix(torch.cat([h, h - m, m, c, r], -1)))
            h = s.drop(h)
            return s.out(h).squeeze(-1)
    return Net()


def forward_np(W, xh, e, M, Pp):
    lin = lambda x, n: x @ W[n + '.weight'].T + (W[n + '.bias'] if n + '.bias' in W else 0)
    xh = (xh - W['xm']) / W['xs']
    e = (e - W['em']) / W['es']
    Pp = (Pp - W['pm']) / W['ps']
    h = np.maximum(lin(xh, 'wh') + e @ W['we.weight'].T, 0)
    h = np.maximum(lin(h, 'l2'), 0)
    K = h.shape[1]
    pre = lin(h, 'pa')[:, :, None, :] + lin(h, 'pb')[:, None, :, :] + lin(Pp, 'pp')
    if 'pe.weight' in W:
        pre = pre + np.abs(e[:, :, None, :] - e[:, None, :, :]) @ W['pe.weight'].T
    msg = np.maximum(pre, 0)
    r = (msg * offdiag(K)[None, :, :, None]).sum(2) / (K - 1)
    m = np.broadcast_to(h.mean(1, keepdims=True), h.shape)
    c = (M @ h) / M.sum(-1, keepdims=True)
    h = np.maximum(lin(np.concatenate([h, h - m, m, c, r], -1), 'mix'), 0)
    return lin(h, 'out')[..., 0]


def pair_inputs(bank, E):
    return V3.pair_features(bank, E.astype(np.float32))
