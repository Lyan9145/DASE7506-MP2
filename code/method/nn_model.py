"""Shared pieces for the listwise NN selector: input building + torch model (training only) and
numpy forward lives in nn_infer.py. Imports features.py/selector.py unchanged."""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import features as F  # noqa: E402
import selector as S  # noqa: E402
from common import parse_answer  # noqa: E402


def same_matrix(bank):
    """(N,K,K) float: 1 where candidates i,j share the same valid parsed answer (diagonal always 1)."""
    N, K = len(bank), max(len(r['candidates']) for r in bank)
    M = np.tile(np.eye(K, dtype=np.float32), (N, 1, 1))
    for g, row in enumerate(bank):
        ans = [parse_answer(c['text']) for c in row['candidates']]
        for i in range(len(ans)):
            for j in range(len(ans)):
                if ans[i] is not None and ans[i] == ans[j]:
                    M[g, i, j] = 1
    return M


def raw_inputs(bank, E):
    """Unstandardized hand+rel+cross features (N,K,dh) and embeddings (N,K,1024)."""
    B = S.blocks(bank, E)
    xh = S.design(B, {'blocks': ['hand', 'cross'], 'rel': True}, None)
    return xh, E.astype(np.float32), same_matrix(bank)


def build_torch(cfg, dh, de):
    import torch
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            H, P = cfg['H'], cfg['P']
            s.use_emb, s.inter = cfg['emb'], cfg['inter']
            s.wh = nn.Linear(dh, H)
            if s.use_emb:
                s.we = nn.Linear(de, H, bias=False)
                s.din = nn.Dropout(cfg['p_emb'])
            s.l2 = nn.Linear(H, H)
            s.drop = nn.Dropout(cfg['p'])
            if s.inter == 'deepsets':
                s.mix = nn.Linear(4 * H, H)
            elif s.inter == 'attn':
                s.nh = 4
                s.q, s.k, s.v, s.o = (nn.Linear(H, H) for _ in range(4))
                s.ln = nn.LayerNorm(H)
                s.f1, s.f2 = nn.Linear(H, 2 * H), nn.Linear(2 * H, H)
                s.ln2 = nn.LayerNorm(H)
            s.out = nn.Linear(H, 1)

        def forward(s, xh, e, M):
            z = s.wh(xh)
            if s.use_emb:
                z = z + s.we(s.din(e))
            h = torch.relu(z)
            h = s.drop(h)
            h = torch.relu(s.l2(h))
            h = s.drop(h)
            if s.inter == 'deepsets':
                m = h.mean(1, keepdim=True).expand_as(h)
                c = (M @ h) / M.sum(-1, keepdim=True)
                h = torch.relu(s.mix(torch.cat([h, h - m, m, c], -1)))
                h = s.drop(h)
            elif s.inter == 'attn':
                B, K, H = h.shape
                sp = lambda t: t.view(B, K, s.nh, H // s.nh).transpose(1, 2)
                q, k, v = sp(s.q(h)), sp(s.k(h)), sp(s.v(h))
                a = torch.softmax(q @ k.transpose(-1, -2) / (H // s.nh) ** 0.5, -1)
                h = s.ln(h + s.drop(s.o((a @ v).transpose(1, 2).reshape(B, K, H))))
                h = s.ln2(h + s.drop(s.f2(torch.relu(s.f1(h)))))
            return s.out(h).squeeze(-1)
    return Net()
