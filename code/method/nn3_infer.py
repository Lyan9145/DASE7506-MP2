"""Numpy-only inference for the v3 selector: v1 hand+cross features + v3 cross-candidate text features
(features_v3), full embedding; ensemble of DeepSets / per-candidate MLP / relational members
(config.json lists them); cluster-sum decode. No torch import."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F  # noqa: E402
import features_v3 as V3  # noqa: E402
import nn_infer  # noqa: E402
import nn_rel  # noqa: E402
import selector as S  # noqa: E402
from nn_model import raw_inputs  # noqa: E402


def inputs(bank, E, feat, need_pairs):
    """Unstandardized model inputs (xh, e, M, pairs) exactly as built for training in exp3.py."""
    xh, e, M = raw_inputs(bank, E)
    v = V3.extra_features(bank)
    if feat.get('relf'):
        v = np.concatenate([v, V3.relation_features(bank)], -1)
    xh = np.concatenate([xh, v, F.group_demean(v)], -1).astype(np.float32)
    P = nn_rel.pair_inputs(bank, E) if need_pairs else None
    return xh, e, M, P


def load(ckdir):
    ckdir = Path(ckdir)
    conf = json.loads((ckdir / 'config.json').read_text())
    with np.load(ckdir / 'nn.npz', allow_pickle=False) as z:
        allw = {k: z[k] for k in z.files}
    members = []
    for i in range(len(conf['members'])):
        pre = f'm{i}/'
        members.append({k[len(pre):]: v for k, v in allw.items() if k.startswith(pre)})
    return conf, members


def probs(bank, E, ckdir):
    conf, members = load(ckdir)
    kinds = [m['cfg']['inter'] if not m['cfg'].get('rel') else 'rel' for m in conf['members']]
    xh, e, M, P = inputs(bank, np.asarray(E, np.float32), conf['feat'], 'rel' in kinds)
    ps = []
    for mc, W, kind in zip(conf['members'], members, kinds):
        lg = nn_rel.forward_np(W, xh, e, M, P) if kind == 'rel' else nn_infer.forward(W, mc['cfg'], xh, e, M)
        ps.append(nn_infer._softmax(lg))
    return np.mean(ps, 0)


def predict(bank, E, ckdir):
    conf, _ = load(ckdir)
    return S.aggregate_choose(bank, probs(bank, E, ckdir), conf['agg'])
