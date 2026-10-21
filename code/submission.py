"""Learned selector: hand-crafted + embedding features -> logistic regression, answer-cluster aggregation.
Checkpoint dir holds selector.npz + config.json (see method/train.py). predict.py puts this file's
directory on sys.path, so `common` and method/ modules are importable.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / 'method'))
import selector as S  # noqa: E402


def predict(bank, arrays, checkpoint=None):
    checkpoint = Path(checkpoint)
    conf = json.loads((checkpoint / 'config.json').read_text())
    with np.load(checkpoint / 'selector.npz', allow_pickle=False) as z:
        ck = {k: z[k] for k in z.files}
    B = S.blocks(bank, arrays['embeddings'])
    X = S.design(B, conf['cfg'], ck)
    p = S.linear_proba(X, ck)
    return S.aggregate_choose(bank, p, conf['agg'])
