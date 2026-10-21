"""Extra verifier features (v2), extending features.py without changing it. numpy/regex only.
extra_features(bank) -> (N,K,len(V2_NAMES)) float32."""
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import features as F  # noqa: E402
from common import parse_answer  # noqa: E402

NUM = r'\d+(?:\.\d+)?'
# chained expression "a op b op c = d" (also "= $12", "= 12 apples" handled by normalisation)
EQ2 = re.compile(r'(?<![\w.])(\(?' + NUM + r'\)?(?:\s*[-+*/]\s*\(?' + NUM + r'\)?)+\)?)\s*=\s*(-?' + NUM + r')(?![\d.]*\d)')
UNIT = re.compile(r'(?<=\d)\s*[A-Za-z]+(?:\s[A-Za-z]+)?(?=\s*[-+*/=)])')

V2_NAMES = ['n_eq2', 'frac_wrong2', 'n_wrong2', 'last_eq_ans', 'ans_last_num', 'q_in_eq', 'last_eq_exists',
            'pl_ratio_max', 'pl_ratio_sum', 'pl_lt_min', 'pl_gt_prod', 'pct_used', 'cl_wrong2']


def _norm2(t):
    t = F._norm_math(t)
    t = re.sub(r'(?<=\d)\s*[xX]\s*(?=[\d(])', '*', t)
    t = re.sub(r'(\d+(?:\.\d+)?)\s*%', r'(\1/100)', t)
    t = t.replace('\\%', '').replace('\\text{', ' ').replace('}', ' ')
    t = re.sub(r'(?<=\d)\s*\$', '', t)
    t = UNIT.sub('', t)
    t = re.sub(r'(?<!\d)(\d+)\.(?!\d)', r'\1 ', t)  # sentence-final periods
    return t


def _eqs(text):
    t = _norm2(text)
    out = []
    for m in EQ2.finditer(t):
        expr = m.group(1)
        if not F.SAFE_RE.match(expr):
            continue
        try:
            lhs = eval(expr, {'__builtins__': {}}, {})
            rhs = float(m.group(2))
        except Exception:
            continue
        ok = abs(lhs - rhs) <= max(0.011 * abs(rhs), 0.006)
        out.append((expr, rhs, ok))
        if len(out) >= 60:
            break
    return out


def _cand(text, qn, val):
    eqs = _eqs(text)
    n = len(eqs); nw = sum(not e[2] for e in eqs)
    last = eqs[-1][1] if eqs else None
    last_eq_ans = 0.5 if (last is None or val is None) else float(abs(last - val) < 1e-6 * max(1, abs(val)))
    nums = [float(s.replace(',', '')) for s in F.NUM_RE.findall(text.rsplit('####', 1)[0])]
    nums = [x for x in nums if True]
    # last number before final answer tail (drop the answer itself if it is the very last token)
    ans_last = 0.5
    if val is not None and nums:
        body = [x for x in nums]
        ans_last = float(abs(body[-1] - val) < 1e-9) if len(body) else 0.5
    qin = 0.0
    if qn and eqs:
        used = set()
        for e in eqs:
            for s in re.findall(NUM, e[0]):
                used.add(float(s))
        qin = len(qn & used) / len(qn)
    qs = sorted(qn)
    if val is not None and qs:
        a = abs(val)
        mx, sm = max(qs), sum(qs)
        pr = 1.0
        for x in qs:
            pr *= max(x, 1.0)
        f = [math.log10(1 + a) - math.log10(1 + mx), math.log10(1 + a) - math.log10(1 + sm),
             float(a < min(qs) / 10 and a != 0), float(a > pr * 10)]
    else:
        f = [0.0, 0.0, 0.0, 0.0]
    pct = float('%' in text)
    return [n and min(n, 20) / 20 or 0.0, (nw / n) if n else 0.0, min(nw, 10), last_eq_ans, ans_last, qin, float(n > 0),
            np.clip(f[0], -4, 4), np.clip(f[1], -4, 4), f[2], f[3], pct], nw


def extra_features(bank):
    N = len(bank); K = max(len(r['candidates']) for r in bank)
    X = np.zeros((N, K, len(V2_NAMES)), np.float32)
    for g, row in enumerate(bank):
        qn = F._nums(row['question'])
        ans = [parse_answer(c['text']) for c in row['candidates']]
        vals = []
        for a in ans:
            try:
                from fractions import Fraction
                vals.append(float(Fraction(a)) if a is not None else None)
            except Exception:
                vals.append(None)
        res = [_cand(c['text'], qn, v) for c, v in zip(row['candidates'], vals)]
        for i, (f, nw) in enumerate(res):
            mem = [j for j in range(len(ans)) if ans[i] is not None and ans[j] == ans[i]]
            cl = np.mean([res[j][1] for j in mem]) if mem else 0.0
            X[g, i, :len(f)] = f
            X[g, i, -1] = cl
    return X
