"""Shared feature extraction (training + inference). Deterministic, CPU-only, no ID use."""
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import parse_answer  # noqa: E402

NUM = r'\d+(?:\.\d+)?'
EQ_RE = re.compile(r'(?<![\w.])(\(?' + NUM + r'\)?(?:\s*[-+*/]\s*\(?' + NUM + r'\)?)+)\s*=\s*(-?' + NUM + r')(?![\d.]*\d)')
NUM_RE = re.compile(r'\d[\d,]*(?:\.\d+)?')
SAFE_RE = re.compile(r'^[\d\s.+\-*/()]+$')
FRAC_RE = re.compile(r'\\[dt]?frac\{([^{}]*)\}\{([^{}]*)\}')
BOXED_RE = re.compile(r'\\boxed\{([^{}]+)\}')
HEDGE = ('wait', 'however', 'actually', 'let me', 're-evaluate', 'recheck', 'mistake', 'hmm', 'but ')

HAND_NAMES = [
    'valid', 'cnt', 'share', 'ndist', 'is_top', 'ntop', 'is_int', 'neg', 'logmag', 'logmag_rel',
    'loglen', 'len_rel', 'len_rank', 'is_longest', 'is_shortest', 'has_boxed', 'has_hash',
    'placeholder', 'agree', 'ans_repeat', 'n_eq', 'eq_frac', 'eq_wrong', 'eq_ok', 'n_lines',
    'q_used', 'q_used_n', 'hedge', 'rep_lines', 'trunc', 'ans_is_qnum', 'n_final',
    'cl_eq_wrong', 'cl_len_rel', 'cl_eq_frac', 'ans_rank', 'tail_after',
]


def _norm_math(t):
    t = t.replace('\\times', '*').replace('\\cdot', '*').replace('×', '*').replace('\\div', '/').replace('÷', '/')
    t = t.replace('\\left', '').replace('\\right', '').replace('\\,', '').replace('\\ ', ' ').replace('\\$', '')
    for _ in range(2):
        t = FRAC_RE.sub(r'(\1)/(\2)', t)
    t = re.sub(r'(?<=\d),(?=\d{3})', '', t)
    t = t.replace('$', '').replace('−', '-').replace('–', '-')
    return t


def arith_check(text):
    """Return (n_expressions, n_correct) of explicit 'a op b ... = c' expressions."""
    t = _norm_math(text)
    n = ok = 0
    for m in EQ_RE.finditer(t):
        expr = m.group(1)
        if not SAFE_RE.match(expr):
            continue
        try:
            lhs = eval(expr, {'__builtins__': {}}, {})  # digits and + - * / ( ) only
            rhs = float(m.group(2))
        except Exception:
            continue
        n += 1
        tol = max(0.011 * abs(rhs), 0.006)
        ok += abs(lhs - rhs) <= tol
        if n >= 60:
            break
    return n, ok


def _nums(text):
    out = set()
    for s in NUM_RE.findall(text):
        try:
            out.add(float(s.replace(',', '')))
        except ValueError:
            pass
    return out


def _cand_basic(text, qnums):
    ans = parse_answer(text)
    val = None
    if ans is not None:
        try:
            from fractions import Fraction
            val = float(Fraction(ans))
        except Exception:
            val = None
    boxed = BOXED_RE.findall(text)
    has_hash = '####' in text
    hash_ans = parse_answer('####' + text.rsplit('####', 1)[1]) if has_hash else None
    box_ans = parse_answer('\\boxed{' + boxed[-1] + '}') if boxed else None
    tail = text.rsplit('####', 1)[1] if has_hash else (boxed[-1] if boxed else '')
    placeholder = int('<numeric answer>' in text or '#### <' in text)
    if hash_ans is not None and box_ans is not None:
        agree = 1.0 if hash_ans == box_ans else 0.0
    else:
        agree = 0.5
    # how often the answer number appears in the reasoning before the final answer
    rep = 0
    if val is not None:
        body = text.rsplit('####', 1)[0] if has_hash else (text[:text.rfind('\\boxed')] if boxed else text)
        for s in NUM_RE.findall(body):
            try:
                if abs(float(s.replace(',', '')) - val) < 1e-9:
                    rep += 1
            except ValueError:
                pass
    n_eq, n_ok = arith_check(text)
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    rep_lines = 1 - len(set(lines)) / len(lines) if lines else 0.0
    tn = _nums(text)
    used = len(qnums & tn) / len(qnums) if qnums else 1.0
    low = text.lower()
    hedge = sum(low.count(h) for h in HEDGE)
    tail_after = len(tail.strip()) if boxed or has_hash else 0
    return dict(ans=ans, val=val, has_boxed=int(bool(boxed)), has_hash=int(has_hash), placeholder=placeholder,
                agree=agree, rep=rep, n_eq=n_eq, n_ok=n_ok, rep_lines=rep_lines, used=used,
                qn=len(qnums), hedge=hedge, n_lines=len(lines), L=len(text),
                trunc=int(ans is None and not boxed and not has_hash),
                n_final=low.count('final answer'), tail_after=min(tail_after, 50) / 50,
                qnum=int(val is not None and val in qnums))


def hand_features(bank):
    """(N, 4, F) float32 hand features, one group per bank row (candidate count taken from row)."""
    N = len(bank)
    K = max(len(r['candidates']) for r in bank)
    X = np.zeros((N, K, len(HAND_NAMES)), np.float32)
    for g, row in enumerate(bank):
        qn = _nums(row['question'])
        B = [_cand_basic(c['text'], qn) for c in row['candidates']]
        k = len(B)
        ans = [b['ans'] for b in B]
        cnt = {}
        for a in ans:
            if a is not None:
                cnt[a] = cnt.get(a, 0) + 1
        top = max(cnt.values()) if cnt else 0
        ntop = sum(v == top for v in cnt.values())
        ndist = len(cnt)
        L = np.array([b['L'] for b in B], float)
        Lm = L.mean() if k else 1.0
        order = np.argsort(np.argsort(L, kind='stable'), kind='stable')
        lm = [math.log10(1 + abs(b['val'])) if b['val'] is not None else np.nan for b in B]
        lmm = np.nanmean(lm) if any(not np.isnan(x) for x in lm) else 0.0
        vals_sorted = sorted({b['val'] for b in B if b['val'] is not None})
        for i, b in enumerate(B):
            a = ans[i]
            mem = [j for j in range(k) if a is not None and ans[j] == a]
            valid = a is not None
            efrac = (b['n_ok'] / b['n_eq']) if b['n_eq'] else 1.0
            cl_wrong = np.mean([B[j]['n_eq'] - B[j]['n_ok'] for j in mem]) if mem else 0.0
            cl_len = np.mean([L[j] for j in mem]) / Lm if mem else 0.0
            cl_frac = np.mean([(B[j]['n_ok'] / B[j]['n_eq']) if B[j]['n_eq'] else 1.0 for j in mem]) if mem else 0.0
            v = b['val']
            row_f = [
                valid, cnt.get(a, 0) if valid else 0, (cnt.get(a, 0) / k) if valid else 0.0, ndist,
                int(valid and cnt[a] == top), ntop,
                int(valid and v is not None and abs(v - round(v)) < 1e-9), int(valid and v is not None and v < 0),
                lm[i] if valid and not np.isnan(lm[i]) else 0.0,
                (lm[i] - lmm) if valid and not np.isnan(lm[i]) else 0.0,
                math.log(1 + L[i]), L[i] / Lm, order[i] / max(k - 1, 1), int(order[i] == k - 1), int(order[i] == 0),
                b['has_boxed'], b['has_hash'], b['placeholder'], b['agree'], min(b['rep'], 5) / 5,
                min(b['n_eq'], 20) / 20, efrac, min(b['n_eq'] - b['n_ok'], 10), int(b['n_eq'] > 0 and b['n_eq'] == b['n_ok']),
                min(b['n_lines'], 60) / 60, b['used'], b['qn'], min(b['hedge'], 10), b['rep_lines'], b['trunc'],
                b['qnum'], min(b['n_final'], 3), cl_wrong, cl_len, cl_frac,
                (vals_sorted.index(v) / max(len(vals_sorted) - 1, 1)) if v is not None else 0.5, b['tail_after'],
            ]
            X[g, i, :] = row_f
    return X


CROSS_NAMES = ['cos_mean', 'cos_same', 'cos_other', 'cos_same_n']


def cross_emb_features(bank, E):
    """Scalar cross-candidate embedding features (N,K,4); E (N,K,D) raw embeddings."""
    N, K, _ = E.shape
    En = E / np.linalg.norm(E, axis=-1, keepdims=True)
    mean = En.mean(1, keepdims=True)
    cos_mean = (En * mean).sum(-1) / np.linalg.norm(mean, axis=-1)
    S = np.einsum('nkd,njd->nkj', En, En)
    out = np.zeros((N, K, 4), np.float32)
    out[:, :, 0] = cos_mean
    for g, row in enumerate(bank):
        ans = [parse_answer(c['text']) for c in row['candidates']]
        for i in range(K):
            same = [j for j in range(K) if j != i and ans[i] is not None and ans[j] == ans[i]]
            oth = [j for j in range(K) if j != i and j not in same]
            out[g, i, 1] = S[g, i, same].mean() if same else 0.0
            out[g, i, 2] = S[g, i, oth].mean() if oth else 0.0
            out[g, i, 3] = len(same)
    return out


def group_demean(X):
    return X - X.mean(1, keepdims=True)


def hard_mask(bank):
    return np.array([len({parse_answer(c['text']) for c in r['candidates']} - {None}) >= 2 for r in bank])
