"""Cross-candidate text features (v3), extending features.py without changing it. numpy/regex only.
Idea: a wrong candidate often stops early (its answer is an intermediate value of another candidate's
chain) or overshoots; whether the OTHER candidates' final answers occur inside a candidate's reasoning
(and where) is evidence that per-candidate embeddings cannot see.
extra_features(bank) -> (N,K,len(V3_NAMES)) float32."""
import re
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import features as F  # noqa: E402
from common import parse_answer  # noqa: E402

V3_NAMES = [
    'ans_in_oth', 'ans_in_oth_any', 'oth_in_me', 'oth_in_me_any', 'oth_in_me_late', 'ans_first_pos',
    'jac_same', 'jac_diff', 'num_grounded', 'n_new_nums', 'qcov_w', 'cl_oth_in_me', 'cl_ans_in_oth',
    'n_none', 'ans_in_oth_cand',
]
WORDNUM = {w: i for i, w in enumerate('zero one two three four five six seven eight nine ten eleven twelve thirteen '
                                      'fourteen fifteen sixteen seventeen eighteen nineteen twenty'.split())}
WORDNUM.update({'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90,
                'hundred': 100, 'thousand': 1000, 'dozen': 12, 'half': 0.5, 'twice': 2, 'double': 2, 'triple': 3,
                'quarter': 0.25, 'third': 3, 'fourth': 4, 'fifth': 5})
WORD_RE = re.compile(r'[a-z]+')
NUMPOS_RE = re.compile(r'\d[\d,]*(?:\.\d+)?')


def _val(a):
    try:
        return float(Fraction(a))
    except Exception:
        return None


def _numpos(text):
    """list of (value, relative position) of numbers in text (commas stripped)."""
    out, L = [], max(len(text), 1)
    for m in NUMPOS_RE.finditer(text):
        try:
            out.append((float(m.group().replace(',', '')), m.start() / L))
        except ValueError:
            pass
    return out


def _qnums_words(q):
    s = F._nums(q)
    for w in WORD_RE.findall(q.lower()):
        if w in WORDNUM:
            s.add(float(WORDNUM[w]))
    return s


def _body(text):
    """reasoning before the final answer marker."""
    if '####' in text:
        return text.rsplit('####', 1)[0]
    i = text.rfind('\\boxed')
    return text[:i] if i >= 0 else text


def extra_features(bank):
    N = len(bank)
    K = max(len(r['candidates']) for r in bank)
    X = np.zeros((N, K, len(V3_NAMES)), np.float32)
    for g, row in enumerate(bank):
        texts = [c['text'] for c in row['candidates']]
        k = len(texts)
        ans = [parse_answer(t) for t in texts]
        vals = [_val(a) if a is not None else None for a in ans]
        bodies = [_body(t) for t in texts]
        npos = [_numpos(b) for b in bodies]
        nset = [set(v for v, _ in p) for p in npos]
        allset = [F._nums(t) for t in texts]
        qn = _qnums_words(row['question'])
        dist = sorted({a for a in ans if a is not None})
        n_none = sum(a is None for a in ans)
        f = np.zeros((k, len(V3_NAMES)), np.float32)
        for i in range(k):
            oth_ans = [a for a in dist if a != ans[i]]
            oth_vals = [_val(a) for a in oth_ans]
            oth_c = [j for j in range(k) if j != i and ans[j] is not None and ans[j] != ans[i]]
            # my answer inside other-answer candidates' reasoning
            if vals[i] is not None and oth_c:
                hits = [vals[i] in nset[j] for j in oth_c]
                f[i, 0], f[i, 1], f[i, 14] = np.mean(hits), float(any(hits)), sum(hits)
            # other answers inside my reasoning (and how late)
            if oth_vals:
                inme = [v is not None and v in nset[i] for v in oth_vals]
                f[i, 2], f[i, 3] = np.mean(inme), float(any(inme))
                late = [p for v, p in npos[i] if any(v == ov for ov in oth_vals if ov is not None)]
                f[i, 4] = max(late) if late else 0.0
            if vals[i] is not None:
                pos = [p for v, p in npos[i] if v == vals[i]]
                f[i, 5] = min(pos) if pos else 1.0
            else:
                f[i, 5] = 1.0
            same = [j for j in range(k) if j != i and ans[i] is not None and ans[j] == ans[i]]
            jac = lambda j: len(allset[i] & allset[j]) / max(len(allset[i] | allset[j]), 1)
            f[i, 6] = np.mean([jac(j) for j in same]) if same else 0.0
            f[i, 7] = np.mean([jac(j) for j in oth_c]) if oth_c else 0.0
            # grounded numbers: in question or in some other candidate
            others = set().union(*[allset[j] for j in range(k) if j != i]) if k > 1 else set()
            mine = allset[i]
            f[i, 8] = len([v for v in mine if v in qn or v in others]) / max(len(mine), 1)
            f[i, 9] = min(len([v for v in mine if v not in qn and v not in others]), 20) / 20
            f[i, 10] = len(qn & mine) / len(qn) if qn else 1.0
            f[i, 13] = n_none
        for i in range(k):
            mem = [j for j in range(k) if ans[i] is not None and ans[j] == ans[i]] or [i]
            f[i, 11] = f[mem, 2].mean()
            f[i, 12] = f[mem, 0].mean()
        X[g, :k] = f
    return X


PAIR_NAMES = ['same', 'aj_in_i', 'aj_in_i_pos', 'ai_in_j', 'jac', 'cos', 'both_valid', 'logratio', 'j_gt_i', 'len_ratio']


def pair_features(bank, E):
    """(N,K,K,len(PAIR_NAMES)) float32 relation features of candidate i (axis 1) towards candidate j (axis 2)."""
    N, K = len(bank), E.shape[1]
    En = E / np.linalg.norm(E, axis=-1, keepdims=True)
    S = np.einsum('nkd,njd->nkj', En, En)
    X = np.zeros((N, K, K, len(PAIR_NAMES)), np.float32)
    X[..., 5] = S
    for g, row in enumerate(bank):
        texts = [c['text'] for c in row['candidates']]
        k = len(texts)
        ans = [parse_answer(t) for t in texts]
        vals = [_val(a) if a is not None else None for a in ans]
        npos = [_numpos(_body(t)) for t in texts]
        nset = [set(v for v, _ in p) for p in npos]
        allset = [F._nums(t) for t in texts]
        for i in range(k):
            for j in range(k):
                f = X[g, i, j]
                f[0] = float(ans[i] is not None and ans[i] == ans[j])
                if vals[j] is not None:
                    pos = [p for v, p in npos[i] if v == vals[j]]
                    f[1], f[2] = float(bool(pos)), (max(pos) if pos else 0.0)
                if vals[i] is not None:
                    f[3] = float(vals[i] in nset[j])
                f[4] = len(allset[i] & allset[j]) / max(len(allset[i] | allset[j]), 1)
                if vals[i] is not None and vals[j] is not None:
                    f[6] = 1.0
                    f[7] = np.clip(np.log10((abs(vals[j]) + 1) / (abs(vals[i]) + 1)), -3, 3)
                    f[8] = float(vals[j] > vals[i])
                f[9] = np.log((len(texts[j]) + 1) / (len(texts[i]) + 1))
    return X


REL_NAMES = ['mul_extra', 'mul_missing', 'add_extra', 'add_missing', 'is_max', 'is_min']


def relation_features(bank):
    """(N,K,6): is my answer another answer times / plus a question number (extra step) or vice versa (missing
    step); is it the largest / smallest distinct answer."""
    N, K = len(bank), max(len(r['candidates']) for r in bank)
    X = np.zeros((N, K, len(REL_NAMES)), np.float32)
    close = lambda x, y: abs(x - y) <= 1e-9 * max(1.0, abs(y))
    for g, row in enumerate(bank):
        qn = {q for q in _qnums_words(row['question']) if q not in (0.0, 1.0)}
        vals = [_val(parse_answer(c['text'])) if parse_answer(c['text']) is not None else None for c in row['candidates']]
        dv = sorted({v for v in vals if v is not None})
        for i, a in enumerate(vals):
            if a is None:
                continue
            f = X[g, i]
            for b in dv:
                if b == a:
                    continue
                if b != 0 and any(close(a / b, q) for q in qn):
                    f[0] = 1
                if a != 0 and any(close(b / a, q) for q in qn):
                    f[1] = 1
                if any(close(a - b, q) for q in qn):
                    f[2] = 1
                if any(close(b - a, q) for q in qn):
                    f[3] = 1
            f[4], f[5] = float(a == dv[-1]), float(a == dv[0])
    return X


PAIR2_NAMES = ['num_prefix', 'div_mine_in_q', 'div_theirs_in_q', 'div_mine_in_them', 'div_theirs_in_me', 'q_cov_diff']


def _numseq(text):
    out = []
    for m in NUMPOS_RE.finditer(text):
        try:
            out.append(float(m.group().replace(',', '')))
        except ValueError:
            pass
    return out


def pair_features2(bank):
    """(N,K,K,6) where the number chains of i and j first diverge: common-prefix fraction of the ordered
    number sequences (question restatement included), whether i's / j's first diverging number is a question
    number, whether it occurs anywhere in the other trace, and the question-coverage difference."""
    N, K = len(bank), max(len(r['candidates']) for r in bank)
    X = np.zeros((N, K, K, len(PAIR2_NAMES)), np.float32)
    for g, row in enumerate(bank):
        qn = _qnums_words(row['question'])
        texts = [c['text'] for c in row['candidates']]
        seqs = [_numseq(t) for t in texts]
        sets = [set(s) for s in seqs]
        cov = [len(qn & s) / len(qn) if qn else 1.0 for s in sets]
        for i in range(len(texts)):
            for j in range(len(texts)):
                if i == j:
                    continue
                a, b = seqs[i], seqs[j]
                p = 0
                while p < min(len(a), len(b)) and a[p] == b[p]:
                    p += 1
                f = X[g, i, j]
                f[0] = p / max(min(len(a), len(b)), 1)
                if p < len(a):
                    f[1], f[3] = float(a[p] in qn), float(a[p] in sets[j])
                if p < len(b):
                    f[2], f[4] = float(b[p] in qn), float(b[p] in sets[i])
                f[5] = cov[i] - cov[j]
    return X
