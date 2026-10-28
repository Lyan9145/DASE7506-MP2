"""Arithmetic-repair features (v4). Idea credited to github.com/Devin-Lo1/dase-7506-mp2 (repair90.py); this code is
an independent implementation. numpy/regex/ast only; candidate text is parsed, never executed.

Walk a trace's `lhs = rhs` equations in order and recompute each lhs with earlier corrections substituted
(a wrong intermediate result is replaced by its corrected value wherever it is reused). Every mismatch records
(written value -> corrected value). If the final answer is a written value that got corrected, the trace's
'repaired answer' is the corrected value: what the same method gives with correct arithmetic.
Cross-candidate evidence: other traces whose repaired answer equals my answer (support for me), and whether my
own repaired answer equals some candidate's answer (I am a slipped copy of it).
repair_features(bank) -> (N,K,len(REPAIR_NAMES)) float32."""
import ast
import math
import operator
import re
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import parse_answer  # noqa: E402

REPAIR_NAMES = ['n_checked', 'n_bad', 'bad_frac', 'max_err', 'n_fix', 'n_propagated', 'ans_repaired', 'log_fix_gap',
                'support_oth', 'support_any', 'own_rep_hits_cand', 'own_rep_hits_oth', 'repaired_vote_share']
OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}
SPAN = re.compile(r'[\d.()+*/\-\s]+')
MAX_FIX = 12


def near(a, b):
    return abs(a - b) <= 1e-6 + 3e-3 * max(abs(a), abs(b))


def _norm(t):
    t = t.replace(',', '').replace('−', '-').replace('×', '*').replace('÷', '/')
    t = re.sub(r'\\(?:text|mathrm)\{[^{}]*\}', ' ', t)
    t = re.sub(r'\\(?:left|right|displaystyle|,|;|!)', '', t)
    for _ in range(6):
        u = re.sub(r'\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}', r'((\1)/(\2))', t)
        u = re.sub(r'\\(?:boxed|mathbf)\s*\{([^{}]*)\}', r'(\1)', u)
        if u == t:
            break
        t = u
    t = t.replace('\\times', '*').replace('\\cdot', '*').replace('\\div', '/').replace('\\$', '').replace('$', ' ')
    t = re.sub(r'(?<=\d)\s*\\?%', '/100', t)
    t = re.sub(r'\*\*(?=[A-Za-z])|(?<=[A-Za-z])\*\*', '', t)  # markdown bold, not a power
    return t.replace('{', '(').replace('}', ')')


def _ev(expr, fix):
    """(written value, value with earlier corrections substituted at every sub-result) or None.
    AST walk over numeric constants and + - * / only."""
    try:
        tree = ast.parse(expr.strip(), mode='eval')
    except (SyntaxError, ValueError):
        return None
    if sum(1 for _ in ast.walk(tree)) > 48:
        return None

    def go(n):
        if isinstance(n, ast.Expression):
            return go(n.body)
        if isinstance(n, ast.Constant) and type(n.value) in (int, float):
            o = p = float(n.value)
        elif isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            o, p = go(n.operand)
            o, p = -o, -p
        elif isinstance(n, ast.BinOp) and type(n.op) in OPS:
            (a, pa), (b, pb) = go(n.left), go(n.right)
            o, p = OPS[type(n.op)](a, b), OPS[type(n.op)](pa, pb)
        else:
            raise ValueError
        if not (math.isfinite(o) and math.isfinite(p)) or abs(p) > 1e15:
            raise ValueError
        for w, c in reversed(fix):
            if near(o, w):
                p = c
                break
        return o, p
    try:
        return go(tree)
    except (ValueError, ZeroDivisionError, OverflowError, RecursionError, TypeError):
        return None


def _side(frag, left):
    """Arithmetic run adjacent to '=': last run of the left fragment / first run of the right one."""
    sp = list(SPAN.finditer(frag))
    for s in (reversed(sp) if left else sp):
        e = s.group().strip().strip('*').strip()
        if not re.search(r'\d', e):
            continue
        tries = [e] + ([e[i:] for i, ch in enumerate(e) if ch.isdigit()][:8] if left else [e.rstrip('.').strip()])
        for x in tries:
            if _ev(x, []) is not None:
                return x, bool(re.search(r'[+*/]|\d\s*-', x))
        return None
    return None


def _val(a):
    try:
        return float(Fraction(a))
    except Exception:
        return None


def trace_repair(text, ans):
    """(stats list, repaired answer or None) for one candidate."""
    t = _norm(text)
    pieces = t.splitlines() + [m.group(1).replace('\n', ' ') for m in re.finditer(r'\\\[(.*?)\\\]', t, re.S)]
    fix, seen, errs, prop = [], set(), [], 0
    for piece in pieces:
        terms = piece.split('=')
        for l, r in zip(terms, terms[1:]):
            L, R = _side(l, True), _side(r, False)
            if L is None or R is None or not (L[1] or R[1]):
                continue
            key = (re.sub(r'\s', '', L[0]), re.sub(r'\s', '', R[0]))
            if key in seen:
                continue
            seen.add(key)
            lv, rv = _ev(L[0], fix), _ev(R[0], [])
            if lv is None or rv is None:
                continue
            errs.append(abs(lv[0] - rv[0]) / max(1.0, abs(lv[0]), abs(rv[0])))
            if not near(lv[1], rv[0]) and len(fix) < MAX_FIX:
                prop += near(lv[0], rv[0])  # locally correct step, wrong only through an earlier slip
                fix.append((rv[0], lv[1]))
    v = _val(ans) if ans is not None else None
    target = None
    if v is not None:
        for w, c in reversed(fix):
            if near(v, w) and not near(v, c):
                target = c
                break
    bad = sum(e > 5e-3 for e in errs)
    stats = [math.log1p(len(errs)), math.log1p(bad), bad / max(len(errs), 1), min(max(errs, default=0.0), 1.0),
             math.log1p(len(fix)), math.log1p(prop), float(target is not None),
             math.log1p(min(abs(target - v), 1e6)) if target is not None else 0.0]
    return stats, target


def repair_features(bank):
    N, K = len(bank), max(len(r['candidates']) for r in bank)
    X = np.zeros((N, K, len(REPAIR_NAMES)), np.float32)
    for g, row in enumerate(bank):
        texts = [c['text'] for c in row['candidates']]
        ans = [parse_answer(t) for t in texts]
        vals = [_val(a) if a is not None else None for a in ans]
        info = [trace_repair(t, a) for t, a in zip(texts, ans)]
        tg = [x[1] for x in info]
        k = len(texts)
        for i in range(k):
            sup = [j for j in range(k) if vals[i] is not None and tg[j] is not None and near(tg[j], vals[i])]
            own = tg[i]
            hits = [j for j in range(k) if own is not None and vals[j] is not None and near(own, vals[j])]
            X[g, i, :8] = info[i][0]
            X[g, i, 8] = sum(j != i for j in sup)
            X[g, i, 9] = float(bool(sup))
            X[g, i, 10] = float(bool(hits))
            X[g, i, 11] = sum(j != i for j in hits)
        # vote share after repair: every trace votes for its repaired answer if it has one, else its own answer
        votes = [tg[j] if tg[j] is not None else vals[j] for j in range(k)]
        for i in range(k):
            if vals[i] is not None:
                X[g, i, 12] = sum(v is not None and near(v, vals[i]) for v in votes) / k
    return X
