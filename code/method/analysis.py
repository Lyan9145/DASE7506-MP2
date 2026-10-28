"""Analysis experiments for the report (validation + train-CV only; the test split is never touched).
Run from code/:  .venv/bin/python method/analysis.py [exp123|exp46|exp5|report]   (default: all)
Part results are cached in $AN_CACHE (json); `report` merges them into analysis.json / analysis_results.md / figures.
Frozen method files are only imported, never modified."""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import nn_train as T  # noqa: E402
import selector as S  # noqa: E402
import features as F  # noqa: E402
import cv as CV  # noqa: E402
from train import acc  # noqa: E402

AN = Path(os.environ.get('AN_CACHE', '/tmp/claude-0/-root-DASE7506-MP2/24ec1cfd-5c48-4fdf-8a0d-54e05b389870/scratchpad/an'))
AN.mkdir(parents=True, exist_ok=True)
H = {n: i for i, n in enumerate(F.HAND_NAMES)}
NH = len(F.HAND_NAMES)           # 37 hand, then 37 group-demeaned, then 4 cross-embedding scalars
CROSS = list(range(2 * NH, 2 * NH + 4))
COS = dict(zip(F.CROSS_NAMES, CROSS))

G = dict(
    vote=['cnt', 'share', 'ndist', 'is_top', 'ntop'],
    ans=['valid', 'is_int', 'neg', 'logmag', 'logmag_rel', 'ans_is_qnum', 'ans_rank'],
    surf=['loglen', 'len_rel', 'len_rank', 'is_longest', 'is_shortest', 'has_boxed', 'has_hash', 'placeholder', 'agree',
          'n_lines', 'rep_lines', 'trunc', 'n_final', 'tail_after', 'cl_len_rel', 'hedge'],
    reas=['ans_repeat', 'n_eq', 'eq_frac', 'eq_wrong', 'eq_ok', 'q_used', 'q_used_n', 'cl_eq_wrong', 'cl_eq_frac'],
)
assert sorted(sum(G.values(), [])) == sorted(F.HAND_NAMES)
LEN = ['loglen', 'len_rel', 'len_rank', 'is_longest', 'is_shortest', 'cl_len_rel']
FMT = ['has_boxed', 'has_hash', 'placeholder', 'agree', 'n_final', 'tail_after', 'trunc', 'n_lines', 'rep_lines']
CLUSTER = ['cnt', 'share', 'ndist', 'is_top', 'ntop', 'cl_len_rel', 'cl_eq_wrong', 'cl_eq_frac']
GROUP_REL = ['ndist', 'ntop', 'len_rel', 'len_rank', 'is_longest', 'is_shortest', 'logmag_rel', 'ans_rank']  # inherently cross-cand


def cols(names, rel=True):
    c = [H[n] for n in names]
    return c + [NH + i for i in c] if rel else c


def keep_only(names):
    k = set(cols(names))
    return [i for i in range(2 * NH + 4) if i not in k]


def drop(names, cross=()):
    return sorted(set(cols(names)) | set(cross))


# ---- specs: (dropped input columns, embedding mode, deepsets/none, decode handled at eval) -----------------------------
def spec(drop_cols=(), emb='raw', inter='deepsets'):
    return dict(drop=sorted(set(drop_cols)), emb=emb, inter=inter)


ALLCOL = list(range(2 * NH + 4))
NOCROSS_EMB = CROSS
S1 = {  # exp 1: information source (cumulative), drop everything not listed
    '1a vote only': spec(keep_only(G['vote']), 'none'),
    '1b + answer-level': spec(keep_only(G['vote'] + G['ans']), 'none'),
    '1c + surface/format': spec(keep_only(G['vote'] + G['ans'] + G['surf']), 'none'),
    '1d + reasoning content': spec(CROSS, 'none'),
    '1e + embeddings (= final arch)': spec((), 'raw'),
    '1f embeddings + vote only': spec(keep_only(G['vote']) + [], 'raw'),
    '1g embeddings only (no hand, no vote)': spec(ALLCOL, 'raw'),
    '1h full - answer-level': spec(drop(G['ans']), 'raw'),
    '1i full - surface/format': spec(drop(G['surf']), 'raw'),
    '1j full - reasoning content': spec(drop(G['reas']), 'raw'),
}
LF = LEN + FMT
S2 = {  # exp 2: shortcut / confound
    '2a full (ref)': spec((), 'raw'),
    '2b full - length/format feats': spec(drop(LF), 'raw'),
    '2c no-emb hand - length/format': spec(drop(LF) + CROSS, 'none'),
    '2d vote+emb raw': S1['1f embeddings + vote only'],
    '2e vote+emb length-residualized': spec(keep_only(G['vote']), 'resid'),
    '2f emb only raw': S1['1g embeddings only (no hand, no vote)'],
    '2g emb only length-residualized': spec(ALLCOL, 'resid'),
    '2h full hand + resid emb': spec((), 'resid'),
    '2i full - length/format + resid emb': spec(drop(LF), 'resid'),
}
NOVOTE = drop(CLUSTER, [COS['cos_same'], COS['cos_same_n']])
STRICT = sorted(set(NOVOTE) | set(range(NH, 2 * NH)) | set(cols(GROUP_REL, False)) | set(CROSS))
S3 = {}
for it in ('none', 'deepsets'):
    S3[f'3 {it} full'] = spec((), 'raw', it)
    S3[f'3 {it} no vote/cluster feats'] = spec(NOVOTE, 'raw', it)
    S3[f'3 {it} no cross-cand info'] = spec(STRICT, 'raw', it)
ALLSPEC = {**S1, **S2, **S3}

RES_COLS = ['loglen', 'has_boxed', 'has_hash', 'placeholder', 'trunc', 'n_lines']


def design_resid(xt_raw, et, fit_mask):
    """Residualize standardized embeddings on [1, loglen, loglen^2, format flags], fit on train candidates of fit_mask."""
    def Z(x):
        z = [x[..., H[n]] for n in RES_COLS]
        return np.concatenate([np.ones(x.shape[:2] + (1,)), np.stack(z + [x[..., H['loglen']] ** 2], -1)], -1)
    Zt = Z(xt_raw)
    f = fit_mask
    mu, sd = et[f].reshape(-1, 1024).mean(0), et[f].reshape(-1, 1024).std(0) + 1e-6
    Zf = Zt[f].reshape(-1, Zt.shape[-1]); Ef = ((et[f] - mu) / sd).reshape(-1, 1024)
    B = np.linalg.lstsq(Zf, Ef, rcond=None)[0]
    return lambda x, e: (((e - mu) / sd) - Z(x) @ B).astype(np.float32), (1 - ((Ef - Zf @ B) ** 2).sum() / (Ef ** 2).sum())


def transform(d, sp, pool=None):
    """Return d' with input columns dropped (zeroed) / embedding mode applied. pool = train-group mask used for fitting."""
    d2 = dict(d)
    keep = np.ones(d['xt'].shape[-1], bool); keep[sp['drop']] = False
    for k in ('xt', 'xv'):
        d2[k] = d[k] * keep
    if sp['emb'] == 'resid':
        f = (pool if pool is not None else np.ones(len(d['xt']), bool)) & d['ht']
        fn, r2 = design_resid(d['xt'], d['et'], f)
        d2['et'] = fn(d['xt'], d['et']); d2['ev'] = fn(d['xv'], d['ev'])
        d2['_r2'] = float(r2)
    return d2


def cfg_of(sp):
    return dict(T.BASE, inter=sp['inter'], emb=sp['emb'] != 'none')


def wilson(k, n, z=1.96):
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


def choose(bank, P, dec):
    return np.array(S.aggregate_choose(bank, P, 'sum' if dec == 'sum' else None))


def run_spec(name, sp, tb, vb, d, fold, vseeds, cseeds):
    out = dict(spec={k: (v if k != 'drop' else len(v)) for k, v in sp.items()})
    cfg = cfg_of(sp)
    Y, hv, ht = d['Yv'], d['hv'], d['ht']
    t0 = time.time()
    dv = transform(d, sp)
    Pv = []
    for s in vseeds:
        net, norm, p, ep = T.train_one(cfg, dv, s)
        Pv.append(p)
    out['sec_per_member'] = (time.time() - t0) / len(vseeds)
    for dec in ('sum', 'argmax'):
        a = [acc(choose(vb, p, dec), Y, hv) for p in Pv]
        ens = acc(choose(vb, np.mean(Pv, 0), dec), Y, hv)
        corr = int(round(np.mean(a) / 100 * hv.sum()))
        out[f'val_{dec}'] = dict(mean=float(np.mean(a)), sd=float(np.std(a)), seeds=a, ens=float(ens),
                                 ci=wilson(corr, int(hv.sum())), all_mean=float(np.mean([acc(choose(vb, p, dec), Y) for p in Pv])))
    # CV (pooled hard accuracy over the 5 held-out folds, mean over cseeds)
    Pc = []
    for s in cseeds:
        P = np.zeros(d['Yt'].shape, np.float32)
        for f in range(CV.NF):
            te = fold == f
            dd = transform(d, sp, ~te)
            dd = dict(dd, xv=dd['xt'][te], ev=dd['et'][te], mv=d['mt'][te])
            net, norm, p, ep = T.train_one(cfg, dd, s * 100 + f, pool=~te)
            P[te] = p
        Pc.append(P)
    hi = np.where(ht)[0]; hb = [tb[i] for i in hi]
    for dec in ('sum', 'argmax'):
        a = [acc(choose(hb, P[hi], dec), d['Yt'][hi]) for P in Pc]
        fa = [acc(choose([tb[i] for i in np.where((fold == f) & ht)[0]], Pc[0][(fold == f) & ht], dec), d['Yt'][(fold == f) & ht])
              for f in range(CV.NF)]
        out[f'cv_{dec}'] = dict(mean=float(np.mean(a)), sd=float(np.std(a)), seeds=a, fold_sd=float(np.std(fa)))
    if sp['emb'] == 'resid':
        out['emb_r2_explained_by_len_fmt'] = dv['_r2']
    out['sec'] = time.time() - t0
    return out


def exp123(vseeds=(0, 1, 2), cseeds=(0, 1)):
    fn = AN / 'exp123.json'
    res = json.loads(fn.read_text()) if fn.exists() else {}
    tb, vb, d = T.load_all()
    fold = CV.folds(len(tb))
    for name, sp in ALLSPEC.items():
        if name in res:
            continue
        res[name] = run_spec(name, sp, tb, vb, d, fold, vseeds, cseeds)
        r = res[name]
        print(f"{name:42s} val(sum) {r['val_sum']['mean']:.2f} val(arg) {r['val_argmax']['mean']:.2f} cv(sum) {r['cv_sum']['mean']:.2f} "
              f"cv(arg) {r['cv_argmax']['mean']:.2f}  {r['sec']:.0f}s", flush=True)
        fn.write_text(json.dumps(res, indent=1))



# ============================================================ exp 4 / 2c / 2d / 6 : analyses of the frozen final model
import nn_infer  # noqa: E402
import baseline_vote  # noqa: E402
from common import load_inputs, parse_answer  # noqa: E402

CK = HERE.parent / 'checkpoint_v2'
CK_LR = HERE.parent / 'checkpoint_lr'


def cluster_table(row, p):
    """[(score, best_idx, answer-or-None)] using the final decode (answer-cluster probability mass; unparseable alone)."""
    ans = [parse_answer(c['text']) for c in row['candidates']]
    out = []
    for a in set(x for x in ans if x is not None):
        mem = [j for j in range(len(ans)) if ans[j] == a]
        out.append((float(p[mem].sum()), max(mem, key=lambda j: p[j]), a))
    out += [(float(p[j]), j, None) for j, a in enumerate(ans) if a is None]
    return sorted(out, key=lambda t: -t[0])


def conf_margin(bank, P):
    conf, marg, ch = [], [], []
    for row, p in zip(bank, P):
        t = cluster_table(row, p)
        conf.append(t[0][0]); marg.append(t[0][0] - (t[1][0] if len(t) > 1 else 0.0)); ch.append(t[0][1])
    return np.array(conf), np.array(marg), np.array(ch)


def ece_bins(conf, corr, nb=10):
    edges = np.linspace(0, 1, nb + 1); idx = np.minimum((conf * nb).astype(int), nb - 1)
    rows, e = [], 0.0
    for b in range(nb):
        m = idx == b
        if m.any():
            rows.append(dict(lo=edges[b], hi=edges[b + 1], n=int(m.sum()), conf=float(conf[m].mean()), acc=float(corr[m].mean())))
            e += m.mean() * abs(conf[m].mean() - corr[m].mean())
    return float(e), rows


def sel_curve(conf, corr):
    o = np.argsort(-conf, kind='stable'); c = corr[o]
    cov = np.arange(1, len(c) + 1) / len(c); a = np.cumsum(c) / np.arange(1, len(c) + 1)
    return cov, a


def stats_len(row):
    t = [c['text'] for c in row['candidates']]
    return dict(L=np.array([len(x) for x in t], float), box=np.array(['\\boxed' in x for x in t], float),
                hs=np.array(['####' in x for x in t], float))


def exp46():
    tb, vb, d = T.load_all()
    _, va = load_inputs('validation')
    E = va['embeddings']
    Y, hv = d['Yv'], d['hv']
    hi = np.where(hv)[0]; hb = [vb[i] for i in hi]; Yh = Y[hi]
    P = nn_infer.probs(vb, E, CK)
    ch = np.array(nn_infer.S.aggregate_choose(vb, P, 'sum'))
    conf, marg, ch2 = conf_margin(vb, P)
    assert (ch == ch2).mean() > 0.995, (ch == ch2).mean()
    corr = Y[np.arange(len(Y)), ch].astype(float)
    ans_of = lambda bank, c: [parse_answer(r['candidates'][i]['text']) for r, i in zip(bank, c)]
    import importlib.util
    spec_ = importlib.util.spec_from_file_location('sublr', HERE.parent / 'submission_lr.py'); m = importlib.util.module_from_spec(spec_); spec_.loader.exec_module(m)
    ch_lr = np.array(m.predict(vb, va, CK_LR)); ch_v = np.array(baseline_vote.predict(vb, va))
    out = {}
    n = len(hi)
    out['val_hard'] = {k: dict(acc=acc(c, Y, hv), correct=int(Y[np.arange(len(Y)), c][hv].sum()), ci=wilson(int(Y[np.arange(len(Y)), c][hv].sum()), n))
                       for k, c in dict(final=ch, lr=ch_lr, vote=ch_v).items()}
    out['n_val_hard'] = int(n)
    # ---- exp 4 calibration (val-hard) + selective accuracy
    c_h, k_h = conf[hi], corr[hi]
    e, rows = ece_bins(c_h, k_h)
    out['cal_val'] = dict(ece=e, bins=rows, mean_conf=float(c_h.mean()), acc=float(k_h.mean()), brier=float(((c_h - k_h) ** 2).mean()))
    # CV calibration on held-out train-hard groups (deepsets_v20 seeds 0-4 mean, honest out-of-fold)
    Pc = np.mean([np.load(CV.CACHE / f'deepsets_v20_s{s}.npy') for s in range(5)], 0)
    hit = np.where(d['ht'])[0]; cc, _, chc = conf_margin([tb[i] for i in hit], Pc[hit])
    kc = d['Yt'][hit][np.arange(len(hit)), chc].astype(float)
    e2, rows2 = ece_bins(cc, kc)
    out['cal_cv'] = dict(ece=e2, bins=rows2, mean_conf=float(cc.mean()), acc=float(kc.mean()), n=int(len(hit)))
    cov, a = sel_curve(c_h, k_h)
    out['sel_val'] = {f'{int(q*100)}%': float(a[int(np.ceil(q * n)) - 1]) for q in (0.3, 0.5, 0.7, 0.8, 0.9, 1.0)}
    out['sel_val_aurc_acc'] = float(a.mean())
    covc, ac = sel_curve(cc, kc)
    out['sel_cv'] = {f'{int(q*100)}%': float(ac[int(np.ceil(q * len(kc))) - 1]) for q in (0.3, 0.5, 0.7, 0.8, 0.9, 1.0)}
    out['curves'] = dict(val_cov=cov.tolist(), val_acc=a.tolist(), cv_cov=covc[::10].tolist(), cv_acc=ac[::10].tolist())
    mb = [(0, .1), (.1, .3), (.3, .5), (.5, .7), (.7, .9), (.9, 1.01)]
    out['margin_bins'] = [dict(lo=lo, hi=min(hi_, 1.0), n=int(((marg[hi] >= lo) & (marg[hi] < hi_)).sum()),
                               acc=float(k_h[(marg[hi] >= lo) & (marg[hi] < hi_)].mean()) if ((marg[hi] >= lo) & (marg[hi] < hi_)).any() else None)
                          for lo, hi_ in mb]
    # agreement of final / LR / vote (by chosen answer)
    aN, aL, aV = ans_of(vb, ch), ans_of(vb, ch_lr), ans_of(vb, ch_v)
    cats = {}
    for g in hi:
        nl, nv, lv = aN[g] == aL[g], aN[g] == aV[g], aL[g] == aV[g]
        key = 'all agree' if nl and nv else 'final=LR != vote' if nl else 'final=vote != LR' if nv else 'LR=vote != final' if lv else 'all differ'
        cats.setdefault(key, []).append(g)
    def r3(c, g): return int(Y[g, c[g]])
    out['agree'] = {k: dict(n=len(v), final=float(np.mean([r3(ch, g) for g in v]) * 100), lr=float(np.mean([r3(ch_lr, g) for g in v]) * 100),
                            vote=float(np.mean([r3(ch_v, g) for g in v]) * 100)) for k, v in cats.items()}
    dis = [g for k, v in cats.items() if k != 'all agree' for g in v]
    out['agree_vs_disagree'] = dict(agree_n=len(cats.get('all agree', [])), disagree_n=len(dis),
                                    final_agree=out['agree'].get('all agree', {}).get('final'),
                                    final_disagree=float(np.mean([r3(ch, g) for g in dis]) * 100))
    # ---- exp 2c: length / format stratification
    L = {g: stats_len(vb[g]) for g in hi}
    corr_any = lambda g: Y[g]
    strata = {}
    for g in hi:
        key = ('longest correct' if Y[g][int(np.argmax(L[g]['L']))] else 'longest wrong') + ' / ' + ('shortest correct' if Y[g][int(np.argmin(L[g]['L']))] else 'shortest wrong')
        strata.setdefault(key, []).append(g)
    out['strata_len'] = {k: dict(n=len(v), final=float(np.mean([r3(ch, g) for g in v]) * 100), lr=float(np.mean([r3(ch_lr, g) for g in v]) * 100),
                                 vote=float(np.mean([r3(ch_v, g) for g in v]) * 100),
                                 pick_longest=float(np.mean([int(np.argmax(L[g]['L'])) == ch[g] for g in v]) * 100)) for k, v in sorted(strata.items())}
    out['always_longest'] = float(np.mean([Y[g][int(np.argmax(L[g]['L']))] for g in hi]) * 100)
    out['always_shortest'] = float(np.mean([Y[g][int(np.argmin(L[g]['L']))] for g in hi]) * 100)
    out['final_pick_longest_rate'] = float(np.mean([int(np.argmax(L[g]['L'])) == ch[g] for g in hi]) * 100)
    out['final_pick_shortest_rate'] = float(np.mean([int(np.argmin(L[g]['L'])) == ch[g] for g in hi]) * 100)
    # wrong picks vs correct alternatives
    wrong = [g for g in hi if not Y[g, ch[g]]]
    right = [g for g in hi if Y[g, ch[g]]]
    def cmp(gs):
        lr_, bx, hs, lon = [], [], [], []
        for g in gs:
            c, o = ch[g], Y[g].astype(bool)
            alt = np.where(o)[0] if not o[c] else np.where(~o)[0]       # wrong pick -> correct alternatives; right pick -> rejected wrong
            alt = alt[alt != c]
            if len(alt) == 0: continue
            lr_.append(np.log(L[g]['L'][c] / L[g]['L'][alt].mean())); bx.append(L[g]['box'][c] - L[g]['box'][alt].mean())
            hs.append(L[g]['hs'][c] - L[g]['hs'][alt].mean()); lon.append(L[g]['L'][c] > L[g]['L'][alt].max())
        return dict(n=len(lr_), mean_log_len_ratio=float(np.mean(lr_)), frac_longer_than_all_alts=float(np.mean(lon)),
                    boxed_diff=float(np.mean(bx)), hash_diff=float(np.mean(hs)))
    out['wrong_vs_correct_alt'] = cmp(wrong)
    out['right_vs_rejected'] = cmp(right)
    ctx = []
    for g in hi:
        o = Y[g].astype(bool)
        ctx.append((np.log(L[g]['L'][o].mean() / L[g]['L'][~o].mean()), L[g]['box'][o].mean() - L[g]['box'][~o].mean(), L[g]['hs'][o].mean() - L[g]['hs'][~o].mean()))
    out['all_groups_correct_minus_wrong'] = dict(mean_log_len_ratio=float(np.mean([c[0] for c in ctx])), boxed_diff=float(np.mean([c[1] for c in ctx])),
                                                 hash_diff=float(np.mean([c[2] for c in ctx])))
    # ---- exp 2d: minority picks, transitions vs vote
    def counts(g):
        a = [parse_answer(c['text']) for c in vb[g]['candidates']]
        cnt = {}
        for x in a:
            if x is not None: cnt[x] = cnt.get(x, 0) + 1
        return a, cnt
    mino, plur, tie_top = [], [], []
    for g in hi:
        a, cnt = counts(g); top = max(cnt.values()); ca = a[ch[g]]
        (mino if ca is None or cnt[ca] < top else tie_top if sum(v == top for v in cnt.values()) > 1 else plur).append(g)
    f = lambda gs: dict(n=len(gs), right=int(sum(Y[g, ch[g]] for g in gs)), wrong=int(sum(1 - Y[g, ch[g]] for g in gs)),
                        acc=float(np.mean([Y[g, ch[g]] for g in gs]) * 100) if gs else None,
                        vote_acc=float(np.mean([Y[g, ch_v[g]] for g in gs]) * 100) if gs else None)
    out['pick_type'] = {'strict plurality answer': f(plur), 'one of tied top answers': f(tie_top), 'minority/unparseable (below top count)': f(mino)}
    tr = lambda a_, b_: int(sum(bool(Y[g, ch[g]]) == a_ and bool(Y[g, ch_v[g]]) == b_ for g in hi))
    out['transitions_vs_vote'] = {'both right': tr(True, True), 'final right, vote wrong': tr(True, False),
                                  'final wrong, vote right': tr(False, True), 'both wrong': tr(False, False)}
    # vote ties: which groups have tie at top; accuracy on tie vs non-tie groups
    ties = [g for g in hi if sum(v == max(counts(g)[1].values()) for v in counts(g)[1].values()) > 1]
    nt = [g for g in hi if g not in set(ties)]
    out['tie_groups'] = dict(n_tie=len(ties), final_tie=float(np.mean([Y[g, ch[g]] for g in ties]) * 100), vote_tie=float(np.mean([Y[g, ch_v[g]] for g in ties]) * 100),
                             n_notie=len(nt), final_notie=float(np.mean([Y[g, ch[g]] for g in nt]) * 100), vote_notie=float(np.mean([Y[g, ch_v[g]] for g in nt]) * 100))
    # ---- exp 6: automatic error categories
    gold = [parse_answer(r['answer']) for r in __import__('common').read_jsonl(__import__('common').DATA / 'gsm8k_validation.jsonl')]
    errs, cat_cnt = [], {}
    for g in wrong:
        a, cnt = counts(g); pat = tuple(sorted(cnt.values(), reverse=True)); top = pat[0]; ca = a[ch[g]]
        n_top = sum(v == top for v in pat)
        corr_ans = gold[g]
        if n_top == 1 and ca is not None and cnt[ca] == top:
            cat = f'plurality answer wrong (pattern {"-".join(map(str, pat))})' if top >= 2 else 'x'
        elif n_top == 1 and cnt.get(corr_ans, 0) == top:
            cat = 'overrode correct plurality'
        elif pat == (2, 2):
            cat = '2-2 tie lost'
        elif top == 1:
            cat = 'all answers distinct (no vote signal)'
        else:
            cat = 'picked non-plurality answer'
        cat_cnt.setdefault(cat.split(' (')[0], []).append(g)
        none_has_gold = any(x is None and corr_ans is not None and corr_ans in c['text'][-120:] for x, c in zip(a, vb[g]['candidates']))
        errs.append(dict(g=int(g), id=vb[g]['id'], cat=cat.split(' (')[0], pattern='-'.join(map(str, pat)), n_correct=int(Y[g].sum()), chosen_len=len(vb[g]['candidates'][ch[g]]['text']),
                         n_unparse=int(sum(x is None for x in a)), unparse_has_gold=bool(none_has_gold), conf=float(conf[g])))
    out['error_cats'] = {k: len(v) for k, v in cat_cnt.items()}
    out['n_errors'] = len(wrong)
    out['errors'] = errs
    out['err_patterns'] = {}
    for e_ in errs: out['err_patterns'][e_['pattern']] = out['err_patterns'].get(e_['pattern'], 0) + 1
    out['err_with_unparse'] = int(sum(e_['n_unparse'] > 0 for e_ in errs))
    # distribution of patterns in all hard groups (context)
    pats = {}
    for g in hi:
        _, cnt = counts(g); k = '-'.join(map(str, sorted(cnt.values(), reverse=True))); pats.setdefault(k, [0, 0]); pats[k][0] += 1; pats[k][1] += int(Y[g, ch[g]])
    out['pattern_acc_all_hard'] = {k: dict(n=v[0], right=v[1]) for k, v in pats.items()}
    (AN / 'exp46.json').write_text(json.dumps(out, indent=1))
    print('exp46 done; val-hard', {k: round(v['acc'], 2) for k, v in out['val_hard'].items()}, 'ECE', round(e, 4), round(e2, 4))



# ============================================================ exp 5 : compute vs accuracy
import subprocess  # noqa: E402
import statistics  # noqa: E402
SCR = Path(os.environ.get('AN_SCRATCH', str(AN.parent)))


def exp5(reps=3):
    tb, vb, d = T.load_all()
    Y, hv = d['Yv'], d['hv']
    env = dict(os.environ, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
    code = HERE.parent
    py = sys.executable
    methods = [('majority vote (tie->longest)', 'method/baseline_vote.py', None),
               ('LR (hand+cross+PCA64)', 'submission_lr.py', 'checkpoint_lr'),
               ('DeepSets NN K=1', 'submission_v2.py', str(SCR / 'ckK1')),
               ('DeepSets NN K=3', 'submission_v2.py', str(SCR / 'ckK3')),
               ('DeepSets NN K=5 (final)', 'submission_v2.py', 'checkpoint_v2')]
    cvr = json.load(open(HERE / 'cv_results.json'))
    cv_hard = {'majority vote (tie->longest)': None, 'LR (hand+cross+PCA64)': float(np.mean(cvr['LR v1cfg']['single'])),
               'DeepSets NN K=1': cvr['deepsets_v20']['K1']['mean'], 'DeepSets NN K=3': cvr['deepsets_v20']['K3']['mean'],
               'DeepSets NN K=5 (final)': cvr['deepsets_v20']['K5']['mean']}
    hi = np.where(d['ht'])[0]
    cv_hard['majority vote (tie->longest)'] = acc(np.array(baseline_vote.predict([tb[i] for i in hi], None)), d['Yt'][hi])
    rows = []
    for name, meth, ck in methods:
        walls = []
        for r in range(reps):
            pred, score = AN / 'pred.json', AN / 'score.json'
            t0 = time.perf_counter()
            cmd = [py, 'predict.py', '--method', meth, '--split', 'validation', '--output', str(pred)] + (['--checkpoint', ck] if ck else [])
            subprocess.run(cmd, cwd=code, env=env, check=True, capture_output=True)
            subprocess.run([py, 'score_predictions.py', '--split', 'validation', '--predictions', str(pred), '--output', str(score)], cwd=code, env=env, check=True, capture_output=True)
            walls.append(time.perf_counter() - t0)
        sel = np.array(json.loads(pred.read_text())['selected'])
        sc = json.loads(score.read_text())
        corr = int(Y[np.arange(len(Y)), sel][hv].sum())
        size = sum(f.stat().st_size for f in Path(code / ck if ck and not os.path.isabs(ck) else ck).glob('*')) if ck else 0
        rows.append(dict(method=name, val_all=float(sc['accuracy']), val_hard=100 * corr / hv.sum(), val_hard_ci=wilson(corr, int(hv.sum())),
                         cv_hard=cv_hard[name], wall_median=statistics.median(walls), wall_all=walls, asset_bytes=int(size)))
        print(name, {k: v for k, v in rows[-1].items() if k != 'wall_all'}, flush=True)
    # training cost (this machine, 2 CPU cores)
    tr = {}
    tb_, ta_ = load_inputs('train')
    t0 = time.time(); Btr = S.blocks(tb_, ta_['embeddings']); tr['hand+cross features, train 4.5k groups (s)'] = time.time() - t0
    from train import fit_lr
    t0 = time.time(); prep = S.fit_prep(Btr, 64); X = S.design(Btr, dict(blocks=['hand', 'cross', 'emb'], npca=64, rel=True), prep)
    ck_ = fit_lr(X[d['ht']], d['Yt'][d['ht']], 10.0); tr['LR PCA+design+fit (s)'] = time.time() - t0
    e1 = json.loads((AN / 'exp123.json').read_text()).get('1e + embeddings (= final arch)', {})
    tr['NN sec per member (s, incl. early stop)'] = e1.get('sec_per_member')
    tr['NN K=5 train (s) = 5 x member'] = 5 * e1['sec_per_member'] if e1 else None
    (AN / 'exp5.json').write_text(json.dumps(dict(rows=rows, train=tr), indent=1))



# ============================================================ report: figures + markdown + json
MANUAL = {  # hand-read val-hard errors of the final model (group index in validation bank) -> cause
    9: 'coin-flip 2-2 tie', 362: 'coin-flip 2-2 tie',
    44: 'shared misreading (wrong cluster >=2 traces)', 52: 'shared misreading (wrong cluster >=2 traces)', 81: 'shared misreading (wrong cluster >=2 traces)',
    318: 'shared misreading (wrong cluster >=2 traces)', 328: 'shared misreading (wrong cluster >=2 traces)', 331: 'shared misreading (wrong cluster >=2 traces)',
    497: 'shared misreading (wrong cluster >=2 traces)', 586: 'shared misreading (wrong cluster >=2 traces)', 617: 'shared misreading (wrong cluster >=2 traces)',
    133: 'no vote signal, hard problem (all answers distinct)', 210: 'no vote signal, hard problem (all answers distinct)', 338: 'no vote signal, hard problem (all answers distinct)',
    397: 'no vote signal, hard problem (all answers distinct)', 550: 'no vote signal, hard problem (all answers distinct)',
    240: 'correct trace lost to format (unparseable / near-tie)', 444: 'correct trace lost to format (unparseable / near-tie)',
    111: 'confident singleton overrides correct plurality', 203: 'confident singleton overrides correct plurality',
}
EXCERPTS = [(444, [0, 1], 'correct answer 40 only in traces that end without ####/boxed (unparseable); chosen 45 at p=0.94'),
            (586, [2, 3], 'three traces share the misreading (all 250-70); the single correct trace (40) gets p=0.06')]


def _tail(text, k=3, w=110):
    ls = [l.strip() for l in text.strip().split('\n') if l.strip() not in ('', '$$', '---', '$$.')]
    return ' / '.join(l[:w] for l in ls[-k:])


def fmt(m, s=None, d=1):
    return f'{m:.{d}f}' if s is None else f'{m:.{d}f}±{s:.{d}f}'


def report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 120, 'pdf.fonttype': 42})
    C = dict(blue='#0072B2', orange='#E69F00', green='#009E73', verm='#D55E00', grey='#777777', sky='#56B4E9')
    FIG = HERE.parent.parent / 'report' / 'figures'; FIG.mkdir(parents=True, exist_ok=True)
    def save(fig, name):
        fig.savefig(FIG / f'{name}.pdf', bbox_inches='tight'); fig.savefig(FIG / f'{name}.png', bbox_inches='tight', dpi=200); plt.close(fig)
    r123 = json.loads((AN / 'exp123.json').read_text()); r46 = json.loads((AN / 'exp46.json').read_text()); r5 = json.loads((AN / 'exp5.json').read_text())
    nh = r46['n_val_hard']
    L = []
    w = L.append
    w('# Analysis results (generated by `method/analysis.py`; validation + train-CV only, test split never read)\n')
    w('Protocol. Val-hard = 355 validation groups with >=2 distinct valid answers (1 group = 0.28 pt; Wilson 95% CI on a single accuracy is about +-4.5 pt, so '
      'differences below ~3 pt are not resolvable on val alone). CV-hard = pooled accuracy on the 2114 hard train groups, 5 group-folds (seed 0), models trained on the other '
      'folds only (early-stopping holdout inside the training pool); fold-to-fold sd is 1.5-2.5 pt, pooled-CV standard error ~0.9 pt. '
      'Exp 1-3 re-train single NNs (DeepSets unless stated; H=128, same hyper-parameters as the final model, hard-train only): val = mean+-sd over 3 seeds (0-2), '
      'CV = mean over 2 seeds (0,1) (seed sd ~0.2-0.5 pt). "sum" = answer-cluster-sum decode, "arg" = plain argmax of candidate score. '
      'Feature ablations zero the dropped input columns (and their group-demeaned copies); "emb none" removes the 1024-d embedding branch. '
      'Final-model rows (exp 4-6) use the frozen K=5 ensemble.\n')
    # ---- exp1
    def row(name, r, dec='sum'):
        v, c = r[f'val_{dec}'], r[f'cv_{dec}']
        return f"| {name} | {fmt(v['mean'], v['sd'])} | [{v['ci'][0]:.1f}, {v['ci'][1]:.1f}] | {fmt(c['mean'])} |"
    w('## Exp 1. Information-source ablation ("does reading the reasoning help?")\n')
    w('| feature set (cumulative) | val-hard (3 seeds) | 95% CI (Wilson) | CV-hard |\n|---|---|---|---|')
    for k in [k for k in r123 if k.startswith('1')]:
        w(row(k[3:], r123[k]))
    w('\nBaselines: vote tie->longest val-hard 70.7 (CI [65.8, 75.2]), final K=5 ensemble 84.8. 1a uses cluster-sum decode on vote features only: ties are broken by index, hence below the tie->longest heuristic.\n')
    w('Rows 1h-1j drop one group from the full model (leave-one-group-out). Groups: vote = cnt/share/ndist/is_top/ntop; answer-level = parse validity, is_int, negative, magnitude, '
      'rank among answers, answer-is-a-question-number; surface/format = length (7 feats), boxed/####/placeholder, truncation, #lines, "final answer" count, hedging words; '
      'reasoning content = arithmetic-check counts/fraction, question-number coverage, answer repeated in body, cluster-level arithmetic stats.\n')
    # ---- exp2
    w('## Exp 2. Shortcut / confound ablation ("what makes a wrong answer convincing?")\n')
    w('| config | val-hard sum | CI | CV-hard sum | val-hard arg | CV-hard arg |\n|---|---|---|---|---|---|')
    for k in [k for k in r123 if k.startswith('2')]:
        r = r123[k]
        extra = f" (emb R2 on len/fmt = {r['emb_r2_explained_by_len_fmt']:.3f})" if 'emb_r2_explained_by_len_fmt' in r else ''
        w(f"| {k[3:]}{extra} | {fmt(r['val_sum']['mean'], r['val_sum']['sd'])} | [{r['val_sum']['ci'][0]:.1f}, {r['val_sum']['ci'][1]:.1f}] | {fmt(r['cv_sum']['mean'])} | "
          f"{fmt(r['val_argmax']['mean'], r['val_argmax']['sd'])} | {fmt(r['cv_argmax']['mean'])} |")
    w('\nLength/format features = log-length, relative length/rank, longest/shortest flags, cluster mean length, boxed/####/placeholder/agree flags, "final answer" count, tail length, truncation, #lines, repeated-line fraction. '
      'Residualization: standardized embedding minus its least-squares fit on [1, log-len, log-len^2, boxed, ####, placeholder, truncated, #lines] (fit on training candidates only).\n')
    w('### 2c. Final model (K=5) on val-hard stratified by trace length\n')
    w('| stratum (longest trace / shortest trace correct?) | n | final | LR | vote | final picks longest |\n|---|---|---|---|---|---|')
    for k, v in r46['strata_len'].items():
        w(f"| {k} | {v['n']} | {v['final']:.1f} | {v['lr']:.1f} | {v['vote']:.1f} | {v['pick_longest']:.0f}% |")
    w(f"\nAlways-longest {r46['always_longest']:.1f}, always-shortest {r46['always_shortest']:.1f} val-hard. Final picks the longest trace in {r46['final_pick_longest_rate']:.0f}% and the shortest in {r46['final_pick_shortest_rate']:.0f}% of groups.\n")
    a, b, c_ = r46['wrong_vs_correct_alt'], r46['right_vs_rejected'], r46['all_groups_correct_minus_wrong']
    w('| comparison (chosen minus alternatives, within group) | n groups | mean log(len ratio) | chosen longer than all alts | boxed diff | #### diff |\n|---|---|---|---|---|---|')
    w(f"| wrong pick vs correct alternatives | {a['n']} | {a['mean_log_len_ratio']:+.3f} | {100*a['frac_longer_than_all_alts']:.0f}% | {a['boxed_diff']:+.2f} | {a['hash_diff']:+.2f} |")
    w(f"| right pick vs rejected wrong alternatives | {b['n']} | {b['mean_log_len_ratio']:+.3f} | {100*b['frac_longer_than_all_alts']:.0f}% | {b['boxed_diff']:+.2f} | {b['hash_diff']:+.2f} |")
    w(f"| (reference) mean correct minus mean wrong candidate, all hard groups | {nh} | {c_['mean_log_len_ratio']:+.3f} | - | {c_['boxed_diff']:+.2f} | {c_['hash_diff']:+.2f} |")
    w('\n### 2d. Minority picks and disagreement with the vote\n')
    w('| final model picks | n | right | wrong | final acc | vote acc on same groups |\n|---|---|---|---|---|---|')
    for k, v in r46['pick_type'].items():
        w(f"| {k} | {v['n']} | {v['right']} | {v['wrong']} | {v['acc']:.1f} | {v['vote_acc']:.1f} |")
    t = r46['transitions_vs_vote']; tg = r46['tie_groups']
    w(f"\nVs vote (tie->longest): both right {t['both right']}, final right/vote wrong {t['final right, vote wrong']}, final wrong/vote right {t['final wrong, vote right']}, both wrong {t['both wrong']}. "
      f"Top-count ties: n={tg['n_tie']} final {tg['final_tie']:.1f} vs vote {tg['vote_tie']:.1f}; no tie: n={tg['n_notie']} final {tg['final_notie']:.1f} vs vote {tg['vote_notie']:.1f}.\n")
    # ---- exp3
    w('## Exp 3. Cross-candidate comparison\n')
    w('| model | feature set | val-hard sum | val-hard arg | CV-hard sum | CV-hard arg |\n|---|---|---|---|---|---|')
    for k in [k for k in r123 if k.startswith('3')]:
        r = r123[k]; _, it, *rest = k.split(' ', 2)[0:1] + k.split(' ')[1:2] + [' '.join(k.split(' ')[2:])]
        w(f"| {it} | {rest[0]} | {fmt(r['val_sum']['mean'], r['val_sum']['sd'])} | {fmt(r['val_argmax']['mean'], r['val_argmax']['sd'])} | {fmt(r['cv_sum']['mean'])} | {fmt(r['cv_argmax']['mean'])} |")
    w('\n"no vote/cluster feats": drops vote counts/share/ndist/top flags, cluster-level stats, same-answer cosine similarity and count. "no cross-cand info": additionally drops every group-demeaned (relative) feature, '
      'relative length/rank/magnitude flags, rank among answers and all cosine features, so the "none" network scores each candidate strictly on its own (argmax decode). '
      'Prior results (results_nn.md, 3 seeds, val-hard): none 83.57+-0.27, DeepSets 83.57+-0.35, attention 82.16+-1.5.\n')
    # ---- exp4
    w('## Exp 4. Uncertainty of the final K=5 ensemble\n')
    cv_, cc = r46['cal_val'], r46['cal_cv']
    w(f"Confidence = probability mass of the chosen answer cluster (sum of member-averaged candidate probabilities of the traces giving that answer). "
      f"Val-hard: n={nh}, accuracy {100*cv_['acc']:.1f}, mean confidence {100*cv_['mean_conf']:.1f}, ECE (10 bins) {cv_['ece']:.3f}, Brier {cv_['brier']:.3f}. "
      f"Out-of-fold train-CV (5 seeds averaged, {cc['n']} hard groups): accuracy {100*cc['acc']:.1f}, mean confidence {100*cc['mean_conf']:.1f}, ECE {cc['ece']:.3f}.\n")
    w('| coverage (most confident first) | 30% | 50% | 70% | 80% | 90% | 100% |\n|---|---|---|---|---|---|---|')
    w('| val-hard selective acc | ' + ' | '.join(f"{100*v:.1f}" for v in r46['sel_val'].values()) + ' |')
    w('| CV-hard selective acc | ' + ' | '.join(f"{100*v:.1f}" for v in r46['sel_cv'].values()) + ' |')
    w('\n| margin (top - 2nd cluster mass) | n (val-hard) | accuracy |\n|---|---|---|')
    for b_ in r46['margin_bins']:
        w(f"| [{b_['lo']:.1f}, {b_['hi']:.1f}) | {b_['n']} | {100*b_['acc']:.1f} |")
    w('\nAgreement of final NN / LR / vote on val-hard (by chosen answer):\n')
    w('| pattern | n | final acc | LR acc | vote acc |\n|---|---|---|---|---|')
    for k, v in r46['agree'].items():
        w(f"| {k} | {v['n']} | {v['final']:.1f} | {v['lr']:.1f} | {v['vote']:.1f} |")
    av = r46['agree_vs_disagree']
    w(f"\nFinal accuracy: {av['final_agree']:.1f} on the {av['agree_n']} groups where all three agree vs {av['final_disagree']:.1f} on the {av['disagree_n']} groups with any disagreement.\n")
    # ---- exp5
    w('## Exp 5. Compute vs accuracy\n')
    w('Inference wall = fresh `predict.py` + `score_predictions.py` processes on validation (all 1319 groups), OMP/OPENBLAS/MKL threads=4, median of 3 runs, this 2-core machine. CV-hard for vote = accuracy on all train-hard groups (no fitting).\n')
    w('| method | val-all | val-hard [95% CI] | CV-hard | wall s | assets |\n|---|---|---|---|---|---|')
    for r in r5['rows']:
        w(f"| {r['method']} | {r['val_all']:.2f} | {r['val_hard']:.1f} [{r['val_hard_ci'][0]:.1f}, {r['val_hard_ci'][1]:.1f}] | {r['cv_hard']:.1f} | {r['wall_median']:.2f} | {r['asset_bytes']/1024:.0f} KiB |")
    w('\nTraining cost (this machine): ' + '; '.join(f"{k}: {v:.1f}" for k, v in r5['train'].items() if v is not None) + '.')
    w('Existing K-curve (CV-hard / wall): K=1 79.96 / 1.11 s, K=3 80.24 / 1.24, K=5 80.45 / 1.35, K=10 80.45 / 1.68 (results_final.md); gains beyond K~5 are inside noise.\n')
    # ---- exp6
    w('## Exp 6. Failure analysis (final model, val-hard, 54 errors of 355)\n')
    w('Automatic categories over all errors (by vote structure of the 4 candidates):\n')
    w('| category | errors |\n|---|---|')
    for k, v in sorted(r46['error_cats'].items(), key=lambda kv: -kv[1]):
        w(f'| {k} | {v} |')
    w('\nVote pattern of errors (answer counts): ' + ', '.join(f'{k}: {v}' for k, v in sorted(r46['err_patterns'].items(), key=lambda kv: -kv[1])) +
      f". Errors in groups with >=1 unparseable trace: {r46['err_with_unparse']}; an unparseable trace that visibly contains the gold answer: {sum(e['unparse_has_gold'] for e in r46['errors'])}.\n")
    w('Accuracy by vote pattern on all val-hard groups (right/n): ' + ', '.join(f"{k}: {v['right']}/{v['n']}" for k, v in sorted(r46['pattern_acc_all_hard'].items(), key=lambda kv: -kv[1]['n'])) + '.\n')
    cats = {}
    for g, c in MANUAL.items():
        cats.setdefault(c, []).append(g)
    w(f'Manual reading of {len(MANUAL)} errors (first 20 of every-second-error in bank order, i.e. a systematic sample; cause judged from the question and the 4 traces):\n')
    w('| cause | n | groups |\n|---|---|---|')
    for k, v in sorted(cats.items(), key=lambda kv: -len(kv[1])):
        w(f"| {k} | {len(v)} | {', '.join(map(str, v))} |")
    vb, _ = load_inputs('validation')
    gold = {r['id']: r for r in __import__('common').read_jsonl(__import__('common').DATA / 'gsm8k_validation.jsonl')}
    w('\nIllustrative excerpts (last lines of the trace):\n')
    for g, idxs, note in EXCERPTS:
        w(f"- group {g}: {note}.")
        for i in idxs:
            w(f"  - trace {i}: `{_tail(vb[g]['candidates'][i]['text']).replace('`', '')}`")
    # ---- figures
    ks = [k for k in r123 if k.startswith('1') and k[1] in 'abcde']
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    y = np.arange(len(ks))[::-1]
    for off, key, col, lab in ((0.18, 'val', C['blue'], 'val-hard (3 seeds, 95% Wilson CI)'), (-0.18, 'cv', C['orange'], 'CV-hard (2 seeds)')):
        if key == 'val':
            m = [r123[k]['val_sum']['mean'] for k in ks]; lo = [m_ - r123[k]['val_sum']['ci'][0] for m_, k in zip(m, ks)]; hi_ = [r123[k]['val_sum']['ci'][1] - m_ for m_, k in zip(m, ks)]
            ax.barh(y + off, m, 0.34, xerr=[lo, hi_], color=col, label=lab, error_kw=dict(lw=0.8, capsize=1.5))
        else:
            m = [r123[k]['cv_sum']['mean'] for k in ks]
            ax.barh(y + off, m, 0.34, color=col, label=lab)
        for yy, mm in zip(y, m): ax.text(mm + 0.4 if key == 'cv' else 60.6, yy + off - 0.02, f'{mm:.1f}', va='center', fontsize=6.5, color='white' if key == 'val' else 'black')
    ax.axvline(r46['val_hard']['vote']['acc'], color=C['grey'], ls=':', lw=0.8); ax.text(r46['val_hard']['vote']['acc'] + 0.2, y[0] + 0.55, 'vote 70.7', fontsize=6.5, color=C['grey'])
    ax.set_yticks(y); ax.set_yticklabels([k[3:] for k in ks]); ax.set_xlim(60, 92); ax.set_xlabel('hard-group accuracy (%)'); ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=6.5, frameon=False)
    ax.set_title('Information-source ablation (cumulative)', fontsize=8)
    save(fig, 'fig_info_ablation')
    fig, axs = plt.subplots(1, 2, figsize=(5.4, 2.5))
    for ax, cal, ttl in ((axs[0], cv_, f"val-hard (n={nh})"), (axs[1], cc, f"out-of-fold train ({cc['n']})")):
        ax.plot([0, 1], [0, 1], color=C['grey'], lw=0.8, ls='--')
        bx = [b['conf'] for b in cal['bins']]; by = [b['acc'] for b in cal['bins']]; bn = np.array([b['n'] for b in cal['bins']])
        ax.scatter(bx, by, s=8 + 90 * bn / bn.max(), color=C['blue'], zorder=3); ax.plot(bx, by, color=C['blue'], lw=0.8)
        ax.set_xlabel('chosen-answer confidence'); ax.set_title(f"{ttl}, ECE={cal['ece']:.3f}", fontsize=7.5); ax.set_xlim(0.3, 1.02); ax.set_ylim(0.3, 1.02)
    axs[0].set_ylabel('accuracy'); fig.tight_layout(); save(fig, 'fig_calibration')
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    cv_c = r46['curves']
    ax.plot(100 * np.array(cv_c['val_cov']), 100 * np.array(cv_c['val_acc']), color=C['blue'], label='val-hard')
    ax.plot(100 * np.array(cv_c['cv_cov']), 100 * np.array(cv_c['cv_acc']), color=C['orange'], label='out-of-fold train-hard')
    ax.axhline(r46['val_hard']['vote']['acc'], color=C['grey'], ls=':', lw=0.8); ax.text(2, r46['val_hard']['vote']['acc'] + 0.8, 'vote (full coverage)', fontsize=6.5, color=C['grey'])
    ax.set_xlabel('coverage (%)'); ax.set_ylabel('selective accuracy (%)'); ax.legend(frameon=False, fontsize=7); ax.set_ylim(65, 101)
    save(fig, 'fig_selective')
    fig, axs = plt.subplots(1, 2, figsize=(5.6, 2.5), sharey=True)
    for ax, key, ttl in ((axs[0], 'val_hard', 'val-hard'), (axs[1], 'cv_hard', 'CV-hard')):
        for r, col in zip(r5['rows'], [C['grey'], C['green'], C['sky'], C['blue'], C['verm']]):
            ax.scatter(r['wall_median'], r[key], color=col, s=22 + 14 * np.log10(max(r['asset_bytes'], 1000) / 1000), zorder=3)
            ax.annotate(r['method'].replace(' (tie->longest)', '').replace('DeepSets NN ', 'NN ').replace(' (hand+cross+PCA64)', '').replace(' (final)', '*'), (r['wall_median'], r[key]), fontsize=6, xytext=(-14, 7) if 'K=3' in r['method'] else (-4, -11) if 'K=5' in r['method'] else (4, 5) if 'vote' in r['method'] else (3, -9), textcoords='offset points')
        ax.set_xlabel('predict+score wall (s, 4 threads)'); ax.set_title(ttl, fontsize=8)
    axs[0].set_ylabel('hard-group accuracy (%)'); fig.tight_layout(); save(fig, 'fig_compute')
    ks2 = [k for k in r123 if k.startswith('2')]
    fig, ax = plt.subplots(figsize=(5.2, 2.8)); y = np.arange(len(ks2))[::-1]
    ax.barh(y + 0.18, [r123[k]['val_sum']['mean'] for k in ks2], 0.34, color=C['blue'], label='val-hard')
    ax.barh(y - 0.18, [r123[k]['cv_sum']['mean'] for k in ks2], 0.34, color=C['orange'], label='CV-hard')
    ax.set_yticks(y); ax.set_yticklabels([k[3:] for k in ks2], fontsize=6.5); ax.set_xlim(55, 90); ax.set_xlabel('hard-group accuracy (%)'); ax.legend(frameon=False, fontsize=7, loc='lower right')
    save(fig, 'fig_shortcut')
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    ec = sorted(r46['error_cats'].items(), key=lambda kv: kv[1]); ax.barh([k for k, _ in ec], [v for _, v in ec], color=C['verm'])
    for i, (_, v) in enumerate(ec): ax.text(v + 0.3, i, str(v), va='center', fontsize=7)
    ax.set_xlabel('errors (of 54)'); ax.tick_params(axis='y', labelsize=6.5); save(fig, 'fig_failure')
    (HERE / 'analysis_results.md').write_text('\n'.join(L) + '\n')
    (HERE / 'analysis.json').write_text(json.dumps(dict(exp123=r123, exp46={k: v for k, v in r46.items() if k != 'curves'}, exp5=r5, manual_error_labels={str(k): v for k, v in MANUAL.items()}), indent=1, default=float))
    print('report written')


if __name__ == '__main__':
    parts = sys.argv[1:] or ['exp123', 'exp46', 'exp5', 'report']
    if 'exp123' in parts:
        exp123()
    if 'exp46' in parts:
        exp46()
    if 'exp5' in parts:
        exp5()
    if 'report' in parts:
        report()
