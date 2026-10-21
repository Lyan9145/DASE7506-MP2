"""Data exploration + trivial baselines. Test: input-side stats only (no references)."""
import sys
from pathlib import Path
from collections import Counter
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common
from common import parse_answer, load_inputs, read_jsonl, DATA


def parsed(bank):
    return [[parse_answer(c['text']) for c in r['candidates']] for r in bank]


def vote(ans, tiebreak, extra=None):
    """Majority vote over valid answers; tiebreak(list of tied cand idx, g) -> idx."""
    cnt = Counter(a for a in ans if a is not None)
    if not cnt:
        return 0, False
    top = max(cnt.values())
    best = [a for a, c in cnt.items() if c == top]
    cands = [i for i, a in enumerate(ans) if a in best]
    tied = len(best) > 1
    return (tiebreak(cands) if tied else cands[0]), tied


def main():
    stats = {}
    for split in ('train', 'validation', 'test'):
        bank, arr = load_inputs(split)
        P = parsed(bank)
        n = len(bank)
        E = arr['embeddings']
        L = np.array([[len(c['text']) for c in r['candidates']] for r in bank])
        nvalid = np.array([len({a for a in p if a is not None}) for p in P])
        print(f'\n=== {split}: N={n} ===')
        norms = np.linalg.norm(E, axis=-1)
        print(f'emb norm mean={norms.mean():.4f} min={norms.min():.4f} max={norms.max():.4f} '
              f'mean|x|={np.abs(E).mean():.4f}')
        print(f'text chars: mean={L.mean():.0f} med={np.median(L):.0f} p5={np.percentile(L,5):.0f} '
              f'p95={np.percentile(L,95):.0f} max={L.max()}; per-cand-idx mean={L.mean(0).round(0).tolist()}')
        unp = np.array([[a is None for a in p] for p in P])
        print(f'unparseable frac={unp.mean():.4f} (per idx {unp.mean(0).round(4).tolist()}); '
              f'has####: {np.mean([["####" in c["text"] for c in r["candidates"]] for r in bank]):.3f} '
              f'has boxed: {np.mean([["\\boxed" in c["text"] for c in r["candidates"]] for r in bank]):.3f}')
        print('#distinct valid answers per group:', dict(sorted(Counter(nvalid.tolist()).items())))
        hard = nvalid >= 2
        print(f'hard groups (>=2 distinct valid): {hard.sum()} / {n} = {hard.mean():.3f}')
        if split == 'test':
            continue
        gold = [parse_answer(g['answer']) for g in read_jsonl(DATA / f'gsm8k_{split}.jsonl')]
        C = np.array([[a is not None and a == g for a in p] for p, g in zip(P, gold)])
        nc = C.sum(1)
        print(f'per-cand acc={C.mean(0).round(4).tolist()} mean={C.mean():.4f}; oracle={np.mean(nc>0):.4f}')
        print('#correct dist:', dict(sorted(Counter(nc.tolist()).items())))
        print('#correct dist (hard):', dict(sorted(Counter(nc[hard].tolist()).items())))
        # tie-breakers
        emb_n = E / np.linalg.norm(E, axis=-1, keepdims=True)
        def tb_first(g): return lambda c: c[0]
        def tb_short(g): return lambda c: min(c, key=lambda i: L[g, i])
        def tb_long(g): return lambda c: max(c, key=lambda i: L[g, i])
        def tb_cent(g):
            S = emb_n[g] @ emb_n[g].T
            return lambda c: max(c, key=lambda i: S[i].sum())
        rules = {'first': tb_first, 'shortest': tb_short, 'longest': tb_long, 'emb-centrality': tb_cent}
        res = {}
        for name, mk in rules.items():
            sel, tied = [], []
            for g in range(n):
                s, t = vote(P[g], mk(g)); sel.append(s); tied.append(t)
            sel = np.array(sel); tied = np.array(tied)
            ok = C[np.arange(n), sel]
            res[name] = (ok.mean(), ok[hard].mean(), tied[hard].mean(), ok[hard & tied].mean() if (hard & tied).any() else float('nan'))
            vote_ok = ok
        print(f'idx0 acc: all={C[:,0].mean():.4f} hard={C[hard,0].mean():.4f}')
        print(f'random(expected) acc: all={C.mean():.4f} hard={C[hard].mean():.4f}')
        print(f'oracle hard={np.mean(nc[hard]>0):.4f}')
        for k, (a, h, tf, tacc) in res.items():
            print(f'vote[tie={k}]: all={a:.4f} hard={h:.4f} | hard tied frac={tf:.4f} acc on hard-tied={tacc:.4f}')
        # all-4-agree etc
        top = np.array([max(Counter(a for a in p if a is not None).values(), default=0) for p in P])
        print('max vote count dist (hard):', dict(sorted(Counter(top[hard].tolist()).items())))
        if split == 'validation':
            ex = np.where(hard)[0][[0, 5, 11, 20]]
            for g in ex:
                r = bank[g]
                print(f'\n-- example {r["id"]} gold={gold[g]} parsed={P[g]} correct={C[g].tolist()}')
                print('Q:', r['question'][:150].replace('\n', ' '))
                for i, c in enumerate(r['candidates']):
                    print(f'  [{i}] len={len(c["text"])} ...{c["text"][-110:]!r}')


if __name__ == '__main__':
    main()
