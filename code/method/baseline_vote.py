"""Training-free baseline: majority vote over parsed final answers.

Ties (and groups with no parseable answer) are broken by the longest trace,
then lowest index. predict.py puts this file's directory on sys.path, so
`common` (the fixed parser) is importable.
"""
from collections import Counter

from common import parse_answer


def predict(bank, arrays, checkpoint=None):
    choices = []
    for row in bank:
        cands = row['candidates']
        ans = [parse_answer(c['text']) for c in cands]
        cnt = Counter(a for a in ans if a is not None)
        if cnt:
            top = max(cnt.values())
            pool = [i for i, a in enumerate(ans) if a is not None and cnt[a] == top]
        else:
            pool = list(range(len(cands)))
        best = max(pool, key=lambda i: (len(cands[i]['text']), -i))
        choices.append(int(best))
    return choices
