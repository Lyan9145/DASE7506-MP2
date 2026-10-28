# v3 experiments: cross-candidate text features (2026-10-28)

Goal: improve on the submitted v2 (DeepSets K=5, test 81.58). Protocol identical to `cv.py`: 5-fold CV over TRAIN
groups (RandomState(0) folds), score = hard-group accuracy of the held-out fold, cluster-sum decode; validation =
models trained on the full train split. Train split only, hard groups only. Raw rows: `exp3_results.jsonl`.
Code: `features_v3.py`, `nn_rel.py`, `exp3.py`, `nn3_infer.py`, `submission_v3.py`.

## Motivation (probe on train hard groups, v2 OOF)
For candidate i with valid answer, among other-answer candidates:
- every other distinct answer occurs inside i's reasoning ("others_in_me"=1): P(correct) 0.75 (n=1353) vs 0.50 when none do.
- i's answer occurs inside every other-answer candidate's reasoning ("ans_in_others"=1): P(correct) 0.40 (n=1428) vs 0.58.
Interpretation: wrong candidates often stop early at an intermediate value of a correct chain. Per-candidate
embeddings cannot see this; the v1 features did not encode it.

## Results (CV-hard: mean over 5 folds; single = mean over seeds; ens = average of seed probabilities)

| variant | CV single | CV ens K=5 | CV ens K=10 | val-hard ens K=5 | val-hard ens K=10 |
|---|---|---|---|---|---|
| v2 DeepSets (submitted) | 79.62 | 80.45 | 80.45 | 84.79 | - |
| **v3 = v2 + features_v3 (15 feats + group-demeaned)** | 80.51 | 81.37 | **81.87** | 84.79 | **85.35** |
| v3, H=256 / H=64 | 80.37 / 80.05 | 81.24 / 81.07 | | 85.07 / 85.35 | |
| v3, dropout .5 / emb-dropout .6 / wd .2 / lr 3e-4 | 80.47 / 80.19 / 80.37 / 80.36 | 81.00 / 81.11 / 81.20 / 81.44 | | 84.51 / 84.79 / 84.51 / 85.07 | |
| v3, no interaction (per-candidate MLP) | 80.54 | 81.52 | | 84.23 | |
| v3 + relational pair-message block (`nn_rel.py`, R=32) | 80.71 | 81.89 | 81.81 | 85.63 | 85.07 |
| relational R=16 / R=64 | 80.54 / 80.65 | 81.58 / 81.80 | | 85.35 / 84.79 | |
| relational + divergence pair feats (`pair_features2`) | 80.73 | 81.81 | | 85.07 | |
| relational + abs(e_i - e_j) pair input | 80.73 | 81.51 | | 85.35 | |
| v3 + answer-relation feats (`relation_features`) | 80.43 | 80.94 | | 85.07 | |
| relational + answer-relation feats | 80.69 | 81.29 | | 84.79 | |
| v3 K=10 + relational K=10 pooled | | | 82.11 (20 members) | | 84.79 |

Also tried, not used: TF-IDF char/word n-gram group-relative LR (alone 71.1 CV, blended with v3 +0.17);
HistGradientBoosting stacker on OOF NN scores + features (+0.5 with max-decode, +0.1 with mean-decode, LR stacker
+0.1: not robust); "unsupported numbers" probe (no signal).

Findings
- The v3 features are the only robust gain: +0.9 CV single-model, +1.4 CV at K=10, and v3 needs a larger ensemble
  than v2 (K=5 -> K=10: +0.5 CV here vs +0.0 for v2).
- Architecture (interaction / relational pair messages / width) and regularisation changes are all within noise
  once v3 features are present: the relational model's K=5 edge (81.89) disappears at K=10 (81.81 vs 81.87).

## Frozen v3 selection (made on CV + validation only)
v3 DeepSets, BASE hyperparameters, K=10 (seeds 0-9), simplest config within noise of the best; 20-member pooling
would exceed 16 MiB. `.venv/bin/python method/exp3.py export K=10 out=checkpoint_v3` (39-47 s CPU, reproduced
bit-for-bit). numpy vs torch max abs diff 1.5e-7.
- `checkpoint_v3/nn.npz` 9,216,382 B sha256 57d13f01e391353260db4efa0c52c0306df29355436adccc580f5d36ef7a6523;
  `config.json` 2,448 B sha256 190f1c27c04f5e09b61561177b1168de616b51ce5772ad83010c1f86fa866024 (total 8.79 MiB).
- Official validation 93.4837 (746/798; v2 93.2331), wall 2.26 s, 124 MB RSS (predict+score, 4 threads, local 2-core EPYC).
- **Official test, run once after freezing: 80.7281 (377/467)**, wall 1.52 s, 95 MB. v2 test was 81.5846 (381/467).
  The CV/validation gain did not transfer to test (-4 groups; binomial s.e. at n=467 is about 1.8 pt, so the
  difference is within noise in both directions). Not used to change the method: re-selecting between v2 and v3
  (or trying the relational/pooled variants) on test would be tuning on test. Files: `results/v3-*.json`.
