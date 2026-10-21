# Final model selection (CV on train only)

Protocol: `method/cv.py`; 5 folds by group (RandomState(0).randint), score = accuracy on HARD groups of the held-out fold (cluster-sum decode), mean±std over the 5 folds (2114 hard train groups total; 1 group ~0.05 pt of the pooled CV, fold std is ~1.5-2 pt). NN early stopping uses a 12% holdout drawn only from the fold's training pool; the LR PCA is refit per fold. Models train on hard groups only. Raw numbers: method/cv_results.json.

| model | single (seed 0) | single, mean over 10 seeds | ens K=5 | ens K=10 |
|---|---|---|---|---|
| v1 LR (shipped v1 config: hand+cross+PCA64, rel, C=10) | 77.31±2.50 | - | - | - |
| v1 LR + v2 features | 76.76±2.17 | - | - | - |
| NN MLP, no interaction (v1 feats) | 79.78±1.51 | 79.64 (seed std 0.21) | 80.44±1.77 | 80.48±1.28 |
| NN MLP, no interaction, + v2 feats | 79.00±1.99 | 79.53 (0.63) | 79.90±1.52 | 80.19±1.44 |
| NN DeepSets (v1 feats) | 79.96±1.52 | 79.74 (0.42) | 80.45±1.90 | 80.45±1.66 |
| NN DeepSets + v2 feats | 79.15±2.01 | 79.72 (0.53) | 80.41±1.76 | 80.70±1.40 |

Findings
- NN (full 1024-d embedding) beats v1 LR by ~2.5 CV pts; interaction (DeepSets) ~ no interaction (differences <=0.3, well inside noise).
- v2 verifier features (`features_v2.py`: chained/unit/%/x arithmetic, wrong-step fraction, last-equation==answer, plausibility vs question numbers, q-number coverage in equations): no CV gain (LR -0.55, NN -0.1 to +0.25). Not used in the final model. Likely reason: the answer-cluster/vote features already carry the signal and equations rarely appear (~2 per candidate).
- Ensembling helps a little: K=1 -> K=5 about +0.5 CV pt; K=10 adds ~0.0-0.3.

## K-curve (DeepSets, v1 feats): CV-hard vs inference cost (validation, OMP/OPENBLAS/MKL=4, predict.py + score_predictions.py wall, fresh processes)

| K | CV-hard | val-all | wall s | RSS MB | nn.npz bytes (approx) |
|---|---|---|---|---|---|
| 1 | 79.96±1.52 | 92.73 | 1.11 | 107 | 0.9 M |
| 3 | 80.24±1.88 | 93.48 | 1.24 | 111 | 2.7 M |
| 5 | 80.45±1.90 | 93.23 | 1.35 | 115 | 4.53 M |
| 10 | 80.45±1.66 | 92.86 | 1.68 | 124 | 9.1 M |

Cost grows ~0.06 s/member; accuracy saturates at K~5. More computation beyond a small ensemble does not pay off (all differences are inside fold noise; val-hard single/ens: 83.7 / 84.8 (K=5) / 83.9 (K=10)).

## Chosen config
DeepSets (H=128, dropout 0.3, input-emb dropout 0.3, AdamW lr 1e-3 wd 0.05, batch 64, <=60 epochs, patience 8), v1 hand+rel+cross features + full standardized embedding, listwise loss -log sum_correct softmax, trained on hard train groups only, ensemble K=5 (seeds 0-4; per-member holdout/init seed = member seed), cluster-sum decode. Selected as the simplest config within CV noise of the best (the CV maximum, DeepSets+v2 K=10 80.70, is +0.25 over it, ~5 groups). Final model trained on the full train split (not train+val).
- Validation (secondary): val-all 93.23 / val-hard 84.79 (official scorer 93.2331). Numpy-vs-torch max abs diff 1.8e-7.
- Official predict+score on validation, 4 threads: wall 1.38 s, RSS 115 MB.
- Assets `code/checkpoint/`: nn.npz 4,530,202 B (sha256 431e70ce6fa5db6a88dcd62a92f2c3201eb5ef3ce3f9261de317c2ff9aceeeaa), config.json 312 B (sha256 f8d852c65c8813ab445d2b5d3002bb68c4669d02ac4bafbf1088627362dbfcb8).
- Reproduce: `.venv/bin/python method/nn_train.py export K=5` (20 s CPU, torch 2.14.1 CPU, 2 cores); CV: `.venv/bin/python method/cv.py 10` (~10 min).
- LR ablation kept: `submission_lr.py` + `checkpoint_lr/`; NN ablation copy `submission_nn.py`.
