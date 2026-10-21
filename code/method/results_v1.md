# Results v1 (train-only fit; val-all / val-hard %, val hard = 355 groups)

C chosen by 5-fold train-CV hard accuracy (cv-hard). Best variant picked by val-hard (slight selection bias; 1 group = 0.28 pt).

| variant | model | train set | C | val-all | val-hard | cv-hard |
|---|---|---|---|---|---|---|
| vote tie->longest (baseline) | - | - | - | 86.97 | 70.70 | - |
| a hand | LR | all | 1.0 | 89.47 | 76.34 | 73.65 |
| a2 hand+cross | LR | all | 0.1 | 89.60 | 76.62 | 73.78 |
| b hand+cross+emb64 | LR | all | 1.0 | 90.23 | 78.03 | 75.00 |
| b2 hand+cross+emb128 | LR | all | 1.0 | 90.35 | 78.31 | 75.44 |
| c emb64 only | LR | all | 0.1 | 84.21 | 67.04 | 69.36 |
| c2 emb128 only | LR | all | 0.1 | 86.09 | 72.11 | 70.79 |
| d1 hand+cross rel | LR | all | 0.01 | 89.97 | 77.46 | 74.85 |
| d2 hand+cross rel, agg=sum | LR | all | 0.1 | 89.47 | 76.34 | 74.32 |
| d3 hand+cross rel, agg=max | LR | all | 0.01 | 89.97 | 77.46 | 74.85 |
| d4 hand+cross+emb64 rel, agg=sum | LR | all | 10.0 | 90.85 | 79.44 | 76.82 |
| d7 hand+cross+emb128 rel, agg=sum | LR | all | 1.0 | 90.73 | 79.15 | 77.04 |
| d8 hand+cross+emb32 rel, agg=sum | LR | all | 1.0 | 90.60 | 78.87 | 76.56 |
| d5 hand+cross, agg=sum | LR | all | 10.0 | 89.35 | 76.06 | 73.53 |
| d6 hand+cross rel, agg=mean | LR | all | 0.1 | 89.72 | 76.90 | 74.96 |
| d4 hand+cross+emb64 rel, agg=sum | LR | hard-only | 10.0 | 91.10 | 80.00 | 77.66 |
| d7 hand+cross+emb128 rel, agg=sum | LR | hard-only | 1.0 | 90.85 | 79.44 | 77.28 |
| d8 hand+cross+emb32 rel, agg=sum | LR | hard-only | 1.0 | 90.73 | 79.15 | 76.89 |
| e hgb hand+cross | HGB | all | - | 89.85 | 77.18 |  |
| e hgb hand+cross agg=sum | HGB | all | - | 89.85 | 77.18 |  |
| e2 hgb hand+cross+emb16 | HGB | all | - | 90.73 | 79.15 |  |
| e2 hgb hand+cross+emb16 agg=sum | HGB | all | - | 90.35 | 78.31 |  |
| e3 hgb hand+cross rel | HGB | all | - | 90.60 | 78.87 |  |
| e3 hgb hand+cross rel agg=sum | HGB | all | - | 90.60 | 78.87 |  |

Shipped: `d4 hand+cross+emb64 rel, agg=sum [hard-only]`, C=10: val-all 91.10, val-hard 80.00 (official scorer 91.10).
Inference: predict+score wall 1.10 s (4 threads, fresh process), RSS 93 MB; checkpoint 275,307 bytes (selector.npz 274,522 + config.json 785).
