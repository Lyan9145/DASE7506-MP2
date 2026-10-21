# Results NN (listwise selector; train-only fit; val-hard = 355 groups, 1 group = 0.28 pt)

Inputs: v1 hand+rel+cross features (standardized) + full 1024-d embedding (standardized, learned linear 1024->128 inside first layer, input dropout 0.3).
Encoder: 2-layer MLP (H=128, dropout 0.3, AdamW lr 1e-3, wd 0.05, batch 64). Loss: -log sum_{correct} softmax. Early stopping on a 12% train-internal holdout (hard groups' loss, patience 8).
Decode: argmax of softmax summed per parsed-answer cluster (agg=sum). Interaction: none / deepsets [h, h-mean, mean, same-answer-cluster mean] / 1 block self-attention (4 heads).
Mean±std over seeds 0,1,2 (single models); "ens3" = mean of the 3 softmaxes. Val used for choosing the variant (selection bias small but present). Raw data: method/nn_grid.json.

| interaction | full emb | train groups | val-all | val-hard | val-hard no cluster agg | ens3 val-all / hard |
|---|---|---|---|---|---|---|
| none (per-cand MLP) | yes | hard | 92.69±0.12 | 83.57±0.27 | 83.57 | 92.73 / 83.66 |
| none | yes | all | 92.23±0.10 | 82.54±0.23 | 82.35 | 92.73 / 83.66 |
| none | no (hand only) | hard | 90.60±0.20 | 78.87±0.46 | 78.31 | 90.48 / 78.59 |
| none | no | all | 89.85±0.18 | 77.18±0.40 | 77.56 | 90.60 / 78.87 |
| deepsets | yes | hard | 92.69±0.16 | 83.57±0.35 | 83.19 | 93.48 / 85.35 |
| deepsets | yes | all | 92.44±0.12 | 83.00±0.27 | 83.57 | 92.61 / 83.38 |
| deepsets | no | hard | 90.52±0.51 | 78.69±1.16 | 78.22 | 90.60 / 78.87 |
| deepsets | no | all | 89.97±0.87 | 77.46±1.97 | 77.65 | 90.73 / 79.15 |
| attention | yes | hard | 92.06±0.67 | 82.16±1.50 | 82.35 | 93.36 / 85.07 |
| attention | yes | all | 92.11±0.45 | 82.25±1.00 | 83.76 | 91.85 / 81.69 |
| attention | no | hard | 90.31±0.12 | 78.22±0.27 | 78.69 | 90.35 / 78.31 |
| attention | no | all | 90.60±0.10 | 78.87±0.23 | 79.06 | 90.60 / 78.87 |
| v1 LR (reference) | PCA64 | hard | 91.10 | 80.00 | | |

Shipped candidate: deepsets + full emb, hard-only, ensemble of 5 seeds (0-4): val-all 93.23 / val-hard 84.79 (official scorer 93.2331).

Findings
- The full 1024-d embedding is what matters (+4.7 hard pts vs hand-only; hand-only NN ~ v1 LR hand-only ~ 78-79). A nonlinear model on the full embedding beats LR on PCA64 by ~3.5 pts (single model).
- Cross-candidate interaction adds nothing for single models (none 83.57 vs deepsets 83.57 vs attention 82.2±1.5, higher variance); the hand/cross features + cluster-sum already encode the cross-candidate info. Ensembling helps deepsets/attention more (+1.8/+2.9) than none (+0.1), but with 3 seeds and 1 group = 0.28 pt this is within noise; do not claim interaction helps.
- Hard-only training >= all-groups. Cluster-sum aggregation: roughly neutral (+0.0 to +0.4) for NN.
- Early stopping triggers at epoch 2-9 (overfits fast), so regularization is critical.

Cost: CPU torch 2.14.1 (local, 2 cores, no GPU/ps used). One model ~3-6 s (hard) / 5-12 s (all); full 12-config x 3-seed grid 3.4 min; 5-member export 20 s. Seeds 0-4 (holdout split and init seeded by same seed).
Numpy vs torch max abs diff on val probs: 1.8e-7 (deepsets ens5), 4.8e-7 (attn), 3.0e-7 (none).
Inference (OMP/OPENBLAS/MKL=4, fresh process, `predict.py --method submission_nn.py` + `score_predictions.py`, /usr/bin/time): wall 1.39 s, RSS 115 MB.
Assets: checkpoint_nn/nn.npz 4,530,202 B + config.json 297 B = 4.32 MiB (<16 MiB).
