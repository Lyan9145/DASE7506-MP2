# DASE7506 MP2: selecting 1 of 4 Qwen3-0.6B GSM8K candidate solutions

Submitted method: a 5-member DeepSets ensemble selector (numpy-only inference) over hand features, group-relative inputs and the full 1024-d supplied embedding, trained with a listwise loss `-log sum_correct p` and decoded by summing probability per distinct parsed answer. Report: [`report/report.pdf`](report/report.pdf) (source `report/report.tex`, figures `report/figures/`). The original course README is untouched at `code/README.md`.

## Scores (official scorer)

| split | metric | value |
|---|---|---|
| **test** (467 groups, run once after freezing) | accuracy | **81.5846 % (381/467)** |
| validation, all 798 groups | accuracy | 93.2331 % (744/798) |
| validation, hard groups (355) | accuracy | 84.79 % (optimistic: variants were selected on it) |
| 5-fold train-only CV, hard groups (2114) | accuracy | 80.45 % +- 1.90 (fold std), honest selection estimate |
| baseline majority vote (tie -> longest) | val-all / val-hard / CV-hard | 86.97 / 70.7 / 69.9 |

Official JSONs: `code/results/test-score.json`, `code/results/validation-score.json` (plus the prediction files). Test is composed only of hard groups, so test accuracy is comparable to the hard-group numbers, not to val-all.

## v3 experiment (not the submitted method)

Cross-candidate text features (`code/method/features_v3.py`: whether the other candidates' final answers occur inside a candidate's reasoning and vice versa, number overlap) raised 5-fold train-CV hard accuracy from 80.45 to 81.87 (K=10) and validation from 93.2331 to 93.4837, but the frozen v3 model scored **80.7281 % (377/467) on test**, below v2's 81.5846. The submitted method stays v2; switching on the basis of the test score would be test-set selection. Details, all variants tried and hashes: `code/method/results_v3.md`. Evaluate: `python predict.py --method submission_v3.py --checkpoint checkpoint_v3 --split test --output runs/v3-test.json`; retrain: `python method/exp3.py export K=10 out=checkpoint_v3` (~45 s CPU, bit-for-bit reproducible locally). Official JSONs: `code/results/v3-*.json`.

## Install

Python 3.12.3 (Linux). From the repo root:

```bash
cd code
python3 -m venv .venv            # on Debian/Ubuntu: apt install python3.12-venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-method.txt        # inference: numpy 2.5.3, scikit-learn 1.9.1, scipy 1.18.1, ...
# training/analysis only (not needed for evaluation):
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-train.txt   # torch 2.14.1 CPU, matplotlib 3.11.2
```

## Evaluate (no retraining; uses `code/checkpoint/`)

```bash
cd code
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
python predict.py --method submission.py --checkpoint checkpoint --split validation --output runs/validation-predictions.json
python score_predictions.py --split validation --predictions runs/validation-predictions.json --output runs/validation-score.json
python predict.py --method submission.py --checkpoint checkpoint --split test --output runs/test-predictions.json
python score_predictions.py --split test --predictions runs/test-predictions.json --output runs/test-score.json
```

Expected `accuracy`: 93.2331 (validation), 81.5846 (test). `submission.py` is the frozen method (DeepSets K=5); `submission_lr.py` + `checkpoint_lr/` is the v1 LR ablation, `submission_nn.py` an NN ablation copy.

## Train (regenerates `code/checkpoint/`)

```bash
cd code
python method/nn_train.py export K=5 out=checkpoint
```

Trains 5 members (seeds 0-4, hard train groups only, train split only; init and early-stopping holdout seeded by the member seed) and writes `checkpoint/nn.npz` and `checkpoint/config.json`; prints the numpy-vs-torch check (1.8e-7) and val-all/val-hard (93.23 / 84.79). Without `out=` it writes to the default `code/checkpoint`. Cost: about 26 s wall on a 2-core CPU, no GPU. **Verified:** running this command into a scratch directory reproduced both files bit-for-bit (sha256 identical to the table below) with torch 2.14.1 CPU on the same machine; bitwise equality on other hardware or torch builds is not guaranteed, but scores should match to within float rounding.

Other scripts (all in `code/method/`, run from `code/`): `train.py` (v1 LR), `nn_train.py grid` (architecture grid, `results_nn.md`), `cv.py 10` (5-fold train-CV, about 10 min, `results_final.md`, `cv_results.json`), `analysis.py` (all report ablations/figures, `analysis_results.md`, `analysis.json`, writes `report/figures/`), `features.py`/`features_v2.py` (features; v2 is a negative result), `baseline_vote.py`, `explore.py`.

## Assets (`code/checkpoint/`, limit 16 MiB; total 4.32 MiB)

| file | bytes | sha256 |
|---|---|---|
| `nn.npz` | 4,530,202 | `431e70ce6fa5db6a88dcd62a92f2c3201eb5ef3ce3f9261de317c2ff9aceeeaa` |
| `config.json` | 312 | `f8d852c65c8813ab445d2b5d3002bb68c4669d02ac4bafbf1088627362dbfcb8` |

## Timing

Fresh `predict.py` + scorer, 4 threads, local AMD EPYC 7A23 (2 cores available): 1.38 s validation (115 MB RSS), 0.97 s test (84 MB). The 30 s eligibility limit is defined on the reference Xeon Platinum 8457C, which we did not have; local margin is more than 20x.

## Seeds, protocol, records

Ensemble seeds 0-4; CV folds `RandomState(0).randint`; ablations: val seeds 0-2, CV seeds 0-1. Fit on the train split only (no train+val refit); validation for secondary comparison; test scored once. Result records: `code/method/results_v1.md`, `results_nn.md`, `results_final.md`, `analysis_results.md`, `cv_results.json`, `nn_grid.json`, `analysis.json`, and `code/results/*.json`.

## AI-use and reuse disclosure

The code, experiments and first draft of the report were produced substantially with Claude Code (Anthropic) AI agents under the student's direction; the student is responsible for the submission. No external data or pretrained weights; no LLM at inference. Reused: the supplied course package, numpy, scikit-learn, scipy, PyTorch, matplotlib. Data derives from GSM8K (MIT licence); candidates and embeddings come from Qwen3-0.6B (supplied by the course; licence/attribution in `code/data/`).
