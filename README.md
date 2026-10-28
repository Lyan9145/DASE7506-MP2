# DASE7506 MP2: selecting 1 of 4 Qwen3-0.6B GSM8K candidate solutions

Submitted method (v4): a 10-member DeepSets ensemble selector (numpy-only inference). Each candidate is scored from three groups of inputs, each also entered group-relative (minus the group mean):
- hand features;
- cross-candidate text features: do the other candidates' final answers appear inside this candidate's reasoning, and vice versa;
- arithmetic-repair features: recompute each trace's equations with earlier slips corrected, then check whether the corrected answer lands on another candidate's answer;

plus the full 1024-d supplied embedding. It is trained with a listwise loss `-log sum_correct p` and decoded by summing probability per distinct parsed answer.

Report: [`report/report.pdf`](report/report.pdf) (source `report/report.tex`, figures `report/figures/`).

## Scores (official scorer)

| split | metric | value |
|---|---|---|
| **test** (467 groups, run once after freezing v4) | accuracy | **82.6552 % (386/467)** |
| validation, all 798 groups | accuracy | 94.2356 % (752/798) |
| validation, hard groups (355) | accuracy | 87.04 % (optimistic: variants were compared on it) |
| 5-fold train-only CV, hard groups (2114) | accuracy | 83.05 % +- 1.49 (fold std), honest selection estimate |
| baseline majority vote (tie -> longest) | val-all / val-hard / CV-hard | 86.97 / 70.7 / 69.9 |

Official JSONs: `code/results/test-score.json`, `code/results/validation-score.json` (plus the prediction files). Test is composed only of hard groups, so test accuracy is comparable to the hard-group numbers, not to val-all.

### Version history (each test run once, after that version was frozen on CV + validation)

| version | entry point + checkpoint | CV-hard | val-all | test |
|---|---|---|---|---|
| v1 LR (hand+cross+PCA64) | `submission_lr.py` + `checkpoint_lr/` | 77.31 | 91.10 | - |
| v2 DeepSets K=5, v1 features | `submission_v2.py` + `checkpoint_v2/` | 80.45 | 93.2331 | 81.5846 (381) |
| v3 = v2 features + cross-candidate text features, K=10 | `submission_v3.py` + `checkpoint_v3/` | 81.87 | 93.4837 | 80.7281 (377) |
| **v4 = v3 + arithmetic-repair features, K=10 (submitted)** | **`submission.py` + `checkpoint/`** | **83.05** | **94.2356** | **82.6552 (386)** |

The test bank was scored three times in total: v2, v3 and v4, each exactly once after freezing. Each version was selected on train-CV and validation only. v4 was developed after v3's test result was known and after reading another student's public repository (see the disclosure below). Old official JSONs: `code/results/v2-*.json`, `code/results/v3-*.json`. Experiment records: `code/method/results_v3.md` (v3 and v4).

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

Expected `accuracy`: 94.2356 (validation), 82.6552 (test). Inference needs numpy only; there is no pickle and no torch.

Ablations use the same commands with these entry points and checkpoints:
- `submission_v3.py` + `checkpoint_v3` (no repair features): 93.4837 / 80.7281;
- `submission_v2.py` + `checkpoint_v2` (v1 features only): 93.2331 / 81.5846;
- `submission_lr.py` + `checkpoint_lr` (logistic regression): 91.1028 on validation.

## Train (regenerates `code/checkpoint/`)

```bash
cd code
python method/exp3.py export K=10 rep=1 out=checkpoint
```

What it does:
- Trains 10 DeepSets members (seeds 0-9) on the hard train groups of the train split only, with early stopping on a 12% train-internal holdout. The init and the holdout are seeded by the member seed.
- Writes `checkpoint/nn.npz` and `checkpoint/config.json`.
- Prints the numpy-vs-torch check (1.5e-7) and val-all / val-hard (94.24 / 87.04).

Cost: about 1 min wall on a 2-core CPU, no GPU (48 s with cached features: about 4 s per member plus validation; uncached feature extraction adds about 15 s).

**Verified:** running this command into a scratch directory reproduced both files bit-for-bit (sha256 identical to the table below), with torch 2.14.1 CPU on the same machine. Bitwise equality on other hardware or torch builds is not guaranteed, but scores should match to within float rounding.

Ablation checkpoints:
- v3: `python method/exp3.py export K=10 rep=0 out=checkpoint_v3`
- v2: `python method/nn_train.py export K=5 out=checkpoint_v2`
- v1 LR: `python method/train.py`

Experiments (`code/method/`, run from `code/`): `exp3.py <variant> [seeds]` runs the 5-fold train-CV plus validation for each v3/v4 variant and appends to `exp3_results.jsonl`. Variants: `v3`, `v3rep` (= v4), relational/pairwise models, hyperparameter sweeps. Full table: `results_v3.md`.

Feature code:
- `features.py` (v1 hand);
- `features_v3.py` (cross-candidate text, answer relations, pair features);
- `features_repair.py` (arithmetic repair);
- `features_v2.py` (negative result).

Model code:
- `nn_model.py` / `nn_train.py` (DeepSets);
- `nn_rel.py` (relational variant);
- `nn_infer.py` / `nn3_infer.py` (numpy inference).

Other scripts (v1/v2 analyses): `train.py`, `nn_train.py grid`, `cv.py`, `analysis.py`, `baseline_vote.py`, `explore.py`.

## Assets (`code/checkpoint/`, limit 16 MiB; total 8.92 MiB)

| file | bytes | sha256 |
|---|---|---|
| `nn.npz` | 9,351,582 | `8a1c1006060db7301901b8b20bea9f9957d074173b5b2e9a2c15b046bde77841` |
| `config.json` | 2,463 | `3f7cd767a02f2efbd98943be4e90e6ae0a14bfb2223678581db4f6e1ff398df4` |

## Timing

Fresh `predict.py` + scorer, 4 threads, on the local AMD EPYC 7A23 (2 cores available):
- validation: 3.1 s, 125 MB RSS;
- test: 2.1-2.4 s, 95 MB.

Feature extraction (regex/AST parsing in pure Python) dominates. The 30 s eligibility limit is defined on the reference Xeon Platinum 8457C, which we did not have; the local margin is about 10x.

## Seeds, protocol, records

- **Seeds:** ensemble seeds 0-9; CV folds `RandomState(0).randint`; early-stopping holdout seeded per member.
- **Fitting:** train split only (no train+val refit).
- **Selection:** 5-fold train-CV (hard groups) as the honest estimate; validation as a secondary check.
- **Test:** scored once per frozen version (see the version history).
- **Records:** `code/method/results_v1.md`, `results_nn.md`, `results_final.md`, `analysis_results.md`, `results_v3.md`, `exp3_results.jsonl`, `cv_results.json`, `nn_grid.json`, `analysis.json`, and `code/results/*.json`.

## AI-use disclosure

The code, experiments and report drafts were produced with Claude Code (Anthropic) AI agents's assist; the student is responsible for the submission. No external data or pretrained weights were used, and no LLM is called at inference.
