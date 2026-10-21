# Research notes — DASE7506 MP2 (LLM reasoning selection)

## Goal
Implement `predict(bank, arrays, checkpoint)` in `code/submission.py` that selects 1 of 4 Qwen3-0.6B GSM8K candidates per question, maximise test selection accuracy under constraints, and produce a ≤10-page report with comparisons, an ablation, failure analysis, resource use. Deadline 2026-10-31 23:59 UTC+8.

## Constraints (from guide/GUIDE.md — do not violate)
- Train on train split only; validation for dev/model selection; test references ONLY for final scoring (never tune on test).
- Don't modify data, `common.py` parser, scorer. No external data/pretrained weights. No LLM calls at inference. IDs only for alignment.
- Inference: CPU, ≤30 s total (fresh predict process + scoring, startup included), 4 threads total. Assets ≤16 MiB uncompressed.
- Training: CPU or ≤1 RTX 4090 (remote `ps` available via ps-gpu-machine skill). Record seeds, deps, training cost.
- Method must work on unseen audit bank (no hard-coded N / IDs).
- Submission repo: public, with code, report, commands, official eval JSON, seeds, asset hashes, AI-use disclosure. Commits must NOT list Claude as author (git author = user).
- `runs/` is gitignored → keep the shipped checkpoint in a committed dir (e.g. `code/checkpoint/`).

## Environment
- Local: Python 3.12.3, 2 CPUs (AMD EPYC 7A23), no numpy/torch installed yet → use `code/.venv`.
- Git remote origin = user's fork git@github.com:Lyan9145/DASE7506-MP2.git; no `gh` CLI.

## Decisions
- Worker model: sonnet (default).
- Final model trained on TRAIN ONLY (guide: "fit only on supplied training inputs and references"; val = selection). Do not refit on train+val.
- Inference implemented numpy-only, checkpoint as npz/json in `code/checkpoint/` (no pickle).

## Findings
- Env: `code/.venv` (needed apt python3.12-venv); pins in `code/requirements-method.txt` (numpy 2.5.3, sklearn 1.9.1, scipy 1.18.1). `submission.py` can `import common` (predict.py adds method dir to sys.path). Exploration: `code/method/explore.py`.
- Oracle = 100% all splits; per-candidate acc ≈0.755. Hard groups (≥2 distinct valid answers) = test definition: train 2114, val 355, test 467 (all). Hard groups have 1–3 correct (never 0/4).
- Unparseable answers ~5% (test 5.9%). Embeddings not normalized (norm ~58). Test texts longer (552 vs 480 chars).
- parse_answer prefers `####` tail; candidates with literal "#### <numeric answer>" placeholder parse as None even if `\boxed{}` present.
- **Test-like metric = val hard-subset accuracy.** Baselines (val all / val hard): idx0 78.2/54.9; random 75.7/51.1; vote tie→lowest 86.2/69.0; vote tie→longest 86.97/70.7 (current submission.py, 0.24 s). Train hard: vote tie→longest 69.9.
- Ties on ~25% of hard groups; longest-trace wins ties 50% vs shortest 27% → short traces often wrong.
- v1 learned selector (`code/method/{features,selector,train}.py`, results `code/method/results_v1.md`): 37 hand feats + cross-cand cosines + PCA64 emb, group-relative (minus group mean), LR, answer-cluster SUM aggregation, trained on hard train groups, C=10 → val-all 91.10 / val-hard 80.00 (official run 91.1028). Train-CV hard ≈77.7 (honest est.). 1.10 s predict+score, ckpt 275 KB.
- Ablation facts: hand-only 76.3 hard; emb-only 67–72; hand+emb 78.3; group-relative+cluster-sum 79.4; HGB 79.2 (no gain). Top weights: is_int(+), truncation(−), negative answer(−), has `####`(−), vote share(+). Arithmetic check rarely fires (~2 exprs/cand).

## Dead ends

## Open questions

## Next steps
2. Listwise neural model (numpy inference) over 4 candidates w/ full emb + hand feats; compare vs v1 LR. Richer arithmetic checks.
3. Ablations per research questions; timing + asset-size measurement.
4. Report (≤10 pages) + README commands + commit/push.
