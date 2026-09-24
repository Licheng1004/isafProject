# Model 3: Tabular Foundation Model (TabPFN)

HEC Paris, MSc DS&AI for Business, *Interpretability, Stability and Algorithmic Fairness* (Prof. C. Pérignon), Fall 2026.

**Client:** a consumer lender. **Target:** `loan_paid_back` (1 = repaid, 0 = default). **Decision:** approve a loan if P(paid back) ≥ threshold.
**Data:** [Kaggle Playground S5E11](https://www.kaggle.com/competitions/playground-series-s5e11) (594k loans, 11 features, synthetic).

This folder contains **model 3, the Tabular Foundation Model (TabPFN)**, with its evaluation toolkit (`src/isaf_eval.py`).
Its 20k test set is **exactly `../data/test_eval_20k.csv`** (same 20,000 loan ids), so the results are directly comparable with the LR and XGBoost notebooks.

## Structure

```
03_tabular_foundation_model.ipynb   full TabPFN analysis: data → performance → interpretability → stability → fairness → robust model
src/isaf_eval.py                    SHARED toolkit: splits, metrics, fairness tests (chi², TOST, FPDP), XPER, LIME, bootstrap
src/tabpfn_model.py                 TabPFN wrapper (batched GPU inference, disk cache, production loader for the app)
outputs/figures/                    PNG figures ready for the slides
outputs/summary_tabpfn.json         every headline number (for the comparison slide / the website)
outputs/preds_tabpfn.csv            test-set predictions (id, y, p) to compare with LR / XGB
outputs/model/                      what the web app needs to rebuild the ROBUST model (5 context sets + config)
outputs/cache/                      cached TabPFN results, committed, so a rerun takes about 1 min instead of about 3 h
```

## Run

```bash
pip install -r requirements.txt
cd TabularFoundationModel && jupyter notebook 03_tabular_foundation_model.ipynb     # run all; reads ../data/train.csv
```

- TabPFN downloads its weights on first use. The first time, it opens a browser to accept the Prior Labs license and stores the token in `~/.cache/tabpfn/`. **Never write the token in the notebook.**
- On a Mac with Apple silicon it uses the GPU (`mps`); on Colab it uses `cuda`.
- To recompute everything from scratch: `Cache(..., recompute=True)` in the first code cell, or delete `outputs/cache/`.

## Toolkit (usable from any notebook: `sys.path.append("TabularFoundationModel/src")`)

```python
import sys; sys.path.append("src")
from isaf_eval import *
df = load_data()
fit_pool, test = make_splits(df)               # test = 20k rows, identical for all models
train_pool, valid = make_validation(fit_pool)  # choose the threshold on `valid`, never on `test`
t = best_threshold(valid.loan_paid_back, p_valid, valid.loan_amount, valid.interest_rate)
statistical_metrics(test.loan_paid_back, p_test)
economic_metrics(test.loan_paid_back, p_test, test.loan_amount, test.interest_rate, t)
fairness_report(test.loan_paid_back, p_test, test[SENSITIVE], t)
prediction_stability(P_refits, t)             # P_refits: (n_refits, n_rows) on the same rows

# course methods, model-agnostic (predict_fn takes a feature table, returns P(paid back))
fairness_tests(test.loan_paid_back, p_test, test.gender, t, test.credit_score)   # 6 definitions, p-values
tost_parity(p_test >= t, test.gender, "Female", "Male", delta=0.02)               # equivalence: certify fairness
fpdp(predict_fn, X, y, group, "employment_status", grid, t)                       # candidate variables
xper(predict_fn, X, {"AUC": ...}, groups)                                         # performance decomposition
lime_explain(predict_fn, x_row, X_ref)                                            # local surrogate
```

Economic assumptions: gain = `loan_amount × interest_rate` (1 year), loss = `loan_amount × LGD`, with LGD = 70%. The notebook includes a sensitivity analysis on LGD.

## Using the model in the web app

```python
import sys; sys.path.append("src")
from tabpfn_model import load_production_model
model = load_production_model("outputs/model")          # builds + caches the robust TabPFN (~1 min, once at app startup)
model.decide(applicants_df)   # raw columns as in train.csv -> p_paid_back, approved (~1 s per applicant)
```

The production model is the **robust TabPFN** of section 8: no gender / marital status, TabPFN's row-hash
"fingerprint" feature switched off, and 5 context sets of 10k rows averaged (context bagging).
