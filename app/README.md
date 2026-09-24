# Web app: loan approval, three models compared

An interactive app for the client: score an applicant with the three models, see *why*, and compare them
on the four dimensions of the brief (performance, fairness, stability, interpretability) plus a recommendation.

```bash
pip install -r app/requirements.txt         # xgboost >= 3 and tabpfn >= 9 are REQUIRED (see below)
streamlit run app/streamlit_app.py          # opens http://localhost:8501
# on Ethan's Mac (conda env "isaf"):  bash app/run_app.sh
```

⚠️ **Use the right environment.** The saved XGBoost model comes from xgboost 3. Read by xgboost 2.x, it loads
*without error* but re-codes the categories and gives wrong probabilities (errors up to 0.33). The app therefore
self-tests every model against its saved predictions at start-up and refuses to run in a wrong environment.

| Tab | What the client can do |
|---|---|
| 🧮 Score an applicant | pick a real test applicant (true outcome shown) or type a new one → P(repay), decision and main reasons for each model |
| 📈 Performance | AUC, Brier, profit, approval rate; profit-vs-threshold curves, ROC, calibration |
| ⚖️ Fairness | approval rates by group, 6 formal fairness tests (p-values), gender equivalence test (TOST) with an adjustable δ |
| 🔁 Stability | the same 1%-noise test for the three models + each notebook's stability evidence |
| 🔍 Interpretability | global drivers of each model, how each one can explain a single decision |
| ✅ Recommendation | trade-off table + recommendation (`app/recommendation.md`, draft to be validated by the team) |

The **LGD slider** (sidebar, default 60% = team convention) changes the economics: each model's approval
threshold is re-optimised on **its own validation sample**; the shared test set is never used to tune anything.

## Where the models come from
`app/build_artifacts.py` rebuilds everything in `app/artifacts/` (already committed; rerun only if a model changes, ~20 s):

- **Logistic regression**: exact replication of `01_logistic_regression.ipynb` (same AUC 0.9061, threshold 0.83, P&L 16.71 M).
- **XGBoost**: the frozen `gender_reweight` variant of `MLModels/XGBoost_ISAF_final_EN.ipynb`, retrained with the
  notebook's exact code. Floating-point differences between machines give AUC 0.916 here vs 0.918 in the notebook.
  To use the notebook's exact trees, copy `gender_reweight.json` from the notebook's run folder to
  `app/artifacts/xgb_notebook_model.json` and rerun the build.
- **TabPFN**: the robust model of `TabularFoundationModel/` (section 8). Test predictions are reused from its outputs.
  Live scoring rebuilds it from `TabularFoundationModel/outputs/model`.

## Technical note
torch (TabPFN) and xgboost ship incompatible OpenMP runtimes on macOS: in the same process they deadlock or
crash. TabPFN therefore runs in a separate background process (`app/tabpfn_worker.py`). It warms up for about 1 min
at start-up without blocking the page, then scores an applicant plus its explanation in about 1 s.
