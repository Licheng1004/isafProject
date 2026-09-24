# ISAF project: which model should approve our loans?

HEC Paris, MSc Data Science & AI for Business, *Interpretability, Stability and Algorithmic Fairness*
(Prof. C. Pérignon), Fall 2026.

**Client:** a consumer lender. **Decision:** approve a personal loan if the predicted probability that it is paid
back, P(repay), is above a threshold. **Data:** [Kaggle Playground S5E11](https://www.kaggle.com/competitions/playground-series-s5e11)
(594k loans, 11 features, synthetic; target `loan_paid_back`, 1 = repaid).

We compare three models on the four dimensions of the brief (performance, interpretability, stability, fairness)
and deliver an interactive web app for the client.

| | White box | Machine learning | Tabular foundation model |
|---|---|---|---|
| Model | Logistic regression | XGBoost (`gender_reweight` variant) | TabPFN-v3.5 (robust: 5-context bagging) |
| Notebook | `01_logistic_regression.ipynb` | `MLModels/XGBoost_ISAF_final_EN.ipynb` | `TabularFoundationModel/03_tabular_foundation_model.ipynb` |
| AUC (shared test set) | 0.906 | 0.916 | 0.916 |
| Profit on 20k test loans (LGD 60%) | €16.7M | €17.4M | €17.4M |
| Decisions flipped by 1% input noise | 0.2% | 2.2% | 2.1% |
| Explanation of one decision | exact (coefficients) | exact (TreeSHAP) | approximate (post-hoc), slow |

*Figures from the web app, same test set and same economics for all models.*

---

## 1. Quick start: run the web app

```bash
git clone https://gitlab.code.hfactory.io/ethan.ben-hamou/isaf-project.git
cd isaf-project
python3.11 -m venv .venv && source .venv/bin/activate    # or: conda create -n isaf python=3.11 && conda activate isaf
pip install -r requirements.txt
streamlit run app/streamlit_app.py                        # opens http://localhost:8501
```

On Ethan's Mac (conda env `isaf` already set up): `bash app/run_app.sh`. It uses the right Python and keeps the Mac awake.

- The page is usable immediately. **TabPFN warms up in the background for about 1 min** (first launch: its weights
  are downloaded and a browser tab asks you to accept the free Prior Labs licence). After that, it scores one
  applicant in about 1 s.
- Nothing to compute beforehand: the models' artifacts are committed in `app/artifacts/`.
- At start-up the app **checks that every model reproduces its saved predictions**. If it refuses to start, you are
  in the wrong environment (see *Troubleshooting*).

**What the client can do:** score a real or a new applicant (P(repay), decision and main reasons for each model),
compare performance and profit, audit fairness (6 formal tests plus an equivalence test), compare stability and
interpretability, and read the recommendation. The **LGD slider** changes the economics; each model's threshold
is re-optimised on its own validation sample. Details: [`app/README.md`](app/README.md).

---

## 2. Run the full analysis

Use the same environment (`pip install -r requirements.txt`). Each notebook is self-contained. **Run each one from
its own folder**, because the paths are relative:

| Step | Notebook | Run from | Runtime |
|---|---|---|---|
| EDA | `EDA/EDA.ipynb` (also needs `pip install ydata-profiling`) | `EDA/` | a few minutes |
| White box | `01_logistic_regression.ipynb` | repo root | ~1 min |
| ML model | `MLModels/XGBoost_ISAF_final_EN.ipynb` (`MODE = "full"`) | `MLModels/` | long: 5 seeds + exact XPER over 512 coalitions. Set `MODE = "smoke"` for a quick check |
| Foundation model | `TabularFoundationModel/03_tabular_foundation_model.ipynb` | `TabularFoundationModel/` | **~1 min** (all TabPFN results are cached in `outputs/cache/`); ~3 h on a laptop GPU if the cache is deleted |
| Web-app artifacts | `python app/build_artifacts.py` | repo root | ~20 s, only needed if a model changes |

In Jupyter or VS Code, select the kernel of the environment where you ran `pip install -r requirements.txt`.

```bash
jupyter notebook                   # or open the .ipynb files in VS Code
# e.g. TabPFN from the command line:
cd TabularFoundationModel && jupyter nbconvert --to notebook --execute --inplace 03_tabular_foundation_model.ipynb
```

TabPFN uses the GPU automatically: Apple-silicon `mps` or NVIDIA `cuda`, otherwise CPU, which is slow if the cache is deleted.

---

## 3. Shared protocol (identical for the three models)

- **Test set:** `data/test_eval_20k.csv`, 20,000 rows removed from all training data and used for reporting only.
- **Economics:** a repaid loan earns `loan_amount × interest_rate / 100`, a default costs `LGD × loan_amount` with
  **LGD = 60%**, a rejected loan earns 0.
- **Threshold:** tuned on each model's own validation split, never on the test set.
- **Inputs:** `gender` and `marital_status` are **never model inputs**, because using them in credit decisions is
  prohibited (EU Directive 2004/113/EC, US ECOA). They are kept only to **audit** fairness, together with
  `education_level` and `employment_status`.
- **Fairness toolkit** (course methods), shared in `TabularFoundationModel/src/isaf_eval.py`:
  - `fairness_tests`: statistical and conditional parity, equal opportunity, predictive equality, equalized odds, sufficiency;
  - `tost_parity`: equivalence test;
  - `fpdp`: fairness PDP and candidate variables;
  - `xper`: performance decomposition;
  - `lime_explain`: local surrogate.

---

## 4. Repository structure

```
data/                         train.csv (Kaggle), test_eval_20k.csv (shared test set)
EDA/                          exploratory analysis
01_logistic_regression.ipynb  white-box model
MLModels/                     XGBoost (final: XGBoost_ISAF_final_EN.ipynb) + earlier experiments
TabularFoundationModel/       TabPFN notebook, src/ (toolkit + TabPFN wrapper), outputs/ (figures, cache, model)
app/                          Streamlit web app, its artifacts, build script, recommendation
project_request/              brief and course slides
requirements.txt              one environment for everything
```

---

## 5. Troubleshooting

| Problem | Fix |
|---|---|
| App stops with *"The models do not reproduce their saved predictions"* | Wrong environment: xgboost 2.x silently misreads the xgboost-3 model (errors up to 0.33). Run `pip install -r requirements.txt` and launch from that environment. |
| `TypeError: altair_chart() got an unexpected keyword argument 'width'` | Old copy of the app: `git pull`. |
| TabPFN stays "warming up" | Normal for about 1 min. The first time, accept the Prior Labs licence in the browser tab that opens. |
| TabPFN notebook or app hangs / crashes when xgboost is imported | torch and xgboost ship conflicting OpenMP runtimes on macOS. Never import both in the same process (the app runs TabPFN in a separate worker for this reason). |
| Long TabPFN computations freeze | The Mac went to sleep. Keep it plugged in with the lid open (`app/run_app.sh` uses `caffeinate`). |
| `FileNotFoundError: data/...` | Run the notebook from its own folder (table in section 2). |
