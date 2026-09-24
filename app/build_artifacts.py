"""
build_artifacts.py — rebuilds the three team models EXACTLY as in their notebooks and saves
everything the web app needs in app/artifacts/ (run once, ~3 min, no GPU needed):

  * white-box   : logistic regression, 01_logistic_regression.ipynb (statsmodels Logit, 21 inputs)
  * ML model    : XGBoost "gender_reweight" (frozen variant), MLModels/XGBoost_ISAF_final_EN.ipynb
  * foundation  : robust TabPFN, TabularFoundationModel/ (predictions reused from its outputs/;
                  live scoring in the app rebuilds it from outputs/model)

Each model keeps its own validation sample, so the app can re-choose each model's threshold
for any LGD without ever touching the shared 20k test set.

    python app/build_artifacts.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "app" / "artifacts"
ART.mkdir(parents=True, exist_ok=True)
TFM = ROOT / "TabularFoundationModel"
sys.path.append(str(TFM / "src"))
from isaf_eval import make_splits, make_validation, add_noise, NUM_COLS  # noqa: E402

SEED, TARGET, ID = 42, "loan_paid_back", "id"
TEAM_LGD = 0.60                       # team convention (Slack): defaulted loan costs 60% of the amount
SENSITIVE = ["gender", "marital_status", "education_level", "employment_status"]
INPUTS = ["annual_income", "debt_to_income_ratio", "credit_score", "loan_amount", "interest_rate",
          "loan_purpose", "grade_subgrade", "employment_status", "education_level"]

full = pd.read_csv(ROOT / "data" / "train.csv")
full[TARGET] = full[TARGET].astype(int)
test = pd.read_csv(ROOT / "data" / "test_eval_20k.csv")
test[TARGET] = test[TARGET].astype(int)
pool = full[~full[ID].isin(test[ID])].copy()


def profit(y, p, amount, rate, t, lgd=TEAM_LGD):
    a = np.asarray(p) >= t
    return float(np.sum(np.where(np.asarray(y) == 1, amount * rate / 100, -lgd * amount)[a]))


def best_threshold(y, p, amount, rate, lgd=TEAM_LGD):
    grid = np.round(np.linspace(0.01, 0.99, 99), 2)
    return float(grid[int(np.argmax([profit(y, p, amount, rate, t, lgd) for t in grid]))])


def valid_frame(df, p):
    return pd.DataFrame({ID: df[ID].to_numpy(), "y": df[TARGET].to_numpy(),
                         "loan_amount": df.loan_amount.to_numpy(), "interest_rate": df.interest_rate.to_numpy(), "p": p})

# =============================================================================
# 1. White-box: logistic regression (01_logistic_regression.ipynb, cells 11–13)
# =============================================================================
def lr_prep(df):
    df = df.copy()
    letter = df["grade_subgrade"].str[0].map({l: i for i, l in enumerate("ABCDEF")})
    df["grade_ord"] = letter * 5 + df["grade_subgrade"].str[1].astype(int) - 1
    return df

LR_NUM = ["annual_income", "debt_to_income_ratio", "credit_score", "loan_amount", "interest_rate", "grade_ord"]
LR_CAT = ["education_level", "employment_status", "loan_purpose"]
tr = lr_prep(pool)
X_all = pd.get_dummies(tr[LR_NUM + LR_CAT], columns=LR_CAT, drop_first=True, dtype=float)
y_all = tr[TARGET].to_numpy()
X_tr, X_val, y_tr, y_val, idx_tr, idx_val = train_test_split(X_all, y_all, tr.index, test_size=0.20,
                                                             stratify=y_all, random_state=SEED)
mu, sd = X_tr[LR_NUM].mean(), X_tr[LR_NUM].std()

def lr_design(df):
    X = pd.get_dummies(lr_prep(df)[LR_NUM + LR_CAT], columns=LR_CAT, dtype=float)
    X = X.reindex(columns=X_all.columns, fill_value=0.0)
    X[LR_NUM] = (X[LR_NUM] - mu) / sd
    return sm.add_constant(X.astype(float), has_constant="add")

X_tr = X_tr.copy(); X_tr[LR_NUM] = (X_tr[LR_NUM] - mu) / sd
logit = sm.Logit(y_tr, sm.add_constant(X_tr.astype(float))).fit(disp=0, maxiter=200)
p_lr_val = logit.predict(lr_design(tr.loc[idx_val])).to_numpy()
p_lr_test = logit.predict(lr_design(test)).to_numpy()
X_test_lr = lr_design(test)
lr_meta = {"params": logit.params.to_dict(), "bse": logit.bse.to_dict(), "pvalues": logit.pvalues.to_dict(),
           "columns": list(X_all.columns), "num": LR_NUM, "cat": LR_CAT,
           "mu": mu.to_dict(), "sd": sd.to_dict(), "xbar": X_test_lr.mean().to_dict()}
(ART / "lr_model.json").write_text(json.dumps(lr_meta, indent=1))
valid_frame(pool.loc[idx_val], p_lr_val).to_csv(ART / "valid_lr.csv.gz", index=False)
t_lr = best_threshold(y_val, p_lr_val, pool.loc[idx_val].loan_amount.to_numpy(), pool.loc[idx_val].interest_rate.to_numpy())
print(f"LR    test AUC {roc_auc_score(test[TARGET], p_lr_test):.4f} | threshold {t_lr} "
      f"| P&L {profit(test[TARGET], p_lr_test, test.loan_amount, test.interest_rate, t_lr)/1e6:.2f} M "
      f"(notebook: 0.9061, 0.83, 16.71 M)")

# =============================================================================
# 2. ML model: XGBoost, frozen "gender_reweight" variant (XGBoost_ISAF_final_EN.ipynb, cells 3, 5, 25)
# =============================================================================
XGB_CAT = ["loan_purpose", "grade_subgrade", "employment_status", "education_level"]
core, held = train_test_split(pool, test_size=.20, stratify=pool[TARGET], random_state=SEED)
fit, rest = train_test_split(core, test_size=.25, stratify=core[TARGET], random_state=SEED)
es, thr = train_test_split(rest, test_size=.5, stratify=rest[TARGET], random_state=SEED + 1)
categories = {c: sorted(fit[c].unique()) for c in INPUTS if c in XGB_CAT}

def xgb_encode(df):
    X = df[INPUTS].copy()
    for c in INPUTS:
        X[c] = pd.Categorical(X[c], categories=categories[c]) if c in XGB_CAT else pd.to_numeric(X[c])
    return X

# gender x Y reweighting (training weights only; gender is NOT a model input)
counts = fit.groupby(["gender", TARGET]).size()
gd, yd = fit.gender.value_counts(normalize=True), fit[TARGET].value_counts(normalize=True)
w = np.array([gd[g] * yd[y] / (counts.loc[(g, y)] / len(fit)) for g, y in zip(fit.gender, fit[TARGET])])
w /= w.mean()
model = XGBClassifier(objective="binary:logistic", eval_metric="auc", n_estimators=1500, max_depth=5, learning_rate=.05,
                      min_child_weight=5, subsample=.8, colsample_bytree=.8, reg_alpha=.5, reg_lambda=5.,
                      scale_pos_weight=1., enable_categorical=True, tree_method="hist", early_stopping_rounds=50,
                      random_state=SEED, n_jobs=4)
SAVED = ART / "xgb_notebook_model.json"   # optional: the exact model saved by the XGBoost notebook
if SAVED.exists():
    # Oscar's run folder (MLModels/blackbox_extended_runs/full/<run>/gender_reweight.json): trees identical to the notebook
    booster = xgb.Booster(); booster.load_model(str(SAVED))
    n_trees = booster.num_boosted_rounds()
else:
    # Retrained with the notebook's exact code. Floating-point differences between machines change the
    # early-stopping point slightly (AUC 0.916 here vs 0.918 in the notebook, within its 95% CI).
    model.fit(xgb_encode(fit), fit[TARGET], sample_weight=w, eval_set=[(xgb_encode(es), es[TARGET])], verbose=False)
    booster = model.get_booster()
    n_trees = model.best_iteration + 1

def xgb_predict(df):
    return expit(booster.predict(xgb.DMatrix(xgb_encode(df), enable_categorical=True), output_margin=True,
                                 iteration_range=(0, n_trees)))

p_xgb_thr, p_xgb_test = xgb_predict(thr), xgb_predict(test)
booster.save_model(str(ART / "xgb_model.json"))
(ART / "xgb_meta.json").write_text(json.dumps({"categories": categories, "n_trees": n_trees, "inputs": INPUTS}, indent=1))
valid_frame(thr, p_xgb_thr).to_csv(ART / "valid_xgb.csv.gz", index=False)
t_xgb = best_threshold(thr[TARGET], p_xgb_thr, thr.loan_amount.to_numpy(), thr.interest_rate.to_numpy())
print(f"XGB   test AUC {roc_auc_score(test[TARGET], p_xgb_test):.4f} | threshold {t_xgb} "
      f"| P&L {profit(test[TARGET], p_xgb_test, test.loan_amount, test.interest_rate, t_xgb)/1e6:.2f} M "
      f"(notebook: 0.9175, 0.84, 17.60 M)")

# =============================================================================
# 3. Tabular foundation model: robust TabPFN (TabularFoundationModel/, section 8)
# =============================================================================
cache = TFM / "outputs" / "cache"
fit_pool, test_tfm = make_splits(full)
_, valid_tfm = make_validation(fit_pool)
p_tfm_val = np.mean([pd.read_pickle(cache / f"robust_A_{k}.pkl")["valid"] for k in range(5)], axis=0)
preds_tfm = pd.read_csv(TFM / "outputs" / "preds_tabpfn.csv")[[ID, "p_tabpfn_robust"]]
valid_frame(valid_tfm, p_tfm_val).to_csv(ART / "valid_tabpfn.csv.gz", index=False)

# =============================================================================
# 4. Shared test set: all predictions + the attributes used by the app
# =============================================================================
out = test[[ID, TARGET, *INPUTS, "gender", "marital_status"]].rename(columns={TARGET: "y"})
out["p_lr"], out["p_xgb"] = p_lr_test, p_xgb_test
out = out.merge(preds_tfm.rename(columns={"p_tabpfn_robust": "p_tabpfn"}), on=ID, how="left")
assert out.p_tabpfn.notna().all(), "TabPFN predictions must cover the shared test set"
out.to_csv(ART / "test_predictions.csv.gz", index=False)
t_tfm = best_threshold(valid_tfm[TARGET], p_tfm_val, valid_tfm.loan_amount.to_numpy(), valid_tfm.interest_rate.to_numpy())
print(f"TabPFN test AUC {roc_auc_score(out.y, out.p_tabpfn):.4f} | threshold {t_tfm} "
      f"| P&L {profit(out.y, out.p_tabpfn, out.loan_amount, out.interest_rate, t_tfm)/1e6:.2f} M (LGD 60%)")

# =============================================================================
# 5. Same input-noise test for the three models (first 2,000 rows of the TabPFN test order,
#    1% Gaussian noise on numeric inputs, identical draw as the TabPFN notebook)
# =============================================================================
rows = test_tfm.iloc[:2000].reset_index(drop=True)
noisy = add_noise(rows, NUM_COLS, 0.01)
p_tfm_noisy = np.mean([pd.read_pickle(cache / f"robust_A_{k}.pkl")["noise1"] for k in range(5)], axis=0)
p_tfm_clean = out.set_index(ID).loc[rows[ID], "p_tabpfn"].to_numpy()
noise = {"lr": (logit.predict(lr_design(rows)).to_numpy(), logit.predict(lr_design(noisy)).to_numpy()),
         "xgb": (xgb_predict(rows), xgb_predict(noisy)),
         "tabpfn": (p_tfm_clean, p_tfm_noisy)}
pd.DataFrame({f"{m}_{k}": v for m, (a, b) in noise.items() for k, v in (("clean", a), ("noisy", b))}) \
  .to_csv(ART / "noise_1pct.csv.gz", index=False)

# typical applicant (reference for TabPFN ablation explanations)
typical = {c: (float(pool[c].median()) if c in NUM_COLS else pool[c].mode()[0]) for c in INPUTS}
(ART / "typical_applicant.json").write_text(json.dumps(typical, indent=1))
print("artifacts saved to", ART)
