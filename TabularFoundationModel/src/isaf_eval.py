"""
isaf_eval.py — shared evaluation toolkit for the ISAF project (HEC, Pérignon, Fall 2026).

Every model (white-box, ML, tabular foundation model) should be evaluated with the SAME
splits and the SAME functions, so the three notebooks are directly comparable in the deck.

    import sys; sys.path.append("src")
    from isaf_eval import *
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, chi2_contingency, spearmanr
from scipy.stats import t as t_dist
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             log_loss, roc_curve)
from sklearn.model_selection import train_test_split
from fairlearn.metrics import (MetricFrame, selection_rate, true_positive_rate,
                               false_positive_rate, demographic_parity_difference,
                               demographic_parity_ratio, equalized_odds_difference)

SEED = 42
TARGET = "loan_paid_back"
NUM_COLS = ["annual_income", "debt_to_income_ratio", "credit_score",
            "loan_amount", "interest_rate"]
CAT_COLS = ["gender", "marital_status", "education_level",
            "employment_status", "loan_purpose", "grade_subgrade"]
SENSITIVE = ["gender", "marital_status", "education_level", "employment_status"]
FEATURES = NUM_COLS + CAT_COLS

# Economic assumptions shared by the three models (sensitivity analysis in the notebooks)
LGD, HORIZON = 0.7, 1.0

# ----------------------------------------------------------------------------
# 1. Data & splits — identical for the three models
# ----------------------------------------------------------------------------
def load_data(root: Path | str = ".") -> pd.DataFrame:
    """Kaggle Playground S5E11 train.csv (594k loans). Kaggle's test.csv has no labels -> unused."""
    root = Path(root)
    for f in (root / "data" / "train.csv.gz", root / "data" / "train.csv",
              root.parent / "data" / "train.csv"):          # repo layout: TabularFoundationModel/ next to data/
        if f.exists():
            df = pd.read_csv(f)
            df[TARGET] = df[TARGET].astype(int)
            return df
    raise FileNotFoundError("data/train.csv.gz not found — see README.md")


def make_splits(df: pd.DataFrame, n_eval: int = 20_000, seed: int = SEED):
    """
    fit_pool : 80 % of rows, from which each model draws its training data
               (LR/XGB may use all of it; TabPFN uses a 10k context sample)
    eval     : a fixed stratified sample of `n_eval` rows from the remaining 20 %
               -> the common TEST set on which all headline numbers are reported.
    """
    fit_pool, holdout = train_test_split(df, test_size=0.20, stratify=df[TARGET],
                                         random_state=seed)
    eval_df, _ = train_test_split(holdout, train_size=n_eval, stratify=holdout[TARGET],
                                  random_state=seed)
    return fit_pool.reset_index(drop=True), eval_df.reset_index(drop=True)


def make_validation(fit_pool: pd.DataFrame, n_valid: int = 5_000, seed: int = SEED):
    """
    Carve a validation set out of fit_pool, used to CHOOSE the decision threshold
    (so that the threshold is never tuned on the test set). Returns (train_pool, valid).
    """
    train_pool, valid = train_test_split(fit_pool, test_size=n_valid,
                                         stratify=fit_pool[TARGET], random_state=seed)
    return train_pool.reset_index(drop=True), valid.reset_index(drop=True)


def stratified_subsample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    sub, _ = train_test_split(df, train_size=n, stratify=df[TARGET], random_state=seed)
    return sub.reset_index(drop=True)

# ----------------------------------------------------------------------------
# 2. Statistical performance
# ----------------------------------------------------------------------------
def statistical_metrics(y, p) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    fpr, tpr, _ = roc_curve(y, p)
    auc = roc_auc_score(y, p)
    return {
        "AUC": auc,
        "Gini": 2 * auc - 1,
        "PR-AUC": average_precision_score(y, p),
        "KS": float(np.max(tpr - fpr)),
        "Brier": brier_score_loss(y, p),
        "LogLoss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)),
    }

# ----------------------------------------------------------------------------
# 3. Economic performance
# ----------------------------------------------------------------------------
# Client = a consumer lender. Decision = approve the loan if P(paid back) >= threshold.
#   * Approved & paid back  -> gain  = loan_amount * interest_rate/100 * horizon_years
#   * Approved & default    -> loss  = loan_amount * LGD
#   * Rejected              -> 0
def _gain_loss(loan_amount, interest_rate, lgd, horizon_years):
    loan_amount, interest_rate = np.asarray(loan_amount), np.asarray(interest_rate)
    return loan_amount * interest_rate / 100.0 * horizon_years, loan_amount * lgd


def loan_profit(y, p, loan_amount, interest_rate, threshold, lgd=LGD, horizon_years=HORIZON):
    gain, loss = _gain_loss(loan_amount, interest_rate, lgd, horizon_years)
    profit = np.where(np.asarray(y) == 1, gain, -loss)
    return float(np.sum(profit[np.asarray(p) >= threshold]))


def profit_curve(y, p, loan_amount, interest_rate, lgd=LGD, horizon_years=HORIZON,
                 thresholds=None):
    thresholds = np.round(np.linspace(0.05, 0.95, 91), 2) if thresholds is None else thresholds
    prof = [loan_profit(y, p, loan_amount, interest_rate, t, lgd, horizon_years)
            for t in thresholds]
    return pd.DataFrame({"threshold": thresholds, "profit": prof})


def best_threshold(y, p, loan_amount, interest_rate, lgd=LGD, horizon_years=HORIZON) -> float:
    """Profit-maximising global threshold. Call it on the VALIDATION set."""
    curve = profit_curve(y, p, loan_amount, interest_rate, lgd, horizon_years)
    return float(curve.loc[curve.profit.idxmax(), "threshold"])


def economic_metrics(y, p, loan_amount, interest_rate, threshold,
                     lgd=LGD, horizon_years=HORIZON) -> dict:
    """Economic performance at a GIVEN threshold (chosen beforehand on validation)."""
    y, p = np.asarray(y), np.asarray(p)
    gain, loss = _gain_loss(loan_amount, interest_rate, lgd, horizon_years)
    approve = p >= threshold
    profit = loan_profit(y, p, loan_amount, interest_rate, threshold, lgd, horizon_years)
    approve_all = loan_profit(y, p, loan_amount, interest_rate, -1, lgd, horizon_years)
    oracle = loan_profit(y, y, loan_amount, interest_rate, 0.5, lgd, horizon_years)
    # Loan-specific expected-value rule: approve iff p*gain > (1-p)*loss (rewards calibration)
    ev_rule = p * gain - (1 - p) * loss > 0
    ev_profit = float(np.sum(np.where(y == 1, gain, -loss)[ev_rule]))
    return {
        "threshold": float(threshold),
        "profit": profit,
        "profit_expected_value_rule": ev_profit,
        "profit_approve_all": approve_all,
        "profit_oracle": oracle,
        "share_of_oracle_captured": (profit - approve_all) / (oracle - approve_all),
        "approval_rate": float(approve.mean()),
        "default_rate_among_approved": float(1 - y[approve].mean()) if approve.any() else np.nan,
        "good_payers_rejected": float(np.mean(~approve[y == 1])),
    }

# ----------------------------------------------------------------------------
# 4. Fairness
# ----------------------------------------------------------------------------
def fairness_report(y, p, sensitive: pd.DataFrame, threshold: float) -> dict:
    """
    For each sensitive attribute:
      - per-group table (n, base rate, mean score, approval rate, TPR, FPR, AUC)
      - demographic parity difference/ratio, equalized odds difference
    'Approval' = predicted positive = p >= threshold.
    """
    y, p = np.asarray(y), np.asarray(p)
    yhat = (p >= threshold).astype(int)
    out = {}
    for col in sensitive.columns:
        g = sensitive[col].to_numpy()
        mf = MetricFrame(metrics={"approval_rate": selection_rate,
                                  "TPR": true_positive_rate,
                                  "FPR": false_positive_rate},
                         y_true=y, y_pred=yhat, sensitive_features=g)
        tab = mf.by_group.copy()
        tab.index.name = col
        tab["n"] = pd.Series(g).value_counts()
        tab["base_rate"] = pd.Series(y).groupby(g).mean()
        tab["mean_score"] = pd.Series(p).groupby(g).mean()
        tab["AUC"] = pd.Series({grp: roc_auc_score(y[g == grp], p[g == grp])
                                if len(np.unique(y[g == grp])) > 1 else np.nan
                                for grp in np.unique(g)})
        out[col] = {
            "by_group": tab[["n", "base_rate", "mean_score", "approval_rate", "TPR", "FPR", "AUC"]],
            "demographic_parity_diff": demographic_parity_difference(y, yhat, sensitive_features=g),
            "demographic_parity_ratio": demographic_parity_ratio(y, yhat, sensitive_features=g),
            "equalized_odds_diff": equalized_odds_difference(y, yhat, sensitive_features=g),
        }
    return out


def fairness_summary_table(report: dict) -> pd.DataFrame:
    rows = [{"attribute": col,
             "DP difference": r["demographic_parity_diff"],
             "DP ratio": r["demographic_parity_ratio"],
             "passes 4/5 rule": r["demographic_parity_ratio"] >= 0.8,
             "EO difference": r["equalized_odds_diff"],
             "min group AUC": r["by_group"]["AUC"].min(),
             "max group AUC": r["by_group"]["AUC"].max()}
            for col, r in report.items()]
    return pd.DataFrame(rows).set_index("attribute")


def _chi2(a, b, strata=None):
    """Chi-square test of independence between a and b; with `strata`, the stratified version
    (sum of per-stratum chi-square statistics and degrees of freedom) = conditional independence."""
    a, b = np.asarray(a), np.asarray(b)
    strata = np.zeros(len(a)) if strata is None else np.asarray(strata)
    stat, dof = 0.0, 0
    for s in np.unique(strata):
        m = strata == s
        tab = pd.crosstab(a[m], b[m])
        tab = tab.loc[tab.sum(axis=1) > 0, tab.sum(axis=0) > 0]
        if tab.shape[0] > 1 and tab.shape[1] > 1:
            st, _, df_, _ = chi2_contingency(tab, correction=False)
            stat, dof = stat + st, dof + df_
    # dof = 0: one of the variables is constant (e.g. everybody approved) -> independence holds trivially
    return stat, dof, (float(chi2.sf(stat, dof)) if dof > 0 else 1.0)


def fairness_tests(y, p, group, threshold, risk_proxy=None, n_bins=5) -> pd.Series:
    """
    Formal fairness tests of Hurlin, Pérignon & Saurin (2026), model- and definition-agnostic.
    H0 = "the model is fair" under each definition; small p-value -> fairness rejected.
      Independence: statistical parity (Yhat ⟂ D), conditional statistical parity (Yhat ⟂ D | risk bins)
      Separation:   equal opportunity (Yhat ⟂ D | Y=1), predictive equality (Yhat ⟂ D | Y=0),
                    equalized odds (Yhat ⟂ D | Y)
      Sufficiency:  Y ⟂ D | score deciles (same score -> same repayment rate in every group)
    Y = 1 is the favourable outcome (paid back), as in the course.
    """
    y, p, g = np.asarray(y), np.asarray(p), np.asarray(group)
    yhat = (p >= threshold).astype(int)
    out = {"statistical parity": _chi2(yhat, g)[2]}
    if risk_proxy is not None:
        bins = pd.qcut(np.asarray(risk_proxy), n_bins, labels=False, duplicates="drop")
        out["conditional statistical parity"] = _chi2(yhat, g, bins)[2]
    out["equal opportunity"] = _chi2(yhat[y == 1], g[y == 1])[2]
    out["predictive equality"] = _chi2(yhat[y == 0], g[y == 0])[2]
    out["equalized odds"] = _chi2(yhat, g, y)[2]
    out["sufficiency"] = _chi2(y, g, pd.qcut(p, 10, labels=False, duplicates="drop"))[2]
    return pd.Series(out, name="p-value")


def tost_parity(approved, group, protected, reference, delta=0.02, alpha=0.05) -> dict:
    """
    Fairness EQUIVALENCE test (TOST, Schuirmann 1987), as in the course (slides 271-277).
    H0: |p_ref - p_prot| >= delta (unfair)  vs  H1: |p_ref - p_prot| < delta (fair within delta).
    Rejecting H0 CERTIFIES that the approval-rate gap is smaller than delta.
    `delta` is a normative tolerance to be fixed BEFORE looking at the data.
    """
    a, g = np.asarray(approved).astype(float), np.asarray(group)
    a_r, a_p = a[g == reference], a[g == protected]
    pr, pp, nr, npr = a_r.mean(), a_p.mean(), len(a_r), len(a_p)
    theta = pr - pp
    se = np.sqrt(pr * (1 - pr) / nr + pp * (1 - pp) / npr)
    z_l, z_u = (theta + delta) / se, (delta - theta) / se
    crit = t_dist.ppf(1 - alpha, nr + npr - 2)
    return {"gap (ref - protected)": theta, "delta": delta,
            "CI 90%": (theta - crit * se, theta + crit * se),
            "z_L": z_l, "z_U": z_u, "critical value": crit,
            "p-value": float(max(t_dist.sf(z_l, nr + npr - 2), t_dist.sf(z_u, nr + npr - 2))),
            "certified fair": bool(z_l > crit and z_u > crit)}


def fpdp(predict_fn, X: pd.DataFrame, y, group, feature, grid, threshold, test="statistical parity",
         risk_proxy=None):
    """
    Fairness Partial Dependence Plot (Hurlin, Pérignon & Saurin): set `feature` to the same value
    for everyone, re-predict, re-run the fairness test. Returns the p-value for each grid value.
    A feature is a CANDIDATE (channel of unfairness) if some value brings the p-value above 5 %.
    The AUC column = cost of the 'mitigation without re-estimation' (fix the candidate to that value).
    All grid points are scored in ONE call to predict_fn (matters for slow models).
    """
    Xs = pd.concat([X.assign(**{feature: v}) for v in grid], ignore_index=True)
    P = np.asarray(predict_fn(Xs)).reshape(len(grid), len(X))
    return pd.DataFrame({"p-value": [fairness_tests(y, P[k], group, threshold, risk_proxy)[test] for k in range(len(grid))],
                         "AUC": [roc_auc_score(y, P[k]) if len(np.unique(P[k])) > 1 else 0.5 for k in range(len(grid))]},
                        index=pd.Index(list(grid), name=feature))


def conditional_parity(p, group, risk_proxy, threshold, n_bins=5):
    """
    Conditional statistical parity (Hurlin, Pérignon & Saurin 2024): approval rate per
    group WITHIN homogeneous risk classes (bins of a risk proxy, e.g. credit_score).
    A fair model shows similar rates within each bin.
    """
    df = pd.DataFrame({"approved": (np.asarray(p) >= threshold).astype(int),
                       "group": np.asarray(group),
                       "risk_bin": pd.qcut(np.asarray(risk_proxy), n_bins, duplicates="drop")})
    return df.pivot_table(index="risk_bin", columns="group", values="approved",
                          aggfunc="mean", observed=True)

# ----------------------------------------------------------------------------
# 5. Stability
# ----------------------------------------------------------------------------
def prediction_stability(pred_matrix, threshold: float) -> dict:
    """
    pred_matrix: shape (n_refits, n_rows) — predicted probabilities of the SAME rows from
    models refitted on different training samples / seeds.
    """
    P = np.asarray(pred_matrix)
    pairs = [(i, j) for i in range(len(P)) for j in range(i + 1, len(P))]
    return {
        "mean_abs_prob_change": float(np.mean([np.mean(np.abs(P[i] - P[j])) for i, j in pairs])),
        "decision_flip_rate": float(np.mean([np.mean((P[i] >= threshold) != (P[j] >= threshold))
                                             for i, j in pairs])),
        "mean_per_row_std": float(np.mean(P.std(axis=0))),
    }


def importance_stability(importance_matrix) -> float:
    """Mean pairwise Spearman correlation of feature-importance vectors across refits."""
    M = np.asarray(importance_matrix)
    return float(np.mean([spearmanr(M[i], M[j]).statistic
                          for i in range(len(M)) for j in range(i + 1, len(M))]))


def add_noise(X: pd.DataFrame, num_cols, noise: float, seed: int = SEED) -> pd.DataFrame:
    """Gaussian noise of `noise` x std on numeric features (input-perturbation stability)."""
    rng = np.random.default_rng(seed)
    Xp = X.copy()
    for c in num_cols:
        Xp[c] = Xp[c] + rng.normal(0, noise * X[c].std(), size=len(X))
    return Xp

# ----------------------------------------------------------------------------
# 6. Model-agnostic interpretability
# ----------------------------------------------------------------------------
def permutation_importance_auc(predict_fn, X: pd.DataFrame, y, n_repeats=2, seed=SEED):
    """
    AUC drop when each column is shuffled. All permuted copies are scored in ONE call to
    `predict_fn`, which matters for slow models (TabPFN). Returns (mean, all repeats).
    """
    rng = np.random.default_rng(seed)
    blocks, keys = [], []
    for c in X.columns:
        for r in range(n_repeats):
            Xp = X.copy()
            Xp[c] = rng.permutation(Xp[c].to_numpy())
            blocks.append(Xp); keys.append((c, r))
    p_all = predict_fn(pd.concat([X] + blocks, ignore_index=True))
    n = len(X)
    base = roc_auc_score(y, p_all[:n])
    drops = pd.Series({k: base - roc_auc_score(y, p_all[(i + 1) * n:(i + 2) * n])
                       for i, k in enumerate(keys)})
    return drops.groupby(level=0).mean().reindex(X.columns), drops


def xper(predict_fn, X: pd.DataFrame, metrics: dict, groups: dict, seed=SEED) -> pd.DataFrame:
    """
    XPER (Hué, Hurlin, Pérignon & Saurin 2026): Shapley decomposition of a PERFORMANCE metric,
    PM = phi_0 + sum_j phi_j. Players = feature groups. Value of a coalition S = metric of the
    predictions where the features NOT in S are replaced by those of a random other applicant
    (one Monte-Carlo draw of the marginal expectation). Exact Shapley over the 2^G coalitions,
    all scored in ONE call to predict_fn.
    metrics: {name: f(p) -> float}   e.g. AUC(y, p), profit(y, p >= t)
    groups:  {name: [columns]}
    """
    from itertools import combinations
    from math import factorial
    rng = np.random.default_rng(seed)
    ref = X.iloc[rng.permutation(len(X))].reset_index(drop=True)
    names, G = list(groups), len(groups)
    coalitions = [frozenset(c) for k in range(G + 1) for c in combinations(range(G), k)]
    blocks = []
    for S in coalitions:
        Xs = X.reset_index(drop=True).copy()
        for j in range(G):
            if j not in S:
                Xs[groups[names[j]]] = ref[groups[names[j]]].to_numpy()
        blocks.append(Xs)
    P = np.asarray(predict_fn(pd.concat(blocks, ignore_index=True))).reshape(len(coalitions), len(X))
    out = {}
    for mname, mf in metrics.items():
        v = {S: mf(P[i]) for i, S in enumerate(coalitions)}
        phi = {"phi_0 (no information)": v[frozenset()]}
        for j in range(G):
            phi[names[j]] = sum(factorial(len(S)) * factorial(G - len(S) - 1) / factorial(G)
                                * (v[S | {j}] - v[S]) for S in coalitions if j not in S)
        phi["= full model"] = v[frozenset(range(G))]
        out[mname] = pd.Series(phi)
    return pd.DataFrame(out)


def bootstrap_ci(stat_fn, n: int, B: int = 500, seed=SEED, level=0.95):
    """Percentile bootstrap CI of stat_fn(idx) over resamples of the n test rows."""
    rng = np.random.default_rng(seed)
    vals = np.array([stat_fn(rng.integers(0, n, n)) for _ in range(B)])
    a = (1 - level) / 2
    return float(np.quantile(vals, a)), float(np.quantile(vals, 1 - a))


def lime_explain(predict_fn, x: pd.DataFrame, X_ref: pd.DataFrame, n_samples=2000, K=5, seed=SEED):
    """
    LIME (Ribeiro et al. 2016), tabular version with a binary interpretable representation:
    z'_j = 1 -> the applicant keeps his own value of feature j, 0 -> value of a random reference
    applicant. Local surrogate g = weighted ridge on z', exponential kernel on the distance to
    the explained point, complexity Omega = K non-zero weights (slides 103-120).
    Returns (weights of the K selected features, local weighted R^2 = fidelity).
    """
    from sklearn.linear_model import Ridge
    rng = np.random.default_rng(seed)
    d = x.shape[1]
    Z = rng.integers(0, 2, size=(n_samples, d)); Z[0] = 1
    refs = X_ref.iloc[rng.integers(0, len(X_ref), n_samples)].to_numpy()
    samples = np.where(Z == 1, np.repeat(x.to_numpy(), n_samples, axis=0), refs)
    y = np.asarray(predict_fn(pd.DataFrame(samples, columns=x.columns)))
    dist = np.sqrt((1 - Z).sum(axis=1) / d)
    w = np.exp(-dist ** 2 / 0.75 ** 2)
    full = Ridge(alpha=1e-3).fit(Z, y, sample_weight=w)
    keep = np.argsort(-np.abs(full.coef_))[:K]
    g = Ridge(alpha=1e-3).fit(Z[:, keep], y, sample_weight=w)
    return (pd.Series(g.coef_, index=x.columns[keep]).sort_values(key=np.abs, ascending=False),
            float(g.score(Z[:, keep], y, sample_weight=w)))


def rank_agreement(a: pd.Series, b: pd.Series, k: int = 3) -> dict:
    """Disagreement problem (Krishna et al.): Spearman rank correlation of |importances| and
    top-k feature agreement between two explanation methods."""
    idx = a.index.intersection(b.index)
    a, b = a[idx].abs(), b[idx].abs()
    return {"spearman": float(spearmanr(a, b).statistic),
            f"top-{k} agreement": len(set(a.nlargest(k).index) & set(b.nlargest(k).index)) / k}
