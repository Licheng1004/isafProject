"""
Loan-approval scoring: compare the team's three models (white box, ML, tabular foundation model).

    streamlit run app/streamlit_app.py

Needs the artifacts built by `python app/build_artifacts.py`.
"""
from __future__ import annotations

import json
from importlib.util import find_spec
import subprocess
import sys
import threading
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb
from scipy.special import expit
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve

APP = Path(__file__).resolve().parent
ROOT = APP.parent
ART = APP / "artifacts"
TFM = ROOT / "TabularFoundationModel"
sys.path.append(str(TFM / "src"))
from isaf_eval import fairness_tests, tost_parity  # noqa: E402

st.set_page_config(page_title="Loan approval: 3 models compared", page_icon="🏦", layout="wide")

# Compatibility with older Streamlit versions (< 1.51 has no **STRETCH nor delta_arrow)
import inspect  # noqa: E402
_NEW_WIDTH = "width" in inspect.signature(st.altair_chart).parameters
STRETCH = {"width": "stretch"} if _NEW_WIDTH else {"use_container_width": True}
_ARROW_OFF = {"delta_arrow": "off"} if "delta_arrow" in inspect.signature(st.metric).parameters else {}

# ----------------------------------------------------------------------------------------------
# Constants: one fixed colour per model (validated categorical palette, slots 1-3)
# ----------------------------------------------------------------------------------------------
KEYS = ["lr", "xgb", "tabpfn"]
NAME = {"lr": "Logistic regression", "xgb": "XGBoost", "tabpfn": "TabPFN"}
KIND = {"lr": "white box", "xgb": "machine learning", "tabpfn": "tabular foundation model"}
COLOR = {"Logistic regression": "#2a78d6", "XGBoost": "#eb6834", "TabPFN": "#1baf7a"}
MODEL_SCALE = alt.Scale(domain=list(COLOR), range=list(COLOR.values()))
RAISES, LOWERS = "#2a78d6", "#e34948"          # diverging pair: raises / lowers P(repay)
INPUTS = ["annual_income", "debt_to_income_ratio", "credit_score", "loan_amount", "interest_rate",
          "loan_purpose", "grade_subgrade", "employment_status", "education_level"]
LABEL = {"annual_income": "Annual income", "debt_to_income_ratio": "Debt-to-income ratio",
         "credit_score": "Credit score", "loan_amount": "Loan amount", "interest_rate": "Interest rate",
         "loan_purpose": "Loan purpose", "grade_subgrade": "Grade", "employment_status": "Employment status",
         "education_level": "Education level"}
SHORT = {"annual_income": "Income", "debt_to_income_ratio": "DTI", "credit_score": "Credit score", "loan_amount": "Amount",
         "interest_rate": "Rate", "loan_purpose": "Purpose", "grade_subgrade": "Grade", "employment_status": "Employment",
         "education_level": "Education"}
SENSITIVE = {"gender": "Gender", "marital_status": "Marital status",
             "education_level": "Education level", "employment_status": "Employment status"}

# ----------------------------------------------------------------------------------------------
# Artifacts
# ----------------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_artifacts():
    test = pd.read_csv(ART / "test_predictions.csv.gz")
    valid = {k: pd.read_csv(ART / f"valid_{k}.csv.gz") for k in KEYS}
    lr = json.loads((ART / "lr_model.json").read_text())
    xmeta = json.loads((ART / "xgb_meta.json").read_text())
    typical = json.loads((ART / "typical_applicant.json").read_text())
    noise = pd.read_csv(ART / "noise_1pct.csv.gz")
    tfm = json.loads((TFM / "outputs" / "summary_tabpfn.json").read_text())
    return test, valid, lr, xmeta, typical, noise, tfm

@st.cache_resource(show_spinner=False)
def load_xgb():
    b = xgb.Booster(); b.load_model(bytearray((ART / "xgb_model.json").read_bytes())); return b

if not (ART / "test_predictions.csv.gz").exists():
    st.error("Artifacts missing: run `python app/build_artifacts.py` first."); st.stop()
TEST, VALID, LR, XMETA, TYPICAL, NOISE, TFM_SUMMARY = load_artifacts()
BOOSTER = load_xgb()
ENV_HINT = ("Run the app with the project environment (xgboost ≥ 3, tabpfn ≥ 9): "
            "`uv sync --locked`, then `uv run --locked streamlit run app/streamlit_app.py`. "
            "For live TabPFN add `--extra foundation` to both commands and arrange authorized weight access.")
LEVELS = {c: sorted(TEST[c].unique()) for c in ["loan_purpose", "grade_subgrade", "employment_status", "education_level"]}

# ----------------------------------------------------------------------------------------------
# Model scoring
# ----------------------------------------------------------------------------------------------
LR_BETA = pd.Series(LR["params"])

def lr_design(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["grade_ord"] = d.grade_subgrade.str[0].map({l: i for i, l in enumerate("ABCDEF")}) * 5 + d.grade_subgrade.str[1].astype(int) - 1
    X = pd.DataFrame(0.0, index=d.index, columns=LR_BETA.index)
    X["const"] = 1.0
    for c in LR["num"]:
        X[c] = (d[c].astype(float) - LR["mu"][c]) / LR["sd"][c]
    for col in LR["columns"]:
        for c in LR["cat"]:
            if col.startswith(c + "_"):
                X[col] = (d[c] == col[len(c) + 1:]).astype(float)
    return X

def lr_proba(df):
    return expit(lr_design(df).to_numpy() @ LR_BETA.to_numpy())

def lr_contrib(row: pd.DataFrame) -> pd.Series:
    """Exact: log-odds(x) - log-odds(average applicant) = sum_j beta_j (x_j - xbar_j)."""
    c = (lr_design(row).iloc[0] - pd.Series(LR["xbar"])) * LR_BETA
    out = {}
    for col, v in c.drop("const").items():
        feat = "grade_subgrade" if col == "grade_ord" else next((k for k in LR["cat"] if col.startswith(k + "_")), col)
        out[feat] = out.get(feat, 0.0) + v
    return pd.Series(out)

def xgb_encode(df):
    X = df[INPUTS].copy()
    for c in INPUTS:
        X[c] = pd.Categorical(X[c], categories=XMETA["categories"][c]) if c in XMETA["categories"] else pd.to_numeric(X[c])
    return X

def xgb_proba(df):
    dm = xgb.DMatrix(xgb_encode(df), enable_categorical=True)
    return expit(BOOSTER.predict(dm, output_margin=True, iteration_range=(0, XMETA["n_trees"])))

def xgb_contrib(df) -> np.ndarray:
    """TreeSHAP (exact for the tree ensemble), in log-odds; last column = bias."""
    dm = xgb.DMatrix(xgb_encode(df), enable_categorical=True)
    return BOOSTER.predict(dm, pred_contribs=True, iteration_range=(0, XMETA["n_trees"]))

@st.cache_resource(show_spinner=False)
def tabpfn_worker():
    """Starts the TabPFN process once; it warms up in the background (≈1 min) without blocking the page."""
    proc = subprocess.Popen([sys.executable, str(APP / "tabpfn_worker.py")], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
    w = {"proc": proc, "lock": threading.Lock(), "ready": threading.Event(), "hello": {}}

    def wait_ready():
        line = proc.stdout.readline()
        if line:
            w["hello"] = json.loads(line)
            w["ready"].set()
    threading.Thread(target=wait_ready, daemon=True).start()
    return w

def tabpfn_ready() -> bool:
    w = tabpfn_worker()
    return w["ready"].is_set() and w["hello"].get("max_diff", 1.0) < 0.01

def tabpfn_proba(df) -> np.ndarray:
    w = tabpfn_worker()
    with w["lock"]:
        w["proc"].stdin.write('{"rows": ' + df[INPUTS].to_json(orient="records") + "}\n")
        w["proc"].stdin.flush()
        resp = json.loads(w["proc"].stdout.readline())
    if "error" in resp:
        raise RuntimeError(resp["error"])
    return np.asarray(resp["p"])

def tabpfn_contrib(row: pd.DataFrame) -> tuple[float, pd.Series]:
    """Ablation: how much P(repay) changes when one feature is replaced by the typical applicant's value.
    One batch of 10 rows = one call to the model (~1 s)."""
    batch = pd.concat([row] + [row.assign(**{f: TYPICAL[f]}) for f in INPUTS], ignore_index=True)
    p = tabpfn_proba(batch)
    return float(p[0]), pd.Series(p[0] - p[1:], index=INPUTS)

# Self-test: a model loaded with another library version can silently give different predictions
# (e.g. xgboost 2.x reading a 3.x model re-codes the categories). Refuse to show wrong numbers.
_chk = TEST.head(500)
_diff = {"Logistic regression": float(np.abs(lr_proba(_chk) - _chk.p_lr).max()),
         "XGBoost": float(np.abs(xgb_proba(_chk) - _chk.p_xgb).max())}
_bad = {m: d for m, d in _diff.items() if d > 1e-3}
if _bad:
    st.error("⚠️ The models do not reproduce their saved predictions in this Python environment "
             f"({', '.join(f'{m}: max error {d:.3f}' for m, d in _bad.items())}; xgboost {xgb.__version__}). " + ENV_HINT)
    st.stop()

# ----------------------------------------------------------------------------------------------
# Economics: each model's threshold is chosen on ITS OWN validation sample for the chosen LGD
# ----------------------------------------------------------------------------------------------
GRID = np.round(np.linspace(0.01, 0.99, 99), 2)

def payoff(y, amount, rate, lgd):
    return np.where(np.asarray(y) == 1, np.asarray(amount) * np.asarray(rate) / 100, -lgd * np.asarray(amount))

@st.cache_data(show_spinner=False)
def thresholds(lgd: float) -> dict:
    out = {}
    for k, v in VALID.items():
        pay = payoff(v.y, v.loan_amount, v.interest_rate, lgd)
        out[k] = float(GRID[int(np.argmax([pay[v.p.to_numpy() >= t].sum() for t in GRID]))])
    return out

@st.cache_data(show_spinner=False)
def performance(lgd: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    thr, y = thresholds(lgd), TEST.y.to_numpy()
    pay = payoff(y, TEST.loan_amount, TEST.interest_rate, lgd)
    rows, curves = [], []
    for k in KEYS:
        p = TEST[f"p_{k}"].to_numpy(); a = p >= thr[k]; auc = roc_auc_score(y, p)
        rows.append({"Model": NAME[k], "Type": KIND[k], "AUC": auc, "Gini": 2 * auc - 1, "Brier": brier_score_loss(y, p),
                     "Threshold": thr[k], "Approval rate": a.mean(), "Default rate among approved": 1 - y[a].mean(),
                     "Profit (M€)": pay[a].sum() / 1e6,
                     "Share of max. profit": (pay[a].sum() - pay.sum()) / (pay[y == 1].sum() - pay.sum())})
        curves += [{"Model": NAME[k], "Threshold": t, "Profit (M€)": pay[p >= t].sum() / 1e6} for t in GRID]
    return pd.DataFrame(rows), pd.DataFrame(curves)

def fmt_eur(x):
    return f"€{x:,.0f}".replace(",", " ")

# ----------------------------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------------------------
with st.sidebar:
    st.title("🏦 Loan approval")
    st.caption("HEC ISAF project · consumer-lender scoring · Kaggle Playground S5E11 (synthetic, 594k loans)")
    lgd = st.slider("Loss given default (LGD)", 0.30, 1.00, 0.60, 0.05,
                    help="Share of the loan lost when a borrower defaults. Team convention: 60%. "
                         "Each model's approval threshold is re-optimised on its own validation sample.")
    st.markdown(f"**Economics per loan**  \n• repaid → + amount × interest rate  \n• default → − {lgd:.0%} × amount  \n• rejected → 0")
    live = st.toggle("Live TabPFN scoring", value=False, disabled=find_spec("tabpfn") is None,
                     help="Starts the robust TabPFN in a background process (≈1 min, once). "
                          "Needs the optional foundation environment and authorized model weights. "
                          "Saved TabPFN results remain available without live scoring.")
    thr = thresholds(lgd)
    st.markdown("**Approval thresholds** (P(repay) ≥ t)")
    for k in KEYS:
        st.markdown(f"<span style='color:{COLOR[NAME[k]]}'>●</span> {NAME[k]}: **{thr[k]:.2f}**", unsafe_allow_html=True)
    st.caption("No model uses gender or marital status (prohibited in credit decisions). "
               "They are used only to audit fairness.")

st.title("Which model should approve our loans?")
st.caption("Three models, same 20,000 held-out applicants (`data/test_eval_20k.csv`), same economics. "
           "White box = logistic regression · ML = XGBoost · Tabular foundation model = TabPFN (robust version).")

tabs = st.tabs(["🧮 Score an applicant", "📈 Performance", "⚖️ Fairness", "🔁 Stability", "🔍 Interpretability", "✅ Recommendation"])

# ==============================================================================================
# 1. Score an applicant
# ==============================================================================================
with tabs[0]:
    left, right = st.columns([1, 2], gap="large")
    with left:
        st.subheader("Applicant")
        src = st.radio("Start from", ["A real applicant from the test set", "A new applicant"], horizontal=True)
        if src.startswith("A real"):
            st.session_state.setdefault("row", 13)
            c1, c2 = st.columns([3, 1])
            if c2.button("🎲", **STRETCH, help="Random applicant"):
                st.session_state.row = int(np.random.default_rng().integers(len(TEST)))
            idx = c1.number_input("Row of the test set", 0, len(TEST) - 1, key="row")
            base = TEST.iloc[idx]
        else:
            base = pd.Series(TYPICAL)
        with st.form("applicant"):
            v = {}
            v["employment_status"] = st.selectbox(LABEL["employment_status"], LEVELS["employment_status"],
                                                  LEVELS["employment_status"].index(base.employment_status))
            v["debt_to_income_ratio"] = st.slider(LABEL["debt_to_income_ratio"], 0.0, 0.7, float(base.debt_to_income_ratio), 0.001, format="%.3f")
            v["credit_score"] = st.slider(LABEL["credit_score"], 395, 850, int(base.credit_score))
            v["grade_subgrade"] = st.selectbox(LABEL["grade_subgrade"], LEVELS["grade_subgrade"], LEVELS["grade_subgrade"].index(base.grade_subgrade))
            c1, c2 = st.columns(2)
            v["annual_income"] = c1.number_input(LABEL["annual_income"] + " (€)", 5_000.0, 400_000.0, float(base.annual_income), 1_000.0)
            v["loan_amount"] = c2.number_input(LABEL["loan_amount"] + " (€)", 500.0, 50_000.0, float(base.loan_amount), 500.0)
            v["interest_rate"] = c1.number_input(LABEL["interest_rate"] + " (%)", 3.0, 25.0, float(base.interest_rate), 0.1)
            v["loan_purpose"] = c2.selectbox(LABEL["loan_purpose"], LEVELS["loan_purpose"], LEVELS["loan_purpose"].index(base.loan_purpose))
            v["education_level"] = st.selectbox(LABEL["education_level"], LEVELS["education_level"], LEVELS["education_level"].index(base.education_level))
            st.form_submit_button("Score this applicant", type="primary", **STRETCH)
        row = pd.DataFrame([v])[INPUTS]
        unchanged = src.startswith("A real") and all(
            (abs(float(row.iloc[0][c]) - float(base[c])) < 1e-9) if c not in LEVELS else row.iloc[0][c] == base[c] for c in INPUTS)
        if unchanged:
            st.info(f"True outcome: **{'repaid ✅' if base.y == 1 else 'defaulted ❌'}** (applicant id {int(base.id)})")

    with right:
        st.subheader("Decisions")
        p = {"lr": float(lr_proba(row)[0]), "xgb": float(xgb_proba(row)[0])}
        tfm_contrib = None
        if live and tabpfn_ready():
            try:
                p["tabpfn"], tfm_contrib = tabpfn_contrib(row)
            except Exception as e:  # noqa: BLE001
                st.warning(f"TabPFN unavailable ({e}).")
        elif live and tabpfn_worker()["ready"].is_set():
            st.warning(f"TabPFN does not reproduce its saved predictions in this environment "
                       f"(max error {tabpfn_worker()['hello'].get('max_diff', float('nan')):.3f}); live scoring disabled. " + ENV_HINT)
        elif live:
            st.info("⏳ TabPFN is warming up in the background (≈1 min, once). "
                    "Its pre-computed score is shown for test-set applicants; click **Score this applicant** again in a minute.")
        if "tabpfn" not in p and unchanged:
            p["tabpfn"] = float(base.p_tabpfn)
        gain, loss = v["loan_amount"] * v["interest_rate"] / 100, lgd * v["loan_amount"]
        cols = st.columns(3)
        for col, k in zip(cols, KEYS):
            with col.container(border=True):
                st.markdown(f"<span style='color:{COLOR[NAME[k]]};font-size:1.3em'>●</span> **{NAME[k]}**  \n"
                            f"<span style='opacity:.7'>{KIND[k]}</span>", unsafe_allow_html=True)
                if k not in p:
                    st.markdown("Turn on *Live TabPFN scoring* in the sidebar."); continue
                ok = p[k] >= thr[k]
                st.metric("P(repay)", f"{p[k]:.1%}")
                st.caption(f"approval threshold: {thr[k]:.0%}")
                st.markdown(f"### {'✅ Approve' if ok else '❌ Reject'}")
                st.caption(f"Expected profit if approved: {fmt_eur(p[k] * gain - (1 - p[k]) * loss)}")
        decisions = {NAME[k]: p[k] >= thr[k] for k in p}
        if len(decisions) == 3:
            if len(set(decisions.values())) == 1:
                st.success(f"The three models agree: **{'approve' if next(iter(decisions.values())) else 'reject'}**.")
            else:
                st.warning("The models **disagree** on this applicant: " +
                           ", ".join(f"{m} → {'approve' if d else 'reject'}" for m, d in decisions.items()))

        st.subheader("Why? Main reasons behind each score")
        st.caption("Blue = pushes towards repayment, red = pushes towards default. "
                   "Logistic regression and XGBoost: exact contributions to the log-odds, relative to an average applicant. "
                   "TabPFN: change in P(repay) when the feature is replaced by a typical applicant's value (approximation).")
        expl = {"lr": lr_contrib(row), "xgb": pd.Series(xgb_contrib(row)[0][:-1], index=INPUTS)}
        if tfm_contrib is not None:
            expl["tabpfn"] = tfm_contrib
        ecols = st.columns(3)
        for col, k in zip(ecols, KEYS):
            with col:
                st.markdown(f"**{NAME[k]}**")
                if k not in expl:
                    st.caption("Needs live TabPFN scoring."); continue
                s = expl[k].reindex(INPUTS).fillna(0)
                val = lambda f: str(row.iloc[0][f]) if f in LEVELS else (f"{row.iloc[0][f]:.3f}" if f == "debt_to_income_ratio" else f"{row.iloc[0][f]:,.0f}")
                d = pd.DataFrame({"Feature": [f"{SHORT[f]}: {val(f)}" for f in INPUTS],
                                  "Contribution": s.to_numpy()})
                d["Effect"] = np.where(d.Contribution >= 0, "raises P(repay)", "lowers P(repay)")
                d = d.reindex(d.Contribution.abs().sort_values(ascending=False).index).head(6)
                unit = "log-odds" if k != "tabpfn" else "Δ P(repay)"
                ch = alt.Chart(d).mark_bar(cornerRadiusEnd=4, height=16).encode(
                    x=alt.X("Contribution:Q", title=unit),
                    y=alt.Y("Feature:N", sort=None, title=None, axis=alt.Axis(labelLimit=180)),
                    color=alt.Color("Effect:N", scale=alt.Scale(domain=["raises P(repay)", "lowers P(repay)"], range=[RAISES, LOWERS]), legend=None),
                    tooltip=["Feature", alt.Tooltip("Contribution:Q", format="+.3f"), "Effect"]).properties(height=230)
                st.altair_chart(ch, **STRETCH)
                top = d.iloc[0]
                st.caption(f"Main reason: **{top.Feature}** ({top.Effect}).")

# ==============================================================================================
# 2. Performance
# ==============================================================================================
with tabs[1]:
    perf, curves = performance(lgd)
    y_all = TEST.y.to_numpy()
    approve_all = payoff(y_all, TEST.loan_amount, TEST.interest_rate, lgd).sum() / 1e6
    best = perf.loc[perf["Profit (M€)"].idxmax()]
    c = st.columns(4)
    c[0].metric("Best AUC", f"{perf.AUC.max():.3f}", perf.loc[perf.AUC.idxmax(), "Model"], delta_color="off", **_ARROW_OFF)
    c[1].metric("Best profit on 20k loans", f"€{best['Profit (M€)']:.2f}M", best.Model, delta_color="off", **_ARROW_OFF)
    c[2].metric("Approve everyone", f"{'−' if approve_all < 0 else ''}€{abs(approve_all):.2f}M", "no model", delta_color="off", **_ARROW_OFF)
    c[3].metric("Profit gap best vs worst model", f"€{perf['Profit (M€)'].max() - perf['Profit (M€)'].min():.2f}M",
                f"{(perf['Profit (M€)'].max() / perf['Profit (M€)'].min() - 1):.1%}", delta_color="off", **_ARROW_OFF)
    st.dataframe(perf.style.format({"AUC": "{:.4f}", "Gini": "{:.3f}", "Brier": "{:.4f}", "Threshold": "{:.2f}",
                                    "Approval rate": "{:.1%}", "Default rate among approved": "{:.1%}",
                                    "Profit (M€)": "{:.2f}", "Share of max. profit": "{:.1%}"}),
                 hide_index=True, **STRETCH)
    st.caption("Profit on the 20,000 test loans at each model's threshold. 'Share of max. profit' = captured share of the gap "
               "between approving everyone and a perfect oracle. XGBoost is retrained with the notebook's exact code "
               "(AUC 0.916 here vs 0.918 in the notebook: floating-point differences between machines).")

    base_line = alt.Chart(curves).mark_line(strokeWidth=2).encode(
        x=alt.X("Threshold:Q", title="Approval threshold on P(repay)"),
        y=alt.Y("Profit (M€):Q", title="Profit on the 20k test loans (M€)"),
        color=alt.Color("Model:N", scale=MODEL_SCALE, legend=alt.Legend(orient="top", title=None)))
    hover = alt.selection_point(fields=["Threshold"], nearest=True, on="pointerover", empty=False)
    rule = alt.Chart(curves).mark_rule(color="#9ca3af").encode(
        x="Threshold:Q", opacity=alt.condition(hover, alt.value(0.6), alt.value(0)),
        tooltip=[alt.Tooltip("Threshold:Q", format=".2f")] + [alt.Tooltip(f"{m}:Q", format=".2f", title=f"{m} (M€)") for m in COLOR]
    ).transform_pivot("Model", value="Profit (M€)", groupby=["Threshold"]).add_params(hover)
    chosen = alt.Chart(perf).mark_point(size=110, filled=True, stroke="white", strokeWidth=2).encode(
        x="Threshold:Q", y="Profit (M€):Q", color=alt.Color("Model:N", scale=MODEL_SCALE, legend=None),
        tooltip=["Model", alt.Tooltip("Threshold:Q", format=".2f"), alt.Tooltip("Profit (M€):Q", format=".2f")])
    zero = alt.Chart(pd.DataFrame({"y": [approve_all]})).mark_rule(strokeDash=[4, 4], color="#9ca3af").encode(y="y:Q")
    st.markdown("**Profit vs approval threshold** (dots = threshold chosen on validation)")
    st.altair_chart((base_line + rule + chosen + zero).properties(height=320), **STRETCH)

    c1, c2 = st.columns(2)
    roc_rows, cal_rows = [], []
    for k in KEYS:
        fpr, tpr, _ = roc_curve(y_all, TEST[f"p_{k}"])
        keep = np.linspace(0, len(fpr) - 1, 300).astype(int)
        roc_rows += [{"Model": NAME[k], "False positive rate": fpr[i], "True positive rate": tpr[i]} for i in keep]
        q = pd.qcut(TEST[f"p_{k}"], 10, labels=False, duplicates="drop")
        g = pd.DataFrame({"p": TEST[f"p_{k}"], "y": y_all, "q": q}).groupby("q").mean()
        cal_rows += [{"Model": NAME[k], "Predicted P(repay)": r.p, "Observed repayment rate": r.y} for r in g.itertuples()]
    def diag(xf, yf):
        return alt.Chart(pd.DataFrame({xf: [0, 1], yf: [0, 1]})).mark_line(strokeDash=[4, 4], color="#9ca3af").encode(x=f"{xf}:Q", y=f"{yf}:Q")
    with c1:
        st.markdown("**ROC curves**")
        roc = alt.Chart(pd.DataFrame(roc_rows)).mark_line(strokeWidth=2).encode(
            x="False positive rate:Q", y="True positive rate:Q",
            color=alt.Color("Model:N", scale=MODEL_SCALE, legend=alt.Legend(orient="bottom", title=None)),
            tooltip=["Model", alt.Tooltip("False positive rate:Q", format=".2f"), alt.Tooltip("True positive rate:Q", format=".2f")])
        st.altair_chart((diag("False positive rate", "True positive rate") + roc).properties(height=300), **STRETCH)
    with c2:
        st.markdown("**Calibration** (deciles of predicted probability)")
        cal = alt.Chart(pd.DataFrame(cal_rows)).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=60, filled=True)).encode(
            x=alt.X("Predicted P(repay):Q", scale=alt.Scale(domain=[0, 1])), y=alt.Y("Observed repayment rate:Q", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Model:N", scale=MODEL_SCALE, legend=alt.Legend(orient="bottom", title=None)),
            tooltip=["Model", alt.Tooltip("Predicted P(repay):Q", format=".3f"), alt.Tooltip("Observed repayment rate:Q", format=".3f")])
        st.altair_chart((diag("Predicted P(repay)", "Observed repayment rate") + cal).properties(height=300), **STRETCH)

# ==============================================================================================
# 3. Fairness
# ==============================================================================================
with tabs[2]:
    st.markdown("Approval decisions of the three models across groups. **No model sees gender or marital status**; "
                "a gap can still appear through correlated features (proxies).")
    attr = st.selectbox("Sensitive attribute", list(SENSITIVE), format_func=SENSITIVE.get)
    rows = []
    for k in KEYS:
        a = TEST[f"p_{k}"].to_numpy() >= thr[k]
        for gname, g in TEST.assign(a=a).groupby(attr):
            rows.append({"Model": NAME[k], "Group": gname, "Applicants": len(g), "Approval rate": g.a.mean(),
                         "True repayment rate": g.y.mean(), "TPR (good payers approved)": g.a[g.y == 1].mean(),
                         "FPR (defaulters approved)": g.a[g.y == 0].mean()})
    fr = pd.DataFrame(rows)
    bars = alt.Chart(fr).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X("Group:N", title=None, axis=alt.Axis(labelAngle=0)), xOffset=alt.XOffset("Model:N", sort=list(COLOR)),
        y=alt.Y("Approval rate:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("Model:N", scale=MODEL_SCALE, legend=alt.Legend(orient="top", title=None)),
        tooltip=["Model", "Group", "Applicants", alt.Tooltip("Approval rate:Q", format=".1%"),
                 alt.Tooltip("True repayment rate:Q", format=".1%"), alt.Tooltip("TPR (good payers approved):Q", format=".1%"),
                 alt.Tooltip("FPR (defaulters approved):Q", format=".1%")])
    st.altair_chart(bars.properties(height=300), **STRETCH)
    with st.expander("Table view"):
        st.dataframe(fr.style.format({c: "{:.1%}" for c in fr.columns if "rate" in c or "TPR" in c or "FPR" in c}),
                     hide_index=True, **STRETCH)

    st.markdown("**Formal tests** (Hurlin, Pérignon & Saurin): p-values, H0 = *the model is fair*. "
                "Below 5% → fairness rejected under that definition.")
    ft = pd.DataFrame({NAME[k]: fairness_tests(TEST.y, TEST[f"p_{k}"], TEST[attr], thr[k], TEST.credit_score) for k in KEYS})
    ft.index = ft.index.str.capitalize()
    st.dataframe(ft.style.format("{:.4f}").map(lambda x: "color:#e34948;font-weight:700" if x < 0.05 else ""), **STRETCH)
    st.caption("Independence = statistical parity (conditional: within credit-score quintiles) · "
               "Separation = equal opportunity, predictive equality, equalized odds · Sufficiency = same score, same repayment rate.")

    st.markdown("**Can we certify gender fairness?** Equivalence test (TOST): H0 = the approval gap is at least δ.")
    delta = st.slider("Tolerance δ (percentage points of approval rate)", 1.0, 5.0, 2.0, 0.5) / 100
    tcols = st.columns(3)
    for col, k in zip(tcols, KEYS):
        r = tost_parity(TEST[f"p_{k}"].to_numpy() >= thr[k], TEST.gender, "Female", "Male", delta)
        col.metric(NAME[k], "✅ certified fair" if r["certified fair"] else "⚠️ not certified",
                   f"gap F−M {-r['gap (ref - protected)']:+.1%}, p = {r['p-value']:.3f}", delta_color="off", **_ARROW_OFF)
    if attr == "employment_status":
        st.info("Employment status is a legitimate risk factor, but it leads to near-total exclusion of students and the unemployed "
                "(8–24% repayment rate). Whether this is acceptable is a policy decision for the lender, not a statistical one.")

# ==============================================================================================
# 4. Stability
# ==============================================================================================
with tabs[3]:
    st.markdown("**Same test for the three models**: add 1% Gaussian noise (× std) to the numeric inputs of the same 2,000 applicants "
                "(measurement error, rounding, stale data). How many decisions change?")
    srows = []
    for k in KEYS:
        a, b = NOISE[f"{k}_clean"].to_numpy(), NOISE[f"{k}_noisy"].to_numpy()
        srows.append({"Model": NAME[k], "Decisions that flip": float(np.mean((a >= thr[k]) != (b >= thr[k]))),
                      "Mean |Δ P(repay)|": float(np.mean(np.abs(a - b)))})
    sd_ = pd.DataFrame(srows)
    c1, c2 = st.columns([2, 3])
    with c1:
        ch = alt.Chart(sd_).mark_bar(cornerRadiusEnd=4, height=28).encode(
            x=alt.X("Decisions that flip:Q", axis=alt.Axis(format="%")), y=alt.Y("Model:N", sort=list(COLOR), title=None),
            color=alt.Color("Model:N", scale=MODEL_SCALE, legend=None),
            tooltip=["Model", alt.Tooltip("Decisions that flip:Q", format=".2%"), alt.Tooltip("Mean |Δ P(repay)|:Q", format=".4f")])
        st.altair_chart(ch.properties(height=170), **STRETCH)
        st.caption(" · ".join(f"{r.Model}: **{r._2:.1%}** of decisions flip" for r in sd_.itertuples()))
    with c2:
        st.markdown("""
Almost all of the sensitivity comes from **debt-to-income**: in this synthetic dataset, repayment jumps in *steps* along DTI.
Flexible models (XGBoost, TabPFN) learn these steps, so a tiny DTI change near a step can flip a decision.
The logistic regression is smooth, which makes it far less sensitive.
""")
    tfm_stab = TFM_SUMMARY["stability"]
    st.markdown("**Stability evidence from each notebook**")
    cr = tfm_stab["context resampling"]
    st.markdown(f"""
| Model | Retraining on another sample (Turney) | Stability of the explanations | Specific risk |
|---|---|---|---|
| Logistic regression | 5-fold CV: AUC sd 0.0013; bootstrap: all significant drivers keep their sign | mean pairwise coefficient distance 0.31 (4% of ‖θ‖) | linear: misses interactions |
| XGBoost | 5 seeds: AUC sd 0.0001 | SHAP ranking Spearman ≥ 0.98 across seeds | step-shaped response to inputs |
| TabPFN | 5 context sets: AUC sd {cr['AUC std']:.4f}, but **{cr['decision_flip_rate']:.1%} of decisions change** (1.45% after context bagging) | importance ranking Spearman {tfm_stab['importance_spearman']:.2f} across contexts | decisions depend on which rows are in the context |
""")

# ==============================================================================================
# 5. Interpretability
# ==============================================================================================
with tabs[4]:
    st.markdown("**Global drivers** — share of each feature in the model's total importance "
                "(units differ across methods, so compare *rankings*).")

    @st.cache_data(show_spinner=False)
    def global_importance():
        sample = TEST.sample(2000, random_state=42)
        xg = pd.Series(np.abs(xgb_contrib(sample)[:, :-1]).mean(axis=0), index=INPUTS)
        X = lr_design(sample)
        lr_imp = {}
        for col in LR["columns"]:
            feat = "grade_subgrade" if col == "grade_ord" else next((k for k in LR["cat"] if col.startswith(k + "_")), col)
            lr_imp[feat] = lr_imp.get(feat, 0) + float(np.mean(np.abs((X[col] - X[col].mean()) * LR_BETA[col])))
        tf = pd.Series(TFM_SUMMARY["interpretability"]["permutation_importance"]).clip(lower=0)
        out = []
        for name, s, method in [("Logistic regression", pd.Series(lr_imp), "mean |β·(x − x̄)| (exact)"),
                                ("XGBoost", xg, "mean |TreeSHAP| (exact)"),
                                ("TabPFN", tf, "permutation importance (ΔAUC)")]:
            s = s.reindex(INPUTS).fillna(0); s = s / s.sum()
            out += [{"Model": name, "Feature": LABEL[f], "Share of importance": v, "Method": method} for f, v in s.items()]
        return pd.DataFrame(out)

    gi = global_importance()
    order = gi.groupby("Feature")["Share of importance"].mean().sort_values(ascending=False).index.tolist()
    ch = alt.Chart(gi).mark_bar(cornerRadiusEnd=4).encode(
        y=alt.Y("Feature:N", sort=order, title=None), yOffset=alt.YOffset("Model:N", sort=list(COLOR)),
        x=alt.X("Share of importance:Q", axis=alt.Axis(format="%")),
        color=alt.Color("Model:N", scale=MODEL_SCALE, legend=alt.Legend(orient="top", title=None)),
        tooltip=["Model", "Feature", alt.Tooltip("Share of importance:Q", format=".1%"), "Method"])
    st.altair_chart(ch.properties(height=420), **STRETCH)
    st.markdown("""
The three models rely on **the same three drivers**: employment status, debt-to-income ratio and credit score.
Where they differ is **how** they can be explained:
""")
    st.markdown("""
| Model | Explaining one decision | Cost per applicant | Global view |
|---|---|---|---|
| Logistic regression | **exact**: each decision = Σ β·(x − x̄) | instant | coefficients, odds ratios, marginal effects |
| XGBoost | **exact for the model**: TreeSHAP | milliseconds | SHAP, XPER, PDP/ICE (post-hoc) |
| TabPFN | **approximate**: permutation SHAP / ablation; LIME local fidelity R² 0.09–0.18 | ≈1 s (ablation) to minutes (SHAP) | permutation importance, XPER, PDP/ICE (post-hoc, slow) |
""")
    st.caption("GDPR art. 22 and the EU AI Act (credit scoring is a high-risk system) require that a rejected applicant can be given the reasons.")

# ==============================================================================================
# 6. Recommendation
# ==============================================================================================
with tabs[5]:
    perf, _ = performance(lgd)
    trade = perf[["Model", "AUC", "Profit (M€)"]].copy()
    trade["Decisions flipped by 1% noise"] = sd_["Decisions that flip"].to_numpy()
    trade["Explanations"] = ["exact, instant", "exact (TreeSHAP), fast", "approximate, slow"]
    trade["Gender fairness (TOST δ = 2 pp)"] = [
        "✅ certified" if tost_parity(TEST[f"p_{k}"].to_numpy() >= thr[k], TEST.gender, "Female", "Male", 0.02)["certified fair"] else "⚠️ not certified"
        for k in KEYS]
    trade["Scoring cost"] = ["microseconds", "milliseconds", "≈1 s on GPU (bagged)"]
    st.markdown(f"**Trade-offs at LGD = {lgd:.0%}**")
    st.dataframe(trade.style.format({"AUC": "{:.4f}", "Profit (M€)": "{:.2f}", "Decisions flipped by 1% noise": "{:.1%}"}),
                 hide_index=True, **STRETCH)
    rec = APP / "recommendation.md"
    if rec.exists():
        st.markdown(rec.read_text())
