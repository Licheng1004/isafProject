"""
tabpfn_model.py — TabPFN wrapper used by the notebook AND reusable by the web app.

TabPFN does in-context learning: "fitting" only stores the context set (instant), all the
cost is at prediction time. Hence: batched inference, GPU memory release, disk cache.

Web app usage:
    from tabpfn_model import load_production_model
    model = load_production_model("outputs/model")        # rebuilds the robust TabPFN (~5 s)
    model.decide(applicants_df)                            # raw columns -> P(paid back), decision
"""
from __future__ import annotations

import gc
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tabpfn import TabPFNClassifier

from isaf_eval import CAT_COLS, FEATURES, SEED, TARGET

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def category_levels(df: pd.DataFrame) -> dict:
    """Sorted levels of each categorical column (grade_subgrade sorts A1 < ... < F5)."""
    return {c: sorted(df[c].unique().tolist()) for c in CAT_COLS}


def encode(df: pd.DataFrame, levels: dict) -> pd.DataFrame:
    """Raw applicant table -> numeric table (categoricals as integer codes), columns = FEATURES.
    Columns the model does not use (e.g. gender, marital status) may be absent: they are coded -1."""
    X = df.reindex(columns=FEATURES).copy()
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=levels[c]).codes
    return X.astype(float)


class TabPFNModel:
    """Takes the ENCODED feature table (see `encode`) so permutation importance / SHAP / PDP
    can manipulate columns directly. Only the columns in `features` are used."""

    def __init__(self, features=FEATURES, n_estimators=4, seed=SEED, device=DEVICE, batch=1000,
                 fingerprint=True, fit_mode="fit_preprocessors"):
        self.features, self.n_estimators, self.seed = list(features), n_estimators, seed
        self.device, self.batch, self.fingerprint, self.fit_mode = device, batch, fingerprint, fit_mode

    def fit(self, X: pd.DataFrame, y):
        cat_idx = [i for i, c in enumerate(self.features) if c in CAT_COLS]
        # fingerprint = TabPFN default: a hash of each row added as a feature (meant for duplicated
        # rows). It makes predictions react to any tiny input change -> we switch it off in production.
        extra = {} if self.fingerprint else {"inference_config": {"FINGERPRINT_FEATURE": False}}
        self.clf = TabPFNClassifier(device=self.device, n_estimators=self.n_estimators,
                                    random_state=self.seed, categorical_features_indices=cat_idx,
                                    ignore_pretraining_limits=True, fit_mode=self.fit_mode, **extra)
        self.clf.fit(X[self.features].to_numpy(), np.asarray(y).astype(int))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """P(paid back), in batches, releasing GPU memory after each one (avoids MPS OOM)."""
        A = X[self.features].to_numpy()
        out = []
        for i in range(0, len(A), self.batch):
            out.append(self.clf.predict_proba(A[i:i + self.batch])[:, 1])
            free_mem()
        return np.concatenate(out)


class BaggedTabPFN:
    """Context bagging: K TabPFN models, each conditioned on a DIFFERENT random context set,
    predictions averaged. Uses K x more training rows and averages away the 'context lottery'
    (variance reduction, like bagging), at K x the inference cost."""

    def __init__(self, members: list):
        self.members = members
        self.features = members[0].features

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.mean([m.predict_proba(X) for m in self.members], axis=0)


def free_mem():
    gc.collect()
    if DEVICE == "mps":
        torch.mps.empty_cache()
    elif DEVICE == "cuda":
        torch.cuda.empty_cache()


class Cache:
    """Tiny disk cache: `cache(name, fn)` returns the stored result or computes and stores it.
    Delete outputs/cache/ (or pass recompute=True) to force a full rerun."""

    def __init__(self, folder: Path, recompute: bool = False):
        self.folder, self.recompute = Path(folder), recompute
        self.folder.mkdir(parents=True, exist_ok=True)

    def __call__(self, name: str, fn):
        f = self.folder / (re.sub(r"[^\w.-]+", "_", name).strip("_") + ".pkl")   # safe file name
        if f.exists() and not self.recompute:
            return pd.read_pickle(f)
        res = fn()
        pd.to_pickle(res, f)
        return res

# ----------------------------------------------------------------------------
# Production bundle for the web app
# ----------------------------------------------------------------------------
def save_production_model(folder, contexts: list, levels: dict, threshold: float, features: list,
                          n_estimators: int, fingerprint: bool, seeds: list, extra: dict | None = None):
    """Saves what is needed to rebuild the (bagged) model: the K context sets + encoding +
    threshold. TabPFN weights are the public pretrained checkpoint, no need to ship them."""
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    for f in folder.glob("tabpfn_context*.csv.gz"):
        f.unlink()
    for k, ctx in enumerate(contexts):
        ctx[FEATURES + [TARGET]].to_csv(folder / f"tabpfn_context_{k}.csv.gz", index=False)
    meta = {"threshold": threshold, "n_estimators": n_estimators, "seeds": list(seeds), "n_contexts": len(contexts),
            "features": list(features), "fingerprint": fingerprint, "levels": levels, **(extra or {})}
    (folder / "tabpfn_config.json").write_text(json.dumps(meta, indent=2, default=float))


class ProductionModel:
    def __init__(self, model, levels: dict, threshold: float):
        self.model, self.levels, self.threshold = model, levels, threshold

    def predict_proba(self, applicants: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(encode(applicants, self.levels))

    def decide(self, applicants: pd.DataFrame) -> pd.DataFrame:
        p = self.predict_proba(applicants)
        return pd.DataFrame({"p_paid_back": p, "approved": p >= self.threshold})


def load_production_model(folder) -> ProductionModel:
    folder = Path(folder)
    meta = json.loads((folder / "tabpfn_config.json").read_text())
    members = []
    for k in range(meta["n_contexts"]):
        ctx = pd.read_csv(folder / f"tabpfn_context_{k}.csv.gz")
        members.append(TabPFNModel(features=meta["features"], n_estimators=meta["n_estimators"],
                                   seed=meta["seeds"][k], fingerprint=meta["fingerprint"],
                                   fit_mode="fit_with_cache")   # caches the context: fast single-applicant scoring
                       .fit(encode(ctx, meta["levels"]), ctx[TARGET]))
    return ProductionModel(BaggedTabPFN(members), meta["levels"], meta["threshold"])
