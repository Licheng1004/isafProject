"""Validate the public clone and retained artifacts without private inputs."""
from pathlib import Path
import ast
import hashlib
import json
import nbformat

ROOT = Path(__file__).resolve().parents[1]

def check_notebooks():
    count = 0
    for path in sorted(ROOT.rglob("*.ipynb")):
        if any(part in {".venv", ".git", ".reproduction", ".ipynb_checkpoints"} for part in path.parts):
            continue
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type != "code":
                continue
            lines = cell.source.splitlines(keepends=True)
            if next((line.lstrip() for line in lines if line.strip()), "").startswith("%%"):
                continue
            code = "".join(line for line in lines if not line.lstrip().startswith(("%", "!")))
            ast.parse(code, filename=f"{path.relative_to(ROOT)}:{index}")
        count += 1
    print(f"Validated {count} notebook(s). This does not execute their training cells.")

def check_manifest():
    path = ROOT / "data_manifest.json"
    if not path.exists():
        return
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for entry in manifest.get("inputs", []):
        relative = entry.get("file")
        expected = entry.get("stored_sha256") or entry.get("sha256")
        if relative and expected:
            actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            if actual != expected:
                raise SystemExit(f"Input hash mismatch: {relative}")
    print("Input manifest hashes checked where recorded.")

def main():
    check_notebooks()
    check_manifest()
    specific_checks()
    print("Public verification passed. See REPRODUCIBILITY.md for its scope.")

def specific_checks():
    import numpy as np
    import pandas as pd
    import xgboost as xgb
    from scipy.special import expit
    from sklearn.metrics import roc_auc_score
    art = ROOT / "app/artifacts"
    test = pd.read_csv(art / "test_predictions.csv.gz")
    assert len(test) == 20000 and test.id.is_unique
    meta = json.loads((art / "xgb_meta.json").read_text())
    X = test[meta["inputs"]].copy()
    for col, levels in meta["categories"].items():
        X[col] = pd.Categorical(X[col], categories=levels)
    model = xgb.Booster()
    model.load_model(bytearray((art / "xgb_model.json").read_bytes()))
    predictions = expit(model.predict(xgb.DMatrix(X, enable_categorical=True),
        output_margin=True, iteration_range=(0, meta["n_trees"])))
    np.testing.assert_allclose(predictions, test.p_xgb, atol=1e-3, rtol=0)
    for key in ["lr", "xgb", "tabpfn"]:
        assert np.isfinite(test["p_" + key]).all()
        print(f"Saved {key} AUC: {roc_auc_score(test.y, test['p_' + key]):.6f}")
    print("Loaded XGBoost reproduces 20,000 saved probabilities (absolute tolerance 0.001).")

if __name__ == "__main__":
    main()
