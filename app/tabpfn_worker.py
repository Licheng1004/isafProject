"""
tabpfn_worker.py — scores applicants with the robust TabPFN in a SEPARATE process.

torch (TabPFN) and xgboost ship incompatible OpenMP runtimes on macOS: loaded in the same
process they deadlock or segfault. The web app therefore talks to this worker through
stdin/stdout, one JSON line per request:  {"rows": [{...applicant...}, ...]}  ->  {"p": [...]}
"""
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "TabularFoundationModel" / "src"))

import pandas as pd  # noqa: E402
from tabpfn_model import load_production_model  # noqa: E402

out = sys.stdout
sys.stdout = sys.stderr          # anything TabPFN prints must not pollute the JSON channel
model = load_production_model(ROOT / "TabularFoundationModel" / "outputs" / "model")
out.write(json.dumps({"ready": True, "threshold": model.threshold}) + "\n"); out.flush()

for line in sys.stdin:
    try:
        rows = pd.DataFrame(json.loads(line)["rows"])
        out.write(json.dumps({"p": model.predict_proba(rows).tolist()}) + "\n")
    except Exception as e:  # keep the worker alive, report the error to the app
        out.write(json.dumps({"error": str(e)}) + "\n")
    out.flush()
