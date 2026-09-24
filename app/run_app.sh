#!/usr/bin/env bash
# Launch the web app with the project environment (xgboost >= 3, tabpfn >= 9, streamlit).
# Usage: bash app/run_app.sh      (set ISAF_PYTHON to use another interpreter)
cd "$(dirname "$0")/.."
PY="${ISAF_PYTHON:-$HOME/miniforge3/envs/isaf/bin/python}"
if [ ! -x "$PY" ]; then PY="$(command -v python3)"; fi
exec caffeinate -i "$PY" -m streamlit run app/streamlit_app.py "$@"
