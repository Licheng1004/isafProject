# Loan Approval: Performance, Explainability, Stability and Fairness

[![CI](https://github.com/Licheng1004/isafProject/actions/workflows/ci.yml/badge.svg)](https://github.com/Licheng1004/isafProject/actions/workflows/ci.yml)

Compare logistic regression, XGBoost and TabPFN for a consumer lender, with a common 20,000-loan evaluation set and an interactive decision-support app. The question is which model offers the best trade-off between predictive performance, explanations, robustness, fairness and a stated profit model.

**Portfolio owner:** Licheng Xiong. **Context:** HEC Paris ISAF team project, Fall 2026, Prof. C. Pérignon. **Stack:** Python, XGBoost, SHAP, scikit-learn, Fairlearn, Streamlit; optional PyTorch/TabPFN.

## Results and artifacts

The committed app artifacts report approximately **0.906 AUC for logistic regression** and **0.916 for XGBoost/TabPFN**, with model-specific validation thresholds and the team's 60% LGD assumption. Profit values represent a modeled outcome on the evaluation snapshot, not revenue realized by a deployed lender.

- [My final English XGBoost notebook](MLModels/XGBoost_ISAF_final_EN.ipynb)
- [Detailed Chinese XGBoost notebook](MLModels/XGBoost_ISAF_final_CN.ipynb)
- [App and evaluation design](app/README.md)
- [Team recommendation](app/recommendation.md), a draft requiring team/client review
- [TabPFN analysis](TabularFoundationModel/03_tabular_foundation_model.ipynb)

## My contribution and team attribution

This is a team project, originally coordinated at `gitlab.code.hfactory.io/ethan.ben-hamou/isaf-project`. The Git history records work by Ethan Benhamou, Najib Nasr, Zixiao Guo and Licheng Xiong. My directly evidenced work is the XGBoost development and evaluation, including annotated course-aligned analysis, sensitive-feature variants, and the final English/Chinese notebook versions. The logistic-regression, foundation-model and app work are shared team deliverables, not claimed as my sole original work.

## Evaluation design

The project compares performance, interpretability, stability and fairness. Development/validation data choose variants and approval thresholds; the shared held-out set reports outcomes. XGBoost analysis includes TreeSHAP, performance decomposition, local explanations, perturbation/seed stability and group/conditional fairness diagnostics. Sensitive-attribute exclusion alone does not remove proxy effects; the recommendation discusses the resulting trade-offs.

## Reproduce the project

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then clone this repository. The committed `.python-version` selects Python 3.12.14; `pyproject.toml` and `uv.lock` lock the direct and transitive Python dependencies. First installation requires internet access.

```sh
git clone https://github.com/Licheng1004/isafProject.git
cd isafProject
uv sync --locked
uv run --locked python -m streamlit run app/streamlit_app.py
```

The default environment supports the public app, live logistic-regression/XGBoost scoring and **saved TabPFN results** without downloading foundation-model weights. Live TabPFN is off by default.

### Verify the installation and published evidence

```sh
uv run --locked python scripts/verify.py
uv run --locked python -m pytest -q
```

`--locked` rejects stale project metadata instead of silently changing the dependency versions. Select `.venv` as the Jupyter kernel in VS Code. Notebook path detection supports execution from the repository root or a notebook subdirectory. Runtime installation and synthetic checks do not reproduce a private-data competition score.

See [the reproducibility record](REPRODUCIBILITY.md) for the actual checks performed, their scope and any external prerequisites.

## Optional live foundation-model scoring

```sh
uv sync --locked --extra foundation
uv run --locked --extra foundation python -m streamlit run app/streamlit_app.py
```

Enable **Live TabPFN scoring** only after accepting the provider's model license and arranging authorized weight access. First use downloads weights; CPU/GPU memory, access and runtime depend on the provider and machine. The worker isolates PyTorch from XGBoost to avoid the team's observed macOS OpenMP conflict. A Python lock cannot grant model access or guarantee identical predictions across hardware. The app checks loaded-model predictions against saved artifacts.

## Data and complete analysis

The synthetic loan data originate from [Kaggle Playground S5E11](https://www.kaggle.com/competitions/playground-series-s5e11). The repository currently includes `data/train.csv` and the team's `data/test_eval_20k.csv`; preserve these matching inputs for the shared protocol. The app uses committed `app/artifacts/` and can run without retraining.

Use `uv run --locked python -m jupyterlab` to inspect the analyses. The final XGBoost notebook locates the project root automatically; the logistic-regression notebook should be run from the root, and the TabPFN notebook from `TabularFoundationModel/` with the foundation extra. Rebuilding the app's artifacts requires the full analyses and the matching saved TabPFN outputs; it is not the quick-start route.

## Repository map

| Path | Purpose |
| --- | --- |
| `MLModels/` | XGBoost final analysis and historical experiments |
| `01_logistic_regression.ipynb` | Team white-box baseline |
| `TabularFoundationModel/` | Team TabPFN analysis, saved results and worker support |
| `app/` | Streamlit app, models and comparison artifacts |
| `data/` | Synthetic loan inputs and shared evaluation set |
| `scripts/`, `tests/` | Public artifact/model and app verification |

## Limits and reuse

This is coursework on synthetic data, not a validated lending policy. Results depend on the split, categorical encoding, library versions and profit assumptions. The final notebooks are the supported research entry points; older experiments may require additional packages or different inputs. Model/data providers retain their terms and the README grants no new reuse license.
