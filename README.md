# Credit Card Fraud Detection

9 classification algorithms × 7 imbalance-handling strategies (58 compared
runs), SHAP explainability, PCA/t-SNE/UMAP embeddings, and an interactive Dash
dashboard.

**Best result:** CatBoost (baseline) — PR-AUC 0.809, precision 0.947, recall
0.735. Full comparison: `results/model_comparison.csv` (sortable in the
dashboard's leaderboard).

## Quick start

```bash
./run.sh
```

Sets up the venv, installs deps (+ `libomp` via Homebrew on macOS if missing),
runs whichever pipeline step hasn't produced output yet (data prep → training →
SHAP → embeddings), and opens the dashboard at **http://localhost:8050**.
Idempotent — re-run anytime; only the first run trains everything (a few
minutes, 58 combinations).

Needs the dataset first — see `data/raw/README.md` (not redistributed here,
Kaggle doesn't allow it).

<details>
<summary>Manual steps</summary>

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

brew install libomp   # macOS only, if missing — needed by XGBoost/LightGBM

python3 -m src.data          # -> data/processed/{train,val,test}.csv
python3 -m src.train         # -> results/model_comparison.csv, models/*.joblib
python3 -m src.explain       # -> results/shap_cache/*.joblib
python3 -m src.embeddings    # -> results/embedding_cache/*.joblib
python3 -m dashboard.app     # -> http://localhost:8050
```

</details>

## Project structure

```
data/raw/           # download creditcard.csv here (see data/raw/README.md)
data/processed/     # generated: train/val/test splits
notebooks/          # EDA
src/
  data.py           # load, feature-engineer, split
  resampling.py     # the 7 imbalance-handling strategies
  models.py         # the 9-algorithm registry
  train.py          # runs the 9x7 grid -> results/, models/
  explain.py        # SHAP per algorithm's best run
  embeddings.py     # PCA/t-SNE/UMAP projections
results/, models/   # generated artifacts
dashboard/
  app.py            # entry point
  layout.py         # card layout
  figures.py        # Plotly figure builders + palette
  data_loaders.py   # shared IO helpers
  callbacks.py      # live interaction
```

## Dashboard, top to bottom

1. **Dataset overview** — what the data is, why PR-AUC.
2. **Fraud volume over time** — fraud count by hour of day.
3. **Leaderboard** — all 58 runs, sortable.
4. **Compare two runs** — side-by-side metric comparison.
5. **Feature importance** — mean |SHAP value|, bar or treemap.
6. **Transaction detail** — SHAP waterfall for a selected transaction.
7. **2D embedding** — PCA/t-SNE/UMAP; click a point to load its SHAP detail above.
8. **Threshold tuner** — live precision/recall/F1, confusion matrix, and a
   toggleable ROC/PR curve with a marker at the current cutoff.
9. **Live transaction stream** — replays held-out test data with real-time
   inference and fraud flagging.
