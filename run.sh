#!/usr/bin/env bash
# One-command bootstrap: venv, deps, pipeline (only what's missing), then the
# dashboard. Safe to re-run — every pipeline step is skipped if its output
# already exists, so after the first run this just launches the dashboard.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  echo "==> Creating virtual environment..."
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install -q -r requirements.txt

if [[ "$OSTYPE" == darwin* ]] && command -v brew >/dev/null && ! brew list libomp >/dev/null 2>&1; then
  echo "==> Installing libomp (required by XGBoost/LightGBM on macOS)..."
  brew install libomp
fi

if [ ! -f "data/raw/creditcard.csv" ]; then
  echo "Missing data/raw/creditcard.csv."
  echo "Download it from https://www.kaggle.com/mlg-ulb/creditcardfraud and place"
  echo "it there (see data/raw/README.md), then re-run this script."
  exit 1
fi

if [ ! -f "data/processed/train.csv" ]; then
  echo "==> Preparing data..."
  python3 -m src.data
fi

if [ ! -f "results/model_comparison.csv" ]; then
  echo "==> Training 9 algorithms x 7 imbalance strategies (this can take a while)..."
  python3 -m src.train
fi

if [ ! -d "results/shap_cache" ] || [ -z "$(ls -A results/shap_cache 2>/dev/null)" ]; then
  echo "==> Computing SHAP explanations..."
  python3 -m src.explain
fi

if [ ! -d "results/embedding_cache" ] || [ -z "$(ls -A results/embedding_cache 2>/dev/null)" ]; then
  echo "==> Computing PCA/t-SNE/UMAP embeddings..."
  python3 -m src.embeddings
fi

echo "==> Starting dashboard at http://localhost:8050"
python3 -m dashboard.app
