#!/usr/bin/env bash
# Builds deploy_artifacts/ — a small (~20MB), git-trackable subset of
# the full local pipeline output (~780MB) that's all the dashboard
# actually reads at runtime. Run this locally after ./run.sh, then
# commit and push deploy_artifacts/ yourself — a git-based host just
# needs to pull and start the app, no separate upload/download step.
#
# This does NOT retrain anything and does NOT touch run.sh's output —
# it only copies from what ./run.sh already produced.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f "results/model_comparison.csv" ]; then
  echo "No results/model_comparison.csv found — run ./run.sh first (full local pipeline)."
  exit 1
fi

source .venv/bin/activate
python3 -m src.prepare_deploy_artifacts
python3 scripts/prepare_deploy_dir.py

echo
echo "==> deploy_artifacts/ is ready. Review it, then:"
echo "      git add deploy_artifacts/ && git commit -m 'Update deploy artifacts' && git push"
