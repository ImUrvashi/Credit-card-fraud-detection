"""Builds deploy_artifacts/ — only what the deployed dashboard actually
reads at runtime — from the full local pipeline output (~780MB), so
that directory alone is small enough (~20MB) to commit and push. Once
`dashboard/data_loaders.py` sees deploy_artifacts/ exists, it reads
every path from there instead of data/, models/, results/.

Run after the full local pipeline (./run.sh) has produced results/.
Normally invoked via ./deploy.sh, not directly.
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dashboard import data_loaders as dl  # noqa: E402

OUTPUT_DIR = Path("deploy_artifacts")

REQUIRED_SOURCES = [
    Path("data/processed/val.csv"),
    Path("results/model_comparison.csv"),
    Path("results/dataset_summary.json"),
    Path("results/live_sim_sequence.csv"),
    Path("results/shap_cache"),
    Path("results/embedding_cache"),
]


def check_sources_exist():
    missing = [p for p in REQUIRED_SOURCES if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} expected artifact(s), run ./run.sh first: {missing}")


def build_deploy_dir(output_dir: Path = OUTPUT_DIR):
    check_sources_exist()

    results_df = dl.load_results()
    best_run = dl.best_run_name(results_df)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    # Generic filename, not the run's actual name — which algorithm
    # wins can change between retrains, but the dashboard code and the
    # committed path here both need something stable.
    shutil.copy(f"models/{best_run}.joblib", output_dir / "model.joblib")
    shutil.copy("data/processed/val.csv", output_dir / "val.csv")
    shutil.copy("results/model_comparison.csv", output_dir / "model_comparison.csv")
    shutil.copy("results/dataset_summary.json", output_dir / "dataset_summary.json")
    shutil.copy("results/live_sim_sequence.csv", output_dir / "live_sim_sequence.csv")
    shutil.copytree("results/shap_cache", output_dir / "shap_cache")
    shutil.copytree("results/embedding_cache", output_dir / "embedding_cache")

    total_size = sum(p.stat().st_size for p in output_dir.rglob("*") if p.is_file())
    print(f"Best run: {best_run}")
    print(f"Wrote {output_dir}/ ({total_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    build_deploy_dir()
