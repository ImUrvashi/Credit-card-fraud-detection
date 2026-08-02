"""Zips up only what the deployed dashboard actually reads at runtime —
one model file, val.csv, and the small precomputed caches/summaries —
instead of the full data/ and models/ directories (~780MB vs ~20MB).

Run after the full local pipeline (./run.sh) has produced results/.
Upload the resulting deploy_bundle.zip somewhere (a GitHub Release
asset, Hugging Face, S3, ...) and have your host unzip it into the
project root before starting `python -m dashboard.app` — the paths
inside the zip already match where the app expects to find them.
"""

import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dashboard import data_loaders as dl  # noqa: E402

OUTPUT_PATH = Path("deploy_bundle.zip")

FIXED_PATHS = [
    Path("data/processed/val.csv"),
    Path("results/model_comparison.csv"),
    Path("results/dataset_summary.json"),
    Path("results/live_sim_sequence.csv"),
]


def collect_paths() -> list[Path]:
    results_df = dl.load_results()
    best_run = dl.best_run_name(results_df)
    paths = list(FIXED_PATHS)
    paths.append(Path("models") / f"{best_run}.joblib")
    paths += sorted(Path("results/shap_cache").glob("*.joblib"))
    paths += sorted(Path("results/embedding_cache").glob("*.joblib"))
    return paths


def build_bundle(paths: list[Path], output_path: Path = OUTPUT_PATH):
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} expected artifact(s), run ./run.sh first: {missing}"
        )

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in paths:
            zf.write(path, arcname=str(path))

    total_size = sum(p.stat().st_size for p in paths)
    print(f"Bundled {len(paths)} files ({total_size / 1e6:.1f} MB uncompressed)")
    print(f"-> {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    build_bundle(collect_paths())
