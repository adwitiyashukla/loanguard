"""Batch-score a CSV of applications with a trained ensemble.

Usage:
    python scripts/score.py --input data/raw/new_apps.csv --output out.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.io import load_joblib  # noqa: E402
from src.utils.logging import setup_logging, get_logger  # noqa: E402

log = get_logger(__name__)


@click.command()
@click.option("--input", "input_path", required=True, type=click.Path(exists=True))
@click.option("--output", "output_path", required=True, type=click.Path())
@click.option("--artifacts", default="artifacts", show_default=True)
@click.option("--threshold", default=0.5, show_default=True)
def main(input_path: str, output_path: str, artifacts: str, threshold: float) -> None:
    setup_logging()
    artifacts_dir = Path(artifacts)

    builder = load_joblib(artifacts_dir / "feature_builder.joblib")
    model_path = artifacts_dir / "model_ensemble.joblib"
    if not model_path.exists():
        model_path = artifacts_dir / "model_xgboost.joblib"
    model = load_joblib(model_path)

    df = pd.read_csv(input_path)
    log.info(f"Scoring {len(df):,} applications from {input_path}")
    X = builder.transform(df)
    proba = model.predict_proba(X)
    df_out = df.copy()
    df_out["fraud_score"] = proba
    df_out["fraud_decision"] = (proba >= threshold).astype(int)
    df_out.to_csv(output_path, index=False)
    log.info(f"Wrote {output_path}. Fraud flag rate: {df_out['fraud_decision'].mean():.2%}")


if __name__ == "__main__":
    main()
