"""Download LendingClub data via Kaggle API.

Requires kaggle CLI configured with credentials. Otherwise falls back
to the synthetic generator (the loader handles that automatically).

Usage:
    python scripts/download_data.py --sample 250000
"""

from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging import setup_logging, get_logger  # noqa: E402

log = get_logger(__name__)


@click.command()
@click.option("--out-dir", default="data/raw", show_default=True)
@click.option("--sample", default=None, type=int, help="Optional: row sample size")
def main(out_dir: str, sample: int | None) -> None:
    setup_logging()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if shutil.which("kaggle") is None:
        log.warning(
            "kaggle CLI not found. Install with `pip install kaggle` and place "
            "your kaggle.json in ~/.kaggle/. Skipping download — the loader "
            "will fall back to the synthetic generator."
        )
        return

    import subprocess
    log.info("Downloading wordsforthewise/lending-club from Kaggle...")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", "wordsforthewise/lending-club", "-p", str(out)],
        check=True,
    )
    zip_path = next(out.glob("*.zip"))
    log.info(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out)
    zip_path.unlink()
    log.info(f"Done. Files in {out}: {[p.name for p in out.iterdir()]}")


if __name__ == "__main__":
    main()
