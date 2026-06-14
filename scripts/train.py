"""CLI entry point for training."""

from __future__ import annotations

import sys
from pathlib import Path

import click

# Ensure src is on path even if package not installed
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.trainer import train_from_config  # noqa: E402
from src.utils.logging import setup_logging  # noqa: E402


@click.command()
@click.option(
    "--config",
    "config_path",
    default="config/config.yaml",
    show_default=True,
    help="Path to YAML config file.",
)
@click.option("--log-level", default="INFO", show_default=True)
def main(config_path: str, log_level: str) -> None:
    """Train LoanGuard end-to-end and persist artifacts."""
    setup_logging(level=log_level)
    results = train_from_config(config_path)
    click.echo("=" * 60)
    click.echo("Training complete. Top-line ensemble metrics:")
    click.echo("=" * 60)
    if results.get("ensemble"):
        for k, v in results["ensemble"].items():
            click.echo(f"  {k:30s}  {v}")
    else:
        click.echo("  (ensemble not trained — only base models)")


if __name__ == "__main__":
    main()
