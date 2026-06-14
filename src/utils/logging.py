"""Structured logging using loguru.

The whole project goes through this module — never call print().
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

_INITIALISED = False


def setup_logging(level: str = "INFO", log_file: str | Path | None = None) -> None:
    """Initialise loguru sinks. Call once at process start.

    Args:
        level: Minimum level to emit (DEBUG / INFO / WARNING / ERROR)
        log_file: Optional path to write structured logs to (JSON).
    """
    global _INITIALISED
    if _INITIALISED:
        return

    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        enqueue=True,
    )

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_path,
            level=level,
            rotation="50 MB",
            retention="14 days",
            serialize=True,
            enqueue=True,
        )

    _INITIALISED = True


def get_logger(name: str | None = None):
    """Return a logger bound to a module name."""
    if not _INITIALISED:
        setup_logging()
    return logger.bind(scope=name) if name else logger
