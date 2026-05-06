"""Structured JSON logging configuration.

Every module in the package retrieves its logger via ``logging.getLogger(__name__)``.
Calling :func:`configure_logging` once at process start (training script,
API startup, test session) is enough to wire structured output everywhere.

We deliberately avoid ``print()`` so logs are machine-readable in production.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

from pythonjsonlogger import jsonlogger


class _StaticFieldFilter(logging.Filter):
    """Inject static fields (e.g. service name) into every log record."""

    def __init__(self, fields: dict[str, Any]) -> None:
        super().__init__()
        self._fields = fields

    def filter(self, record: logging.LogRecord) -> bool:
        for key, value in self._fields.items():
            setattr(record, key, value)
        return True


def configure_logging(
    level: str | int = "INFO",
    *,
    service: str = "churn",
    json_format: bool | None = None,
) -> None:
    """Configure the root logger with a structured JSON formatter.

    Parameters
    ----------
    level:
        Logging level (e.g. ``"INFO"``, ``"DEBUG"``).
    service:
        Service name added as a static field on every record.
    json_format:
        Force JSON output on/off. When ``None`` (default), JSON is enabled
        unless ``LOG_FORMAT=text`` is set in the environment - useful for
        readable local development logs.
    """
    if json_format is None:
        json_format = os.environ.get("LOG_FORMAT", "json").lower() != "text"

    handler = logging.StreamHandler(stream=sys.stdout)
    if json_format:
        formatter: logging.Formatter = jsonlogger.JsonFormatter(
            "%(asctime)s %(name)s %(levelname)s %(message)s",
            rename_fields={"asctime": "timestamp", "levelname": "level"},
        )
    else:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        )
    handler.setFormatter(formatter)
    handler.addFilter(_StaticFieldFilter({"service": service}))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # Quiet down noisy third-party loggers by default.
    for noisy in ("urllib3", "matplotlib", "PIL", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Convenience wrapper so callers don't need to import ``logging``."""
    return logging.getLogger(name)
