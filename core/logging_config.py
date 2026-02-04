"""Structured logging configuration for the RAG application.

Supports two output formats controlled by the LOG_FORMAT environment variable:
- "text" (default): human-readable log lines
- "json": machine-parseable JSON log lines (for log aggregators)

Usage:
    from logging_config import setup_logging
    setup_logging()
"""
import json
import logging
import sys
import os
from datetime import datetime, timezone

from .config import Config


class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects.

    Output fields: timestamp, level, logger, message, module, function, line,
    and request_id (if present on the record via RequestIDMiddleware).
    """

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        # Include request_id when available (set by RequestIDMiddleware)
        request_id = getattr(record, "request_id", None)
        if request_id:
            log_entry["request_id"] = request_id

        # Include exception info if present
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, ensure_ascii=False)


def setup_logging() -> None:
    """Configure the root logger based on environment variables.

    Reads:
        LOG_LEVEL  - logging level name (default "INFO")
        LOG_FORMAT - "text" or "json" (default "text")
    """
    log_level = getattr(logging, Config.LOG_LEVEL.upper(), logging.INFO)
    log_format = Config.LOG_FORMAT.lower()

    # Create handler writing to stdout
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)

    if log_format == "json":
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-8s [%(name)s] %(message)s"
        ))

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove any existing handlers to avoid duplicates on reload
    root_logger.handlers.clear()
    root_logger.addHandler(handler)

    # Quieten noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
