import logging
import os
from typing import Optional

from ray._common.formatters import JSONFormatter
from ray.anyscale.lineage.common.constants import (
    LOG_ENABLE_CONSOLE,
    LOG_ENCODING,
    LOG_LEVEL,
)


class LineageSessionFileHandler(logging.Handler):
    """Logging handler to write to files in the Ray session directory."""

    def __init__(self, filename: str):
        super().__init__()
        self._filename = filename
        self._handler: Optional[logging.FileHandler] = None
        self._formatter: Optional[logging.Formatter] = None
        self._path: Optional[str] = None

    def emit(self, record: logging.LogRecord) -> None:
        if self._handler is None:
            self._try_create_handler()
        if self._handler is not None:
            self._handler.emit(record)

    def setFormatter(self, fmt: Optional[logging.Formatter]) -> None:
        if self._handler is not None:
            self._handler.setFormatter(fmt)
        self._formatter = fmt

    def _try_create_handler(self):
        log_directory = self._get_log_directory()

        if log_directory is None:
            return

        self._path = os.path.join(log_directory, self._filename)
        self._handler = logging.FileHandler(self._path)
        if self._formatter is not None:
            self._handler.setFormatter(self._formatter)

    @staticmethod
    def _get_log_directory() -> Optional[str]:
        """Returns '{session_dir}/logs/lineage/' or None."""
        # Lazy import to avoid circular dependency (utils.py imports logging.py)
        from ray.anyscale.lineage.common.utils import get_lineage_logs_dir

        try:
            return get_lineage_logs_dir()
        except RuntimeError:
            return None


_logging_configured = False


def configure_logging(
    level: Optional[str] = LOG_LEVEL,
    enable_console: bool = LOG_ENABLE_CONSOLE,
    enable_file: bool = True,
) -> None:
    """Configure logging for ray.anyscale.lineage.

    This should be called explicitly before using lineage loggers.
    File logs are written to Ray session directory: {session_dir}/logs/lineage/
    """
    global _logging_configured

    # Skip if already configured to avoid duplicate handlers
    if _logging_configured:
        return

    effective_level = level or "INFO"

    # Set OpenLineage client logging level to match our logging level.
    # This must be set before initializing the OpenLineage client.
    os.environ["OPENLINEAGE_CLIENT_LOGGING"] = effective_level

    logger = logging.getLogger("ray.anyscale.lineage")
    logger.setLevel(effective_level)

    # Clear existing handlers to avoid duplicates on re-configuration
    logger.handlers.clear()
    logger.propagate = False

    # Create formatter based on environment variable
    formatter: logging.Formatter
    if LOG_ENCODING.upper() == "JSON":
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s\t%(levelname)s %(filename)s:%(lineno)s -- %(message)s"
        )

    if enable_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(effective_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if enable_file:
        file_handler = LineageSessionFileHandler(filename="ray-lineage.log")
        file_handler.setLevel(effective_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _logging_configured = True


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance.

    Call configure_logging() first to set up handlers and formatters.
    If not called, the logger will use Python's default configuration.
    """
    return logging.getLogger(name)


def reset_logging() -> None:
    """Reset logging configuration.

    Clears all handlers and resets the logging state.
    Useful for testing or reconfiguration.
    """
    global _logging_configured
    logger = logging.getLogger("ray.anyscale.lineage")
    logger.handlers.clear()
    logger.setLevel(logging.NOTSET)
    _logging_configured = False
