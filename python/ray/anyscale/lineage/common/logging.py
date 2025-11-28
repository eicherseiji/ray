import logging
from typing import Optional

from ray.anyscale.lineage.common.constants import LOG_ENCODING, LOG_LEVEL


_logging_configured = False


def configure_logging(
    level: Optional[str] = None,
    enable_console: bool = True,
    enable_file: bool = True,
) -> None:
    """Configure logging for ray.anyscale.lineage.

    This should be called explicitly before using lineage loggers.
    File logs are written to Ray session directory: {session_dir}/logs/lineage/
    """
    global _logging_configured

    if level is None:
        level = LOG_LEVEL

    # Lazy imports to avoid issues if Ray isn't initialized yet
    from ray._common.formatters import JSONFormatter
    from ray.data._internal.logging import SessionFileHandler

    logger = logging.getLogger("ray.anyscale.lineage")
    logger.setLevel(level)

    # Clear existing handlers to avoid duplicates on re-configuration
    logger.handlers.clear()
    logger.propagate = False

    # Create formatter based on environment variable
    log_encoding = LOG_ENCODING
    if log_encoding.upper() == "JSON":
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s\t%(levelname)s %(filename)s:%(lineno)s -- %(message)s"
        )

    if enable_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if enable_file:
        file_handler = SessionFileHandler(filename="lineage/ray-lineage.log")
        file_handler.setLevel(level)
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
