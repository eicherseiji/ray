import logging
import os
from unittest import mock

import pytest

from ray._common.formatters import JSONFormatter
from ray.anyscale.lineage.common import logging as logging_module


@pytest.fixture
def preserve_logger_state():
    """Preserve and restore logger state for test isolation."""
    logger = logging.getLogger("ray.anyscale.lineage")

    original_handlers = logger.handlers[:]
    original_level = logger.level
    original_propagate = logger.propagate

    yield

    logger.handlers = original_handlers
    logger.setLevel(original_level)
    logger.propagate = original_propagate


@pytest.mark.parametrize(
    "level,enable_console,enable_file,expected_level,handler_type",
    [
        ("INFO", True, False, logging.INFO, logging.StreamHandler),
        ("DEBUG", False, True, logging.DEBUG, logging.Handler),
    ],
)
def test_configure_logging_handlers(
    preserve_logger_state,
    level,
    enable_console,
    enable_file,
    expected_level,
    handler_type,
):
    """Test logging configuration with various handler combinations."""
    logging_module.reset_logging()

    logging_module.configure_logging(
        level=level, enable_console=enable_console, enable_file=enable_file
    )

    logger = logging.getLogger("ray.anyscale.lineage")
    assert logger.level == expected_level
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], handler_type)


def test_log_encoding_json_default(preserve_logger_state):
    """Test that JSON encoding is used by default."""
    logging_module.reset_logging()

    logging_module.configure_logging(enable_console=True, enable_file=False)

    logger = logging.getLogger("ray.anyscale.lineage")
    assert len(logger.handlers) > 0
    assert isinstance(logger.handlers[0].formatter, JSONFormatter)


@mock.patch.dict(os.environ, {"ANYSCALE_LINEAGE_LOG_ENCODING": "TEXT"})
def test_log_encoding_text(preserve_logger_state):
    """Test that TEXT encoding uses standard formatter when configured."""
    import importlib
    from ray.anyscale.lineage.common import constants

    importlib.reload(constants)
    importlib.reload(logging_module)

    logging_module.reset_logging()

    logging_module.configure_logging(enable_console=True, enable_file=False)

    logger = logging.getLogger("ray.anyscale.lineage")
    assert len(logger.handlers) > 0
    assert not isinstance(logger.handlers[0].formatter, JSONFormatter)
    assert isinstance(logger.handlers[0].formatter, logging.Formatter)

    importlib.reload(constants)
    importlib.reload(logging_module)


if __name__ == "__main__":
    import sys
    import pytest

    sys.exit(pytest.main(["-v", "-x", __file__]))
