import logging
import os
import tempfile
from unittest import mock

import pytest

from ray._common.formatters import JSONFormatter
from ray.anyscale.lineage.common import logging as logging_module
from ray.anyscale.lineage.common.logging import LineageSessionFileHandler


@pytest.fixture
def preserve_logger_state():
    """Preserve and restore logger state for test isolation."""
    logger = logging.getLogger("ray.anyscale.lineage")

    original_handlers = logger.handlers[:]
    original_level = logger.level
    original_propagate = logger.propagate
    original_configured = logging_module._logging_configured

    yield

    logger.handlers = original_handlers
    logger.setLevel(original_level)
    logger.propagate = original_propagate
    logging_module._logging_configured = original_configured


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


class TestLineageSessionFileHandler:
    """Tests for LineageSessionFileHandler class."""

    def test_emit_with_ray_session(self):
        """Handler creates file and applies formatter on first emit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lineage_logs_dir = os.path.join(tmpdir, "logs", "lineage")
            os.makedirs(lineage_logs_dir, exist_ok=True)

            with mock.patch(
                "ray.anyscale.lineage.common.utils.get_lineage_logs_dir",
                return_value=lineage_logs_dir,
            ):
                handler = LineageSessionFileHandler(filename="test.log")
                formatter = logging.Formatter("%(message)s")
                handler.setFormatter(formatter)

                record = logging.LogRecord(
                    name="test",
                    level=logging.INFO,
                    pathname="test.py",
                    lineno=1,
                    msg="Test message",
                    args=(),
                    exc_info=None,
                )
                handler.emit(record)
                handler._handler.flush()

                expected_path = os.path.join(lineage_logs_dir, "test.log")
                assert handler._path == expected_path
                assert handler._handler.formatter is formatter
                with open(expected_path) as f:
                    assert "Test message" in f.read()

    def test_emit_without_ray_session(self):
        """Handler gracefully handles missing Ray session."""
        with mock.patch(
            "ray.anyscale.lineage.common.utils.get_lineage_logs_dir",
            side_effect=RuntimeError("Ray global node is not initialized"),
        ):
            handler = LineageSessionFileHandler(filename="test.log")

            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="test.py",
                lineno=1,
                msg="Test message",
                args=(),
                exc_info=None,
            )
            handler.emit(record)

            assert handler._handler is None


if __name__ == "__main__":
    import sys
    import pytest

    sys.exit(pytest.main(["-v", "-x", __file__]))
