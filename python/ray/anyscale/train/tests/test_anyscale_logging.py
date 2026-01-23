import builtins
import logging

import pytest

import ray
from ray.runtime_context import get_runtime_context
from ray.train.v2._internal.logging import LoggingManager
from ray.train.v2.tests.test_logging import get_file_contents
from ray.train.v2.tests.util import create_dummy_run_context, create_dummy_train_context


@pytest.fixture(name="worker_logging")
def worker_logging_fixture():
    # Save the current root logger settings
    root_logger = logging.getLogger()
    root_original_handlers = root_logger.handlers[:]
    original_level = root_logger.level
    # Save the current train logger settings
    train_logger = logging.getLogger("ray.anyscale.train")
    train_original_handlers = train_logger.handlers[:]
    train_original_level = train_logger.level
    # Save the original print function
    original_print = builtins.print
    yield
    # Reset the root logger back to its original state
    root_logger.handlers = root_original_handlers
    root_logger.setLevel(original_level)
    # Reset the train logger back to its original state
    train_logger.handlers = train_original_handlers
    train_logger.setLevel(train_original_level)
    # Reset the print function back to its original state
    builtins.print = original_print


@pytest.fixture(name="controller_logging")
def controller_logging_fixture():
    # Save the current root logger settings
    train_logger = logging.getLogger("ray.anyscale.train")
    original_handlers = train_logger.handlers[:]
    original_level = train_logger.level
    yield
    # Reset the root logger back to its original state
    train_logger.handlers = original_handlers
    train_logger.setLevel(original_level)


@pytest.fixture(autouse=True)
def ray_start():
    ray.init()
    yield
    ray.shutdown()


def test_worker_sys_structured_log_to_file(worker_logging):
    LoggingManager.configure_worker_logger(create_dummy_train_context())
    worker_id = get_runtime_context().get_worker_id()
    train_logger = logging.getLogger("ray.anyscale.train")
    train_logger.info("ham")

    log_contents = get_file_contents(f"ray-train-sys-worker-{worker_id}.log")
    assert "ham" in log_contents


def test_worker_app_structured_log_to_file(worker_logging):
    LoggingManager.configure_worker_logger(create_dummy_train_context())
    worker_id = get_runtime_context().get_worker_id()
    train_logger = logging.getLogger("ray.anyscale.train")
    train_logger.info("ham")

    log_contents = get_file_contents(f"ray-train-app-worker-{worker_id}.log")
    assert "ham" in log_contents


def test_controller_sys_logged_to_file(controller_logging):
    """
    Test that system messages are logged to the correct file on Controller process.
    """
    LoggingManager.configure_controller_logger(create_dummy_run_context())
    worker_id = get_runtime_context().get_worker_id()
    train_logger = logging.getLogger("ray.anyscale.train")
    train_logger.info("ham")

    log_contents = get_file_contents(f"ray-train-sys-controller-{worker_id}.log")
    assert "ham" in log_contents


def test_worker_sys_not_logged_to_file(worker_logging):
    """
    Test that system messages are not logged on Worker process when logging not configured.
    """
    worker_id = get_runtime_context().get_worker_id()
    train_logger = logging.getLogger("ray.anyscale.train")
    train_logger.info("ham")

    with pytest.raises(FileNotFoundError):
        get_file_contents(f"ray-train-sys-worker-{worker_id}.log")


def test_controller_sys_not_logged_to_file(controller_logging):
    """
    Test that system messages are not logged on Controller process when logging not configured.
    """
    worker_id = get_runtime_context().get_worker_id()
    train_logger = logging.getLogger("ray.anyscale.train")
    train_logger.info("ham")

    with pytest.raises(FileNotFoundError):
        get_file_contents(f"ray-train-sys-controller-{worker_id}.log")


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", "-x", __file__]))
