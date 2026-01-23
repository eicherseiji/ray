"""
This file tests previously known circular imports between Ray Train and Rayturbo Train to prevent regressions.
"""

import subprocess
import sys

import pytest


@pytest.fixture
def enable_train_v2(monkeypatch):
    monkeypatch.setenv("RAY_TRAIN_V2_ENABLED", "1")
    yield


def run_isolated(code: str):
    """
    Helper function to run code in separate subprocesses to isolate module imports.
    """
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"Subprocess failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


def test_config_import(enable_train_v2):
    run_isolated("from ray.anyscale.train.api.config import ScalingConfig")


def test_data_config_import(enable_train_v2):
    run_isolated(
        "from ray.anyscale.train.api.data_config import DataConfig, DatasetCheckpointConfig"
    )


def test_train_fn_utils_import(enable_train_v2):
    run_isolated("from ray.anyscale.train.api.train_fn_utils import get_dataset_shard")


def test_scaling_policy_import():
    run_isolated(
        "from ray.anyscale.train._internal.execution.scaling_policy.factory import create_scaling_policy"
    )


def test_failure_policy_import():
    run_isolated(
        "from ray.anyscale.train._internal.execution.failure_handling.factory import create_failure_policy"
    )


def test_logging_manager_import():
    run_isolated(
        "from ray.anyscale.train._internal.logging import AnyscaleLoggingManager as LoggingManager"
    )


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", "-x", __file__]))
