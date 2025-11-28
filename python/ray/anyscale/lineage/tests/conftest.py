"""Shared fixtures for lineage tests."""

import pytest

from ray.anyscale.lineage.common import utils
from ray.anyscale.lineage.tests.test_constants import (
    TEST_CLOUD_ID,
    TEST_JOB_ID,
    TEST_ORG_ID,
    TEST_PROJECT_ID,
    TEST_SERVICE_ID,
    TEST_USER_EMAIL,
    TEST_WORKLOAD_NAME,
    TEST_WORKLOAD_TYPE,
    TEST_WORKSPACE_ID,
)


def _reset_utils_globals():
    """Reset global utility variables."""
    utils._anyscale_workload_id = None
    utils._anyscale_workload_ol_job_name = None
    utils._anyscale_workload_ol_job_namespace = None
    utils._anyscale_workload_ol_run_id = None
    utils._anyscale_workload_os_version = None
    utils._anyscale_workload_python_version = None
    utils._anyscale_workload_ray_version = None
    utils._anyscale_workload_uri = None


@pytest.fixture(autouse=True, scope="function")
def reset_utils_globals():
    """Reset global utility variables before and after each test."""
    _reset_utils_globals()
    yield
    _reset_utils_globals()


@pytest.fixture
def clean_environment(monkeypatch):
    """Clean Anyscale environment variables."""
    env_vars_to_clean = [v.value for v in utils.AnyscaleEnvironmentVariables]
    env_vars_to_clean.append("ANYSCALE_WORKLOAD_VERSION_ID")

    for env_var in env_vars_to_clean:
        monkeypatch.delenv(env_var, raising=False)

    return monkeypatch


@pytest.fixture
def sample_anyscale_env(monkeypatch):
    """Set up sample Anyscale environment variables."""
    env_vars = {
        "ANYSCALE_ORGANIZATION_ID": TEST_ORG_ID,
        "ANYSCALE_CLOUD_ID": TEST_CLOUD_ID,
        "ANYSCALE_PROJECT_ID": TEST_PROJECT_ID,
        "ANYSCALE_USER_EMAIL": TEST_USER_EMAIL,
        "ANYSCALE_WORKLOAD_NAME": TEST_WORKLOAD_NAME,
        "ANYSCALE_WORKLOAD_TYPE": TEST_WORKLOAD_TYPE,
        "ANYSCALE_JOB_ID": TEST_JOB_ID,
    }

    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)

    return env_vars


@pytest.fixture
def service_workload_env(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Set up service workload environment."""
    env_vars = {
        "ANYSCALE_ORGANIZATION_ID": TEST_ORG_ID,
        "ANYSCALE_CLOUD_ID": TEST_CLOUD_ID,
        "ANYSCALE_PROJECT_ID": TEST_PROJECT_ID,
        "ANYSCALE_USER_EMAIL": TEST_USER_EMAIL,
        "ANYSCALE_WORKLOAD_NAME": TEST_WORKLOAD_NAME,
        "ANYSCALE_WORKLOAD_TYPE": "service",
        "ANYSCALE_SERVICE_ID": TEST_SERVICE_ID,
    }

    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)

    return env_vars


@pytest.fixture
def workspace_workload_env(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Set up workspace workload environment."""
    env_vars = {
        "ANYSCALE_ORGANIZATION_ID": TEST_ORG_ID,
        "ANYSCALE_CLOUD_ID": TEST_CLOUD_ID,
        "ANYSCALE_PROJECT_ID": TEST_PROJECT_ID,
        "ANYSCALE_USER_EMAIL": TEST_USER_EMAIL,
        "ANYSCALE_WORKLOAD_NAME": TEST_WORKLOAD_NAME,
        "ANYSCALE_WORKLOAD_TYPE": "workspace",
        "ANYSCALE_WORKSPACE_ID": TEST_WORKSPACE_ID,
    }

    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)

    return env_vars
