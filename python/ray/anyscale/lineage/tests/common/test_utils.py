import uuid

import pytest

from ray.anyscale.lineage.common import utils
from ray.anyscale.lineage.common.exceptions import AnyscaleLineageClientError
from ray.anyscale.lineage.tests.test_constants import (
    SIMPLE_CLOUD,
    SIMPLE_IDENTIFIER,
    SIMPLE_JOB,
    SIMPLE_ORG,
    SIMPLE_PROJECT,
    TEST_CLOUD_ID,
    TEST_JOB_ID,
    TEST_OL_DATASET_NAME,
    TEST_OL_JOB_NAME,
    TEST_OL_NAMESPACE,
    TEST_SERVICE_ID,
    TEST_WORKSPACE_ID,
)

pytestmark = [
    pytest.mark.timeout(60),
]


def test_get_anyscale_workload_ol_job_namespace_success(monkeypatch, clean_environment):
    monkeypatch.setenv("ANYSCALE_ORGANIZATION_ID", SIMPLE_ORG)
    monkeypatch.setenv("ANYSCALE_CLOUD_ID", SIMPLE_CLOUD)
    monkeypatch.setenv("ANYSCALE_PROJECT_ID", SIMPLE_PROJECT)

    namespace = utils.get_anyscale_workload_ol_job_namespace()

    assert namespace == f"{SIMPLE_ORG}.{SIMPLE_CLOUD}.{SIMPLE_PROJECT}"


def test_get_anyscale_workload_ol_job_namespace_missing_env(clean_environment):
    with pytest.raises(ValueError):
        utils.get_anyscale_workload_ol_job_namespace()


@pytest.mark.parametrize(
    "workload_type,env_key,env_value",
    [
        ("JOB", "ANYSCALE_JOB_ID", SIMPLE_JOB),
        ("SERVICE", "ANYSCALE_SERVICE_ID", SIMPLE_IDENTIFIER),
        ("WORKSPACE", "ANYSCALE_WORKSPACE_ID", SIMPLE_IDENTIFIER),
    ],
)
def test_get_anyscale_workload_ol_job_name(
    monkeypatch, clean_environment, workload_type, env_key, env_value
):
    monkeypatch.setenv("ANYSCALE_WORKLOAD_TYPE", workload_type)
    monkeypatch.setenv(env_key, env_value)

    name = utils.get_anyscale_workload_ol_job_name()

    assert name == f"{workload_type.lower()}.{env_value}"


def test_get_anyscale_workload_ol_job_name_missing(clean_environment):
    with pytest.raises(ValueError):
        utils.get_anyscale_workload_ol_job_name()


def test_get_anyscale_workload_ol_run_id(sample_anyscale_env):
    run_id = utils.get_anyscale_workload_ol_run_id()

    assert isinstance(run_id, str)
    uuid_obj = uuid.UUID(run_id)
    assert uuid_obj.version == 5

    run_id_2 = utils.get_anyscale_workload_ol_run_id()
    assert run_id == run_id_2


def test_get_anyscale_workload_ol_run_id_missing(clean_environment):
    utils._anyscale_workload_ol_run_id = None

    with pytest.raises(ValueError):
        utils.get_anyscale_workload_ol_run_id()


def test_catch_ol_client_exception_wraps_errors():
    @utils.catch_ol_client_exception
    def faulty():
        raise RuntimeError("boom")

    with pytest.raises(AnyscaleLineageClientError) as exc_info:
        faulty()

    assert "boom" in str(exc_info.value)


def test_get_anyscale_workload_id_job(monkeypatch, clean_environment):
    monkeypatch.setenv("ANYSCALE_WORKLOAD_TYPE", "JOB")
    monkeypatch.setenv("ANYSCALE_JOB_ID", SIMPLE_JOB)

    assert utils.get_anyscale_workload_id() == SIMPLE_JOB


@pytest.mark.parametrize(
    "workload_type,env_key",
    [
        ("SERVICE", "ANYSCALE_SERVICE_ID"),
        ("WORKSPACE", "ANYSCALE_WORKSPACE_ID"),
    ],
)
def test_get_anyscale_workload_id_other_types(
    monkeypatch, clean_environment, workload_type, env_key
):
    monkeypatch.setenv("ANYSCALE_WORKLOAD_TYPE", workload_type)
    monkeypatch.setenv(env_key, SIMPLE_IDENTIFIER)

    assert utils.get_anyscale_workload_id() == SIMPLE_IDENTIFIER


def test_get_anyscale_workload_id_missing(monkeypatch, clean_environment):
    monkeypatch.setenv("ANYSCALE_WORKLOAD_TYPE", "")

    with pytest.raises(ValueError):
        utils.get_anyscale_workload_id()


@pytest.mark.parametrize("with_facets", [False, True])
def test_create_openlineage_job_from_args(with_facets):
    from unittest import mock

    facets = {"test": mock.Mock()} if with_facets else None
    job = utils.create_openlineage_job_from_args(
        TEST_OL_NAMESPACE, TEST_OL_JOB_NAME, facets=facets
    )

    assert job.namespace == TEST_OL_NAMESPACE
    assert job.name == TEST_OL_JOB_NAME
    assert job.facets == facets


def test_create_openlineage_run_from_args():
    from unittest import mock

    run_id = str(uuid.uuid4())
    facets = {"test": mock.Mock()}

    run = utils.create_openlineage_run_from_args(run_id, facets=facets)

    assert run.runId == run_id
    assert run.facets == facets


@pytest.mark.parametrize(
    "function_name",
    [
        "create_openlineage_dataset_from_args",
        "create_openlineage_input_dataset_from_args",
        "create_openlineage_output_dataset_from_args",
    ],
)
def test_create_openlineage_dataset_variants(function_name):
    create_func = getattr(utils, function_name)
    result = create_func(TEST_OL_NAMESPACE, TEST_OL_DATASET_NAME)

    assert result.namespace == TEST_OL_NAMESPACE
    assert result.name == TEST_OL_DATASET_NAME


@pytest.mark.parametrize(
    "env_var,env_value,default,expected",
    [
        ("TEST_ENV_VAR", "test_value", None, "test_value"),
        ("NONEXISTENT_ENV_VAR", None, "default_value", "default_value"),
        ("NONEXISTENT_ENV_VAR", None, None, ""),
    ],
)
def test_get_os_env(monkeypatch, env_var, env_value, default, expected):
    if env_value:
        monkeypatch.setenv(env_var, env_value)

    if default is not None:
        result = utils.get_os_env(env_var, default=default)
    else:
        result = utils.get_os_env(env_var)

    assert result == expected


def test_catch_class_method_exception_catches_exceptions():
    error_messages = []

    def handler(e, func_name, func_args, func_kwargs):
        error_messages.append(f"{func_name}: {e}")
        return "handled"

    @utils.catch_class_method_exception(handler)
    def faulty_function(x):
        raise ValueError("test error")

    result = faulty_function(10)

    assert result == "handled"
    assert len(error_messages) == 1
    assert "faulty_function: test error" in error_messages[0]


def test_catch_class_method_exception_only_catches_specified_exceptions():
    def handler(e, func_name, func_args, func_kwargs):
        return "handled"

    @utils.catch_class_method_exception(handler, exceptions=ValueError)
    def faulty_function():
        raise TypeError("not caught")

    with pytest.raises(TypeError):
        faulty_function()


def test_catch_class_method_exception_preserves_metadata():
    def handler(e, func_name, func_args, func_kwargs):
        return "handled"

    @utils.catch_class_method_exception(handler)
    def original_function(x, y):
        """This is the original function docstring."""
        return x + y

    assert original_function.__name__ == "original_function"
    assert original_function.__doc__ == "This is the original function docstring."


def test_wrap_class_methods_wraps_instance_methods():
    call_log = []

    def logger(func):
        def wrapper(*args, **kwargs):
            call_log.append(func.__name__)
            return func(*args, **kwargs)

        return wrapper

    @utils.wrap_class_methods(decorator=logger)
    class TestClass:
        def method1(self):
            return "method1"

        def method2(self):
            return "method2"

    obj = TestClass()
    obj.method1()
    obj.method2()

    assert "method1" in call_log
    assert "method2" in call_log


def test_wrap_class_methods_excludes_specified_methods():
    call_log = []

    def logger(func):
        def wrapper(*args, **kwargs):
            call_log.append(func.__name__)
            return func(*args, **kwargs)

        return wrapper

    @utils.wrap_class_methods(decorator=logger, exclude=("excluded_method",))
    class TestClass:
        def included_method(self):
            return "included"

        def excluded_method(self):
            return "excluded"

    obj = TestClass()
    obj.included_method()
    obj.excluded_method()

    assert "included_method" in call_log
    assert "excluded_method" not in call_log


def test_wrap_class_methods_with_inherited_methods():
    call_log = []

    def logger(func):
        def wrapper(*args, **kwargs):
            call_log.append(func.__name__)
            return func(*args, **kwargs)

        return wrapper

    class ParentClass:
        def parent_method(self):
            return "parent"

    @utils.wrap_class_methods(decorator=logger, include_inherited=True)
    class ChildClass(ParentClass):
        def child_method(self):
            return "child"

    obj = ChildClass()
    obj.parent_method()
    obj.child_method()

    assert "parent_method" in call_log
    assert "child_method" in call_log


@pytest.mark.parametrize(
    "storage_type,expected_id",
    [
        ("user_storage", TEST_JOB_ID),
        ("cluster_storage", TEST_JOB_ID),
        ("shared_storage", TEST_CLOUD_ID),
    ],
)
def test_transform_anyscale_mnt_path_basic(
    sample_anyscale_env, storage_type, expected_id
):
    """Test transformation of /mnt storage paths."""
    path = f"/mnt/{storage_type}/data/file.csv"
    result = utils.transform_anyscale_mnt_path(path)
    assert result == f"{expected_id}:/mnt/{storage_type}/data/file.csv"


@pytest.mark.parametrize("scheme", ["file:", "local:"])
def test_transform_anyscale_mnt_path_with_scheme(sample_anyscale_env, scheme):
    """Test transformation with file: and local: schemes."""
    path = f"{scheme}/mnt/user_storage/data/file.csv"
    result = utils.transform_anyscale_mnt_path(path)
    assert result == f"{scheme}{TEST_JOB_ID}:/mnt/user_storage/data/file.csv"


@pytest.mark.parametrize(
    "path",
    [
        "/tmp/data/file.csv",
        "s3://bucket/path/to/data",
        "",
        "/mnt/other_storage/data",
    ],
)
def test_transform_anyscale_mnt_path_unchanged(sample_anyscale_env, path):
    """Test that non-transformable paths remain unchanged."""
    result = utils.transform_anyscale_mnt_path(path)
    assert result == path


def test_transform_anyscale_mnt_path_none():
    """Test None handling."""
    result = utils.transform_anyscale_mnt_path(None)
    assert result is None


@pytest.mark.parametrize(
    "workload_env_fixture,storage_type,expected_id",
    [
        ("service_workload_env", "user_storage", TEST_SERVICE_ID),
        ("workspace_workload_env", "cluster_storage", TEST_WORKSPACE_ID),
    ],
)
def test_transform_anyscale_mnt_path_workload_types(
    request, workload_env_fixture, storage_type, expected_id
):
    """Test transformation with different workload types."""
    request.getfixturevalue(workload_env_fixture)
    path = f"/mnt/{storage_type}/data/file.csv"
    result = utils.transform_anyscale_mnt_path(path)
    assert result == f"{expected_id}:/mnt/{storage_type}/data/file.csv"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", "-x", __file__]))
