from unittest import mock

import mlflow

from ray.anyscale.lineage.mlflow_lineage import facet_constructor


def test_construct_mlflow_workload_details_run_facet_uses_defaults() -> None:
    constructor = facet_constructor.MLflowFacetConstructor()

    with mock.patch.object(mlflow, "__version__", "2.9.0"):
        facet_dict = constructor.construct_mlflow_workload_details_run_facet(
            host="mlflow.local", experiment_id="1", run_id="abc"
        )

    facet = facet_dict["mlflowWorkloadDetails"]
    assert facet.host == "mlflow.local"
    assert facet.experiment_id == "1"
    assert facet.run_id == "abc"
    assert facet.mlflow_version == "2.9.0"


def test_construct_mlflow_workload_details_run_facet_custom_version() -> None:
    constructor = facet_constructor.MLflowFacetConstructor()

    facet_dict = constructor.construct_mlflow_workload_details_run_facet(
        host="mlflow.local",
        experiment_id="2",
        run_id="def",
        mlflow_version="3.0.0",
    )

    facet = facet_dict["mlflowWorkloadDetails"]
    assert facet.mlflow_version == "3.0.0"


def test_construct_input_schema_dataset_facet_multiple_fields() -> None:
    """Test input schema facet with multiple fields."""
    constructor = facet_constructor.MLflowFacetConstructor()

    facet_dict = constructor.construct_input_schema_dataset_facet(
        fields=[
            {"name": "col1", "type": "string"},
            {"name": "col2", "type": "int"},
            {"name": "col3", "type": "float"},
        ]
    )

    facet = facet_dict["inputSchema"]
    assert len(facet.fields) == 3
    assert facet.fields[0].name == "col1"
    assert facet.fields[0].type == "string"
    assert facet.fields[1].name == "col2"
    assert facet.fields[1].type == "int"
    assert facet.fields[2].name == "col3"
    assert facet.fields[2].type == "float"


def test_construct_output_schema_dataset_facet_multiple_fields() -> None:
    """Test output schema facet with multiple fields."""
    constructor = facet_constructor.MLflowFacetConstructor()

    facet_dict = constructor.construct_output_schema_dataset_facet(
        fields=[
            {"name": "result", "type": "string"},
            {"name": "confidence", "type": "float"},
        ]
    )

    facet = facet_dict["outputSchema"]
    assert len(facet.fields) == 2
    assert facet.fields[0].name == "result"
    assert facet.fields[0].type == "string"
    assert facet.fields[1].name == "confidence"
    assert facet.fields[1].type == "float"


def test_mlflow_facet_constructor_inheritance() -> None:
    """Test that MLflowFacetConstructor inherits from the expected base classes."""
    from ray.anyscale.lineage.common.facet_constructor import (
        DatasetFacetConstructor,
        JobFacetConstructor,
        RunFacetConstructor,
    )

    constructor = facet_constructor.MLflowFacetConstructor()
    assert isinstance(constructor, DatasetFacetConstructor)
    assert isinstance(constructor, JobFacetConstructor)
    assert isinstance(constructor, RunFacetConstructor)


def test_mlflow_facet_error_handler_logs_errors() -> None:
    """Test that the error handler function logs errors."""
    error = ValueError("test error")
    func_name = "test_function"
    func_args = ("arg1", "arg2")
    func_kwargs = {"kwarg1": "value1"}

    from ray.anyscale.lineage.common.exceptions import (
        AnyscaleLineageMLflowError,
    )

    # Test the error handler directly
    try:
        facet_constructor.mlflow_facet_error_handler(
            error, func_name, func_args, func_kwargs
        )
        raise AssertionError("Should have raised AnyscaleLineageMLflowError")
    except AnyscaleLineageMLflowError as e:
        assert isinstance(e.__cause__, ValueError)
        assert str(e.__cause__) == "test error"
