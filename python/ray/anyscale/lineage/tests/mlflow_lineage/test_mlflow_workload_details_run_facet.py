from ray.anyscale.lineage.mlflow_lineage.facets.run.mlflow_workload_details import (
    MLflowWorkloadDetailsRunFacet,
    create_mlflow_workload_details_run_facet,
)


def test_create_mlflow_workload_details_run_facet() -> None:
    facet = create_mlflow_workload_details_run_facet(
        host="mlflow.local",
        experiment_id="123",
        run_id="456",
        mlflow_version="2.7.0",
    )

    assert isinstance(facet, MLflowWorkloadDetailsRunFacet)
    assert facet.host == "mlflow.local"
    assert facet.experiment_id == "123"
    assert facet.run_id == "456"
    assert facet.mlflow_version == "2.7.0"
