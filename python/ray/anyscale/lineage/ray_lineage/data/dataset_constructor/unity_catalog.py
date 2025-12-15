from typing import Any, Dict

from openlineage.client.event_v2 import InputDataset

from ray.anyscale.lineage.common.facets.dataset import DatasetType
from ray.anyscale.lineage.common.utils import (
    create_openlineage_input_dataset_from_args,
)
from ray.anyscale.lineage.ray_lineage.data.facet_constructor import (
    RayDataFacetConstructor,
)


def process_databricks_uc_datasource(
    host: str,
    warehouse_id: str,
    catalog: str,
    schema: str,
) -> InputDataset:
    """Process a databricks UC datasource."""
    facets: Dict[str, Any] = {}
    facets.update(
        RayDataFacetConstructor.construct_dataset_type_dataset_facet(
            dataset_type=DatasetType.FILE
        )
    )
    facets.update(
        RayDataFacetConstructor.construct_datasource_dataset_facet(
            uri=f"https://{host}/api/2.0/sql/statements/{warehouse_id}/{catalog}/{schema}"
        )
    )
    return create_openlineage_input_dataset_from_args(
        dataset_namespace=host,
        dataset_name=f"{warehouse_id}.{catalog}.{schema}",
        facets=facets,
    )


def process_unity_catalog_datasource(
    base_url: str,
    table_name: str,
) -> InputDataset:
    """Process a unity catalog datasource."""
    facets: Dict[str, Any] = {}
    facets.update(
        RayDataFacetConstructor.construct_dataset_type_dataset_facet(
            dataset_type=DatasetType.FILE
        )
    )
    facets.update(
        RayDataFacetConstructor.construct_datasource_dataset_facet(
            uri=f"{base_url}/api/2.1/unity-catalog/tables/{table_name}"
        )
    )
    return create_openlineage_input_dataset_from_args(
        dataset_namespace=base_url, dataset_name=table_name, facets=facets
    )
