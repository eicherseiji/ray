from typing import Any, Dict

from openlineage.client.event_v2 import InputDataset, OutputDataset

from ray.anyscale.lineage.common.facets.dataset import DatasetType
from ray.anyscale.lineage.common.utils import (
    create_openlineage_input_dataset_from_args,
    create_openlineage_output_dataset_from_args,
    parse_uri,
)
from ray.anyscale.lineage.ray_lineage.data.facet_constructor import (
    RayDataFacetConstructor,
)


def get_mongo_source_common_facets(
    uri: str,
    database: str,
    collection: str,
) -> Dict[str, Any]:
    """Get common facets for a mongo datasource or datasink."""
    facets: Dict[str, Any] = {}
    facets.update(
        RayDataFacetConstructor.construct_dataset_type_dataset_facet(
            dataset_type=DatasetType.FILE
        )
    )
    facets.update(RayDataFacetConstructor.construct_datasource_dataset_facet(uri=uri))
    return facets


def process_mongo_datasource(
    uri: str,
    database: str,
    collection: str,
) -> InputDataset:
    """Process a mongo datasource."""
    facets: Dict[str, Any] = get_mongo_source_common_facets(uri, database, collection)

    url_attributes = parse_uri(uri)
    dataset_namespace = "mongodb://" + url_attributes["netloc"]
    dataset_name = f"{database}.{collection}"

    return create_openlineage_input_dataset_from_args(
        dataset_namespace=dataset_namespace,
        dataset_name=dataset_name,
        facets=facets,
    )


def process_mongo_datasink(
    uri: str,
    database: str,
    collection: str,
) -> OutputDataset:
    """Process a mongo datasink."""
    facets: Dict[str, Any] = get_mongo_source_common_facets(uri, database, collection)

    url_attributes = parse_uri(uri)
    dataset_namespace = "mongodb://" + url_attributes["netloc"]
    dataset_name = f"{database}.{collection}"

    return create_openlineage_output_dataset_from_args(
        dataset_namespace=dataset_namespace,
        dataset_name=dataset_name,
        facets=facets,
    )
