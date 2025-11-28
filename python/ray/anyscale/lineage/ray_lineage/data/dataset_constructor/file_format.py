from typing import Any, Dict

from openlineage.client.event_v2 import InputDataset, OutputDataset

from ray.anyscale.lineage.common.dataset_naming import (
    resolve_dataset_naming_type_and_attributes,
    resolve_ol_dataset_namespace_and_name,
)
from ray.anyscale.lineage.common.facets.dataset import DatasetType, FileFormats
from ray.anyscale.lineage.common.utils import (
    create_openlineage_input_dataset_from_args,
    create_openlineage_output_dataset_from_args,
    transform_anyscale_mnt_path,
)
from ray.anyscale.lineage.ray_lineage.data.facet_constructor import (
    RayDataFacetConstructor,
)


def get_file_format_source_common_facets(
    path: str,
    file_format: FileFormats,
) -> Dict[str, Any]:
    """Get common facets for a file format datasource or datasink."""
    facets: Dict[str, Any] = {}
    facets.update(
        RayDataFacetConstructor.construct_dataset_type_dataset_facet(
            dataset_type=DatasetType.FILE
        )
    )
    facets.update(RayDataFacetConstructor.construct_datasource_dataset_facet(uri=path))
    facets.update(
        RayDataFacetConstructor.construct_file_format_dataset_facet(format=file_format)
    )
    return facets


def process_file_format_datasource(path: str, file_format: FileFormats) -> InputDataset:
    """Process a file format datasource."""
    # Transform Anyscale-specific /mnt paths
    transformed_path = transform_anyscale_mnt_path(path)

    facets: Dict[str, Any] = get_file_format_source_common_facets(
        transformed_path, file_format
    )
    (
        dataset_naming_type,
        dataset_attributes,
    ) = resolve_dataset_naming_type_and_attributes(transformed_path)
    dataset_namespace, dataset_name = resolve_ol_dataset_namespace_and_name(
        dataset_naming_type, **dataset_attributes
    )

    return create_openlineage_input_dataset_from_args(
        dataset_namespace=dataset_namespace,
        dataset_name=dataset_name,
        facets=facets,
    )


def process_file_format_datasink(path: str, file_format: FileFormats) -> OutputDataset:
    """Process a file format datasink."""
    # Transform Anyscale-specific /mnt paths
    transformed_path = transform_anyscale_mnt_path(path)

    facets: Dict[str, Any] = get_file_format_source_common_facets(
        transformed_path, file_format
    )
    (
        dataset_naming_type,
        dataset_attributes,
    ) = resolve_dataset_naming_type_and_attributes(transformed_path)
    dataset_namespace, dataset_name = resolve_ol_dataset_namespace_and_name(
        dataset_naming_type, **dataset_attributes
    )
    return create_openlineage_output_dataset_from_args(
        dataset_namespace=dataset_namespace,
        dataset_name=dataset_name,
        facets=facets,
    )
