from typing import Any, Dict, List

from openlineage.client.event_v2 import InputDataset

from ray.anyscale.lineage.common.dataset_naming import (
    resolve_dataset_naming_type_and_attributes,
    resolve_ol_dataset_namespace_and_name,
)
from ray.anyscale.lineage.common.facets.dataset import DatasetType, FileFormats
from ray.anyscale.lineage.common.logging import get_logger
from ray.anyscale.lineage.common.utils import (
    create_openlineage_input_dataset_from_args,
    transform_anyscale_mnt_path,
)
from ray.anyscale.lineage.ray_lineage.data.facet_constructor import (
    RayDataFacetConstructor,
)
from ray.anyscale.lineage.ray_lineage.data.utils import (
    FILE_EXTENSIONS_REGISTRY,
    FILE_FORMATS_REGISTRY,
)

logger = get_logger(__name__)


def get_list_files_common_facets(
    path: str,
    file_extensions: List[str],
) -> Dict[str, Any]:
    """Get common facets for a file format datasource or datasink."""
    facets: Dict[str, Any] = {}
    facets.update(
        RayDataFacetConstructor.construct_dataset_type_dataset_facet(
            dataset_type=DatasetType.FILE
        )
    )
    facets.update(RayDataFacetConstructor.construct_datasource_dataset_facet(uri=path))

    file_format = None

    if file_extensions:
        # Infer file format from the first file extension
        first_file_extension = file_extensions[0]
        file_datasource_type = None

        for ds_name, ds_file_extensions in FILE_EXTENSIONS_REGISTRY.items():
            if first_file_extension in ds_file_extensions:
                file_datasource_type = ds_name
                break

        if file_datasource_type:
            file_format = FILE_FORMATS_REGISTRY.get(file_datasource_type)

    facets.update(
        RayDataFacetConstructor.construct_file_format_dataset_facet(
            format=file_format or FileFormats.UNKNOWN
        )
    )

    return facets


def process_list_files_operator_path(
    path: str, file_extensions: List[str]
) -> InputDataset:
    """Process a ListFiles operator path."""
    # Transform Anyscale-specific /mnt paths
    transformed_path = transform_anyscale_mnt_path(path)

    facets: Dict[str, Any] = get_list_files_common_facets(
        transformed_path, file_extensions
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
