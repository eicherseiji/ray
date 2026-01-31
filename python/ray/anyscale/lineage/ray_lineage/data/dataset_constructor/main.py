from typing import Any, Dict, List, Tuple

from openlineage.client.event_v2 import InputDataset, OutputDataset

from ray.anyscale.lineage.common.facets.dataset import FileFormats
from ray.anyscale.lineage.common.logging import get_logger
from ray.anyscale.lineage.common.utils import evaluate_and_transform_uri
from ray.anyscale.lineage.ray_lineage.data.dataset_constructor.file_format import (
    process_file_format_datasink,
    process_file_format_datasource,
)
from ray.anyscale.lineage.ray_lineage.data.dataset_constructor.list_files import (
    process_list_files_operator_path,
)
from ray.anyscale.lineage.ray_lineage.data.dataset_constructor.mongo import (
    process_mongo_datasink,
    process_mongo_datasource,
)
from ray.anyscale.lineage.ray_lineage.data.dataset_constructor.unity_catalog import (
    process_databricks_uc_datasource,
    process_unity_catalog_datasource,
)
from ray.anyscale.lineage.ray_lineage.data.facet_constructor import (
    RayDataFacetConstructor,
)
from ray.anyscale.lineage.ray_lineage.data.utils import (
    FILE_FORMATS_REGISTRY,
    Datasinks,
    Datasources,
    get_file_format_datasinks,
    get_file_format_datasources,
)
from ray.data._internal.execution.streaming_executor import StreamingExecutor
from ray.data._internal.logical.interfaces.logical_operator import LogicalOperator
from ray.data._internal.logical.operators.read_operator import Read as ReadOperator
from ray.data._internal.logical.operators.write_operator import Write as WriteOperator
from ray.data.datasource.datasink import Datasink
from ray.data.datasource.datasource import Datasource

logger = get_logger(__name__)


LIST_FILES_LOGICAL_OPERATOR_NAME = "ListFiles"


def process_datasource(
    datasource: Datasource, seen_input_uris: set[str]
) -> Tuple[List[InputDataset], set[str]]:
    """Process a datasource."""
    input_datasets = []

    if (ds_class := datasource.__class__) in get_file_format_datasources():
        paths = getattr(datasource, "_source_paths", [])
        file_format = FILE_FORMATS_REGISTRY.get(ds_class.__name__, FileFormats.UNKNOWN)
        for path in paths:
            if path not in seen_input_uris:
                seen_input_uris.add(path)
                should_track, transformed_path = evaluate_and_transform_uri(path)
                if should_track:
                    input_datasets.append(
                        process_file_format_datasource(transformed_path, file_format)
                    )

    if isinstance(datasource, Datasources.MONGO_DATASOURCE.value):
        ds_uri = f"{datasource._uri}/{datasource._database}/{datasource._collection}"
        if ds_uri not in seen_input_uris:
            seen_input_uris.add(ds_uri)
            input_datasets.append(
                process_mongo_datasource(
                    uri=datasource._uri,
                    database=datasource._database,
                    collection=datasource._collection,
                )
            )

    if isinstance(datasource, Datasources.DATABRICKS_UC_DATASOURCE.value):
        ds_uri = f"{datasource.host}/{datasource.warehouse_id}/{datasource.catalog}/{datasource.schema}"
        if ds_uri not in seen_input_uris:
            seen_input_uris.add(ds_uri)
            input_datasets.append(
                process_databricks_uc_datasource(
                    host=datasource.host,
                    warehouse_id=datasource.warehouse_id,
                    catalog=datasource.catalog,
                    schema=datasource.schema,
                )
            )

    if isinstance(datasource, Datasources.UNITY_CATALOG_DATASOURCE.value):
        ds_uri = f"{datasource.base_url}/{datasource.table_full_name}"
        if ds_uri not in seen_input_uris:
            seen_input_uris.add(ds_uri)
            input_datasets.append(
                process_unity_catalog_datasource(
                    base_url=datasource.base_url,
                    table_name=datasource.table_full_name,
                )
            )

    return input_datasets, seen_input_uris


def process_datasink(
    datasink: Datasink,  # type: ignore[type-arg]
    seen_output_uris: set[str],
) -> Tuple[List[OutputDataset], set[str]]:
    """Process a datasink."""
    output_datasets = []

    if (ds_class := datasink.__class__) in get_file_format_datasinks():
        path = datasink.unresolved_path  # type: ignore[attr-defined]
        file_format = FILE_FORMATS_REGISTRY.get(ds_class.__name__, FileFormats.UNKNOWN)
        if path not in seen_output_uris:
            seen_output_uris.add(path)
            should_track, transformed_path = evaluate_and_transform_uri(path)
            if should_track:
                output_datasets.append(
                    process_file_format_datasink(transformed_path, file_format)
                )

    if isinstance(datasink, Datasinks.MONGO_DATASINK.value):
        ds_uri = f"{datasink.uri}/{datasink.database}/{datasink.collection}"
        if ds_uri not in seen_output_uris:
            seen_output_uris.add(ds_uri)
            output_datasets.append(
                process_mongo_datasink(
                    uri=datasink.uri,
                    database=datasink.database,
                    collection=datasink.collection,
                )
            )

    return output_datasets, seen_output_uris


def process_list_files_operator(
    operator: LogicalOperator,
    seen_input_uris: set[str],
) -> Tuple[List[InputDataset], set[str]]:
    """Process a list files operator."""
    paths = getattr(operator, "_source_paths", [])
    if isinstance(paths, str):
        paths = [paths]
    file_extensions = getattr(operator, "file_extensions", [])

    input_datasets = []

    for path in paths:
        if path not in seen_input_uris:
            seen_input_uris.add(path)
            should_track, transformed_path = evaluate_and_transform_uri(path)
            if should_track:
                input_datasets.append(
                    process_list_files_operator_path(transformed_path, file_extensions)
                )
    return input_datasets, seen_input_uris


def construct_input_output_datasets(
    executor: StreamingExecutor,
) -> tuple[list[InputDataset], list[OutputDataset]]:
    input_datasets: List[InputDataset] = []
    output_datasets: List[OutputDataset] = []

    seen_input_uris: set[str] = set()  # avoid duplicate input uris
    seen_output_uris: set[str] = set()  # avoid duplicate output uris

    topology = executor._topology
    if topology is None:
        return input_datasets, output_datasets

    physical_ops = list(topology.keys())

    for physical_op in physical_ops:
        logical_ops = getattr(physical_op, "_logical_operators", [])

        for logical_op in logical_ops:
            # construct common dataset facets
            common_dataset_facets: Dict[str, Any] = {}

            # dataset schema facet
            dataset_fields = []
            if (
                hasattr(logical_op, "infer_schema")
                and (schema := logical_op.infer_schema()) is not None
            ):
                for name, stype in zip(schema.names, schema.types):
                    dataset_fields.append({"name": name, "type": stype})
            common_dataset_facets.update(
                RayDataFacetConstructor.construct_schema_dataset_facet(
                    fields=dataset_fields
                )
            )

            # dataset ownership facet
            common_dataset_facets.update(
                RayDataFacetConstructor.construct_ownership_dataset_facet()
            )

            if getattr(logical_op, "_name", None) == LIST_FILES_LOGICAL_OPERATOR_NAME:
                (
                    processed_input_datasets,
                    updated_seen_input_uris,
                ) = process_list_files_operator(logical_op, seen_input_uris)
                for input_dataset in processed_input_datasets:
                    if input_dataset.facets is not None:
                        input_dataset.facets.update(common_dataset_facets)
                    input_datasets.append(input_dataset)
                seen_input_uris = updated_seen_input_uris

            elif isinstance(logical_op, ReadOperator):
                try:
                    datasource = (
                        logical_op._datasource
                        or logical_op._datasource_or_legacy_reader
                    )
                    (
                        processed_input_datasets,
                        updated_seen_input_uris,
                    ) = process_datasource(
                        datasource, seen_input_uris
                    )  # type: ignore[arg-type]
                    for input_dataset in processed_input_datasets:
                        if input_dataset.facets is not None:
                            input_dataset.facets.update(common_dataset_facets)
                        input_datasets.append(input_dataset)
                    seen_input_uris = updated_seen_input_uris
                except AttributeError as e:
                    logger.error(f"Error processing datasource: {e}")

            elif isinstance(logical_op, WriteOperator):
                try:
                    datasink = logical_op._datasink_or_legacy_datasource
                    (
                        processed_output_datasets,
                        updated_seen_output_uris,
                    ) = process_datasink(
                        datasink, seen_output_uris
                    )  # type: ignore[arg-type]
                    for output_dataset in processed_output_datasets:
                        if output_dataset.facets is not None:
                            output_dataset.facets.update(common_dataset_facets)
                        output_datasets.append(output_dataset)
                    seen_output_uris = updated_seen_output_uris
                except AttributeError as e:
                    logger.error(f"Error processing datasink: {e}")

    return input_datasets, output_datasets
