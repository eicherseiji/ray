from typing import Any, Optional, Union

import openlineage.client.facet_v2 as ol_facets

from ray.anyscale.lineage.common.facets import dataset as dataset_facets
from ray.anyscale.lineage.common.utils import AnyscaleEnvironmentVariables, get_os_env


class DatasetFacetConstructor:
    @staticmethod
    def construct_dataset_type_dataset_facet(
        dataset_type: dataset_facets.DatasetType,
    ) -> dict[str, ol_facets.dataset_type_dataset.DatasetTypeDatasetFacet]:
        return {
            dataset_facets.DATASET_TYPE_DATASET_FACET_KEY: dataset_facets.create_dataset_type_dataset_facet(
                dataset_type=dataset_type,
            )
        }

    @staticmethod
    def construct_datasource_dataset_facet(
        name: Optional[str] = None,
        uri: Optional[str] = None,
    ) -> dict[str, ol_facets.datasource_dataset.DatasourceDatasetFacet]:
        return {
            dataset_facets.DATA_SOURCE_DATASET_FACET_KEY: dataset_facets.create_datasource_dataset_facet(
                name=name,
                uri=uri,
            )
        }

    @staticmethod
    def construct_file_format_dataset_facet(
        format: Union[dataset_facets.FileFormats, dataset_facets.FreeFormFileFormat],
    ) -> dict[str, dataset_facets.FileFormatDatasetFacet]:
        return {
            dataset_facets.FILE_FORMAT_DATASET_FACET_KEY: dataset_facets.create_file_format_dataset_facet(
                format=format,
            )
        }

    @staticmethod
    def construct_ownership_dataset_facet() -> dict[
        str, ol_facets.ownership_dataset.OwnershipDatasetFacet
    ]:
        owner_emails = []
        anyscale_user_email = get_os_env(
            AnyscaleEnvironmentVariables.ANYSCALE_USER_EMAIL.value
        )

        if anyscale_user_email:
            owner_emails.append(anyscale_user_email)

        return {
            dataset_facets.OWNERSHIP_DATASET_FACET_KEY: dataset_facets.create_ownership_dataset_facet(
                owner_emails=owner_emails,
            )
        }

    @staticmethod
    def construct_schema_dataset_facet(
        fields: list[dict[str, Any]],
    ) -> dict[str, ol_facets.schema_dataset.SchemaDatasetFacet]:
        return {
            dataset_facets.SCHEMA_DATASET_FACET_KEY: dataset_facets.create_schema_dataset_facet(
                fields=fields,
            )
        }

    @staticmethod
    def construct_storage_dataset_facet(
        storage_layer: dataset_facets.StorageDatasetStorageLayer,
        file_format: dataset_facets.StorageDatasetFileFormat,
    ) -> dict[str, ol_facets.storage_dataset.StorageDatasetFacet]:
        return {
            dataset_facets.STORAGE_DATASET_FACET_KEY: dataset_facets.create_storage_dataset_facet(
                storage_layer=storage_layer,
                file_format=file_format,
            )
        }

    @staticmethod
    def construct_tags_dataset_facet(
        tags: dict[str, str],
    ) -> dict[str, ol_facets.tags_dataset.TagsDatasetFacet]:
        return {
            dataset_facets.TAGS_DATASET_FACET_KEY: dataset_facets.create_tags_dataset_facet(
                tags=tags,
            )
        }

    @staticmethod
    def construct_version_dataset_facet(
        version: str,
    ) -> dict[str, ol_facets.dataset_version_dataset.DatasetVersionDatasetFacet]:
        return {
            dataset_facets.VERSION_DATASET_FACET_KEY: dataset_facets.create_version_dataset_facet(
                version=version,
            )
        }
