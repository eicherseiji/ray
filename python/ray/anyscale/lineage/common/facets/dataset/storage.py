from enum import Enum

from openlineage.client.facet_v2 import storage_dataset

STORAGE_DATASET_FACET_KEY: str = "storage"


class StorageDatasetStorageLayer(Enum):
    """Storage layer."""

    AZURE = "azure"
    DELTA = "delta"
    GCS = "gcs"
    HDFS = "hdfs"
    S3 = "s3"
    ICEBERG = "iceberg"


class StorageDatasetFileFormat(Enum):
    """File format."""

    AVRO = "avro"
    CSV = "csv"
    JSON = "json"
    ORC = "orc"
    PARQUET = "parquet"
    TEXT = "text"
    XML = "xml"
    YAML = "yaml"


def create_storage_dataset_facet(
    storage_layer: StorageDatasetStorageLayer,
    file_format: StorageDatasetFileFormat,
) -> storage_dataset.StorageDatasetFacet:
    return storage_dataset.StorageDatasetFacet(
        storageLayer=storage_layer.value, fileFormat=file_format.value
    )
