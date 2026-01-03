from enum import Enum

from openlineage.client.facet_v2 import dataset_type_dataset

DATASET_TYPE_DATASET_FACET_KEY: str = "datasetType"


class DatasetType(Enum):
    """Dataset type."""

    FILE = "file"
    MODEL = "model"
    STREAM = "stream"
    TABLE = "table"
    VIEW = "view"


def create_dataset_type_dataset_facet(
    dataset_type: DatasetType,
) -> dataset_type_dataset.DatasetTypeDatasetFacet:
    return dataset_type_dataset.DatasetTypeDatasetFacet(datasetType=dataset_type.value)
