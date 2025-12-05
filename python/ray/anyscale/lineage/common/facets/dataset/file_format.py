from enum import Enum, unique
from typing import ClassVar, Union

import attr
from openlineage.client.facet_v2 import DatasetFacet

from ray.anyscale.lineage.common.constants import REPO_URL

FILE_FORMAT_DATASET_FACET_KEY: str = "fileFormat"


@unique
class FileFormats(str, Enum):
    """File formats."""

    AVRO = "Avro"
    AUDIO = "Audio"
    BINARY = "Binary"
    CSV = "CSV"
    IMAGE = "Image"
    JSON = "JSON"
    NUMPY = "Numpy"
    PARQUET = "Parquet"
    TEXT = "Text"
    TFRECORD = "TFRecord"
    VIDEO = "Video"
    WEB_DATASET = "WebDataset"
    UNKNOWN = "Unknown"


class FreeFormFileFormat:
    """Free form file format. Accepts any string value."""

    def __init__(self, format: str):
        self.value = format


@attr.define
class FileFormatDatasetFacet(DatasetFacet):
    """File format dataset facet."""

    format: str
    """file format"""

    _additional_skip_redact: ClassVar[list[str]] = [
        "format",
    ]

    @staticmethod
    def _get_schema() -> str:
        return f"{REPO_URL}/blob/main/lineage/common/facets/dataset/file_format"


def create_file_format_dataset_facet(
    format: Union[FileFormats, FreeFormFileFormat],
) -> FileFormatDatasetFacet:
    return FileFormatDatasetFacet(format=format.value)
