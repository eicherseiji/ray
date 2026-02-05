from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Type

from ray.anyscale.lineage.common.constants import IGNORE_ERRORS
from ray.anyscale.lineage.common.exceptions import AnyscaleLineageRayDataError
from ray.anyscale.lineage.common.facets.dataset import FileFormats
from ray.anyscale.lineage.common.logging import get_logger

# Import all datasource classes
from ray.data._internal.datasource.audio_datasource import AudioDatasource
from ray.data._internal.datasource.avro_datasource import AvroDatasource

# Import all datasink classes
from ray.data._internal.datasource.bigquery_datasink import BigQueryDatasink
from ray.data._internal.datasource.bigquery_datasource import BigQueryDatasource
from ray.data._internal.datasource.binary_datasource import BinaryDatasource
from ray.data._internal.datasource.clickhouse_datasink import ClickHouseDatasink
from ray.data._internal.datasource.clickhouse_datasource import ClickHouseDatasource
from ray.data._internal.datasource.csv_datasink import CSVDatasink
from ray.data._internal.datasource.csv_datasource import CSVDatasource
from ray.data._internal.datasource.databricks_uc_datasource import (
    DatabricksUCDatasource,
)
from ray.data._internal.datasource.delta_sharing_datasource import (
    DeltaSharingDatasource,
)
from ray.data._internal.datasource.hudi_datasource import HudiDatasource
from ray.data._internal.datasource.huggingface_datasource import HuggingFaceDatasource
from ray.data._internal.datasource.iceberg_datasink import IcebergDatasink
from ray.data._internal.datasource.iceberg_datasource import IcebergDatasource
from ray.data._internal.datasource.image_datasink import ImageDatasink
from ray.data._internal.datasource.image_datasource import ImageDatasource
from ray.data._internal.datasource.json_datasink import JSONDatasink
from ray.data._internal.datasource.json_datasource import (
    ArrowJSONDatasource,
    PandasJSONDatasource,
)
from ray.data._internal.datasource.lance_datasink import LanceDatasink
from ray.data._internal.datasource.lance_datasource import LanceDatasource
from ray.data._internal.datasource.mongo_datasink import MongoDatasink
from ray.data._internal.datasource.mongo_datasource import MongoDatasource
from ray.data._internal.datasource.numpy_datasink import NumpyDatasink
from ray.data._internal.datasource.numpy_datasource import NumpyDatasource
from ray.data._internal.datasource.parquet_datasink import ParquetDatasink
from ray.data._internal.datasource.parquet_datasource import ParquetDatasource
from ray.data._internal.datasource.range_datasource import RangeDatasource
from ray.data._internal.datasource.sql_datasink import SQLDatasink
from ray.data._internal.datasource.sql_datasource import SQLDatasource
from ray.data._internal.datasource.text_datasource import TextDatasource
from ray.data._internal.datasource.tfrecords_datasink import TFRecordDatasink
from ray.data._internal.datasource.tfrecords_datasource import TFRecordDatasource
from ray.data._internal.datasource.torch_datasource import TorchDatasource
from ray.data._internal.datasource.uc_datasource import UnityCatalogConnector
from ray.data._internal.datasource.video_datasource import VideoDatasource
from ray.data._internal.datasource.webdataset_datasink import WebDatasetDatasink
from ray.data._internal.datasource.webdataset_datasource import WebDatasetDatasource
from ray.data.datasource.datasink import Datasink
from ray.data.datasource.datasource import Datasource

logger = get_logger(__name__)


class Datasources(Enum):
    """Enum of all available Ray Data datasource classes."""

    # File Format Datasources
    AVRO_DATASOURCE = AvroDatasource
    AUDIO_DATASOURCE = AudioDatasource
    BINARY_DATASOURCE = BinaryDatasource
    CSV_DATASOURCE = CSVDatasource
    IMAGE_DATASOURCE = ImageDatasource
    NUMPY_DATASOURCE = NumpyDatasource
    PARQUET_DATASOURCE = ParquetDatasource
    TEXT_DATASOURCE = TextDatasource
    TFRECORD_DATASOURCE = TFRecordDatasource
    VIDEO_DATASOURCE = VideoDatasource
    WEB_DATASET_DATASOURCE = WebDatasetDatasource
    ARROW_JSON_DATASOURCE = ArrowJSONDatasource
    PANDAS_JSON_DATASOURCE = PandasJSONDatasource

    # Database Datasources
    BIG_QUERY_DATASOURCE = BigQueryDatasource
    CLICK_HOUSE_DATASOURCE = ClickHouseDatasource
    MONGO_DATASOURCE = MongoDatasource
    SQL_DATASOURCE = SQLDatasource

    # Data Lake and Warehouse Datasources
    DATABRICKS_UC_DATASOURCE = DatabricksUCDatasource
    DELTA_SHARING_DATASOURCE = DeltaSharingDatasource
    HUDI_DATASOURCE = HudiDatasource
    ICEBERG_DATASOURCE = IcebergDatasource
    LANCE_DATASOURCE = LanceDatasource
    UNITY_CATALOG_DATASOURCE = UnityCatalogConnector

    # ML and AI Datasources
    HUGGING_FACE_DATASOURCE = HuggingFaceDatasource
    TORCH_DATASOURCE = TorchDatasource

    # Utility Datasources
    RANGE_DATASOURCE = RangeDatasource


class Datasinks(Enum):
    """Enum of all available Ray Data datasink classes."""

    # File Format Datasinks
    CSV_DATASINK = CSVDatasink
    IMAGE_DATASINK = ImageDatasink
    JSON_DATASINK = JSONDatasink
    NUMPY_DATASINK = NumpyDatasink
    PARQUET_DATASINK = ParquetDatasink
    TFRECORD_DATASINK = TFRecordDatasink
    WEB_DATASET_DATASINK = WebDatasetDatasink

    # Database Datasinks
    BIG_QUERY_DATASINK = BigQueryDatasink
    CLICK_HOUSE_DATASINK = ClickHouseDatasink
    MONGO_DATASINK = MongoDatasink
    SQL_DATASINK = SQLDatasink

    # Data Lake and Warehouse Datasinks
    ICEBERG_DATASINK = IcebergDatasink
    LANCE_DATASINK = LanceDatasink


def get_file_format_datasources() -> List[Type[Datasource]]:
    """Get file format datasources."""
    file_format_datasources = [
        Datasources.AUDIO_DATASOURCE.value,
        Datasources.AVRO_DATASOURCE.value,
        Datasources.ARROW_JSON_DATASOURCE.value,
        Datasources.BINARY_DATASOURCE.value,
        Datasources.CSV_DATASOURCE.value,
        Datasources.IMAGE_DATASOURCE.value,
        Datasources.NUMPY_DATASOURCE.value,
        Datasources.PANDAS_JSON_DATASOURCE.value,
        Datasources.PARQUET_DATASOURCE.value,
        Datasources.TEXT_DATASOURCE.value,
        Datasources.TFRECORD_DATASOURCE.value,
        Datasources.VIDEO_DATASOURCE.value,
        Datasources.WEB_DATASET_DATASOURCE.value,
    ]
    return file_format_datasources


def get_database_datasources() -> List[Type[Datasource]]:
    """Get database datasources."""
    database_datasources = [
        Datasources.BIG_QUERY_DATASOURCE.value,
        Datasources.CLICK_HOUSE_DATASOURCE.value,
        Datasources.MONGO_DATASOURCE.value,
        Datasources.SQL_DATASOURCE.value,
    ]
    return database_datasources


def get_data_lake_datasources() -> List[Type[Datasource]]:
    """Get data lake datasources."""
    data_lake_datasources = [
        Datasources.DATABRICKS_UC_DATASOURCE.value,
        Datasources.DELTA_SHARING_DATASOURCE.value,
        Datasources.HUDI_DATASOURCE.value,
        Datasources.ICEBERG_DATASOURCE.value,
        Datasources.LANCE_DATASOURCE.value,
        Datasources.UNITY_CATALOG_DATASOURCE.value,
    ]
    return data_lake_datasources


def get_file_format_datasinks() -> List[Type[Datasink]]:
    """Get file format datasinks."""
    file_format_datasinks = [
        Datasinks.CSV_DATASINK.value,
        Datasinks.IMAGE_DATASINK.value,
        Datasinks.JSON_DATASINK.value,
        Datasinks.NUMPY_DATASINK.value,
        Datasinks.PARQUET_DATASINK.value,
        Datasinks.TFRECORD_DATASINK.value,
        Datasinks.WEB_DATASET_DATASINK.value,
    ]
    return file_format_datasinks


def get_database_datasinks() -> List[Type[Datasink]]:
    """Get database datasinks."""
    database_datasinks = [
        Datasinks.BIG_QUERY_DATASINK.value,
        Datasinks.CLICK_HOUSE_DATASINK.value,
        Datasinks.MONGO_DATASINK.value,
        Datasinks.SQL_DATASINK.value,
    ]
    return database_datasinks


def get_data_lake_datasinks() -> List[Type[Datasink]]:
    """Get data lake datasinks."""
    data_lake_datasinks = [
        Datasinks.ICEBERG_DATASINK.value,
        Datasinks.LANCE_DATASINK.value,
    ]
    return data_lake_datasinks


# Mapping from datasource classes to file formats
_DATASOURCE_TO_FILE_FORMAT: Dict[Type[Datasource], FileFormats] = {
    AvroDatasource: FileFormats.AVRO,
    AudioDatasource: FileFormats.AUDIO,
    BinaryDatasource: FileFormats.BINARY,
    CSVDatasource: FileFormats.CSV,
    ImageDatasource: FileFormats.IMAGE,
    NumpyDatasource: FileFormats.NUMPY,
    ParquetDatasource: FileFormats.PARQUET,
    TextDatasource: FileFormats.TEXT,
    TFRecordDatasource: FileFormats.TFRECORD,
    VideoDatasource: FileFormats.VIDEO,
    WebDatasetDatasource: FileFormats.WEB_DATASET,
    ArrowJSONDatasource: FileFormats.JSON,
    PandasJSONDatasource: FileFormats.JSON,
}

# Mapping from datasink classes to file formats
_DATASINK_TO_FILE_FORMAT: Dict[Type[Datasink], FileFormats] = {
    CSVDatasink: FileFormats.CSV,
    ImageDatasink: FileFormats.IMAGE,
    JSONDatasink: FileFormats.JSON,
    NumpyDatasink: FileFormats.NUMPY,
    ParquetDatasink: FileFormats.PARQUET,
    TFRecordDatasink: FileFormats.TFRECORD,
    WebDatasetDatasink: FileFormats.WEB_DATASET,
}


def build_file_extensions_registry() -> Dict[str, List[str]]:
    """Build file extension registry from file format datasources."""
    registry = {}
    for datasource in get_file_format_datasources():
        datasource_name = datasource.__name__
        if getattr(datasource, "_FILE_EXTENSIONS", None) is not None:
            registry[datasource_name] = datasource._FILE_EXTENSIONS
        elif getattr(datasource, "_FUTURE_FILE_EXTENSIONS", None) is not None:
            registry[datasource_name] = datasource._FUTURE_FILE_EXTENSIONS
    return registry


def build_file_formats_registry() -> Dict[str, FileFormats]:
    """Build file formats registry from file format datasources and datasinks."""
    registry = {}

    # Process datasources
    for datasource in get_file_format_datasources():
        datasource_name = datasource.__name__
        file_format = _DATASOURCE_TO_FILE_FORMAT.get(datasource)
        if file_format is not None:
            registry[datasource_name] = file_format

    # Process datasinks
    for datasink in get_file_format_datasinks():
        datasink_name = datasink.__name__
        file_format = _DATASINK_TO_FILE_FORMAT.get(datasink)
        if file_format is not None:
            registry[datasink_name] = file_format

    return registry


FILE_EXTENSIONS_REGISTRY = build_file_extensions_registry()
FILE_FORMATS_REGISTRY = build_file_formats_registry()


def catch_lineage_callback_exception(func: Callable[..., Any]) -> Callable[..., Any]:
    """Wrapper to catch and handle lineage execution callback exceptions."""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_msg = (
                f"Error in lineage execution callback method '{func.__name__}' "
                f"for args '{args!s}' and kwargs '{kwargs!s}': {e!r}"
            )
            logger.error(error_msg)

            # If IGNORE_ERRORS=True, suppress error and allow workload to continue
            if not IGNORE_ERRORS:
                raise AnyscaleLineageRayDataError(error_msg) from e

    return wrapper
