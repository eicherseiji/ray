################################################################
# NOTE: PLEASE READ CAREFULLY
#
# Anyscale patches have to be applied *before* any imports such
# that to make sure that all of the patched classes are being
# appropriately picked up by Ray OSS components.
#
# Otherwise, importing Ray OSS modules first brings exposure to
# overridden OSS components being bound during this early imports
# rendering Anyscale patching ineffective.
################################################################

from ray.anyscale.data.apply_anyscale_patches import (  # noqa: E402, isort:skip
    apply_anyscale_patches,
)

apply_anyscale_patches()

### Imports ####################################################

# Short term workaround for https://github.com/ray-project/ray/issues/32435
# Dataset has a hard dependency on pandas, so it doesn't need to be delayed.
import pandas  # noqa
from packaging.version import parse as parse_version

from ray.data._internal.utils.arrow_utils import get_pyarrow_version

from ray.data._internal.compute import ActorPoolStrategy, TaskPoolStrategy
from ray.data._internal.datasource.tfrecords_datasource import TFXReadOptions
from ray.data._internal.execution.interfaces import (
    ExecutionOptions,
    ExecutionResources,
    NodeIdStr,
)
from ray.data._internal.logging import configure_logging
from ray.data.context import DataContext, DatasetContext
from ray.data.dataset import (
    Dataset,
    Schema,
    SinkMode,
    ClickHouseTableSettings,
    SaveMode,
)
from ray.data.stats import DatasetSummary
from ray.data.datasource import (
    BlockBasedFileDatasink,
    Datasink,
    Datasource,
    FileShuffleConfig,
    ReadTask,
    RowBasedFileDatasink,
)
from ray.data.iterator import DataIterator, DatasetIterator
from ray.data.preprocessor import Preprocessor
from ray.data.read_api import (  # noqa: F401
    KafkaAuthConfig,  # noqa: F401
    from_arrow,
    from_arrow_refs,
    from_blocks,
    from_daft,
    from_dask,
    from_huggingface,
    from_items,
    from_mars,
    from_modin,
    from_numpy,
    from_numpy_refs,
    from_pandas,
    from_pandas_refs,
    from_spark,
    from_tf,
    from_torch,
    range,
    range_tensor,
    read_audio,
    read_avro,
    read_bigquery,
    read_binary_files,
    read_clickhouse,
    read_csv,
    read_databricks_tables,
    read_datasource,
    read_delta,
    read_delta_sharing_tables,
    read_kafka,
    read_hudi,
    read_iceberg,
    read_images,
    read_json,
    read_lance,
    read_mcap,
    read_mongo,
    read_numpy,
    read_parquet,
    read_snowflake,
    read_sql,
    read_text,
    read_tfrecords,
    read_unity_catalog,
    read_videos,
    read_webdataset,
)

# Module-level cached global functions for callable classes. It needs to be defined here
# since it has to be process-global across cloudpickled funcs.
_map_actor_context = None

configure_logging()

try:
    import pyarrow as pa

    # Import these arrow extension types to ensure that they are registered.
    from ray.data._internal.tensor_extensions.arrow import (  # noqa
        ArrowTensorType,
        ArrowVariableShapedTensorType,
    )

    # https://github.com/apache/arrow/pull/38608 deprecated `PyExtensionType`, and
    # disabled it's deserialization by default. To ensure that users can load data
    # written with earlier version of Ray Data, we enable auto-loading of serialized
    # tensor extensions.
    #
    # NOTE: `PyExtensionType` is deleted from Arrow >= 21.0
    pyarrow_version = get_pyarrow_version()
    if pyarrow_version is None or pyarrow_version >= parse_version("21.0.0"):
        pass
    else:
        from ray._private.ray_constants import env_bool

        RAY_DATA_AUTOLOAD_PYEXTENSIONTYPE = env_bool(
            "RAY_DATA_AUTOLOAD_PYEXTENSIONTYPE", False
        )

        if (
            pyarrow_version >= parse_version("14.0.1")
            and RAY_DATA_AUTOLOAD_PYEXTENSIONTYPE
        ):
            pa.PyExtensionType.set_auto_load(True)

except ModuleNotFoundError:
    pass


__all__ = [
    "ActorPoolStrategy",
    "BlockBasedFileDatasink",
    "ClickHouseTableSettings",
    "Dataset",
    "DataContext",
    "DatasetContext",  # Backwards compatibility alias.
    "DatasetSummary",
    "DataIterator",
    "DatasetIterator",  # Backwards compatibility alias.
    "Datasink",
    "Datasource",
    "ExecutionOptions",
    "ExecutionResources",
    "FileShuffleConfig",
    "NodeIdStr",
    "ReadTask",
    "RowBasedFileDatasink",
    "Schema",
    "SinkMode",
    "SaveMode",
    "TaskPoolStrategy",
    "from_daft",
    "from_dask",
    "from_items",
    "from_arrow",
    "from_arrow_refs",
    "from_mars",
    "from_modin",
    "from_numpy",
    "from_numpy_refs",
    "from_pandas",
    "from_pandas_refs",
    "from_spark",
    "from_tf",
    "from_torch",
    "from_huggingface",
    "range",
    "range_tensor",
    "read_audio",
    "read_avro",
    "read_text",
    "read_binary_files",
    "read_clickhouse",
    "read_csv",
    "read_datasource",
    "read_delta",
    "read_delta_sharing_tables",
    "read_kafka",
    "KafkaAuthConfig",
    "read_hudi",
    "read_iceberg",
    "read_images",
    "read_json",
    "read_lance",
    "read_mcap",
    "read_numpy",
    "read_mongo",
    "read_parquet",
    "read_snowflake",
    "read_sql",
    "read_tfrecords",
    "read_unity_catalog",
    "read_videos",
    "read_webdataset",
    "Preprocessor",
    "TFXReadOptions",
]

# Append Anyscale proprietary APIs and apply patches
from ray.anyscale.data.api import StreamingAggFn  # noqa: E402, isort:skip
from ray.anyscale.data.api.read_api import (  # noqa: E402, F811, isort:skip
    read_audio,
    read_binary_files,
    read_csv,
    read_images,
    read_json,
    read_numpy,
    read_parquet,
    read_snowflake,
    read_text,
    read_videos,
    read_webdataset,
    read_delta,
)


__all__ += [
    "StreamingAggFn",
    "read_snowflake",
    "read_audio",
    "read_videos",
    "read_delta",
]
