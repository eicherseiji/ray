from collections import namedtuple
from enum import Enum, unique
from typing import Any, Callable, Optional, Tuple
from urllib.parse import urlsplit

import openlineage.client.naming.dataset as dataset_naming

from ray.anyscale.lineage.common import uri_parsing
from ray.anyscale.lineage.common.exceptions import AnyscaleLineageSDKError
from ray.anyscale.lineage.common.logging import get_logger
from ray.anyscale.lineage.common.utils import parse_uri

logger = get_logger(__name__)


@unique
class OpenLineageDatasetNamingTypes(Enum):
    """OpenLineage dataset naming types."""

    ATHENA = "Athena"
    AWS_GLUE = "AWSGlue"
    AZURE_COSMOS_DB = "AzureCosmosDB"
    AZURE_DATA_EXPLORER = "AzureDataExplorer"
    AZURE_SYNAPSE = "AzureSynapse"
    BIG_QUERY = "BigQuery"
    CASSANDRA = "Cassandra"
    MY_SQL = "MySQL"
    CRATE_DB = "CrateDB"
    DB2 = "DB2"
    OCEAN_BASE = "OceanBase"
    ORACLE = "Oracle"
    POSTGRES = "Postgres"
    TERADATA = "Teradata"
    REDSHIFT = "Redshift"
    SNOWFLAKE = "Snowflake"
    TRINO = "Trino"
    ABFSS = "ABFSS"
    DBFS = "DBFS"
    GCS = "GCS"
    HDFS = "HDFS"
    HIVE = "Hive"
    KAFKA = "Kafka"
    LOCAL_FILE_SYSTEM = "LocalFileSystem"
    REMOTE_FILE_SYSTEM = "RemoteFileSystem"
    S3 = "S3"
    WASBS = "WASBS"
    PUB_SUB_NAMING = "PubSubNaming"


DatasetStorageLayer = namedtuple(
    "DatasetStorageLayer", ["name", "schemes"], defaults=[[]]
)


@unique
class DatasetStorageLayerTypes(Enum):
    """Dataset storage layer types."""

    ATHENA = DatasetStorageLayer(
        name=OpenLineageDatasetNamingTypes.ATHENA.value, schemes=["awsathena://"]
    )
    AWS_GLUE = DatasetStorageLayer(
        name=OpenLineageDatasetNamingTypes.AWS_GLUE.value, schemes=["arn:aws:glue:"]
    )
    AZURE_COSMOS_DB = DatasetStorageLayer(
        name=OpenLineageDatasetNamingTypes.AZURE_COSMOS_DB.value,
        schemes=["azurecosmos://"],
    )
    AZURE_DATA_EXPLORER = DatasetStorageLayer(
        name=OpenLineageDatasetNamingTypes.AZURE_DATA_EXPLORER.value,
        schemes=["azurekusto://"],
    )
    AZURE_SYNAPSE = DatasetStorageLayer(
        name=OpenLineageDatasetNamingTypes.AZURE_SYNAPSE.value, schemes=["sqlserver://"]
    )
    BIG_QUERY = DatasetStorageLayer(
        name=OpenLineageDatasetNamingTypes.BIG_QUERY.value, schemes=["bigquery/"]
    )
    CASSANDRA = DatasetStorageLayer(
        name=OpenLineageDatasetNamingTypes.CASSANDRA.value, schemes=["cassandra://"]
    )
    MY_SQL = DatasetStorageLayer(
        name=OpenLineageDatasetNamingTypes.MY_SQL.value, schemes=["mysql://"]
    )
    CRATE_DB = DatasetStorageLayer(
        name=OpenLineageDatasetNamingTypes.CRATE_DB.value, schemes=["crate://"]
    )
    DB2 = DatasetStorageLayer(
        name=OpenLineageDatasetNamingTypes.DB2.value, schemes=["db2://"]
    )
    OCEAN_BASE = DatasetStorageLayer(
        name=OpenLineageDatasetNamingTypes.OCEAN_BASE.value, schemes=["oceanbase://"]
    )
    ORACLE = DatasetStorageLayer(
        name=OpenLineageDatasetNamingTypes.ORACLE.value, schemes=["oracle://"]
    )
    POSTGRES = DatasetStorageLayer(
        name=OpenLineageDatasetNamingTypes.POSTGRES.value, schemes=["postgres://"]
    )
    TERADATA = DatasetStorageLayer(
        name=OpenLineageDatasetNamingTypes.TERADATA.value, schemes=["teradata://"]
    )
    REDSHIFT = DatasetStorageLayer(
        name=OpenLineageDatasetNamingTypes.REDSHIFT.value, schemes=["redshift://"]
    )
    SNOWFLAKE = DatasetStorageLayer(
        name=OpenLineageDatasetNamingTypes.SNOWFLAKE.value, schemes=["snowflake://"]
    )
    TRINO = DatasetStorageLayer(
        name=OpenLineageDatasetNamingTypes.TRINO.value, schemes=["trino://"]
    )
    ABFSS = DatasetStorageLayer(
        name=OpenLineageDatasetNamingTypes.ABFSS.value, schemes=["abfs://", "abfss://"]
    )
    DBFS = DatasetStorageLayer(
        name=OpenLineageDatasetNamingTypes.DBFS.value, schemes=["dbfs://"]
    )
    GCS = DatasetStorageLayer(
        name=OpenLineageDatasetNamingTypes.GCS.value, schemes=["gs://", "gcs://"]
    )
    HDFS = DatasetStorageLayer(
        name=OpenLineageDatasetNamingTypes.HDFS.value, schemes=["hdfs://"]
    )
    HIVE = DatasetStorageLayer(
        name=OpenLineageDatasetNamingTypes.HIVE.value, schemes=["hive://"]
    )
    KAFKA = DatasetStorageLayer(
        name=OpenLineageDatasetNamingTypes.KAFKA.value, schemes=["kafka://"]
    )
    LOCAL_FILE_SYSTEM = DatasetStorageLayer(
        name=OpenLineageDatasetNamingTypes.LOCAL_FILE_SYSTEM.value, schemes=["local://"]
    )
    REMOTE_FILE_SYSTEM = DatasetStorageLayer(
        name=OpenLineageDatasetNamingTypes.REMOTE_FILE_SYSTEM.value, schemes=["file://"]
    )
    S3 = DatasetStorageLayer(
        name=OpenLineageDatasetNamingTypes.S3.value, schemes=["s3://", "s3a://"]
    )
    WASBS = DatasetStorageLayer(
        name=OpenLineageDatasetNamingTypes.WASBS.value, schemes=["wasbs://"]
    )
    PUB_SUB_NAMING = DatasetStorageLayer(
        name=OpenLineageDatasetNamingTypes.PUB_SUB_NAMING.value, schemes=["pubsub/"]
    )


# URI parser mapping for each dataset type
PARSER_MAP = {
    OpenLineageDatasetNamingTypes.ATHENA: uri_parsing.parse_athena_uri,
    OpenLineageDatasetNamingTypes.AWS_GLUE: uri_parsing.parse_aws_glue_uri,
    OpenLineageDatasetNamingTypes.AZURE_COSMOS_DB: uri_parsing.parse_azure_cosmos_db_uri,
    OpenLineageDatasetNamingTypes.AZURE_DATA_EXPLORER: uri_parsing.parse_azure_data_explorer_uri,
    OpenLineageDatasetNamingTypes.AZURE_SYNAPSE: uri_parsing.parse_azure_synapse_uri,
    OpenLineageDatasetNamingTypes.BIG_QUERY: uri_parsing.parse_big_query_uri,
    OpenLineageDatasetNamingTypes.CASSANDRA: uri_parsing.parse_cassandra_uri,
    OpenLineageDatasetNamingTypes.MY_SQL: uri_parsing.parse_mysql_uri,
    OpenLineageDatasetNamingTypes.CRATE_DB: uri_parsing.parse_crate_db_uri,
    OpenLineageDatasetNamingTypes.DB2: uri_parsing.parse_db2_uri,
    OpenLineageDatasetNamingTypes.HIVE: uri_parsing.parse_hive_uri,
    OpenLineageDatasetNamingTypes.OCEAN_BASE: uri_parsing.parse_ocean_base_uri,
    OpenLineageDatasetNamingTypes.ORACLE: uri_parsing.parse_oracle_uri,
    OpenLineageDatasetNamingTypes.POSTGRES: uri_parsing.parse_postgres_uri,
    OpenLineageDatasetNamingTypes.TERADATA: uri_parsing.parse_teradata_uri,
    OpenLineageDatasetNamingTypes.REDSHIFT: uri_parsing.parse_redshift_uri,
    OpenLineageDatasetNamingTypes.SNOWFLAKE: uri_parsing.parse_snowflake_uri,
    OpenLineageDatasetNamingTypes.TRINO: uri_parsing.parse_trino_uri,
    OpenLineageDatasetNamingTypes.ABFSS: uri_parsing.parse_abfss_uri,
    OpenLineageDatasetNamingTypes.DBFS: uri_parsing.parse_dbfs_uri,
    OpenLineageDatasetNamingTypes.GCS: uri_parsing.parse_gcs_uri,
    OpenLineageDatasetNamingTypes.HDFS: uri_parsing.parse_hdfs_uri,
    OpenLineageDatasetNamingTypes.KAFKA: uri_parsing.parse_kafka_uri,
    OpenLineageDatasetNamingTypes.LOCAL_FILE_SYSTEM: uri_parsing.parse_local_file_system_uri,
    OpenLineageDatasetNamingTypes.REMOTE_FILE_SYSTEM: uri_parsing.parse_remote_file_system_uri,
    OpenLineageDatasetNamingTypes.S3: uri_parsing.parse_s3_uri,
    OpenLineageDatasetNamingTypes.WASBS: uri_parsing.parse_wasbs_uri,
    OpenLineageDatasetNamingTypes.PUB_SUB_NAMING: uri_parsing.parse_pub_sub_uri,
}


def parse_dataset_uri(
    uri: str,
    dataset_type: OpenLineageDatasetNamingTypes,
) -> dict[str, Any]:
    """Parse a URI according to the OpenLineage dataset naming specification."""
    parsed = urlsplit(uri)
    parser_func: Callable[[str, Any], dict[str, Any]] = PARSER_MAP.get(
        dataset_type, lambda u, p: {}
    )
    return parser_func(uri, parsed)


def get_ol_dataset_naming_class(
    dataset_naming_type: OpenLineageDatasetNamingTypes,
) -> Optional[dataset_naming.DatasetNaming]:
    """Get the dataset naming class for a dataset naming type."""
    return getattr(dataset_naming, dataset_naming_type.value, None)


def resolve_ol_dataset_namespace_and_name(
    dataset_naming_type: OpenLineageDatasetNamingTypes, **kwargs: Any
) -> Tuple[str, str]:
    """Resolve the OpenLineage dataset namespace and name for the given dataset."""
    dataset_class = get_ol_dataset_naming_class(dataset_naming_type)

    if dataset_class is None:
        raise AnyscaleLineageSDKError(
            "Error resolving OpenLineage dataset namespace and name "
            f"for dataset naming type: '{dataset_naming_type}'."
        )

    try:
        # schema validation
        dataset = dataset_class(**kwargs)  # type: ignore[operator]
    except (TypeError, ValueError) as e:
        logger.error(
            f"Invalid dataset arguments for dataset naming type: '{dataset_naming_type}', "
            f"args: '{kwargs!s}', error: '{e!r}'"
        )
        raise AnyscaleLineageSDKError(
            "Error resolving OpenLineage dataset namespace and name"
        ) from e
    return dataset.get_namespace(), dataset.get_name()


def resolve_dataset_naming_type_and_attributes(
    uri: str,
) -> Tuple[OpenLineageDatasetNamingTypes, dict[str, Any]]:
    """Resolve the dataset naming type and extract all attributes from a URI."""
    parsed_uri = parse_uri(uri)
    scheme = parsed_uri["scheme"]
    path = parsed_uri["path"].lstrip("/")

    # Handle special cases for known dataset types
    if scheme == "arn" and path.startswith("aws:glue:"):
        scheme = DatasetStorageLayerTypes.AWS_GLUE.value.schemes[0]
    elif scheme == "" and path.startswith("bigquery"):
        scheme = DatasetStorageLayerTypes.BIG_QUERY.value.schemes[0]
    elif scheme == "" and path.startswith("pubsub"):
        scheme = DatasetStorageLayerTypes.PUB_SUB_NAMING.value.schemes[0]
    elif scheme == "file" and not parsed_uri.get("netloc"):
        scheme = DatasetStorageLayerTypes.LOCAL_FILE_SYSTEM.value.schemes[0]
    elif scheme == "":
        scheme = DatasetStorageLayerTypes.LOCAL_FILE_SYSTEM.value.schemes[0]
    else:
        scheme = scheme + "://" if scheme else ""

    # Identify dataset type from scheme
    dataset_naming_type = None
    for storage_layer_type in DatasetStorageLayerTypes:
        if scheme in storage_layer_type.value.schemes:
            dataset_naming_type = OpenLineageDatasetNamingTypes(
                storage_layer_type.value.name
            )
            break
    else:
        # If no match, assume local file system and return the URI as path attribute
        return OpenLineageDatasetNamingTypes.LOCAL_FILE_SYSTEM, {"path": uri}

    # Parse URI according to dataset type
    attributes = parse_dataset_uri(uri, dataset_naming_type)
    return dataset_naming_type, attributes
