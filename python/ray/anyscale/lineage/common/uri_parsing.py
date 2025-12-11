import re
from typing import Any

import openlineage.client.naming.dataset as ol_dataset_naming


def parse_athena_uri(uri: str, parsed: Any) -> dict[str, Any]:
    """Parse Athena URI: awsathena://athena.{region_name}.amazonaws.com/{catalog}.{database}.{table}"""
    attributes: dict[str, Any] = {}
    netloc = parsed.netloc
    path = parsed.path.lstrip("/")

    match = re.match(r"athena\.([^.]+)\.amazonaws\.com", netloc)
    if match:
        attributes["region_name"] = match.group(1)
    name_match = re.match(r"([^.]+)\.([^.]+)\.(.+)", path)
    if name_match:
        attributes["catalog"] = name_match.group(1)
        attributes["database"] = name_match.group(2)
        attributes["table"] = name_match.group(3)

    return attributes


def parse_aws_glue_uri(uri: str, parsed: Any) -> dict[str, Any]:
    """Parse AWS Glue URI: arn:aws:glue:{region}:{account_id}/table/{database_name}/{table_name}"""
    attributes: dict[str, Any] = {}

    match = re.match(r"aws:glue:([^:]+):([^/]+)", parsed.path)
    if match:
        attributes["region_name"] = match.group(1)
        attributes["account_id"] = match.group(2)
        table_match = re.search(r"/table/([^/]+)/(.+)", uri)
        if table_match:
            attributes["database_name"] = table_match.group(1)
            attributes["table_name"] = table_match.group(2)

    return attributes


def parse_azure_cosmos_db_uri(uri: str, parsed: Any) -> dict[str, Any]:
    """Parse Azure Cosmos DB URI: azurecosmos://{host}/dbs/{database}/colls/{table}"""
    attributes: dict[str, Any] = {}
    netloc = parsed.netloc
    path = parsed.path.lstrip("/")

    attributes["host"] = netloc
    parts_match = re.match(r"dbs/([^/]+)/colls/(.+)", path)
    if parts_match:
        attributes["database"] = parts_match.group(1)
        attributes["table"] = parts_match.group(2)

    return attributes


def parse_azure_data_explorer_uri(uri: str, parsed: Any) -> dict[str, Any]:
    """Parse Azure Data Explorer URI: azurekusto://{host}.kusto.windows.net/{database}/{table}"""
    attributes: dict[str, Any] = {}
    netloc = parsed.netloc
    path = parsed.path.lstrip("/")

    match = re.match(r"([^.]+)\.kusto\.windows\.net", netloc)
    if match:
        attributes["host"] = match.group(1)
    parts = path.split("/")
    if len(parts) >= 2:
        attributes["database"] = parts[0]
        attributes["table"] = parts[1]

    return attributes


def parse_azure_synapse_uri(uri: str, parsed: Any) -> dict[str, Any]:
    """Parse Azure Synapse URI: sqlserver://{host}:{port}/{schema}.{table}"""
    attributes: dict[str, Any] = {}
    netloc = parsed.netloc
    path = parsed.path.lstrip("/")

    if ":" in netloc:
        host, port = netloc.rsplit(":", 1)
        attributes["host"] = host
        attributes["port"] = port
    name_match = re.match(r"([^.]+)\.(.+)", path)
    if name_match:
        attributes["schema"] = name_match.group(1)
        attributes["table"] = name_match.group(2)

    return attributes


def parse_big_query_uri(uri: str, parsed: Any) -> dict[str, Any]:
    """Parse BigQuery URI: bigquery/{project_id}.{dataset_name}.{table_name}"""
    attributes: dict[str, Any] = {}
    path = parsed.path.lstrip("/")

    # Note: BigQuery namespace is just "bigquery", name is the full path
    # Remove "bigquery/" prefix if present
    name_part = path if path else uri
    if name_part.startswith("bigquery/"):
        name_part = name_part[9:]  # Remove "bigquery/"
    name_match = re.match(r"([^.]+)\.([^.]+)\.(.+)", name_part)
    if name_match:
        attributes["project_id"] = name_match.group(1)
        attributes["dataset_name"] = name_match.group(2)
        attributes["table_name"] = name_match.group(3)

    return attributes


def parse_cassandra_uri(uri: str, parsed: Any) -> dict[str, Any]:
    """Parse Cassandra URI: cassandra://{host}:{port}/{keyspace}.{table}"""
    attributes: dict[str, Any] = {}
    netloc = parsed.netloc
    path = parsed.path.lstrip("/")

    if ":" in netloc:
        host, port = netloc.rsplit(":", 1)
        attributes["host"] = host
        attributes["port"] = port
    name_match = re.match(r"([^.]+)\.(.+)", path)
    if name_match:
        attributes["keyspace"] = name_match.group(1)
        attributes["table"] = name_match.group(2)

    return attributes


def parse_mysql_uri(uri: str, parsed: Any) -> dict[str, Any]:
    """Parse MySQL URI: mysql://{host}:{port}/{database}.{table}"""
    attributes: dict[str, Any] = {}
    netloc = parsed.netloc
    path = parsed.path.lstrip("/")

    if ":" in netloc:
        host, port = netloc.rsplit(":", 1)
        attributes["host"] = host
        attributes["port"] = port
    name_match = re.match(r"([^.]+)\.(.+)", path)
    if name_match:
        attributes["database"] = name_match.group(1)
        attributes["table"] = name_match.group(2)

    return attributes


def parse_crate_db_uri(uri: str, parsed: Any) -> dict[str, Any]:
    """Parse CrateDB URI: crate://{host}:{port}/{database}.{schema}.{table}"""
    attributes: dict[str, Any] = {}
    netloc = parsed.netloc
    path = parsed.path.lstrip("/")

    if ":" in netloc:
        host, port = netloc.rsplit(":", 1)
        attributes["host"] = host
        attributes["port"] = port
    name_match = re.match(r"([^.]+)\.([^.]+)\.(.+)", path)
    if name_match:
        attributes["database"] = name_match.group(1)
        attributes["schema"] = name_match.group(2)
        attributes["table"] = name_match.group(3)

    return attributes


def parse_db2_uri(uri: str, parsed: Any) -> dict[str, Any]:
    """Parse DB2 URI: db2://{host}:{port}/{database}.{schema}.{table}"""
    attributes: dict[str, Any] = {}
    netloc = parsed.netloc
    path = parsed.path.lstrip("/")

    if ":" in netloc:
        host, port = netloc.rsplit(":", 1)
        attributes["host"] = host
        attributes["port"] = port
    name_match = re.match(r"([^.]+)\.([^.]+)\.(.+)", path)
    if name_match:
        attributes["database"] = name_match.group(1)
        attributes["schema"] = name_match.group(2)
        attributes["table"] = name_match.group(3)

    return attributes


def parse_hive_uri(uri: str, parsed: Any) -> dict[str, Any]:
    """Parse Hive URI: hive://{host}:{port}/{database}.{table}"""
    attributes: dict[str, Any] = {}
    netloc = parsed.netloc
    path = parsed.path.lstrip("/")

    if ":" in netloc:
        host, port = netloc.rsplit(":", 1)
        attributes["host"] = host
        attributes["port"] = port
    name_match = re.match(r"([^.]+)\.(.+)", path)
    if name_match:
        attributes["database"] = name_match.group(1)
        attributes["table"] = name_match.group(2)

    return attributes


def parse_ocean_base_uri(uri: str, parsed: Any) -> dict[str, Any]:
    """Parse OceanBase URI: oceanbase://{host}:{port}/{database}.{table}"""
    attributes: dict[str, Any] = {}
    netloc = parsed.netloc
    path = parsed.path.lstrip("/")

    if ":" in netloc:
        host, port = netloc.rsplit(":", 1)
        attributes["host"] = host
        attributes["port"] = port
    name_match = re.match(r"([^.]+)\.(.+)", path)
    if name_match:
        attributes["database"] = name_match.group(1)
        attributes["table"] = name_match.group(2)

    return attributes


def parse_oracle_uri(uri: str, parsed: Any) -> dict[str, Any]:
    """Parse Oracle URI: oracle://{host}:{port}/{serviceName}.{schema}.{table}"""
    attributes: dict[str, Any] = {}
    netloc = parsed.netloc
    path = parsed.path.lstrip("/")

    if ":" in netloc:
        host, port = netloc.rsplit(":", 1)
        attributes["host"] = host
        attributes["port"] = port
    name_match = re.match(r"([^.]+)\.([^.]+)\.(.+)", path)
    if name_match:
        attributes["service_name"] = name_match.group(1)
        attributes["schema"] = name_match.group(2)
        attributes["table"] = name_match.group(3)

    return attributes


def parse_postgres_uri(uri: str, parsed: Any) -> dict[str, Any]:
    """Parse Postgres URI: postgres://{host}:{port}/{database}.{schema}.{table}"""
    attributes: dict[str, Any] = {}
    netloc = parsed.netloc
    path = parsed.path.lstrip("/")

    if ":" in netloc:
        host, port = netloc.rsplit(":", 1)
        attributes["host"] = host
        attributes["port"] = port
    name_match = re.match(r"([^.]+)\.([^.]+)\.(.+)", path)
    if name_match:
        attributes["database"] = name_match.group(1)
        attributes["schema"] = name_match.group(2)
        attributes["table"] = name_match.group(3)

    return attributes


def parse_teradata_uri(uri: str, parsed: Any) -> dict[str, Any]:
    """Parse Teradata URI: teradata://{host}:{port}/{database}.{table}"""
    attributes: dict[str, Any] = {}
    netloc = parsed.netloc
    path = parsed.path.lstrip("/")

    if ":" in netloc:
        host, port = netloc.rsplit(":", 1)
        attributes["host"] = host
        attributes["port"] = port
    name_match = re.match(r"([^.]+)\.(.+)", path)
    if name_match:
        attributes["database"] = name_match.group(1)
        attributes["table"] = name_match.group(2)

    return attributes


def parse_redshift_uri(uri: str, parsed: Any) -> dict[str, Any]:
    """Parse Redshift URI: redshift://{cluster_identifier}.{region_name}:{port}/{database}.{schema}.{table}"""
    attributes: dict[str, Any] = {}
    netloc = parsed.netloc
    path = parsed.path.lstrip("/")

    netloc_match = re.match(r"([^.]+)\.([^:]+):(.+)", netloc)
    if netloc_match:
        attributes["cluster_identifier"] = netloc_match.group(1)
        attributes["region"] = netloc_match.group(2)
        attributes["port"] = netloc_match.group(3)
    name_match = re.match(r"([^.]+)\.([^.]+)\.(.+)", path)
    if name_match:
        attributes["database"] = name_match.group(1)
        attributes["schema"] = name_match.group(2)
        attributes["table"] = name_match.group(3)

    return attributes


def parse_snowflake_uri(uri: str, parsed: Any) -> dict[str, Any]:
    """Parse Snowflake URI: snowflake://{organization_name}-{account_name}/{database}.{schema}.{table}"""
    attributes: dict[str, Any] = {}
    netloc = parsed.netloc
    path = parsed.path.lstrip("/")

    if "-" in netloc:
        parts = netloc.split("-", 1)
        attributes["organization_name"] = parts[0]
        attributes["account_name"] = parts[1]
    name_match = re.match(r"([^.]+)\.([^.]+)\.(.+)", path)
    if name_match:
        attributes["database"] = name_match.group(1)
        attributes["schema"] = name_match.group(2)
        attributes["table"] = name_match.group(3)

    return attributes


def parse_trino_uri(uri: str, parsed: Any) -> dict[str, Any]:
    """Parse Trino URI: trino://{host}:{port}/{catalog}.{schema}.{table}"""
    attributes: dict[str, Any] = {}
    netloc = parsed.netloc
    path = parsed.path.lstrip("/")

    if ":" in netloc:
        host, port = netloc.rsplit(":", 1)
        attributes["host"] = host
        attributes["port"] = port
    name_match = re.match(r"([^.]+)\.([^.]+)\.(.+)", path)
    if name_match:
        attributes["catalog"] = name_match.group(1)
        attributes["schema"] = name_match.group(2)
        attributes["table"] = name_match.group(3)

    return attributes


def parse_abfss_uri(uri: str, parsed: Any) -> dict[str, Any]:
    """Parse ABFSS URI: abfss://{container_name}@{service_name}.dfs.core.windows.net/{path}"""
    attributes: dict[str, Any] = {}
    netloc = parsed.netloc
    path = parsed.path.lstrip("/")

    match = re.match(r"([^@]+)@([^.]+)\.dfs\.core\.windows\.net", netloc)
    if match:
        attributes["container"] = match.group(1)
        attributes["service"] = match.group(2)
    attributes["path"] = path

    return attributes


def parse_dbfs_uri(uri: str, parsed: Any) -> dict[str, Any]:
    """Parse DBFS URI: dbfs://{workspace_name}/{path}"""
    attributes: dict[str, Any] = {}
    netloc = parsed.netloc
    path = parsed.path.lstrip("/")

    attributes["workspace"] = netloc
    attributes["path"] = path

    return attributes


def parse_gcs_uri(uri: str, parsed: Any) -> dict[str, Any]:
    """Parse GCS URI: gs://{bucket_name}/{object_key}"""
    attributes: dict[str, Any] = {}
    netloc = parsed.netloc
    path = parsed.path.lstrip("/")

    attributes["bucket_name"] = netloc
    attributes["object_key"] = path

    return attributes


def parse_hdfs_uri(uri: str, parsed: Any) -> dict[str, Any]:
    """Parse HDFS URI: hdfs://{namenode_host}:{namenode_port}/{path}"""
    attributes: dict[str, Any] = {}
    netloc = parsed.netloc
    path = parsed.path.lstrip("/")

    if ":" in netloc:
        host, port = netloc.rsplit(":", 1)
        attributes["namenode_host"] = host
        attributes["namenode_port"] = port
    attributes["path"] = path

    return attributes


def parse_kafka_uri(uri: str, parsed: Any) -> dict[str, Any]:
    """Parse Kafka URI: kafka://{bootstrap_server_host}:{port}/{topic}"""
    attributes: dict[str, Any] = {}
    netloc = parsed.netloc
    path = parsed.path.lstrip("/")

    if ":" in netloc:
        host, port = netloc.rsplit(":", 1)
        attributes["bootstrap_server_host"] = host
        attributes["port"] = port
    attributes["topic"] = path

    return attributes


def parse_local_file_system_uri(uri: str, parsed: Any) -> dict[str, Any]:
    """Parse Local File System URI: file:{path}, local://{path}, or absolute paths."""
    attributes: dict[str, Any] = {}
    path = parsed.path

    attributes["path"] = path if path else uri

    return attributes


def parse_remote_file_system_uri(uri: str, parsed: Any) -> dict[str, Any]:
    """Parse Remote File System URI: file://{host}/{path}"""
    attributes: dict[str, Any] = {}
    netloc = parsed.netloc
    path = parsed.path.lstrip("/")

    attributes["host"] = netloc
    attributes["path"] = path

    return attributes


def parse_s3_uri(uri: str, parsed: Any) -> dict[str, Any]:
    """Parse S3 URI: s3://{bucket_name}/{object_key}"""
    attributes: dict[str, Any] = {}
    netloc = parsed.netloc
    path = parsed.path.lstrip("/")

    attributes["bucket_name"] = netloc
    attributes["object_key"] = path

    return attributes


def parse_wasbs_uri(uri: str, parsed: Any) -> dict[str, Any]:
    """Parse WASBS URI: wasbs://{container_name}@{service_name}.blob.core.windows.net/{object_key}"""
    attributes: dict[str, Any] = {}
    netloc = parsed.netloc
    path = parsed.path.lstrip("/")

    match = re.match(r"([^@]+)@([^.]+)\.blob\.core\.windows\.net", netloc)
    if match:
        attributes["container"] = match.group(1)
        attributes["service_name"] = match.group(2)
    attributes["object_key"] = path

    return attributes


def parse_pub_sub_uri(uri: str, parsed: Any) -> dict[str, Any]:
    """Parse PubSub URI: pubsub/topic:{project_id}:{topic_id} or pubsub/subscription:{project_id}:{subscription_id}"""
    attributes: dict[str, Any] = {}
    path = parsed.path.lstrip("/")

    # Remove 'pubsub/' prefix if present
    name_part = path if path else uri
    if name_part.startswith("pubsub/"):
        name_part = name_part[7:]  # Remove "pubsub/"

    name_match = re.match(r"(topic|subscription):([^:]+):(.+)", name_part)
    if name_match:
        resource_type_str = name_match.group(1)
        attributes["resource_type"] = ol_dataset_naming.PubSubResourceType(
            resource_type_str
        )
        attributes["project_id"] = name_match.group(2)
        attributes["resource_id"] = name_match.group(3)

    return attributes
