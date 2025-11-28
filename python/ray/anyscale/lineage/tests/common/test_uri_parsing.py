"""Tests for URI parsing functions."""

import pytest
from urllib.parse import urlsplit

from ray.anyscale.lineage.common import uri_parsing


@pytest.mark.parametrize(
    "uri,parse_func,expected",
    [
        (
            "azurecosmos://myaccount.documents.azure.com/dbs/mydb/colls/mycollection",
            uri_parsing.parse_azure_cosmos_db_uri,
            {
                "host": "myaccount.documents.azure.com",
                "database": "mydb",
                "table": "mycollection",
            },
        ),
        (
            "azurekusto://mycluster.kusto.windows.net/mydb/mytable",
            uri_parsing.parse_azure_data_explorer_uri,
            {"host": "mycluster", "database": "mydb", "table": "mytable"},
        ),
        (
            "sqlserver://myserver.database.windows.net:1433/dbo.mytable",
            uri_parsing.parse_azure_synapse_uri,
            {
                "host": "myserver.database.windows.net",
                "port": "1433",
                "schema": "dbo",
                "table": "mytable",
            },
        ),
        (
            "cassandra://localhost:9042/mykeyspace.mytable",
            uri_parsing.parse_cassandra_uri,
            {
                "host": "localhost",
                "port": "9042",
                "keyspace": "mykeyspace",
                "table": "mytable",
            },
        ),
        (
            "mysql://mysql.example.com:3306/mydb.mytable",
            uri_parsing.parse_mysql_uri,
            {
                "host": "mysql.example.com",
                "port": "3306",
                "database": "mydb",
                "table": "mytable",
            },
        ),
        (
            "crate://localhost:4200/mydb.myschema.mytable",
            uri_parsing.parse_crate_db_uri,
            {
                "host": "localhost",
                "port": "4200",
                "database": "mydb",
                "schema": "myschema",
                "table": "mytable",
            },
        ),
        (
            "db2://db2host:50000/testdb.testschema.testtable",
            uri_parsing.parse_db2_uri,
            {
                "host": "db2host",
                "port": "50000",
                "database": "testdb",
                "schema": "testschema",
                "table": "testtable",
            },
        ),
        (
            "hive://hiveserver:10000/mydb.mytable",
            uri_parsing.parse_hive_uri,
            {
                "host": "hiveserver",
                "port": "10000",
                "database": "mydb",
                "table": "mytable",
            },
        ),
        (
            "oceanbase://obhost:2881/mydb.mytable",
            uri_parsing.parse_ocean_base_uri,
            {"host": "obhost", "port": "2881", "database": "mydb", "table": "mytable"},
        ),
        (
            "postgres://postgres.example.com:5432/mydb.public.users",
            uri_parsing.parse_postgres_uri,
            {
                "host": "postgres.example.com",
                "port": "5432",
                "database": "mydb",
                "schema": "public",
                "table": "users",
            },
        ),
        (
            "teradata://tdhost:1025/mydb.mytable",
            uri_parsing.parse_teradata_uri,
            {"host": "tdhost", "port": "1025", "database": "mydb", "table": "mytable"},
        ),
        (
            "trino://trinohost:8080/mycatalog.myschema.mytable",
            uri_parsing.parse_trino_uri,
            {
                "host": "trinohost",
                "port": "8080",
                "catalog": "mycatalog",
                "schema": "myschema",
                "table": "mytable",
            },
        ),
        (
            "dbfs://myworkspace/path/to/data",
            uri_parsing.parse_dbfs_uri,
            {"workspace": "myworkspace", "path": "path/to/data"},
        ),
        (
            "hdfs://namenode:8020/user/hadoop/data",
            uri_parsing.parse_hdfs_uri,
            {
                "namenode_host": "namenode",
                "namenode_port": "8020",
                "path": "user/hadoop/data",
            },
        ),
        (
            "kafka://broker:9092/mytopic",
            uri_parsing.parse_kafka_uri,
            {"bootstrap_server_host": "broker", "port": "9092", "topic": "mytopic"},
        ),
        (
            "local:///path/to/file",
            uri_parsing.parse_local_file_system_uri,
            {"path": "path/to/file"},
        ),
        (
            "file://remotehost/path/to/file",
            uri_parsing.parse_remote_file_system_uri,
            {"host": "remotehost", "path": "path/to/file"},
        ),
        (
            "wasbs://mycontainer@mystorageaccount.blob.core.windows.net/path/to/blob",
            uri_parsing.parse_wasbs_uri,
            {
                "container": "mycontainer",
                "service_name": "mystorageaccount",
                "object_key": "path/to/blob",
            },
        ),
    ],
)
def test_basic_uri_parsing(uri, parse_func, expected):
    """Test URI parsing functions with standard URIs."""
    parsed = urlsplit(uri)
    result = parse_func(uri, parsed)
    assert result == expected


class TestAthenaURI:
    """Test Athena URI parsing."""

    def test_parse_athena_uri(self):
        uri = "awsathena://athena.us-west-2.amazonaws.com/awsdatacatalog.mydb.mytable"
        parsed = urlsplit(uri)
        result = uri_parsing.parse_athena_uri(uri, parsed)

        assert result["region_name"] == "us-west-2"
        assert result["catalog"] == "awsdatacatalog"
        assert result["database"] == "mydb"
        assert result["table"] == "mytable"


class TestAWSGlueURI:
    """Test AWS Glue URI parsing."""

    def test_parse_aws_glue_uri(self):
        uri = "arn:aws:glue:us-west-2:123456789012/table/mydb/mytable"
        parsed = urlsplit(uri)
        result = uri_parsing.parse_aws_glue_uri(uri, parsed)

        assert result["region_name"] == "us-west-2"
        assert result["account_id"] == "123456789012"
        assert result["database_name"] == "mydb"
        assert result["table_name"] == "mytable"


class TestBigQueryURI:
    """Test BigQuery URI parsing."""

    def test_parse_bigquery_uri_with_prefix(self):
        uri = "bigquery/my-project.my_dataset.my_table"
        parsed = urlsplit(uri)
        result = uri_parsing.parse_big_query_uri(uri, parsed)

        assert result["project_id"] == "my-project"
        assert result["dataset_name"] == "my_dataset"
        assert result["table_name"] == "my_table"

    def test_parse_bigquery_uri_without_prefix(self):
        uri = "my-project-2.dataset_2.table_2"
        parsed = urlsplit(uri)
        result = uri_parsing.parse_big_query_uri(uri, parsed)

        assert result["project_id"] == "my-project-2"
        assert result["dataset_name"] == "dataset_2"
        assert result["table_name"] == "table_2"


class TestOracleURI:
    """Test Oracle URI parsing."""

    def test_parse_oracle_uri(self):
        uri = "oracle://oraclehost:1521/ORCL.SCHEMA1.TABLE1"
        parsed = urlsplit(uri)
        result = uri_parsing.parse_oracle_uri(uri, parsed)

        assert result["host"] == "oraclehost"
        assert result["port"] == "1521"
        assert result["service_name"] == "ORCL"
        assert result["schema"] == "SCHEMA1"
        assert result["table"] == "TABLE1"


class TestRedshiftURI:
    """Test Redshift URI parsing with complex netloc format."""

    def test_parse_redshift_uri(self):
        uri = "redshift://mycluster.us-west-2:5439/mydb.public.mytable"
        parsed = urlsplit(uri)
        result = uri_parsing.parse_redshift_uri(uri, parsed)

        assert result["cluster_identifier"] == "mycluster"
        assert result["region"] == "us-west-2"
        assert result["port"] == "5439"
        assert result["database"] == "mydb"
        assert result["schema"] == "public"
        assert result["table"] == "mytable"


class TestSnowflakeURI:
    """Test Snowflake URI parsing with organization-account format."""

    def test_parse_snowflake_uri(self):
        uri = "snowflake://myorg-myaccount/mydb.myschema.mytable"
        parsed = urlsplit(uri)
        result = uri_parsing.parse_snowflake_uri(uri, parsed)

        assert result["organization_name"] == "myorg"
        assert result["account_name"] == "myaccount"
        assert result["database"] == "mydb"
        assert result["schema"] == "myschema"
        assert result["table"] == "mytable"


class TestABFSSURI:
    """Test ABFSS URI parsing."""

    def test_parse_abfss_uri(self):
        uri = "abfss://mycontainer@myaccount.dfs.core.windows.net/path/to/data"
        parsed = urlsplit(uri)
        result = uri_parsing.parse_abfss_uri(uri, parsed)

        assert result["container"] == "mycontainer"
        assert result["service"] == "myaccount"
        assert result["path"] == "path/to/data"


class TestGCSURI:
    """Test GCS URI parsing."""

    def test_parse_gcs_uri(self):
        uri = "gs://my-bucket/path/to/object.parquet"
        parsed = urlsplit(uri)
        result = uri_parsing.parse_gcs_uri(uri, parsed)

        assert result["bucket_name"] == "my-bucket"
        assert result["object_key"] == "path/to/object.parquet"


class TestLocalFileSystemURI:
    """Test Local File System URI parsing."""

    def test_parse_local_file_system_uri_without_scheme(self):
        uri = "/absolute/path/to/file"
        parsed = urlsplit(uri)
        result = uri_parsing.parse_local_file_system_uri(uri, parsed)

        assert result["path"] == "absolute/path/to/file"


class TestS3URI:
    """Test S3 URI parsing."""

    def test_parse_s3_uri(self):
        uri = "s3://my-bucket/path/to/object.parquet"
        parsed = urlsplit(uri)
        result = uri_parsing.parse_s3_uri(uri, parsed)

        assert result["bucket_name"] == "my-bucket"
        assert result["object_key"] == "path/to/object.parquet"


class TestPubSubURI:
    """Test PubSub URI parsing with non-standard format."""

    def test_parse_pub_sub_uri_topic(self):
        uri = "pubsub/topic:my-project:my-topic"
        parsed = urlsplit(uri)
        result = uri_parsing.parse_pub_sub_uri(uri, parsed)

        assert result["resource_type"].value == "topic"
        assert result["project_id"] == "my-project"
        assert result["resource_id"] == "my-topic"

    def test_parse_pub_sub_uri_subscription(self):
        uri = "pubsub/subscription:my-project:my-subscription"
        parsed = urlsplit(uri)
        result = uri_parsing.parse_pub_sub_uri(uri, parsed)

        assert result["resource_type"].value == "subscription"
        assert result["project_id"] == "my-project"
        assert result["resource_id"] == "my-subscription"
