import csv
import os
import random
import urllib.parse
from typing import List, Union
from unittest.mock import MagicMock

import pandas as pd
import pyarrow
import pytest
from pyarrow.fs import FileSelector, LocalFileSystem
from pytest_lazy_fixtures import lf as lazy_fixture

import ray
from ray.anyscale.data._internal.logical.operators.read_files_operator import (
    ReadFiles,
)
from ray.anyscale.data._internal.planner.checkpoint import (
    plan_from_op_with_checkpoint_filter,
    plan_read_files_op_with_checkpoint_filter,
    plan_read_op_with_checkpoint_filter,
    plan_write_op_with_checkpoint_writer,
)
from ray.anyscale.data._internal.readers import FileReader
from ray.anyscale.data.checkpoint.checkpoint_filter import BatchBasedCheckpointFilter
from ray.anyscale.data.checkpoint.checkpoint_writer import (
    BatchBasedCheckpointWriter,
)
from ray.anyscale.data.checkpoint.interfaces import (
    CheckpointBackend,
    CheckpointConfig,
    InvalidCheckpointingConfig,
)
from ray.data._internal.datasource.csv_datasource import CSVDatasource
from ray.data._internal.datasource.parquet_datasink import ParquetDatasink
from ray.data._internal.execution.operators.input_data_buffer import (
    InputDataBuffer,
)
from ray.data._internal.logical.interfaces.logical_plan import LogicalPlan
from ray.data._internal.logical.operators.from_operators import AbstractFrom
from ray.data._internal.logical.operators.input_data_operator import InputData
from ray.data._internal.logical.operators.read_operator import Read
from ray.data._internal.logical.operators.write_operator import Write
from ray.data._internal.logical.optimizers import get_execution_plan
from ray.data.block import BlockAccessor
from ray.data.datasource import Datasink
from ray.data.datasource.datasource import Datasource
from ray.data.datasource.path_util import _unwrap_protocol
from ray.data.tests.conftest import *  # noqa
from ray.tests.conftest import *  # noqa
from ray.data import DataContext

# User-provided ID column name
ID_COL = "id"

# Generated ID column name
GENERATED_ID_COL = "row_id"

# Number of rows in the sample data
SAMPLE_DATA_NUM_ROWS = 5

# Auto-use `restore_data_context` for each test and apply 300-second timeout to all tests.
pytestmark = [
    pytest.mark.usefixtures("restore_data_context"),
    pytest.mark.timeout(300),
]


@pytest.fixture
def generate_sample_data_csv(tmp_path):
    def _generate():
        # Generate a dummy dataset with SAMPLE_DATA_NUM_ROWS rows and columns [ID_COL, "col1"]
        data = [
            {ID_COL: i, "col1": random.random()} for i in range(SAMPLE_DATA_NUM_ROWS)
        ]

        f_path = os.path.join(tmp_path, "sample_data.csv")
        with open(f_path, mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
        return f_path

    return _generate


@pytest.fixture
def checkpoint_path(tmp_path):
    """Fixture to provide a temporary checkpoint path."""
    return str(tmp_path / "checkpoint")


@pytest.fixture
def generate_sample_data_parquet(tmp_path):
    def _generate():
        f_dir = os.path.join(tmp_path, "sample_data_parquet")
        os.makedirs(f_dir, exist_ok=True)
        # Generate a dummy dataset with SAMPLE_DATA_NUM_ROWS rows and columns [ID_COL, "col1"]
        df = pd.DataFrame(
            [{ID_COL: i, "col1": random.random()} for i in range(SAMPLE_DATA_NUM_ROWS)]
        )

        f_path = os.path.join(f_dir, "sample_data.parquet")
        df.to_parquet(f_path)
        return f_dir

    return _generate


@pytest.fixture
def generate_sample_physical_plan(generate_sample_data_csv, tmp_path):
    ctx = ray.data.DataContext.get_current()

    datasource = CSVDatasource(generate_sample_data_csv())

    read_op = Read(datasource, datasource, -1, None)
    write_path = os.path.join(tmp_path, "output")
    write_op = Write(read_op, ParquetDatasink(write_path))
    logical_plan = LogicalPlan(write_op, ctx)
    physical_plan = get_execution_plan(logical_plan)
    yield physical_plan


def _get_row_based_files(ckpt_path: str, fs) -> List[str]:
    """Get checkpoint filenames for row-based backends."""
    if fs is None:
        if not os.path.exists(ckpt_path):
            return []
        return [f for f in os.listdir(ckpt_path) if f.endswith(".jsonl")]
    else:
        files = fs.get_file_info(
            FileSelector(_unwrap_protocol(ckpt_path), allow_not_found=True)
        )
        return [
            os.path.basename(file_info.path) for file_info in files if file_info.is_file
        ]


def _parse_filename_to_id(filename: str, is_generated_id: bool) -> Union[int, str]:
    """Parse a filename to extract ID."""
    basename = (
        filename.replace(".jsonl", "") if filename.endswith(".jsonl") else filename
    )

    if is_generated_id:
        return basename

    # For regular IDs, the filename should be the integer ID
    return int(basename)


def _get_batch_based_files(ckpt_path: str, fs) -> List[str]:
    """Get checkpoint file paths for batch-based backends."""
    if fs is None:
        if not os.path.exists(ckpt_path):
            return []
        return [os.path.join(ckpt_path, f) for f in os.listdir(ckpt_path)]
    else:
        files = fs.get_file_info(
            FileSelector(_unwrap_protocol(ckpt_path), allow_not_found=True)
        )
        return [file_info.path for file_info in files if file_info.is_file]


def _read_batch_file_ids(file_paths: List[str], id_column: str, fs) -> List[int]:
    """Read IDs from batch-based checkpoint files."""
    ids = []
    for file_path in file_paths:
        if fs is None:
            with open(file_path, "r") as f:
                df = pd.read_parquet(f)
        else:
            with fs.open_input_file(file_path) as f:
                df = pd.read_parquet(f)
        ids.extend(df[id_column].tolist())
    return ids


def read_ids_from_checkpoint_files(config: CheckpointConfig) -> List[Union[int, str]]:
    """Reads the checkpoint files and returns a sorted list of IDs which have been checkpointed."""
    is_generated_id = config.generated_id_column is not None

    # Row-based backends
    if config.backend in (
        CheckpointBackend.FILE_STORAGE_ROW,
        CheckpointBackend.CLOUD_OBJECT_STORAGE_ROW,
    ):
        filenames = _get_row_based_files(config.checkpoint_path, config.filesystem)
        parsed_ids = [_parse_filename_to_id(f, is_generated_id) for f in filenames]

        if is_generated_id:
            return sorted(
                parsed_ids, key=lambda x: int(urllib.parse.unquote(x).split("/")[-1])
            )
        else:
            return sorted(parsed_ids)

    # Batch-based backends
    elif config.backend in (
        CheckpointBackend.FILE_STORAGE,
        CheckpointBackend.CLOUD_OBJECT_STORAGE,
    ):
        file_paths = _get_batch_based_files(config.checkpoint_path, config.filesystem)
        return sorted(
            _read_batch_file_ids(file_paths, config.id_column, config.filesystem)
        )

    else:
        raise ValueError(f"Invalid backend: {config.backend}")


class TestCheckpointConfig:
    @pytest.mark.parametrize("id_column", ["", 1])
    def test_invalid_id_column(self, id_column, local_path):
        with pytest.raises(
            InvalidCheckpointingConfig,
            match="Checkpoint ID column",
        ):
            CheckpointConfig(id_column, local_path)

    def test_override_backend_emits_deprecation_warning(self):
        with pytest.warns(FutureWarning, match="deprecated"):
            CheckpointConfig(
                ID_COL,
                "s3://bucket/path",
                override_backend=CheckpointBackend.FILE_STORAGE,
            )

    def test_default_checkpoint_path(self, s3_path, monkeypatch):
        with pytest.raises(
            InvalidCheckpointingConfig,
            match="CheckpointConfig.checkpoint_path",
        ):
            CheckpointConfig(ID_COL, None)

        default_bucket = s3_path
        monkeypatch.setenv(
            CheckpointConfig.DEFAULT_CHECKPOINT_PATH_BUCKET_ENV_VAR, default_bucket
        )

        config = CheckpointConfig(ID_COL, None)
        assert (
            config.checkpoint_path
            == f"{default_bucket}/{CheckpointConfig.DEFAULT_CHECKPOINT_PATH_DIR}"
        )

    @pytest.mark.parametrize("checkpoint_path", ["tmp/", "s3:/tmp", "s4://tmp"])
    def test_invalid_checkpoint_path(self, checkpoint_path):
        with pytest.raises(
            InvalidCheckpointingConfig,
            match="Invalid checkpoint path",
        ):
            CheckpointConfig(ID_COL, checkpoint_path)

    @pytest.mark.parametrize(
        "checkpoint_path",
        [
            lazy_fixture("local_path"),
            lazy_fixture("s3_path"),
        ],
    )
    def test_infer_filesystem_and_backend(self, checkpoint_path):
        config = CheckpointConfig(ID_COL, checkpoint_path)
        if checkpoint_path.startswith("/"):
            assert isinstance(config.filesystem, pyarrow.fs.LocalFileSystem)
            assert config.backend == CheckpointBackend.FILE_STORAGE
        else:
            assert isinstance(config.filesystem, pyarrow.fs.S3FileSystem)
            assert config.backend == CheckpointBackend.CLOUD_OBJECT_STORAGE

    @pytest.mark.parametrize(
        "checkpoint_path,fs,backend",
        [
            (
                lazy_fixture("local_path"),
                lazy_fixture("local_fs"),
                CheckpointBackend.FILE_STORAGE_ROW,
            ),
            (
                lazy_fixture("s3_path"),
                lazy_fixture("s3_fs"),
                CheckpointBackend.FILE_STORAGE_ROW,
            ),
            (
                lazy_fixture("local_path"),
                lazy_fixture("local_fs"),
                CheckpointBackend.CLOUD_OBJECT_STORAGE_ROW,
            ),
            (
                lazy_fixture("s3_path"),
                lazy_fixture("s3_fs"),
                CheckpointBackend.CLOUD_OBJECT_STORAGE_ROW,
            ),
        ],
    )
    def test_override_filesystem_and_backend(self, checkpoint_path, fs, backend):
        config = CheckpointConfig(
            ID_COL, checkpoint_path, override_filesystem=fs, override_backend=backend
        )
        assert config.filesystem is fs
        assert config.backend is backend

    def test_skip_inference_with_overrides(self):
        """Test that filesystem inference is skipped when override is provided."""
        # Inferring filesystem will fail if the path doesn't exist.
        path = "s3://non-existing-bucket/"
        fs = pyarrow.fs.S3FileSystem()
        config = CheckpointConfig(
            ID_COL,
            path,
            override_filesystem=fs,
        )
        assert config.filesystem is fs
        assert config.backend is CheckpointBackend.CLOUD_OBJECT_STORAGE

    def test_generated_id_column_default_column(self, checkpoint_path):
        """Test CheckpointConfig with id_column missing and no generated_id_column provided."""
        # id_column is None, generated_id_column is None - should raise error
        with pytest.raises(
            InvalidCheckpointingConfig,
            match="Either `id_column` or `generated_id_column` must be provided",
        ):
            CheckpointConfig(
                None,
                checkpoint_path,
            )

    def test_generated_id_column_custom_column(self, checkpoint_path):
        """Test CheckpointConfig with id_column missing, but with user provided generated_id_column."""
        # id_column is None, generated_id_column is "custom_id"
        config = CheckpointConfig(
            None,
            checkpoint_path,
            generated_id_column="custom_id",
        )
        assert config.id_column == "custom_id"
        assert config.generated_id_column == "custom_id"

    def test_generated_id_column_with_existing_id_column(self, checkpoint_path):
        """Test CheckpointConfig with both id_column and generated_id_column provided."""
        with pytest.raises(
            InvalidCheckpointingConfig,
            match="Cannot specify both `id_column` and `generated_id_column`",
        ):
            CheckpointConfig(
                "existing_id",
                checkpoint_path,
                generated_id_column="generated_id",
            )


@pytest.mark.parametrize("read_code_path", ["runtime", "oss_fallback"])
@pytest.mark.parametrize("generated_id_column", [None, "row_id"])
@pytest.mark.parametrize(
    "backend,fs,data_path",
    [
        (CheckpointBackend.FILE_STORAGE_ROW, None, lazy_fixture("local_path")),
        (
            CheckpointBackend.FILE_STORAGE_ROW,
            lazy_fixture("local_fs"),
            lazy_fixture("local_path"),
        ),
        (
            CheckpointBackend.CLOUD_OBJECT_STORAGE_ROW,
            lazy_fixture("s3_fs"),
            lazy_fixture("s3_path"),
        ),
        (CheckpointBackend.FILE_STORAGE, None, lazy_fixture("local_path")),
        (
            CheckpointBackend.FILE_STORAGE,
            lazy_fixture("local_fs"),
            lazy_fixture("local_path"),
        ),
        (
            CheckpointBackend.CLOUD_OBJECT_STORAGE,
            lazy_fixture("s3_fs"),
            lazy_fixture("s3_path"),
        ),
    ],
)
def test_checkpoint(
    ray_start_10_cpus_shared,
    generate_sample_data_csv,
    read_code_path,
    backend,
    fs,
    data_path,
    generated_id_column,
):
    class TestActor:
        def __init__(self):
            pass

        def __call__(self, batch):
            return batch

    ctx = ray.data.DataContext.get_current()
    ckpt_path = os.path.join(data_path, "test_checkpoint_output_files")

    if generated_id_column is not None:
        ctx.checkpoint_config = CheckpointConfig(
            generated_id_column=generated_id_column,
            checkpoint_path=ckpt_path,
            override_filesystem=fs,
            override_backend=backend,
        )
    else:
        ctx.checkpoint_config = CheckpointConfig(
            id_column=ID_COL,
            checkpoint_path=ckpt_path,
            override_filesystem=fs,
            override_backend=backend,
        )

    csv_file = generate_sample_data_csv()

    if read_code_path == "runtime":
        ds = ray.data.read_csv(csv_file)
    elif read_code_path == "oss_fallback":
        ds = ray.data.read_api.read_csv(csv_file)
    else:
        raise Exception(f"Invalid `read_code_path`: {read_code_path}")

    # Execute the dataset with checkpointing enabled.
    ds = ds.map_batches(TestActor, concurrency=1)
    data_output_path = os.path.join(data_path, "output")

    if generated_id_column is not None:
        # For CSV datasets with generated_id_column, the read operation fails with an AssertionError
        # because CSV datasources don't support auto-generated row IDs
        with pytest.raises(
            AssertionError,
            match="For checkpointing with `generated_id_column`, .* operator must use a ParquetReader",
        ):
            ds.write_parquet(data_output_path, filesystem=fs)
        pytest.skip("`generated_id_column` is not supported for CSV datasets")
    else:
        ds.write_parquet(data_output_path, filesystem=fs)

    # Ensure that the written data is correct.
    ds_readback = ray.data.read_parquet(data_output_path, filesystem=fs)
    actual_output = sorted([row[ID_COL] for row in ds_readback.iter_rows()])
    expected_output = sorted([row[ID_COL] for row in ds.iter_rows()])
    assert actual_output == expected_output

    # When execution succeeds, checkpoint data should be automatically deleted.
    # TODO(haochen): Also delete checkpoint for row-based backends.
    ctx.checkpoint_enabled_override = False
    checkpoint_ids = read_ids_from_checkpoint_files(ctx.checkpoint_config)
    if ctx.checkpoint_config.is_batch_based():
        assert checkpoint_ids == []
    else:
        expected_checkpoint_ids = sorted([row[ID_COL] for row in ds.iter_rows()])
        assert checkpoint_ids == expected_checkpoint_ids


@pytest.mark.parametrize("read_code_path", ["runtime", "oss_fallback"])
@pytest.mark.parametrize("generated_id_column", [None, "row_id"])
@pytest.mark.parametrize(
    "backend,fs,data_path",
    [
        (CheckpointBackend.FILE_STORAGE_ROW, None, lazy_fixture("local_path")),
        (
            CheckpointBackend.FILE_STORAGE_ROW,
            lazy_fixture("local_fs"),
            lazy_fixture("local_path"),
        ),
        (
            CheckpointBackend.CLOUD_OBJECT_STORAGE_ROW,
            lazy_fixture("s3_fs"),
            lazy_fixture("s3_path"),
        ),
        (CheckpointBackend.FILE_STORAGE, None, lazy_fixture("local_path")),
        (
            CheckpointBackend.FILE_STORAGE,
            lazy_fixture("local_fs"),
            lazy_fixture("local_path"),
        ),
        (
            CheckpointBackend.CLOUD_OBJECT_STORAGE,
            lazy_fixture("s3_fs"),
            lazy_fixture("s3_path"),
        ),
    ],
)
def test_full_dataset_executed_for_non_write(
    ray_start_10_cpus_shared,
    generate_sample_data_parquet,
    read_code_path,
    backend,
    fs,
    data_path,
    generated_id_column,
):
    """Tests that for an already fully checkpointed Dataset,
    calling `schema()` and `count()` should not skip checkpointing
    and should execute the full Dataset to get the correct information.
    """

    ctx = ray.data.DataContext.get_current()
    ckpt_path = os.path.join(data_path, "test_checkpoint_output_files")

    if generated_id_column is not None:
        ctx.checkpoint_config = CheckpointConfig(
            generated_id_column=generated_id_column,
            checkpoint_path=ckpt_path,
            override_filesystem=fs,
            override_backend=backend,
        )
    else:
        ctx.checkpoint_config = CheckpointConfig(
            id_column=ID_COL,
            checkpoint_path=ckpt_path,
            override_filesystem=fs,
            override_backend=backend,
        )

    parquet_dir = generate_sample_data_parquet()

    error_expected = (
        read_code_path == "oss_fallback" and generated_id_column is not None
    )
    if read_code_path == "runtime":
        ds = ray.data.read_parquet(parquet_dir)
    elif read_code_path == "oss_fallback":
        ds = ray.data.read_api.read_parquet(parquet_dir, concurrency=1)

    ds = ds.map(lambda row: row)

    # Get the schema and count prior to writing the dataset.
    schema_before_write = ds.schema()
    count_before_write = ds.count()

    data_output_path = os.path.join(data_path, "output")
    if error_expected:
        # For generated_id_column with oss_fallback, the read operation fails with an AssertionError
        # because the datasource is not a ParquetReader
        with pytest.raises(
            AssertionError,
            match="For checkpointing with `generated_id_column`, Read operator must use a ParquetReader",
        ):
            ds.write_parquet(data_output_path, filesystem=fs)
        pytest.skip(
            "`generated_id_column` is not supported for Parquet datasets with OSS fallback"
        )
    else:
        ds.write_parquet(data_output_path, filesystem=fs)

    # Recreate the same dataset, so that it will skip checkpointed rows.
    ctx.checkpoint_enabled_override = False
    ds2 = ray.data.read_parquet(parquet_dir)
    ds2 = ds2.map(lambda row: row)

    # Check that when re-running a dataset which has already been completely
    # checkpointed, it does not skip any rows during `schema()` and `count()` calls.
    assert ds2.schema() == schema_before_write
    if generated_id_column is not None:
        assert generated_id_column in ds2.schema().names
    assert ds2.count() == count_before_write


@pytest.mark.parametrize(
    "ds_factory,generated_id_column",
    [
        (lazy_fixture("generate_sample_data_parquet"), None),
        (lazy_fixture("generate_sample_data_parquet"), "generated_id"),
    ],
)
@pytest.mark.parametrize(
    "backend,fs,data_path",
    [
        (CheckpointBackend.FILE_STORAGE_ROW, None, lazy_fixture("local_path")),
        (
            CheckpointBackend.FILE_STORAGE_ROW,
            lazy_fixture("local_fs"),
            lazy_fixture("local_path"),
        ),
        (
            CheckpointBackend.CLOUD_OBJECT_STORAGE_ROW,
            lazy_fixture("s3_fs"),
            lazy_fixture("s3_path"),
        ),
        (CheckpointBackend.FILE_STORAGE, None, lazy_fixture("local_path")),
        (
            CheckpointBackend.FILE_STORAGE,
            lazy_fixture("local_fs"),
            lazy_fixture("local_path"),
        ),
        (
            CheckpointBackend.CLOUD_OBJECT_STORAGE,
            lazy_fixture("s3_fs"),
            lazy_fixture("s3_path"),
        ),
    ],
)
def test_recovery_skips_checkpointed_rows(
    ray_start_10_cpus_shared,
    ds_factory,
    backend,
    fs,
    data_path,
    generated_id_column,
):
    """Tests that for a Dataset which fails partway and is recovered,
    it skips rows which have already been checkpointed."""

    ctx = ray.data.DataContext.get_current()
    ctx.execution_options.preserve_order = True
    ckpt_path = os.path.join(data_path, "test_checkpoint_output_files")

    # Ensure checkpoint directory exists
    os.makedirs(ckpt_path, exist_ok=True)

    if generated_id_column is not None:
        ctx.checkpoint_config = CheckpointConfig(
            generated_id_column=generated_id_column,
            checkpoint_path=ckpt_path,
            override_filesystem=fs,
            override_backend=backend,
        )
    else:
        ctx.checkpoint_config = CheckpointConfig(
            id_column=ID_COL,
            checkpoint_path=ckpt_path,
            override_filesystem=fs,
            override_backend=backend,
        )

    # Catch the custom TestException raised by FailActor.
    ctx.raise_original_map_exception = True

    @ray.remote(num_cpus=0)
    class Coordinator:
        def __init__(self):
            self._should_fail = True

        def disable_failure(self):
            self._should_fail = False

        def should_fail(self):
            return self._should_fail

    coordinator_actor = Coordinator.remote()

    class TestException(Exception):
        pass

    class FailActor:
        """Simple passthrough actor, which fails after a certain number of rows."""

        def __init__(self, coordinator_actor, max_num_items, checkpoint_config):
            self._should_fail = ray.get(coordinator_actor.should_fail.remote())
            self._max_num_items = max_num_items
            self._checkpoint_config = checkpoint_config

        def __call__(self, batch):
            # Get the ID column name from the checkpoint config
            id_col = self._checkpoint_config.id_column

            # Process each row in the batch
            ids = batch[id_col]

            for i, id in enumerate(ids):
                # Extract numeric ID - handle both string and integer cases
                if isinstance(id, str):
                    # String case: '/sample_data.parquet/2' -> 2
                    numeric_id = int(id.split("/")[-1])
                else:
                    # Integer case: already a number
                    numeric_id = int(id)

                if self._should_fail and numeric_id == 2:
                    raise TestException(f"FailActor: Failing on row {id}")

            return batch

        def _wait_until_checkpoint_written(self):
            checkpointed_ids = set(
                read_ids_from_checkpoint_files(self._checkpoint_config)
            )

            # Wait for any checkpointing to happen
            return len(checkpointed_ids) > 0

    # Use the ds_factory to create the dataset
    local_data_path = ds_factory()
    ds = ray.data.read_parquet(local_data_path)

    # Get the actual number of items from the dataset
    max_num_items = ds.count()

    ds = ds.map_batches(
        FailActor,
        fn_constructor_args=[coordinator_actor, max_num_items, ctx.checkpoint_config],
        concurrency=1,
        batch_size=None,
        num_cpus=1.1,  # Use a different num_cpus to avoid operater fusion.
    )

    data_output_path = os.path.join(data_path, "output")
    # Should fail in the middle.
    with pytest.raises(TestException):
        ds.write_parquet(data_output_path, filesystem=fs, concurrency=1)

    ray.get(coordinator_actor.disable_failure.remote())
    # When executing the same dataset again, this should skip the already
    # checkpointed rows.
    ds.write_parquet(data_output_path, filesystem=fs, concurrency=1)

    # When execution succeeds, checkpoint data should be automatically deleted.
    # TODO(haochen): Also delete checkpoint for row-based backends.
    ctx.checkpoint_enabled_override = False
    if ctx.checkpoint_config.is_batch_based():
        assert read_ids_from_checkpoint_files(ctx.checkpoint_config) == []
    else:
        # For row-based backends, check that all rows are checkpointed
        checkpointed_ids = read_ids_from_checkpoint_files(ctx.checkpoint_config)
        if generated_id_column is not None:
            # For generated IDs, we expect string IDs with absolute paths
            # The exact paths depend on the temporary directory, so we just check the count
            assert len(checkpointed_ids) == max_num_items
        else:
            # For existing ID column, expect integer IDs
            assert checkpointed_ids == list(range(max_num_items))

    # Get the ID column name from the checkpoint config
    id_col = ctx.checkpoint_config.id_column

    # Disable checkpointing prior to reading back the data, so we don't skip any rows.
    ctx.checkpoint_config = None

    # Ensure that the written data is correct.
    ds_readback = ray.data.read_parquet(data_output_path, filesystem=fs)

    actual_output = sorted([row[id_col] for row in ds_readback.iter_rows()])

    # Handle both integer and string ID cases
    if generated_id_column is not None:
        # For generated_id_column, expect string IDs with absolute path like
        # '/tmp/.../sample_data.parquet/0'
        # Get the actual path from the dataset to construct expected IDs
        actual_paths = [row[id_col] for row in ds_readback.iter_rows()]
        # Extract the base path from the first actual path
        first_path = actual_paths[0]
        # Remove the row number and add trailing slash
        base_path = first_path.rsplit("/", 1)[0] + "/"
        expected_output = sorted([f"{base_path}{i}" for i in range(max_num_items)])
    else:
        # For existing id column, expect integer IDs
        expected_output = sorted(range(max_num_items))

    assert actual_output == expected_output


@pytest.mark.parametrize("read_code_path", ["runtime", "oss_fallback"])
@pytest.mark.parametrize(
    "backend,fs,data_path",
    [
        (CheckpointBackend.FILE_STORAGE_ROW, None, lazy_fixture("local_path")),
        (
            CheckpointBackend.FILE_STORAGE_ROW,
            lazy_fixture("local_fs"),
            lazy_fixture("local_path"),
        ),
        (
            CheckpointBackend.CLOUD_OBJECT_STORAGE_ROW,
            lazy_fixture("s3_fs"),
            lazy_fixture("s3_path"),
        ),
        (CheckpointBackend.FILE_STORAGE, None, lazy_fixture("local_path")),
        (
            CheckpointBackend.FILE_STORAGE,
            lazy_fixture("local_fs"),
            lazy_fixture("local_path"),
        ),
        (
            CheckpointBackend.CLOUD_OBJECT_STORAGE,
            lazy_fixture("s3_fs"),
            lazy_fixture("s3_path"),
        ),
    ],
)
def test_skip_checkpoint_flag(
    ray_start_10_cpus_shared,
    generate_sample_data_csv,
    read_code_path,
    backend,
    fs,
    data_path,
):
    """Test that for a valid Dataset with checkpointing enabled, calling methods like
    `schema()` and `count()` should skip checkpointing and not create any checkpoint
    files. Subsequently calling `write_xxx()` on the same dataset should have
    checkpointing enabled."""

    ctx = ray.data.DataContext.get_current()
    ckpt_path = os.path.join(data_path, "test_checkpoint_output_files")
    ctx.checkpoint_config = CheckpointConfig(
        ID_COL,
        ckpt_path,
        delete_checkpoint_on_success=False,
        override_filesystem=fs,
        override_backend=backend,
    )

    def generate_ds():
        if read_code_path == "runtime":
            ds = ray.data.read_csv(generate_sample_data_csv())
        elif read_code_path == "oss_fallback":
            ds = ray.data.read_api.read_csv(generate_sample_data_csv())

        ds = ds.map(lambda row: row)
        return ds

    ds = generate_ds()

    # Calling `ds.schema()` should skip checkpointing.
    assert ds.schema() is not None
    assert len(read_ids_from_checkpoint_files(ctx.checkpoint_config)) == 0

    # Calling `ds.count()` should skip checkpointing.
    ds = generate_ds()
    assert ds.count() is not None
    assert len(read_ids_from_checkpoint_files(ctx.checkpoint_config)) == 0

    # Calling `ds.write_xxx()` afterwards should enable checkpointing.
    ds.write_parquet(os.path.join(data_path, "output"), filesystem=fs)

    # Check what checkpoint files exist
    checkpoint_files = read_ids_from_checkpoint_files(ctx.checkpoint_config)

    assert len(checkpoint_files) == SAMPLE_DATA_NUM_ROWS


def test_checkpoint_with_missing_id_column(
    ray_start_10_cpus_shared,
    generate_sample_data_csv,
    local_path,
):
    """Test that checkpointing fails gracefully when the configured id_column doesn't exist in the data."""

    ctx = ray.data.DataContext.get_current()
    ckpt_path = os.path.join(local_path, "test_checkpoint_output_files")
    # Configure checkpointing with an id_column that doesn't exist in the CSV data
    ctx.checkpoint_config = CheckpointConfig(
        id_column="nonexistent_column",
        checkpoint_path=ckpt_path,
        delete_checkpoint_on_success=False,
    )

    def generate_ds():
        ds = ray.data.read_csv(generate_sample_data_csv())
        ds = ds.map(lambda row: row)
        return ds

    ds = generate_ds()
    data_output_path = os.path.join(local_path, "output")

    # The write operation should fail because the id_column doesn't exist
    with pytest.raises(
        ValueError,
        match="ID column nonexistent_column is absent in the block to be written",
    ):
        ds.write_parquet(data_output_path)


def test_dict_checkpoint_config(checkpoint_path):
    """Test that a dict checkpoint config can be used to create a CheckpointConfig."""
    context = ray.data.DataContext.get_current()
    fs = LocalFileSystem()
    context.checkpoint_config = {
        "id_column": ID_COL,
        "checkpoint_path": checkpoint_path,
        "override_filesystem": fs,
        "override_backend": "CLOUD_OBJECT_STORAGE_ROW",
    }
    assert context.checkpoint_config.id_column == ID_COL
    assert context.checkpoint_config.checkpoint_path == checkpoint_path
    assert context.checkpoint_config.filesystem is fs
    assert (
        context.checkpoint_config.backend == CheckpointBackend.CLOUD_OBJECT_STORAGE_ROW
    )


class TestPlanner:
    def test_plan_from_op_with_checkpoint_filter(self):
        op = AbstractFrom([], [])

        physical_op = plan_from_op_with_checkpoint_filter(
            op, [], ray.data.DataContext.get_current()
        )

        # TODO: (Here and elsewhere) testing against representations is brittle. We
        # should expose a seam to enable more explicit testing.
        assert "FilterCheckpointedRows" in physical_op.dag_str

    def test_plan_read_files_op_with_checkpoint_filter(self):
        input_data = InputData([])
        op = ReadFiles(
            input_data, reader=MagicMock(spec=FileReader), filesystem=MagicMock()
        )

        input_data_buffer = InputDataBuffer(ray.data.DataContext.get_current(), [])
        physical_op = plan_read_files_op_with_checkpoint_filter(
            op, [input_data_buffer], ray.data.DataContext.get_current()
        )

        assert "filter_checkpointed_rows" in str(
            physical_op._map_transformer._transform_fns
        )

    def test_plan_read_op_with_checkpoint_filter(self):
        op = Read(MagicMock(spec=Datasource), None, -1, None)

        physical_op = plan_read_op_with_checkpoint_filter(
            op, [], ray.data.DataContext.get_current()
        )

        assert "filter_checkpointed_rows" in str(
            physical_op._map_transformer._transform_fns
        )

    def test_plan_write_op_with_checkpoint_writer(self, checkpoint_path):
        class FakeDatasink(Datasink):
            def write(self, blocks, ctx):
                return None

        # Configure checkpointing.
        ctx = ray.data.DataContext.get_current()
        ctx.checkpoint_config = CheckpointConfig(ID_COL, checkpoint_path)

        # Construct a logical DAG.
        input_data = InputData([])
        op = Write(input_data, FakeDatasink())

        # Plan the physical DAG.
        input_data_buffer = InputDataBuffer(ray.data.DataContext.get_current(), [])
        physical_op = plan_write_op_with_checkpoint_writer(
            op, [input_data_buffer], ray.data.DataContext.get_current()
        )

        # Verify that the checkpoint writer is inserted.
        assert "write_checkpoint_for_block" in str(
            physical_op._map_transformer._transform_fns
        )


def create_string_test_data(row_id: int, path: str) -> str:
    """Helper function to create string test data for row IDs."""
    return f"{path}/{row_id}"


@pytest.mark.parametrize("generated_id_column", [False, True])
def test_write_block_checkpoint_with_pandas_df(
    restore_data_context, tmp_path, generated_id_column
):
    ctx = ray.data.DataContext.get_current()

    if generated_id_column:
        ctx.checkpoint_config = CheckpointConfig(
            generated_id_column=GENERATED_ID_COL,
            checkpoint_path=str(tmp_path),
        )
        # For struct IDs, we need to create a DataFrame with struct data
        df = pd.DataFrame(
            {
                GENERATED_ID_COL: [
                    create_string_test_data(0, "/data/file1.parquet"),
                    create_string_test_data(1, "/data/file2.parquet"),
                ]
            }
        )
        expected_ids = [
            create_string_test_data(0, "/data/file1.parquet"),
            create_string_test_data(1, "/data/file2.parquet"),
        ]
    else:
        ctx.checkpoint_config = CheckpointConfig(
            ID_COL,
            str(tmp_path),
        )
        df = pd.DataFrame({ID_COL: [0, 1]})
        expected_ids = [0, 1]

    checkpoint_writer = BatchBasedCheckpointWriter(ctx.checkpoint_config)
    checkpoint_writer.write_block_checkpoint(BlockAccessor.for_block(df))

    assert len(os.listdir(tmp_path)) == 1
    checkpoint_filename = os.listdir(tmp_path)[0]
    checkpoint_path = tmp_path / checkpoint_filename
    written_ids = pd.read_parquet(checkpoint_path)[
        GENERATED_ID_COL if generated_id_column else ID_COL
    ].tolist()
    assert written_ids == expected_ids


@pytest.mark.parametrize("generated_id_column", [False, True])
def test_filter_rows_for_block(generated_id_column):
    """Test BatchBasedCheckpointFilter.filter_rows_for_block."""

    # Common test setup
    checkpoint_path = "/mock/path"

    if generated_id_column:
        # Test with struct ID column (generated_id_column)
        config = CheckpointConfig(
            generated_id_column=GENERATED_ID_COL,
            checkpoint_path=checkpoint_path,
        )

        # Create a mock block with string ID column
        # Each string contains /path/to/file/row_id
        string_data = [
            create_string_test_data(0, "/data/file1.parquet"),
            create_string_test_data(1, "/data/file1.parquet"),
            create_string_test_data(2, "/data/file1.parquet"),
            create_string_test_data(0, "/data/file2.parquet"),  # Different file
            create_string_test_data(1, "/data/file2.parquet"),
            create_string_test_data(2, "/data/file2.parquet"),
        ]

        block = pyarrow.table(
            {
                GENERATED_ID_COL: string_data,
                "data": [str(i) for i in range(6)],
            }
        )

        # Create a mock checkpointed_ids with string columns
        # Checkpointed: row_id=1 from file1.parquet, row_id=0 from file2.parquet
        checkpointed_string_data = [
            create_string_test_data(1, "/data/file1.parquet"),
            create_string_test_data(0, "/data/file2.parquet"),
        ]

        chunk1 = pyarrow.table({GENERATED_ID_COL: [checkpointed_string_data[0]]})
        chunk2 = pyarrow.table({GENERATED_ID_COL: [checkpointed_string_data[1]]})
        checkpointed_ids = pyarrow.concat_tables([chunk1, chunk2])
        assert len(checkpointed_ids[GENERATED_ID_COL].chunks) == 2

        # Expected: keep row_id=0,2 from file1.parquet and row_id=1,2 from file2.parquet
        expected_string_data = [
            create_string_test_data(0, "/data/file1.parquet"),
            create_string_test_data(2, "/data/file1.parquet"),
            create_string_test_data(1, "/data/file2.parquet"),
            create_string_test_data(2, "/data/file2.parquet"),
        ]

        expected_block = pyarrow.table(
            {
                GENERATED_ID_COL: expected_string_data,
                "data": ["0", "2", "4", "5"],
            }
        )
    else:
        # Test with simple ID column
        config = CheckpointConfig(
            id_column=ID_COL,
            checkpoint_path=checkpoint_path,
        )

        # Create a mock block.
        block = pyarrow.table(
            {
                ID_COL: list(range(10)),
                "data": [str(i) for i in range(10)],
            }
        )
        # Create a mock checkpointed_ids with multiple chunks.
        chunk1 = pyarrow.table({ID_COL: [1, 2, 4]})
        chunk2 = pyarrow.table({ID_COL: [6, 8, 9, 11]})
        chunk3 = pyarrow.table({ID_COL: [12, 13]})
        checkpointed_ids = pyarrow.concat_tables([chunk1, chunk2, chunk3])
        assert len(checkpointed_ids[ID_COL].chunks) == 3

        expected_block = pyarrow.table(
            {
                ID_COL: [0, 3, 5, 7],
                "data": ["0", "3", "5", "7"],
            }
        )

    # Common test execution and verification
    filter_instance = BatchBasedCheckpointFilter(config)
    filtered_block = filter_instance.filter_rows_for_block(block, checkpointed_ids)
    assert filtered_block.equals(expected_block)


@pytest.mark.parametrize("generated_id_column", [False, True])
def test_checkpoint_restore_after_full_execution(
    ray_start_10_cpus_shared,
    tmp_path,
    generate_sample_data_parquet,
    checkpoint_path,
    generated_id_column,
):
    """Test checkpoint restore after full execution of data pipeline. This is
    done by retaining the checkpoint metadata files with
    delete_checkpoint_on_success=False.
    """

    def run_simple_pipeline(
        checkpoint_config: CheckpointConfig, input_path: str, output_path: str
    ) -> int:
        """Run a simple pipeline with checkpointing."""
        ctx = DataContext.get_current()
        ctx.checkpoint_config = checkpoint_config
        ctx.checkpoint_enabled_override = False
        ds = ray.data.read_parquet(input_path)
        ds.write_parquet(output_path)
        return ds.count()

    # Create test paths
    input_data_path = generate_sample_data_parquet()
    data_output_path = str(tmp_path / "output")

    # Create checkpoint config
    checkpoint_config = CheckpointConfig(
        generated_id_column=GENERATED_ID_COL if generated_id_column else None,
        id_column=None if generated_id_column else ID_COL,
        checkpoint_path=checkpoint_path,
        override_backend=CheckpointBackend.FILE_STORAGE,
        delete_checkpoint_on_success=False,
    )

    # First run: create checkpoint
    num_rows_first = run_simple_pipeline(
        checkpoint_config, input_data_path, data_output_path
    )
    assert (
        num_rows_first == SAMPLE_DATA_NUM_ROWS
    ), f"Expected {SAMPLE_DATA_NUM_ROWS} rows, got {num_rows_first}"

    # Check if checkpoint files were created
    assert os.path.exists(checkpoint_path), "No checkpoint directory created!"

    # Second run: should use checkpoint
    num_rows_second = run_simple_pipeline(
        checkpoint_config, input_data_path, data_output_path
    )
    assert (
        num_rows_second == SAMPLE_DATA_NUM_ROWS
    ), f"Expected {SAMPLE_DATA_NUM_ROWS} rows, got {num_rows_second}"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
