import csv
import os
import pathlib
import random
import urllib.parse
from typing import List, Union
from unittest.mock import MagicMock

import pandas as pd
import pyarrow
import pyarrow as pa
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
from ray.anyscale.data.checkpoint.checkpoint_filter import (
    BatchBasedCheckpointFilter,
)
from ray.anyscale.data.checkpoint.util import (
    get_generated_id_column,
    ROW_ID_FIELD,
    PATH_PREFIX_FIELD,
    FILE_NAME_FIELD,
    ROW_GROUP_FIELD,
    NUM_ROWS_FIELD,
    GENERATED_ID_COLUMN_FIELD_NAMES,
    CHECKPOINTED_FRAGMENT_COLUMN_NAME,
    CHECKPOINTED_ROW_COUNT_COLUMN_NAME,
    CHECKPOINTED_ROW_IDS_COLUMN_NAME,
    get_checkpointed_fragment_info,
    exclude_checkpointed_rows,
    GENERATED_ID_COLUMN_TYPE,
    normalize_id,
)

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
from ray.data.block import BlockAccessor, Block
from ray.types import ObjectRef
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
SAMPLE_DATA_NUM_ROWS = 10

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
def data_output_path(data_path):
    """Fixture to provide a standardized data output path."""
    return os.path.join(data_path, "output")


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
        # Write 3 row groups per file with uneven distribution of rows per row group
        df.to_parquet(f_path, row_group_size=SAMPLE_DATA_NUM_ROWS // 3)
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
                parsed_ids,
                key=lambda x: int(urllib.parse.unquote(x).split("row_id=")[-1]),
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
                generated_id_column=GENERATED_ID_COL,
            )


@pytest.mark.parametrize("read_code_path", ["runtime", "oss_fallback"])
@pytest.mark.parametrize("generated_id_column", [None, GENERATED_ID_COL])
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
    data_output_path,
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
@pytest.mark.parametrize("generated_id_column", [None, GENERATED_ID_COL])
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
    data_output_path,
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
        (lazy_fixture("generate_sample_data_parquet"), GENERATED_ID_COL),
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
    data_output_path,
    generated_id_column,
):
    """Tests that for a Dataset which fails partway and is recovered,
    it skips rows which have already been checkpointed."""

    ctx = ray.data.DataContext.get_current()
    ctx.execution_options.preserve_order = True
    ctx.default_hash_shuffle_parallelism = 1
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

            for _, id in enumerate(ids):
                # Extract numeric ID - handle both dict and integer cases
                if isinstance(id, dict):
                    # Dict case: extract row_id from dict
                    numeric_id = id[ROW_ID_FIELD]
                else:
                    # Integer case: already a number
                    numeric_id = int(id)

                if self._should_fail and numeric_id == 2:
                    raise TestException(f"FailActor: Failing on row {id}")

            return batch

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

    # Handle both integer and dict ID cases
    if generated_id_column is not None:
        # For generated_id_column, expect complete dict IDs
        # Get the actual complete dicts from the dataset
        actual_dicts = sorted(
            [row[id_col] for row in ds_readback.iter_rows()],
            key=lambda d: (d[ROW_GROUP_FIELD], d[ROW_ID_FIELD]),
        )

        # With multiple row groups, generate expected complete dicts
        # With SAMPLE_DATA_NUM_ROWS=10 and row_group_size=3,
        # we get 4 row groups: [3,3,3,1]
        rows_per_group = SAMPLE_DATA_NUM_ROWS // 3
        last_group_rows = SAMPLE_DATA_NUM_ROWS % 3

        actual_path_prefix = actual_dicts[0][PATH_PREFIX_FIELD]
        actual_file_name = actual_dicts[0][FILE_NAME_FIELD]

        # Build list of (row_group_idx, num_rows_in_group) for all row groups
        row_groups = [(i, rows_per_group) for i in range(rows_per_group)]
        if last_group_rows > 0:
            row_groups.append((rows_per_group, last_group_rows))

        # Generate all expected dicts
        expected_dicts = [
            {
                PATH_PREFIX_FIELD: actual_path_prefix,
                FILE_NAME_FIELD: actual_file_name,
                ROW_GROUP_FIELD: row_group_idx,
                NUM_ROWS_FIELD: num_rows_in_group,
                ROW_ID_FIELD: row_id,
            }
            for row_group_idx, num_rows_in_group in row_groups
            for row_id in range(num_rows_in_group)
        ]

        expected_dicts = sorted(
            expected_dicts, key=lambda d: (d[ROW_GROUP_FIELD], d[ROW_ID_FIELD])
        )

        assert actual_dicts == expected_dicts
    else:
        # For existing id column, expect integer IDs
        actual_output = sorted([row[id_col] for row in ds_readback.iter_rows()])
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
    tmp_path,
):
    """Test that checkpointing fails gracefully when the configured id_column doesn't exist in the data."""

    ctx = ray.data.DataContext.get_current()
    ckpt_path = os.path.join(tmp_path, "test_checkpoint_output_files")
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

    # The write operation should fail because the id_column doesn't exist
    with pytest.raises(
        ValueError,
        match="ID column nonexistent_column is absent in the block to be written",
    ):
        ds.write_parquet(os.path.join(tmp_path, "output"))


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
        struct_array = get_generated_id_column(
            path="/data/file1.parquet",
            row_group_idx=0,
            total_num_rows=2,
            current_row_offset=0,
            current_num_rows=2,
        )
        assert struct_array.type == GENERATED_ID_COLUMN_TYPE
        df = pyarrow.table({GENERATED_ID_COL: struct_array}).to_pandas()
        expected_ids = [
            normalize_id(struct_array[0].as_py()),
            normalize_id(struct_array[1].as_py()),
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
    if generated_id_column:
        # For generated IDs, reconstruct the normalized ID from the individual fields
        df = pd.read_parquet(checkpoint_path)
        written_ids = [normalize_id(row) for row in df[GENERATED_ID_COL]]
    else:
        # For regular IDs, read the original ID column
        written_ids = pd.read_parquet(checkpoint_path)[ID_COL].tolist()
    assert written_ids == expected_ids


def test_filter_rows_for_block():
    """Test BatchBasedCheckpointFilter.filter_rows_for_block."""

    # Common test setup
    checkpoint_path = "/mock/path"

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

    filter_instance = BatchBasedCheckpointFilter(config)
    filtered_block = filter_instance.filter_rows_for_block(
        block=block,
        checkpointed_ids=checkpointed_ids,
    )

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
        from ray.data.datasource import WriteResult

        ctx = DataContext.get_current()
        ctx.checkpoint_config = checkpoint_config
        ctx.checkpoint_enabled_override = False
        ctx.default_hash_shuffle_parallelism = 1
        ds = ray.data.read_parquet(input_path)

        # Patch `on_write_complete` to get the WriteResult.
        num_rows_written = None
        original_on_write_complete = ParquetDatasink.on_write_complete

        def patched_on_write_complete(self, write_result: WriteResult[None]):
            nonlocal num_rows_written
            num_rows_written = write_result.num_rows
            return original_on_write_complete(self, write_result)

        ParquetDatasink.on_write_complete = patched_on_write_complete

        ds.write_parquet(output_path)
        return int(num_rows_written)

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
        num_rows_second == 0  # No rows should be written
    ), f"Expected 0 rows, got {num_rows_second}"


class TestCheckpointFragmentRestore:
    """Test the get_checkpointed_fragment_info and exclude_checkpointed_rows functions."""

    def _create_checkpointed_table(
        self,
        fragment_ids: list[str],
        row_ids_per_fragment: list[list[int]],
        expected_row_counts: list[int],
    ) -> pyarrow.Table:
        """Helper to create checkpointed_ids table with fragment IDs.

        Args:
            fragment_ids: List of fragment IDs in format /path/to/file/row_group=X
            row_ids_per_fragment: List of row ID lists for each fragment
            expected_row_counts: List of expected total row counts per fragment

        Returns:
            pyarrow.Table: A table containing checkpointed fragment information with columns:
                - CHECKPOINTED_FRAGMENT_COLUMN_NAME: sorted fragment IDs
                - NUM_ROWS_FIELD: expected row counts per fragment
                - CHECKPOINTED_ROW_COUNT_COLUMN_NAME: actual checkpointed row counts per fragment
                - CHECKPOINTED_ROW_IDS_COLUMN_NAME: list of checkpointed row IDs per fragment
        """
        if not fragment_ids:
            return pyarrow.table(
                {
                    CHECKPOINTED_FRAGMENT_COLUMN_NAME: [],
                    NUM_ROWS_FIELD: [],
                    CHECKPOINTED_ROW_COUNT_COLUMN_NAME: [],
                    CHECKPOINTED_ROW_IDS_COLUMN_NAME: [],
                }
            )

        # Sort the fragments for binary search to work correctly
        sorted_fragment_ids = sorted(fragment_ids)

        return pyarrow.table(
            {
                CHECKPOINTED_FRAGMENT_COLUMN_NAME: sorted_fragment_ids,
                NUM_ROWS_FIELD: pyarrow.array(expected_row_counts),
                CHECKPOINTED_ROW_COUNT_COLUMN_NAME: pyarrow.array(
                    [
                        # If row_ids is empty, it means all rows are checkpointed
                        expected_row_counts[i] if len(row_ids) == 0 else len(row_ids)
                        for i, row_ids in enumerate(row_ids_per_fragment)
                    ]
                ),
                CHECKPOINTED_ROW_IDS_COLUMN_NAME: pyarrow.array(
                    [
                        []
                        if len(row_ids) == 0
                        else [i in row_ids for i in range(expected_row_counts[i])]
                        for i, row_ids in enumerate(row_ids_per_fragment)
                    ],
                    type=pyarrow.list_(pyarrow.bool_()),
                ),
            }
        )

    def _create_fragment(self, path: str, expected_num_rows: int = 3) -> MagicMock:
        """Helper to create mock fragment."""
        fragment = MagicMock()
        fragment.path = path
        # Set up metadata.num_rows and metadata.row_group().num_rows for the mock fragment
        fragment.metadata = MagicMock()
        fragment.metadata.num_rows = expected_num_rows
        row_group_mock = MagicMock()
        row_group_mock.num_rows = expected_num_rows
        fragment.metadata.row_group.return_value = row_group_mock
        return fragment

    def _test_get_checkpointed_fragment_info(
        self,
        fragment_path: str,
        row_group_idx: int,
        checkpointed_fragments: Union[pyarrow.Table, None],
        expected_fully_checkpointed: bool,
        expected_checkpointed_row_ids: list[int],
        expected_num_rows: int = 0,
    ) -> None:
        """Helper to test get_checkpointed_fragment_info with given inputs."""
        fragment = self._create_fragment(fragment_path, expected_num_rows)
        result = get_checkpointed_fragment_info(
            fragment=fragment,
            row_group_idx=row_group_idx,
            checkpointed_ids=checkpointed_fragments,
        )
        assert result.fragment.path == fragment_path
        assert result.fully_checkpointed is expected_fully_checkpointed
        assert result.num_rows == expected_num_rows

        # Validate checkpointed_row_ids
        if result.checkpointed_row_ids is None:
            # Handle None case - no checkpoint data
            actual_row_ids = []
        elif len(result.checkpointed_row_ids) == 0:
            # Handle empty arrays
            actual_row_ids = []

        else:
            # Handle boolean array - convert True positions to row IDs
            actual_row_ids = [
                i
                for i, is_checkpointed in enumerate(
                    result.checkpointed_row_ids[0].as_py()
                )
                if is_checkpointed
            ]

        assert actual_row_ids == expected_checkpointed_row_ids, (
            f"Expected checkpointed row IDs {expected_checkpointed_row_ids}, "
            f"but got {actual_row_ids}"
        )

    def _generate_fragment_ids(
        self, base_path: str, file_name: str, row_groups: list[int]
    ) -> list[str]:
        """Helper to generate fragment IDs for a file with multiple row groups."""
        return [f"{base_path}/{file_name}/row_group={rg}" for rg in row_groups]

    def test_get_checkpointed_fragment_info_full_match_multiple_files_row_groups(
        self,
    ) -> None:
        """Test full match with multiple files and row groups - all rows checkpointed."""
        # Create checkpointed table with multiple files and row groups
        all_fragments = (
            self._generate_fragment_ids(
                base_path="/data",
                file_name="file1.parquet",
                row_groups=[0, 1, 2],
            )
            + self._generate_fragment_ids(
                base_path="/data",
                file_name="file2.parquet",
                row_groups=[0, 1],
            )
            + self._generate_fragment_ids(
                base_path="/data",
                file_name="file3.parquet",
                row_groups=[0],
            )
        )
        checkpointed_table = self._create_checkpointed_table(
            fragment_ids=all_fragments,
            # All rows checkpointed for each fragment - use empty lists when fully checkpointed
            row_ids_per_fragment=[
                [],  # file1, row_group=0 - fully checkpointed (3 rows)
                [],  # file1, row_group=1 - fully checkpointed (3 rows)
                [],  # file1, row_group=2 - fully checkpointed (3 rows)
                [],  # file2, row_group=0 - fully checkpointed (3 rows)
                [],  # file2, row_group=1 - fully checkpointed (3 rows)
                [],  # file3, row_group=0 - fully checkpointed (3 rows)
            ],
            # Expected total rows for each fragment
            expected_row_counts=[3, 3, 3, 3, 3, 3],
        )

        # Test that existing fragments are fully checkpointed
        self._test_get_checkpointed_fragment_info(
            fragment_path="/data/file1.parquet",
            row_group_idx=0,
            checkpointed_fragments=checkpointed_table,
            expected_fully_checkpointed=True,
            expected_num_rows=3,
            expected_checkpointed_row_ids=[],  # Empty when fully checkpointed
        )
        self._test_get_checkpointed_fragment_info(
            fragment_path="/data/file1.parquet",
            row_group_idx=1,
            checkpointed_fragments=checkpointed_table,
            expected_fully_checkpointed=True,
            expected_num_rows=3,
            expected_checkpointed_row_ids=[],  # Empty when fully checkpointed
        )
        self._test_get_checkpointed_fragment_info(
            fragment_path="/data/file1.parquet",
            row_group_idx=2,
            checkpointed_fragments=checkpointed_table,
            expected_fully_checkpointed=True,
            expected_num_rows=3,
            expected_checkpointed_row_ids=[],  # Empty when fully checkpointed
        )

        # Test that existing fragments are fully checkpointed
        self._test_get_checkpointed_fragment_info(
            fragment_path="/data/file2.parquet",
            row_group_idx=0,
            checkpointed_fragments=checkpointed_table,
            expected_fully_checkpointed=True,
            expected_num_rows=3,
            expected_checkpointed_row_ids=[],  # Empty when fully checkpointed
        )

        self._test_get_checkpointed_fragment_info(
            fragment_path="/data/file3.parquet",
            row_group_idx=0,
            checkpointed_fragments=checkpointed_table,
            expected_fully_checkpointed=True,
            expected_num_rows=3,
            expected_checkpointed_row_ids=[],  # Empty when fully checkpointed
        )

    def test_get_checkpointed_fragment_info_partial_match_multiple_files_row_groups(
        self,
    ) -> None:
        """Test partial match with multiple files and row groups - some rows checkpointed."""
        # Create checkpointed table with multiple files and row groups, but only some rows checkpointed
        all_fragments = (
            self._generate_fragment_ids(
                base_path="/data",
                file_name="file1.parquet",
                row_groups=[0, 1, 2],
            )
            + self._generate_fragment_ids(
                base_path="/data",
                file_name="file2.parquet",
                row_groups=[0, 1],
            )
            + self._generate_fragment_ids(
                base_path="/data",
                file_name="file3.parquet",
                row_groups=[0],
            )
        )
        checkpointed_table = self._create_checkpointed_table(
            fragment_ids=all_fragments,
            row_ids_per_fragment=[
                [0, 1, 2],  # file1, row_group=0
                [0, 1],  # file1, row_group=1
                [0, 1, 2],  # file1, row_group=2
                [0, 1],  # file2, row_group=0
                [0, 1],  # file2, row_group=1
                [0],  # file3, row_group=0
            ],  # Custom row IDs for 6 fragments
            expected_row_counts=[
                5,
                5,
                5,
                5,
                5,
                5,
            ],  # Expected total rows for each fragment (6 fragments)
        )

        # Test that fragments are partially checkpointed
        self._test_get_checkpointed_fragment_info(
            fragment_path="/data/file1.parquet",
            row_group_idx=0,
            checkpointed_fragments=checkpointed_table,
            expected_fully_checkpointed=False,
            expected_num_rows=5,
            expected_checkpointed_row_ids=[0, 1, 2],
        )
        self._test_get_checkpointed_fragment_info(
            fragment_path="/data/file1.parquet",
            row_group_idx=1,
            checkpointed_fragments=checkpointed_table,
            expected_fully_checkpointed=False,
            expected_num_rows=5,
            expected_checkpointed_row_ids=[0, 1],
        )
        self._test_get_checkpointed_fragment_info(
            fragment_path="/data/file1.parquet",
            row_group_idx=2,
            checkpointed_fragments=checkpointed_table,
            expected_fully_checkpointed=False,
            expected_num_rows=5,
            expected_checkpointed_row_ids=[0, 1, 2],
        )

        # Test that fragments are not checkpointed
        self._test_get_checkpointed_fragment_info(
            fragment_path="/data/file2.parquet",
            row_group_idx=0,
            checkpointed_fragments=checkpointed_table,
            expected_fully_checkpointed=False,
            expected_num_rows=5,
            expected_checkpointed_row_ids=[0, 1],
        )

        self._test_get_checkpointed_fragment_info(
            fragment_path="/data/file3.parquet",
            row_group_idx=0,
            checkpointed_fragments=checkpointed_table,
            expected_fully_checkpointed=False,
            expected_num_rows=5,
            expected_checkpointed_row_ids=[0],
        )

    def test_get_checkpointed_fragment_info_no_match(self) -> None:
        """Test no match scenarios - fragment not found in checkpointed data."""
        # Test with empty checkpointed_ids Block
        self._test_get_checkpointed_fragment_info(
            fragment_path="/data/file1.parquet",
            row_group_idx=0,
            checkpointed_fragments=pa.table({}),
            expected_fully_checkpointed=False,
            expected_checkpointed_row_ids=[],
            expected_num_rows=10,  # Should return actual fragment row count
        )

        # Test with empty checkpointed_ids
        empty_table = self._create_checkpointed_table([], [], expected_row_counts=[])
        self._test_get_checkpointed_fragment_info(
            fragment_path="/data/file1.parquet",
            row_group_idx=0,
            checkpointed_fragments=empty_table,
            expected_fully_checkpointed=False,
            expected_checkpointed_row_ids=[],
            expected_num_rows=10,  # Should return actual fragment row count
        )

        # Test with non-matching file path
        checkpointed_table = self._create_checkpointed_table(
            [
                "/data/file2.parquet/row_group=0",
                "/data/file3.parquet/row_group=0",
            ],
            [[0, 1, 2], [0, 1, 2]],  # All rows checkpointed for each fragment
            expected_row_counts=[3, 3],
        )
        self._test_get_checkpointed_fragment_info(
            fragment_path="/data/file1.parquet",
            row_group_idx=0,
            checkpointed_fragments=checkpointed_table,
            expected_fully_checkpointed=False,
            expected_checkpointed_row_ids=[],
            expected_num_rows=10,  # Should return actual fragment row count
        )

        # Test with matching file path but non-matching row group
        checkpointed_table = self._create_checkpointed_table(
            [
                "/data/file1.parquet/row_group=1",  # Different row group
            ],
            [[0, 1, 2]],  # All rows checkpointed
            expected_row_counts=[3],
        )
        self._test_get_checkpointed_fragment_info(
            fragment_path="/data/file1.parquet",
            row_group_idx=0,
            checkpointed_fragments=checkpointed_table,
            expected_fully_checkpointed=False,
            expected_checkpointed_row_ids=[],
            expected_num_rows=10,  # Should return actual fragment row count
        )

    def test_exclude_checkpointed_rows_full_match(self) -> None:
        """Test exclude_checkpointed_rows with full match - all rows checkpointed."""
        from ray.anyscale.data.checkpoint.util import (
            CheckpointedFragmentInfo,
        )

        # Create a test table with row IDs 0-9
        test_table = pyarrow.table(
            {"row_id": list(range(10)), "data": [f"data_{i}" for i in range(10)]}
        )

        # All rows are checkpointed - with optimization, this creates an empty ListArray
        fragment_info = CheckpointedFragmentInfo(
            fragment=None,  # Not used in this test
            row_group_idx=0,
            num_rows=10,
            fully_checkpointed=True,
            checkpointed_row_ids=pyarrow.array(
                [[]]
            ),  # Empty ListArray (optimized case)
            checkpointed_row_count=0,
        )

        result = exclude_checkpointed_rows(test_table, fragment_info, 0, 10)
        assert len(result) == 0  # No rows should remain - all checkpointed

    def test_exclude_checkpointed_rows_partial_match(self) -> None:
        """Test exclude_checkpointed_rows with partial match - some rows checkpointed."""
        from ray.anyscale.data.checkpoint.util import (
            CheckpointedFragmentInfo,
        )

        # Create a test table with row IDs 0-9
        test_table = pyarrow.table(
            {"row_id": list(range(10)), "data": [f"data_{i}" for i in range(10)]}
        )

        # Only some rows are checkpointed
        # Create boolean array: True = checkpointed, False = not checkpointed
        # Rows 2, 5, 8 are checkpointed, others are not
        boolean_array = [
            False,
            False,
            True,
            False,
            False,
            True,
            False,
            False,
            True,
            False,
        ]
        fragment_info = CheckpointedFragmentInfo(
            fragment=None,
            row_group_idx=0,
            num_rows=10,
            fully_checkpointed=False,
            checkpointed_row_ids=pyarrow.array(
                [boolean_array], type=pyarrow.list_(pyarrow.bool_())
            ),
            checkpointed_row_count=3,
        )

        result = exclude_checkpointed_rows(test_table, fragment_info, 0, 10)
        assert len(result) == 7  # 10 - 3 = 7 rows should remain

        # Check that checkpointed rows are excluded and non-checkpointed rows remain
        remaining_row_ids = result["row_id"].to_numpy()
        checkpointed_ids = [2, 5, 8]
        non_checkpointed_ids = [0, 1, 3, 4, 6, 7, 9]

        assert not any(
            rid in remaining_row_ids for rid in checkpointed_ids
        ), f"Checkpointed rows {checkpointed_ids} should be excluded"
        assert all(
            rid in remaining_row_ids for rid in non_checkpointed_ids
        ), f"Non-checkpointed rows {non_checkpointed_ids} should remain"

    def test_exclude_checkpointed_rows_with_offset(self) -> None:
        """Test exclude_checkpointed_rows with non-zero row offset."""
        from ray.anyscale.data.checkpoint.util import (
            CheckpointedFragmentInfo,
        )

        # Create a test table with row IDs 0-9
        test_table = pyarrow.table(
            {"row_id": list(range(10)), "data": [f"data_{i}" for i in range(10)]}
        )

        # Test with offset - checkpointed rows are 12, 15, 18 (offset 10 + 2, 5, 8)
        # Create boolean array for rows 0-19: True = checkpointed, False = not checkpointed
        # Rows 12, 15, 18 are checkpointed, others are not
        # The boolean array must cover the full range that exclude_checkpointed_rows will access
        boolean_array = [False] * 10 + [
            False,
            False,
            True,
            False,
            False,
            True,
            False,
            False,
            True,
            False,
        ]
        fragment_info = CheckpointedFragmentInfo(
            fragment=None,
            row_group_idx=0,
            num_rows=20,  # Must cover rows 0-19 for the function to work correctly
            fully_checkpointed=False,
            checkpointed_row_ids=pyarrow.array(
                [boolean_array], type=pyarrow.list_(pyarrow.bool_())
            ),
            checkpointed_row_count=3,
        )

        # Use offset 10, so row_indices will be [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        result = exclude_checkpointed_rows(test_table, fragment_info, 10, 10)
        assert len(result) == 7  # 10 - 3 = 7 rows should remain

        # Check that checkpointed rows are excluded and non-checkpointed rows remain
        remaining_row_ids = result["row_id"].to_numpy()
        # These correspond to rows 12, 15, 18 in the offset space
        checkpointed_ids = [2, 5, 8]
        # These correspond to rows 10, 11, 13, 14, 16, 17, 19
        non_checkpointed_ids = [0, 1, 3, 4, 6, 7, 9]

        assert not any(
            rid in remaining_row_ids for rid in checkpointed_ids
        ), f"Checkpointed rows {checkpointed_ids} should be excluded"
        assert all(
            rid in remaining_row_ids for rid in non_checkpointed_ids
        ), f"Non-checkpointed rows {non_checkpointed_ids} should remain"

    def test_exclude_checkpointed_rows_no_match_with_valid_checkpoint(self) -> None:
        """Test exclude_checkpointed_rows when fragment doesn't match checkpointed
        data, but checkpoint has valid row IDs.
        """
        from ray.anyscale.data.checkpoint.util import (
            CheckpointedFragmentInfo,
        )

        # Create a test table with row IDs 5-9 (different range than checkpoint)
        test_table = pyarrow.table(
            {"row_id": list(range(5, 10)), "data": [f"data_{i}" for i in range(5, 10)]}
        )

        # Create checkpointed fragment info with non-overlapping row IDs
        # Checkpoint has rows 0-4, table has rows 5-9 (no overlap)
        # Boolean array covers all 10 rows: 0-4 True (checkpointed), 5-9 False (not checkpointed)
        boolean_array = [
            True,
            True,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
        ]
        fragment_info = CheckpointedFragmentInfo(
            fragment=None,
            row_group_idx=0,
            num_rows=10,  # 10 rows in this fragment (0-9)
            fully_checkpointed=False,  # Not fully checkpointed
            checkpointed_row_ids=pyarrow.array(
                [boolean_array], type=pyarrow.list_(pyarrow.bool_())
            ),
            checkpointed_row_count=5,  # 5 rows checkpointed (0-4)
        )

        # When there are no overlapping IDs, exclude_checkpointed_rows should return the table as-is
        # Use offset 5 to match the table's row IDs
        result = exclude_checkpointed_rows(test_table, fragment_info, 5, 5)
        assert len(result) == 5  # All 5 rows should remain unchanged

        # Verify the result is identical to the input table
        assert result.equals(
            test_table
        ), "Table should be unchanged when no overlapping row IDs found"


class TestLoadCheckpointAndProcessGeneratedId:
    """Test the CheckpointFilter load_checkpoint."""

    def _create_checkpoint_directory(
        self, tmp_path: pytest.TempPathFactory
    ) -> pathlib.Path:
        """Helper to create a temporary checkpoint directory.

        Args:
            tmp_path: Pytest temporary path fixture

        Returns:
            Path to the created checkpoint directory
        """
        checkpoint_path = tmp_path / "checkpoint"
        checkpoint_path.mkdir()
        return checkpoint_path

    def _create_test_data_row(
        self,
        path_prefix: str,
        file_name: str,
        row_group: int,
        num_rows: int,
        row_id: int,
    ) -> dict[str, Union[str, int]]:
        """Helper to create a single test data row.

        Args:
            path_prefix: Path prefix for the file
            file_name: Name of the parquet file
            row_group: Row group identifier
            num_rows: Expected number of rows for this file/row_group
            row_id: Unique row identifier

        Returns:
            Dictionary containing test data row with all required fields
        """
        return {
            PATH_PREFIX_FIELD: path_prefix,
            FILE_NAME_FIELD: file_name,
            ROW_GROUP_FIELD: row_group,
            NUM_ROWS_FIELD: num_rows,
            ROW_ID_FIELD: row_id,
        }

    def _create_checkpoint_filter(
        self,
        checkpoint_path: pathlib.Path,
        id_column: str,
        generated_id_column: bool = True,
    ) -> BatchBasedCheckpointFilter:
        """Helper to create a checkpoint filter instance.

        Args:
            checkpoint_path: Path to the checkpoint directory
            id_column: Name of the ID column
            generated_id_column: Whether to use generated ID column

        Returns:
            Configured BatchBasedCheckpointFilter instance
        """
        from ray.anyscale.data.checkpoint.interfaces import CheckpointConfig
        from ray.anyscale.data.checkpoint.checkpoint_filter import (
            BatchBasedCheckpointFilter,
        )

        if generated_id_column:
            # When using generated_id_column, don't specify id_column
            config = CheckpointConfig(
                checkpoint_path=str(checkpoint_path),
                generated_id_column=id_column,  # Use id_column as the name for generated ID column
                override_filesystem=pa.fs.LocalFileSystem(),
            )
        else:
            # When not using generated_id_column, specify id_column
            config = CheckpointConfig(
                checkpoint_path=str(checkpoint_path),
                id_column=id_column,
                override_filesystem=pa.fs.LocalFileSystem(),
            )

        return BatchBasedCheckpointFilter(config)

    def _write_generated_id_column_checkpoint_data(
        self, checkpoint_path: pathlib.Path, test_data: List[dict[str, Union[str, int]]]
    ) -> None:
        """Helper to write test data with generated ID structure to parquet file.

        Args:
            checkpoint_path: Path to the checkpoint directory
            test_data: List of test data dictionaries with generated ID structure
        """
        import pyarrow as pa

        # Create the struct column for the generated ID
        id_arrays = []
        for field_name in GENERATED_ID_COLUMN_FIELD_NAMES:
            field_values = [row[field_name] for row in test_data]
            id_arrays.append(pa.array(field_values))

        id_column_struct = pa.StructArray.from_arrays(
            id_arrays, names=GENERATED_ID_COLUMN_FIELD_NAMES
        )

        # Create the table with just the id_column (struct)
        table = pa.table({ROW_ID_FIELD: id_column_struct})

        parquet_file = checkpoint_path / "checkpoint.parquet"
        pa.parquet.write_table(table, parquet_file)

    def _write_id_column_checkpoint_data(
        self, checkpoint_path: pathlib.Path, test_data: List[dict[str, Union[str, int]]]
    ) -> None:
        """Helper to write simple test data to parquet file (without generated ID structure).

        Args:
            checkpoint_path: Path to the checkpoint directory
            test_data: List of simple test data dictionaries to write
        """
        import pyarrow as pa

        # Extract column names from the first row
        if test_data:
            column_names = list(test_data[0].keys())
            # Create arrays for each column
            arrays = []
            for col_name in column_names:
                col_values = [row[col_name] for row in test_data]
                arrays.append(pa.array(col_values))
            table = pa.table(arrays, names=column_names)
        else:
            # Handle empty data case
            table = pa.table([], names=[])

        parquet_file = checkpoint_path / "checkpoint.parquet"
        pa.parquet.write_table(table, parquet_file)

    def _verify_result(self, result: ObjectRef[Block], id_column: str) -> None:
        """Helper method to verify the checkpoint block structure and sorting.

        Args:
            result: ObjectRef[Block] result from load_checkpoint
            id_column: Name of the ID column

        Raises:
            AssertionError: If the checkpoint block is not properly structured or sorted
        """
        # Get the actual block from the ObjectRef
        checkpoint_block = ray.get(result)

        if checkpoint_block.num_rows == 0:
            return

        if id_column == GENERATED_ID_COL:
            # Verify required columns exist for generated ID column
            required_columns = [
                CHECKPOINTED_FRAGMENT_COLUMN_NAME,
                NUM_ROWS_FIELD,
                CHECKPOINTED_ROW_COUNT_COLUMN_NAME,
                CHECKPOINTED_ROW_IDS_COLUMN_NAME,
            ]
            for col in required_columns:
                assert (
                    col in checkpoint_block.column_names
                ), f"Required column {col} not found in generated ID checkpoint block. Available columns: {checkpoint_block.column_names}"

            # Verify checkpointed fragment column is sorted
            checkpoint_fragments = checkpoint_block[
                CHECKPOINTED_FRAGMENT_COLUMN_NAME
            ].to_pylist()
            assert checkpoint_fragments == sorted(
                checkpoint_fragments
            ), f"Checkpoint block {CHECKPOINTED_FRAGMENT_COLUMN_NAME} column is not sorted: {checkpoint_fragments}"
        else:
            # Verify required columns exist for regular ID column
            required_columns = [
                id_column,
            ]
            for col in required_columns:
                assert (
                    col in checkpoint_block.column_names
                ), f"Required column {col} not found in checkpoint block. Available columns: {checkpoint_block.column_names}"

            # Verify ID column is sorted
            assert checkpoint_block[id_column].to_pylist() == sorted(
                checkpoint_block[id_column].to_pylist()
            ), f"Checkpoint block {id_column} column is not sorted: {checkpoint_block[id_column].to_pylist()}"

    def _create_test_data_with_row_groups(
        self, file_specs: List[tuple[str, str, List[tuple[int, int, int, int]]]]
    ) -> List[dict[str, Union[str, int]]]:
        """Helper to create test data with multiple files and row groups.

        Args:
            file_specs: List of tuples (path_prefix, file_name, row_group_specs)
                where row_group_specs is a list of (row_group, expected_rows, actual_rows, row_id_offset)

        Returns:
            List of test data dictionaries ready for PyArrow table creation
        """
        test_data = []
        for path_prefix, file_name, row_group_specs in file_specs:
            for row_group, expected_rows, actual_rows, row_id_offset in row_group_specs:
                for i in range(actual_rows):
                    test_data.append(
                        self._create_test_data_row(
                            path_prefix,
                            file_name,
                            row_group,
                            expected_rows,
                            row_id_offset + i,
                        )
                    )
        return test_data

    def test_load_checkpoint_with_generated_id_column(self, tmp_path):
        """Test load_checkpoint with generated_id_column enabled."""
        # Create test data with some completed and some incomplete file/row_group combinations
        file_specs = [
            (
                "/data",
                "file1.parquet",
                [(0, 3, 3, 0)],
            ),  # File 1: 3 rows, expected 3 rows (completed)
            (
                "/data",
                "file2.parquet",
                [(0, 3, 2, 0)],
            ),  # File 2: 2 rows, expected 3 rows (incomplete)
            (
                "/data",
                "file3.parquet",
                [(0, 1, 1, 0)],
            ),  # File 3: 1 row, expected 1 row (completed)
        ]
        test_data = self._create_test_data_with_row_groups(file_specs)

        # Setup checkpoint
        checkpoint_path = self._create_checkpoint_directory(tmp_path)
        self._write_generated_id_column_checkpoint_data(checkpoint_path, test_data)
        filter_instance = self._create_checkpoint_filter(
            checkpoint_path, GENERATED_ID_COL, True
        )

        # load_checkpoint
        result = filter_instance.load_checkpoint()
        checkpoint_block = ray.get(result)
        assert checkpoint_block.num_rows > 0

        # Verify the checkpoint block structure and sorting
        self._verify_result(result, GENERATED_ID_COL)

    def test_load_checkpoint_multiple_row_groups_per_file(self, tmp_path):
        """Test load_checkpoint with files that have multiple row groups."""
        # Create a temporary checkpoint directory
        checkpoint_path = tmp_path / "checkpoint"
        checkpoint_path.mkdir()

        # Create test data with multiple row groups per file
        file_specs = [
            # File 1: 3 row groups, each with varying completion status
            (
                "/data",
                "file1.parquet",
                [
                    (0, 3, 3, 0),  # Row group 0: 3 rows, expected 3 rows (completed)
                    (1, 3, 2, 100),  # Row group 1: 2 rows, expected 3 rows (incomplete)
                    (2, 3, 3, 200),  # Row group 2: 3 rows, expected 3 rows (completed)
                ],
            ),
            # File 2: 2 row groups
            (
                "/data",
                "file2.parquet",
                [
                    (0, 1, 1, 300),  # Row group 0: 1 row, expected 1 row (completed)
                    (1, 2, 2, 400),  # Row group 1: 2 rows, expected 2 rows (completed)
                ],
            ),
        ]

        # Setup checkpoint
        test_data = self._create_test_data_with_row_groups(file_specs)
        self._write_generated_id_column_checkpoint_data(checkpoint_path, test_data)
        filter_instance = self._create_checkpoint_filter(
            checkpoint_path, GENERATED_ID_COL, True
        )

        # load_checkpoint
        result = filter_instance.load_checkpoint()

        # Verify the checkpoint block structure and sorting
        self._verify_result(result, GENERATED_ID_COL)

    def test_load_checkpoint_large_number_of_row_groups(self, tmp_path):
        """Test load_checkpoint with a large number of row groups per file."""
        # Create a temporary checkpoint directory
        checkpoint_path = tmp_path / "checkpoint"
        checkpoint_path.mkdir()

        # Create test data with many row groups
        # File with 20 row groups, alternating between completed and incomplete
        file_specs = []
        for row_group in range(20):
            expected_rows = 5  # Each row group expects 5 rows
            actual_rows = (
                5 if row_group % 2 == 0 else 3
            )  # Even row groups are complete, odd are incomplete
            file_specs.append((row_group, expected_rows, actual_rows, row_group * 1000))

        test_data = self._create_test_data_with_row_groups(
            [("/data", "large_file.parquet", file_specs)]
        )

        # Setup checkpoint
        self._write_generated_id_column_checkpoint_data(checkpoint_path, test_data)
        filter_instance = self._create_checkpoint_filter(
            checkpoint_path, GENERATED_ID_COL, True
        )

        # load_checkpoint
        result = filter_instance.load_checkpoint()

        # Verify the checkpoint block structure and sorting
        self._verify_result(result, GENERATED_ID_COL)

    def test_load_checkpoint_mixed_completion_patterns(self, tmp_path):
        """Test load_checkpoint with complex completion patterns across multiple files and row groups."""
        # Create a temporary checkpoint directory
        checkpoint_path = tmp_path / "checkpoint"
        checkpoint_path.mkdir()

        # Create test data with complex patterns
        file_specs = [
            # File 1: Mixed completion pattern
            (
                "/data",
                "file1.parquet",
                [
                    (0, 3, 2, 0),  # Row group 0: 2/3 rows (incomplete)
                    (1, 3, 3, 100),  # Row group 1: 3/3 rows (complete)
                    (2, 3, 1, 200),  # Row group 2: 1/3 rows (incomplete)
                ],
            ),
            # File 2: All row groups complete
            (
                "/data",
                "file2.parquet",
                [
                    (0, 2, 2, 0),  # Row group 0: 2/2 rows (complete)
                    (1, 2, 2, 2),  # Row group 1: 2/2 rows (complete)
                    (2, 2, 2, 4),  # Row group 2: 2/2 rows (complete)
                ],
            ),
            # File 3: All row groups incomplete
            (
                "/data",
                "file3.parquet",
                [
                    (0, 2, 1, 0),  # Row group 0: 1/2 rows (incomplete)
                    (1, 2, 1, 1),  # Row group 1: 1/2 rows (incomplete)
                ],
            ),
        ]

        # Setup checkpoint
        test_data = self._create_test_data_with_row_groups(file_specs)
        self._write_generated_id_column_checkpoint_data(checkpoint_path, test_data)
        filter_instance = self._create_checkpoint_filter(
            checkpoint_path, GENERATED_ID_COL, True
        )

        # load_checkpoint
        result = filter_instance.load_checkpoint()

        # Verify the checkpoint block structure and sorting
        self._verify_result(result, GENERATED_ID_COL)

    def test_load_checkpoint_without_generated_id_column(self, tmp_path):
        """Test load_checkpoint without generated_id_column (should return None for completed/remaining)."""
        # Create simple test data without generated ID structure
        test_data = [{ID_COL: i, "value": f"row_{i}"} for i in range(5)]

        # Setup checkpoint
        checkpoint_path = self._create_checkpoint_directory(tmp_path)
        self._write_id_column_checkpoint_data(checkpoint_path, test_data)
        filter_instance = self._create_checkpoint_filter(checkpoint_path, ID_COL, False)

        # load_checkpoint
        result = filter_instance.load_checkpoint()

        # Verify the checkpoint block structure and sorting
        self._verify_result(result, ID_COL)

    def test_load_checkpoint_empty_checkpoint(self, tmp_path):
        """Test load_checkpoint with an empty checkpoint directory."""
        # Create checkpoint filter
        checkpoint_path = self._create_checkpoint_directory(tmp_path)
        filter_instance = self._create_checkpoint_filter(
            checkpoint_path, GENERATED_ID_COL, True
        )

        # load checkpoint
        result = filter_instance.load_checkpoint()

        # Verify the checkpoint block structure and sorting
        self._verify_result(result, ID_COL)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
