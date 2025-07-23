import csv
import os
import random
from typing import List
from unittest.mock import MagicMock

import pandas as pd
import pyarrow
import pytest
from pyarrow.fs import FileSelector, LocalFileSystem
from pytest_lazy_fixtures import lf as lazy_fixture

import ray
from ray._common.test_utils import wait_for_condition
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

ID_COL = "id"

# Auto-use `restore_data_context` for each test and apply 300-second timeout to all tests.
pytestmark = [
    pytest.mark.usefixtures("restore_data_context"),
    pytest.mark.timeout(300),
]


@pytest.fixture
def generate_sample_data_csv(tmp_path):
    def _generate(with_id_column=True):
        if with_id_column:
            # Generate a dummy dataset with 5 rows and columns ["id", "col1"]
            data = [{"id": i, "col1": random.random()} for i in range(5)]
        else:
            # Generate a dummy dataset with 5 rows and columns ["col1"]
            data = [{"col1": random.random()} for _ in range(5)]

        f_path = os.path.join(tmp_path, "sample_data.csv")
        with open(f_path, mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
        return f_path

    return _generate


@pytest.fixture
def generate_sample_data_parquet(tmp_path):
    def _generate(with_id_column=True):
        f_dir = os.path.join(tmp_path, "sample_data_parquet")
        os.makedirs(f_dir, exist_ok=True)
        if with_id_column:
            # Generate a dummy dataset with 5 rows and columns ["id", "col1"]
            df = pd.DataFrame([{"id": i, "col1": random.random()} for i in range(5)])
        else:
            # Generate a dummy dataset with 5 rows and columns ["col1"]
            df = pd.DataFrame([{"col1": random.random()} for _ in range(5)])

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


def read_ids_from_checkpoint_files(config: CheckpointConfig) -> List[int]:
    """Reads the checkpoint files and returns a sorted list of IDs
    which have been checkpointed."""
    backend = config.backend
    ckpt_path = config.checkpoint_path
    fs = config.filesystem

    if backend in (
        CheckpointBackend.FILE_STORAGE_ROW,
        CheckpointBackend.CLOUD_OBJECT_STORAGE_ROW,
    ):
        if fs is None:
            try:
                actual_checkpoint_file_paths = [
                    os.path.join(ckpt_path, fname) for fname in os.listdir(ckpt_path)
                ]
            except FileNotFoundError:
                return []
        else:
            files = fs.get_file_info(
                FileSelector(_unwrap_protocol(ckpt_path), allow_not_found=True)
            )
            actual_checkpoint_file_paths = []
            for file_info in files:
                if file_info.is_file:
                    actual_checkpoint_file_paths.append(file_info.path)
        # Parse the checkpoint file paths to get the ID.
        # Paths are of form `.../id.jsonl`.
        # Split the path by / and jsonl file extension.
        return sorted(
            [
                int(os.path.basename(f).split(".")[0])
                for f in actual_checkpoint_file_paths
            ]
        )

    if backend in (
        CheckpointBackend.FILE_STORAGE,
        CheckpointBackend.CLOUD_OBJECT_STORAGE,
    ):
        if fs is None:
            try:
                actual_checkpoint_file_paths = [
                    os.path.join(ckpt_path, fname) for fname in os.listdir(ckpt_path)
                ]
            except FileNotFoundError:
                return []

            read_ckpt_ids = []
            for file in actual_checkpoint_file_paths:
                with open(file, "r") as f:
                    ckpt_df = pd.read_parquet(f)
                    read_ckpt_ids.extend(ckpt_df[ID_COL].tolist())
            return sorted(read_ckpt_ids)
        else:
            actual_checkpoint_file_paths = []
            files = fs.get_file_info(
                FileSelector(_unwrap_protocol(ckpt_path), allow_not_found=True)
            )
            for file_info in files:
                if file_info.is_file:
                    actual_checkpoint_file_paths.append(file_info.path)

            read_ckpt_ids = []
            for fpath in actual_checkpoint_file_paths:
                with fs.open_input_file(fpath) as f:
                    ckpt_df = pd.read_parquet(f)
                    read_ckpt_ids.extend(ckpt_df[ID_COL].tolist())

            return sorted(read_ckpt_ids)
    raise Exception(f"Invalid backend: {backend}")


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
                "id",
                "s3://bucket/path",
                override_backend=CheckpointBackend.FILE_STORAGE,
            )

    def test_default_checkpoint_path(self, s3_path, monkeypatch):
        with pytest.raises(
            InvalidCheckpointingConfig,
            match="CheckpointConfig.checkpoint_path",
        ):
            CheckpointConfig("id", None)

        default_bucket = s3_path
        monkeypatch.setenv(
            CheckpointConfig.DEFAULT_CHECKPOINT_PATH_BUCKET_ENV_VAR, default_bucket
        )

        config = CheckpointConfig("id", None)
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
            CheckpointConfig("id", checkpoint_path)

    @pytest.mark.parametrize(
        "checkpoint_path",
        [
            lazy_fixture("local_path"),
            lazy_fixture("s3_path"),
        ],
    )
    def test_infer_filesystem_and_backend(self, checkpoint_path):
        config = CheckpointConfig("id", checkpoint_path)
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
            "id", checkpoint_path, override_filesystem=fs, override_backend=backend
        )
        assert config.filesystem is fs
        assert config.backend is backend

    def test_skip_inference_with_overrides(self):
        """Test that filesystem inference is skipped when override is provided."""
        # Inferring filesystem will fail if the path doesn't exist.
        path = "s3://non-existing-bucket/"
        fs = pyarrow.fs.S3FileSystem()
        config = CheckpointConfig(
            "id",
            path,
            override_filesystem=fs,
        )
        assert config.filesystem is fs
        assert config.backend is CheckpointBackend.CLOUD_OBJECT_STORAGE

    def test_generate_row_id_default_column(self):
        """Test CheckpointConfig with id_column missing and no generate_row_id provided."""
        # id_column is None, generate_row_id is None - should raise error
        with pytest.raises(
            InvalidCheckpointingConfig,
            match="Either `id_column` or `generate_row_id` must be provided",
        ):
            CheckpointConfig(
                None,
                "/tmp/checkpoint",
            )

    def test_generate_row_id_custom_column(self):
        """Test CheckpointConfig with id_column missing, but with user provided generate_row_id."""
        # id_column is None, generate_row_id is "custom_id"
        config = CheckpointConfig(
            None,
            "/tmp/checkpoint",
            generate_row_id="custom_id",
        )
        assert config.id_column == "custom_id"
        assert config.generate_row_id == "custom_id"

    def test_generate_row_id_with_existing_id_column(self):
        """Test CheckpointConfig with both id_column and generate_row_id provided."""
        with pytest.raises(
            InvalidCheckpointingConfig,
            match="Cannot specify both `id_column` and `generate_row_id`",
        ):
            CheckpointConfig(
                "existing_id",
                "/tmp/checkpoint",
                generate_row_id="generated_id",
            )


@pytest.mark.parametrize("read_code_path", ["runtime", "oss_fallback"])
@pytest.mark.parametrize(
    "with_id_column,generate_row_id", [("existing", None), ("generated", "row_id")]
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
def test_checkpoint(
    ray_start_10_cpus_shared,
    generate_sample_data_csv,
    read_code_path,
    backend,
    fs,
    data_path,
    with_id_column,
    generate_row_id,
):
    class TestActor:
        def __init__(self):
            pass

        def __call__(self, batch):
            return batch

    ctx = ray.data.DataContext.get_current()
    ckpt_path = os.path.join(data_path, "test_checkpoint_output_files")

    if generate_row_id is not None:
        ctx.checkpoint_config = CheckpointConfig(
            generate_row_id=generate_row_id,
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

    csv_file = generate_sample_data_csv(with_id_column=(with_id_column == "existing"))

    if read_code_path == "runtime":
        ds = ray.data.read_csv(csv_file)
    elif read_code_path == "oss_fallback":
        ds = ray.data.read_api.read_csv(csv_file)
    else:
        raise Exception(f"Invalid `read_code_path`: {read_code_path}")

    # Execute the dataset with checkpointing enabled.
    ds = ds.map_batches(TestActor, concurrency=1)
    data_output_path = os.path.join(data_path, "output")

    if generate_row_id is not None:
        # For CSV datasets with generate_row_id, the read operation fails with an AssertionError
        # because CSV datasources don't support auto-generated row IDs
        with pytest.raises(
            AssertionError,
            match="For checkpointing with `generate_row_id`, .* operator must use a ParquetReader",
        ):
            ds.write_parquet(data_output_path, filesystem=fs)
        pytest.skip("`generate_row_id` is not supported for CSV datasets")
    else:
        ds.write_parquet(data_output_path, filesystem=fs)

    # Disable checkpointing prior to reading back the data, so we don't skip any rows.
    ctx.checkpoint_config.enabled = False

    # Ensure that the written data is correct.
    ds_readback = ray.data.read_parquet(data_output_path, filesystem=fs)
    actual_output = sorted([row["id"] for row in ds_readback.iter_rows()])
    expected_output = sorted([row["id"] for row in ds.iter_rows()])
    assert actual_output == expected_output

    # When execution succeeds, checkpoint data should be automatically deleted.
    # TODO(haochen): Also delete checkpoint for row-based backends.
    checkpoint_ids = read_ids_from_checkpoint_files(ctx.checkpoint_config)
    if ctx.checkpoint_config.is_batch_based():
        assert checkpoint_ids == []
    else:
        expected_checkpoint_ids = sorted([row[ID_COL] for row in ds.iter_rows()])
        assert checkpoint_ids == expected_checkpoint_ids


@pytest.mark.parametrize("read_code_path", ["runtime", "oss_fallback"])
@pytest.mark.parametrize(
    "with_id_column,generate_row_id", [("existing", None), ("generated", "row_id")]
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
def test_full_dataset_executed_for_non_write(
    ray_start_10_cpus_shared,
    generate_sample_data_parquet,
    read_code_path,
    backend,
    fs,
    data_path,
    with_id_column,
    generate_row_id,
):
    """Tests that for an already fully checkpointed Dataset,
    calling `schema()` and `count()` should not skip checkpointing
    and should execute the full Dataset to get the correct information.
    """

    ctx = ray.data.DataContext.get_current()
    ckpt_path = os.path.join(data_path, "test_checkpoint_output_files")

    if generate_row_id is not None:
        ctx.checkpoint_config = CheckpointConfig(
            generate_row_id=generate_row_id,
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

    parquet_dir = generate_sample_data_parquet(
        with_id_column=(with_id_column == "existing")
    )

    error_expected = read_code_path == "oss_fallback" and generate_row_id is not None
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
        # For generate_row_id with oss_fallback, the read operation fails with an AssertionError
        # because the datasource is not a ParquetReader
        with pytest.raises(
            AssertionError,
            match="For checkpointing with `generate_row_id`, Read operator must use a ParquetReader",
        ):
            ds.write_parquet(data_output_path, filesystem=fs)
        pytest.skip(
            "`generate_row_id` is not supported for Parquet datasets with OSS fallback"
        )
    else:
        ds.write_parquet(data_output_path, filesystem=fs)

    # Recreate the same dataset, so that it will skip checkpointed rows.
    ds2 = ray.data.read_parquet(parquet_dir)
    ds2 = ds2.map(lambda row: row)

    # Check that when re-running a dataset which has already been completely
    # checkpointed, it does not skip any rows during `schema()` and `count()` calls.
    assert ds2.schema() == schema_before_write
    assert ds2.count() == count_before_write


@pytest.mark.parametrize(
    "ds_factory",
    [
        lambda max_num_items: ray.data.range(
            max_num_items, override_num_blocks=max_num_items
        ),
        lambda max_num_items: ray.data.from_items(
            [{"id": i} for i in range(max_num_items)], override_num_blocks=max_num_items
        ),
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
):
    """Tests that for a Dataset which fails partway and is recovered,
    it skips rows which have already been checkpointed."""

    ctx = ray.data.DataContext.get_current()
    ctx.execution_options.preserve_order = True
    ckpt_path = os.path.join(data_path, "test_checkpoint_output_files")
    ctx.checkpoint_config = CheckpointConfig(
        ID_COL,
        ckpt_path,
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
            assert len(batch[ID_COL]) == 1
            id = batch[ID_COL][0]
            if self._should_fail and id == self._max_num_items // 2:
                # Fail the Dataset when the first half of rows are
                # finished and checkpointed.
                wait_for_condition(self._wait_until_checkpoint_written)
                raise TestException(f"FailActor: Failing on row {batch['id']}")

            return batch

        def _wait_until_checkpoint_written(self):
            checkpointed_ids = set(
                read_ids_from_checkpoint_files(self._checkpoint_config)
            )
            return checkpointed_ids == set(range(self._max_num_items // 2))

    max_num_items = 10
    ds = ds_factory(max_num_items)
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
    if ctx.checkpoint_config.is_batch_based():
        assert read_ids_from_checkpoint_files(ctx.checkpoint_config) == []
    else:
        assert read_ids_from_checkpoint_files(ctx.checkpoint_config) == list(
            range(max_num_items)
        )

    # Disable checkpointing prior to reading back the data, so we don't skip any rows.
    ctx.checkpoint_config.enabled = False

    # Ensure that the written data is correct.
    ds_readback = ray.data.read_parquet(data_output_path, filesystem=fs)
    actual_output = sorted([row["id"] for row in ds_readback.iter_rows()])
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
    assert len(read_ids_from_checkpoint_files(ctx.checkpoint_config)) == 5


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


def test_dict_checkpoint_config():
    """Test that a dict checkpoint config can be used to create a CheckpointConfig."""
    context = ray.data.DataContext.get_current()
    checkpoint_path = "/tmp/checkpoint"
    fs = LocalFileSystem()
    context.checkpoint_config = {
        "id_column": "id",
        "checkpoint_path": checkpoint_path,
        "override_filesystem": fs,
        "override_backend": "CLOUD_OBJECT_STORAGE_ROW",
    }
    assert context.checkpoint_config.id_column == "id"
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

    def test_plan_write_op_with_checkpoint_writer(self):
        class FakeDatasink(Datasink):
            def write(self, blocks, ctx):
                return None

        # Configure checkpointing.
        ctx = ray.data.DataContext.get_current()
        ctx.checkpoint_config = CheckpointConfig("id", "/tmp/checkpoint")

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


def test_write_block_checkpoint_with_pandas_df(restore_data_context, tmp_path):
    ctx = ray.data.DataContext.get_current()
    ctx.checkpoint_config = CheckpointConfig(
        "id",
        str(tmp_path),
    )
    checkpoint_writer = BatchBasedCheckpointWriter(ctx.checkpoint_config)
    df = pd.DataFrame({"id": [0]})

    checkpoint_writer.write_block_checkpoint(BlockAccessor.for_block(df))

    assert len(os.listdir(tmp_path)) == 1
    checkpoint_filename = os.listdir(tmp_path)[0]
    checkpoint_path = tmp_path / checkpoint_filename
    written_ids = pd.read_parquet(checkpoint_path)["id"].tolist()
    assert written_ids == [0]


def test_filter_rows_for_block():
    """Test BatchBasedCheckpointFilter.filter_rows_for_block."""

    # Mock CheckpointConfig
    config = CheckpointConfig(
        id_column="id",
        checkpoint_path="/mock/path",
    )

    filter_instance = BatchBasedCheckpointFilter(config)

    # Create a mock block.
    block = pyarrow.table(
        {
            "id": list(range(10)),
            "data": [str(i) for i in range(10)],
        }
    )
    # Create a mock checkpointed_ids with multiple chunks.
    chunk1 = pyarrow.table({"id": [1, 2, 4]})
    chunk2 = pyarrow.table({"id": [6, 8, 9, 11]})
    chunk3 = pyarrow.table({"id": [12, 13]})
    checkpointed_ids = pyarrow.concat_tables([chunk1, chunk2, chunk3])
    assert len(checkpointed_ids["id"].chunks) == 3

    # Call the method
    filtered_block = filter_instance.filter_rows_for_block(block, checkpointed_ids)

    # Verify the output
    expected_block = pyarrow.table(
        {
            "id": [0, 3, 5, 7],
            "data": ["0", "3", "5", "7"],
        }
    )
    assert filtered_block.equals(expected_block)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
