import logging
from unittest.mock import patch

import pyarrow as pa
import pyarrow.fs as pa_fs
import pytest
from pyarrow.fs import FileSystem

import ray
from ray.anyscale.data._internal.file_indexer import NonSamplingFileIndexer
from ray.data._internal.util import RetryingPyFileSystem
from ray.data.context import DataContext
from ray.data.tests.conftest import (  # noqa
    CoreExecutionMetrics,
    assert_core_execution_metrics_equals,
    get_initial_core_execution_metrics_snapshot,
    restore_data_context,
)
from ray.tests.conftest import *  # noqa


@pytest.mark.parametrize("max_num_list_files_tasks", [1, 8])
def test_configure_max_num_list_tasks(
    ray_start_regular_shared,
    tmp_path,
    restore_data_context,  # noqa: F811
    max_num_list_files_tasks,
):
    snapshot = get_initial_core_execution_metrics_snapshot()

    # Create a large number of files to read from.
    paths = []
    for i in range(1024):
        path = tmp_path / f"{i}.dat"
        with open(path, "w"):
            pass

        paths.append(str(path))

    DataContext.get_current().set_config(
        "max_num_list_files_tasks", max_num_list_files_tasks
    )
    ray.data.read_binary_files(paths).materialize()

    assert_core_execution_metrics_equals(
        CoreExecutionMetrics(
            task_count={"ListFiles": lambda count: count <= max_num_list_files_tasks}
        ),
        last_snapshot=snapshot,
    )


def test_non_sampling_file_indexer_logs_warning_for_zero_size_files(
    tmp_path, caplog, propagate_logs
):
    indexer = NonSamplingFileIndexer(ignore_missing_paths=True)
    path = str(tmp_path / "file.dat")
    with open(path, "w"):
        pass

    block = pa.Table.from_pydict({"path": [path]})
    filesystem, _ = FileSystem.from_uri(tmp_path)

    with caplog.at_level(logging.WARNING):
        list(indexer.list_files(block["path"], filesystem=filesystem))

    assert "Skipping zero-size file" in caplog.text


@pytest.mark.parametrize("preserve_order", [False, True])
@pytest.mark.parametrize("use_threading", [False, True])
def test_file_indexer_threading_identical_manifests(
    tmp_path, preserve_order, use_threading
):
    """Test threaded and non-threaded versions produce identical manifests."""
    # Create 1K files
    num_files = 1000
    expected_file_paths = []
    for i in range(num_files):
        file_path = tmp_path / f"file_{i:05d}.txt"
        with open(file_path, "w") as f:
            f.write(f"content_{i}")
        expected_file_paths.append(str(file_path))

    indexer = NonSamplingFileIndexer(ignore_missing_paths=False)
    paths_column = pa.array(expected_file_paths)
    local_fs = pa_fs.LocalFileSystem()
    filesystem = RetryingPyFileSystem.wrap(local_fs, retryable_errors=[])

    max_paths_per_output = 11
    num_workers = 4 if use_threading else 1

    with patch.object(NonSamplingFileIndexer, "_THREADED_NUM_WORKERS", num_workers):
        with patch.object(
            NonSamplingFileIndexer,
            "_MAX_PATHS_PER_LIST_FILES_OUTPUT",
            max_paths_per_output,
        ):
            manifests = list(
                indexer.list_files(
                    paths_column, filesystem=filesystem, preserve_order=preserve_order
                )
            )

    # Flatten all paths from manifests
    paths = []
    for manifest in manifests:
        paths.extend(list(manifest.paths))

    # Verify identical file sets
    assert set(paths) == set(expected_file_paths)
    assert len(paths) == num_files

    if preserve_order:
        # With preserve_order, paths should be in deterministic order
        assert paths == expected_file_paths


@pytest.mark.parametrize("preserve_order", [False, True])
@pytest.mark.parametrize("use_threading", [False, True])
def test_file_indexer_threading_exception_propagation(
    tmp_path, preserve_order, use_threading
):
    """Test that exceptions raised in worker threads are propagated to main thread."""
    # Create some valid files
    valid_paths = []
    for i in range(5):
        file_path = tmp_path / f"file_{i}.txt"
        with open(file_path, "w") as f:
            f.write(f"content_{i}")
        valid_paths.append(str(file_path))

    # Add a non-existent path that will raise FileNotFoundError
    nonexistent_path = str(tmp_path / "nonexistent_file.txt")
    paths = valid_paths + [nonexistent_path]

    indexer = NonSamplingFileIndexer(ignore_missing_paths=False)
    paths_column = pa.array(paths)
    local_fs = pa_fs.LocalFileSystem()
    filesystem = RetryingPyFileSystem.wrap(local_fs, retryable_errors=[])

    # Test with or without threading - should propagate exception
    num_workers = 4 if use_threading else 1
    with patch.object(NonSamplingFileIndexer, "_THREADED_NUM_WORKERS", num_workers):
        with pytest.raises(FileNotFoundError) as exc_info:
            list(
                indexer.list_files(
                    paths_column, filesystem=filesystem, preserve_order=preserve_order
                )
            )
        # Verify exception message contains the path
        assert nonexistent_path in str(exc_info.value)


def test_list_files_operator_throttling_disabled():
    """Test that the ListFiles operator has throttling disabled."""
    from unittest.mock import MagicMock

    from ray.anyscale.data._internal.logical.operators.list_files_operator import (
        ListFiles,
    )
    from ray.anyscale.data._internal.planner.plan_list_files_op import (
        plan_list_files_op,
    )

    # Create a minimal ListFiles logical operator
    list_files_op = ListFiles(
        paths=["/tmp/test"],
        file_indexer=NonSamplingFileIndexer(ignore_missing_paths=True),
        filesystem=MagicMock(),
        source_paths=["/tmp/test"],
    )

    data_context = DataContext.get_current()
    physical_op = plan_list_files_op(list_files_op, [], data_context)

    # Verify throttling is disabled for ListFiles operator
    assert physical_op.throttling_disabled() is True


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
