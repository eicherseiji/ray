import logging

import pyarrow as pa
import pytest
from pyarrow.fs import FileSystem

import ray
from ray.anyscale.data._internal.file_indexer import NonSamplingFileIndexer
from ray.data import DataContext
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


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
