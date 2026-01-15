from typing import Iterable, Iterator, Optional, Set
from unittest.mock import MagicMock

import pytest

from ray.anyscale.data._internal.file_indexer import (
    NonSamplingFileIndexer,
    WholeFileChunker,
)
from ray.anyscale.data._internal.logical.operators.read_files_operator import (
    ListFiles,
    ReadFiles,
)
from ray.anyscale.data._internal.logical.rules import PushdownCountFiles
from ray.anyscale.data._internal.partitioners.file_partitioner import (
    FilePartitioner,
)
from ray.anyscale.data._internal.readers import SupportsMetadata
from ray.anyscale.data._internal.readers.file_reader import FileReader
from ray.anyscale.data._internal.readers.supports_metadata import MetadataType
from ray.data import DataContext
from ray.data._internal.logical.interfaces import LogicalPlan
from ray.data._internal.logical.operators.count_operator import Count
from ray.data._internal.logical.operators.map_operator import MapBatches


class StubReaderWithMetadata(FileReader, SupportsMetadata):
    def __init__(
        self,
        available_metadata: Set[MetadataType],
        target_metadata_batch_size: Optional[int] = None,
    ):
        self._available_metadata = available_metadata
        self._target_metadata_batch_size = target_metadata_batch_size

    def read_files(self, file_manifest, *, filesystem) -> Iterable:
        yield from ()

    def read_metadata(self, file_manifest, *, filesystem) -> Iterator:
        yield from ()

    def available_metadata(self: FileReader) -> Set[MetadataType]:
        return self._available_metadata

    def get_target_metadata_batch_size(self: FileReader) -> Optional[int]:
        return self._target_metadata_batch_size


def test_rule_produces_plan_with_expected_types():
    list_files = ListFiles(
        paths=["/tmp/test"],
        file_indexer=NonSamplingFileIndexer(ignore_missing_paths=True),
        filesystem=MagicMock(),
        source_paths=["/tmp/test"],
    )
    read_files = ReadFiles(
        list_files,
        reader=StubReaderWithMetadata(available_metadata={MetadataType.NUM_ROWS}),
        filesystem=MagicMock(),
    )
    count = Count(read_files)

    plan = LogicalPlan(count, DataContext.get_current())
    rule = PushdownCountFiles()
    optimized_plan = rule.apply(plan)

    # TODO: We should make it easier to compare if two logical plans are equal.
    optimized_plan_types = [type(op) for op in optimized_plan.dag.post_order_iter()]
    assert optimized_plan_types == [ListFiles, MapBatches]


@pytest.mark.parametrize("target_metadata_batch_size", (None, 1))
def test_optimized_plan_reads_metadata_with_specified_batch_size(
    target_metadata_batch_size,
):
    list_files = ListFiles(
        paths=["/tmp/test"],
        file_indexer=NonSamplingFileIndexer(ignore_missing_paths=True),
        filesystem=MagicMock(),
        source_paths=["/tmp/test"],
    )
    read_files = ReadFiles(
        list_files,
        reader=StubReaderWithMetadata(
            available_metadata={MetadataType.NUM_ROWS},
            target_metadata_batch_size=target_metadata_batch_size,
        ),
        filesystem=MagicMock(),
    )
    count = Count(read_files)

    plan = LogicalPlan(count, DataContext.get_current())
    rule = PushdownCountFiles()
    optimized_plan = rule.apply(plan)

    optimized_list_files, optimized_map_batches = list(
        optimized_plan.dag.post_order_iter()
    )
    assert optimized_map_batches._batch_size == target_metadata_batch_size


def test_optimized_plan_does_not_partition_or_chunk_files():
    list_files = ListFiles(
        paths=["/tmp/test"],
        file_indexer=NonSamplingFileIndexer(ignore_missing_paths=True),
        filesystem=MagicMock(),
        file_partitioner=MagicMock(spec=FilePartitioner),
        source_paths=["/tmp/test"],
    )
    read_files = ReadFiles(
        list_files,
        reader=StubReaderWithMetadata(available_metadata={MetadataType.NUM_ROWS}),
        filesystem=MagicMock(),
    )
    count = Count(read_files)

    plan = LogicalPlan(count, DataContext.get_current())
    rule = PushdownCountFiles()
    optimized_plan = rule.apply(plan)

    optimized_list_files, optimized_map_batches = list(
        optimized_plan.dag.post_order_iter()
    )
    # Partitioning files for counting isn't incorrect, but it usually isn't necessary
    # and can hurt performance.
    assert optimized_list_files.file_partitioner is None
    # If the file lister chunks files, it might create file manifests with repeated
    # file paths (e.g., different byte ranges of the same file). If we don't disable
    # this, the downstream count tasks can double-count rows.
    assert isinstance(
        optimized_list_files.file_indexer, NonSamplingFileIndexer
    ) and isinstance(optimized_list_files.file_indexer.file_chunker, WholeFileChunker)


def test_does_not_pushdown_if_row_metadata_is_not_available():
    list_files = ListFiles(
        paths=["/tmp/test"],
        file_indexer=NonSamplingFileIndexer(ignore_missing_paths=True),
        filesystem=MagicMock(),
        source_paths=["/tmp/test"],
    )
    read_files = ReadFiles(
        list_files,
        reader=StubReaderWithMetadata(available_metadata={}),
        filesystem=MagicMock(),
    )
    count = Count(read_files)

    plan = LogicalPlan(count, DataContext.get_current())
    rule = PushdownCountFiles()
    optimized_plan = rule.apply(plan)

    # Rule should no-op.
    assert plan == optimized_plan


def test_does_not_pushdown_if_reader_does_not_support_metadata():
    class StubReader(FileReader):
        def read_files(self, file_manifest, *, filesystem) -> Iterable:
            yield from ()

    list_files = ListFiles(
        paths=["/tmp/test"],
        file_indexer=NonSamplingFileIndexer(ignore_missing_paths=True),
        filesystem=MagicMock(),
        source_paths=["/tmp/test"],
    )
    read_files = ReadFiles(list_files, reader=StubReader(), filesystem=MagicMock())
    count = Count(read_files)

    plan = LogicalPlan(count, DataContext.get_current())
    rule = PushdownCountFiles()
    optimized_plan = rule.apply(plan)

    # Rule should no-op.
    assert plan == optimized_plan
