from ray.anyscale.data._internal.logical.operators.list_files_operator import (
    PATH_COLUMN_NAME,
    FileManifest,
    ListFiles,
)
from ray.anyscale.data._internal.logical.operators.read_files_operator import ReadFiles
from ray.anyscale.data._internal.readers import SupportsMetadata
from ray.anyscale.data._internal.readers.supports_metadata import MetadataType
from ray.data._internal.logical.interfaces import LogicalPlan, Rule
from ray.data._internal.logical.operators.count_operator import Count
from ray.data._internal.logical.operators.map_operator import MapBatches
from ray.data.block import DataBatch
from ray.anyscale.data._internal.file_indexer import WholeFileChunker


class PushdownCountFiles(Rule):
    """Optimization rule that pushes down counting to the file reader.

    `FileReader` subclasses can implement `SupportsRowCounting` to efficiently count the
    number of rows in a file. If a `ReadFiles` operator is followed by a `Count`
    operator, this rule replaces the `ReadFiles` with an operator that calls
    `count_rows` and outputs the number of rows. This avoids reading any actual data.
    """

    # NOTE: Default CPU allocation is 1, so we're lowering this to allow
    #       at least 2 tasks to run per CPU core
    _PER_TASK_NUM_CPUS_ALLOCATION = 0.5

    def apply(self, plan: LogicalPlan) -> LogicalPlan:
        count = plan.dag
        if not isinstance(count, Count):
            return plan

        assert len(count.input_dependencies) == 1, len(count.input_dependencies)
        read_files = count.input_dependencies[0]

        if (
            not isinstance(read_files, ReadFiles)
            or not isinstance(read_files.reader, SupportsMetadata)
            or MetadataType.NUM_ROWS not in read_files.reader.available_metadata()
            # If `ReadFiles` op was optimized by predicate pushdown, this
            # PushdownCountFiles based on file stats won't work, so skip this rule.
            or read_files.filter_expr is not None
        ):
            return plan

        assert len(read_files.input_dependencies) == 1, len(
            read_files.input_dependencies
        )
        list_files = read_files.input_dependencies[0]

        assert isinstance(list_files, ListFiles), list_files

        # Disable file partitioning.
        # TODO: Replace with copy to avoid modifying the original operator in-place.
        list_files.file_partitioner = None

        # Also disable file chunking so that each file is listed exactly once.
        # Otherwise, the same file may appear multiple times (once per chunk)
        # and be processed in different batches/tasks, leading to overcounting.
        # NOTE: We mutate the indexer in-place for this optimized count path only.
        list_files.file_indexer._file_chunker = WholeFileChunker()

        def count_rows(batch: DataBatch) -> DataBatch:
            import pyarrow as pa

            assert PATH_COLUMN_NAME in batch.column_names, batch.column_names

            block_metadata_generator = read_files.reader.read_metadata(
                FileManifest(batch),
                filesystem=read_files.filesystem,
            )
            total_rows = 0
            for block_metadata in block_metadata_generator:
                total_rows += block_metadata.num_rows

            return pa.table({Count.COLUMN_NAME: pa.array([total_rows])})

        count_rows_op = MapBatches(
            list_files,
            count_rows,
            batch_format="pyarrow",
            batch_size=read_files.reader.get_target_metadata_batch_size(),
            min_rows_per_bundled_input=read_files.reader.get_target_metadata_batch_size(),
            zero_copy_batch=True,
            ray_remote_args={"num_cpus": self._PER_TASK_NUM_CPUS_ALLOCATION},
        )

        return LogicalPlan(count_rows_op, plan._context)
