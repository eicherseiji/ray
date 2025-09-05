import functools
import logging
import os
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, Iterator, Iterable, List, Optional, Set, Tuple

import numpy as np
import pyarrow
import pyarrow as pa
import pyarrow.dataset

from ray.anyscale.data._internal.logical.operators.list_files_operator import (
    FileManifest,
)
from ray.anyscale.data._internal.file_indexer import (
    ChunkMetadata,
    create_chunk_metadata,
)
from ray.data._internal.datasource.parquet_datasource import (
    PARQUET_ENCODING_RATIO_ESTIMATE_DEFAULT,
    ParquetDatasource,
    _infer_schema,
    check_for_legacy_tensor_type,
    emit_file_extensions_future_warning,
    get_parquet_dataset,
    _read_batches_from,
)
from ray.data._internal.util import (
    call_with_retry,
    iterate_with_retry,
    make_async_gen,
)
from ray.data.block import Block, BlockMetadata, DataBatch, Schema
from ray.data.context import DataContext, MAX_SAFE_BLOCK_SIZE_FACTOR
from ray.data.datasource import Partitioning, PathPartitionParser
from ray.data.datasource.path_util import _has_file_extension
from ray.util.debug import log_once
from ray.anyscale.data.checkpoint.util import (
    CheckpointedFragmentInfo,
    get_checkpointed_fragment_info,
    get_generated_id_column,
    GENERATED_ID_COLUMN_TYPE,
    exclude_checkpointed_rows,
)

from .file_reader import FileReader
from .in_memory_size_estimator import (
    InMemorySizeEstimator,
)
from .supports_metadata import MetadataType, SupportsMetadata, SupportsSchema


logger = logging.getLogger(__name__)


class ParquetFileChunkMetadata(ChunkMetadata):
    """Metadata for Parquet file chunks.

    For a parquet file, the chunks are based on the total size of the file, not on the
    underlying row groups. We will split a file into potentially many chunks of the
    target chunk size. This may correspond to  0, 1, or more row groups per chunk.
    """

    chunk_idx: int
    total_num_chunks: int


class ParquetFileChunker:
    """File chunker for Parquet files.

    This chunker splits Parquet files into estimated number of chunks. We will not fetch the metadata for the
    file so we might overestimate the number of chunks to be more than the actual number of underlying row
    groups. The partitioner will create groupings based on the assumptions we make here, and the reader will
    fetch the metadata and ensure that all row groups are read / any overestimated row groups are ignored.
    """

    # This was chosen so that we can effectively chunk files but will also not result in
    # OOMs if the compression ratio is high.
    #
    # If the compression ratio is high and this chunk size is large, we will end up with
    # larger chunks than we need and when we go to read we might get an OOM. By reducing
    # the chunk size we can get better memory performance by reading a smaller fraction
    # of row groups at a time.
    #
    # We also want to keep this large enough such that we don't end up reading too much
    # data if we underestimate the number of chunks. If the row groups are larger than
    # the chunk size and we place many of them in the same read task, the total amount
    # of data read by the read task might be larger than expected. By increasing the size
    # of the chunks, we will be less likely to put many such row groups in the same task.
    _DEFAULT_TARGET_CHUNK_SIZE = 256 * 1024 * 1024  # 256MB

    def __init__(self, target_chunk_size: Optional[int] = None):
        # Initialize the chunker with a target chunk size, use this order of precedence:
        # 1. Target chunk size passed in to the constructor
        # 2. Environment variable RAY_TURBO_PARQUET_CHUNKER_TARGET_CHUNK_SIZE
        # 3. Default target chunk size
        if target_chunk_size is not None:
            self._target_chunk_size = target_chunk_size
        elif os.environ.get("RAY_TURBO_PARQUET_CHUNKER_TARGET_CHUNK_SIZE"):
            self._target_chunk_size = int(
                os.environ.get("RAY_TURBO_PARQUET_CHUNKER_TARGET_CHUNK_SIZE")
            )
        else:
            self._target_chunk_size = self._DEFAULT_TARGET_CHUNK_SIZE

    def generate_chunk_metadatas(
        self, path: str, file_size: int
    ) -> Iterable[Tuple[Optional[ChunkMetadata], int]]:
        if file_size <= self._target_chunk_size:
            # do not chunk if the file is smaller than the target chunk size, when we read the file this
            # will prevent us from additional metadata fetching since we want to read the entire file.
            yield None, file_size
            return

        num_chunks = math.ceil(file_size / self._target_chunk_size)
        for i in range(num_chunks):
            yield create_chunk_metadata(
                ParquetFileChunkMetadata,
                chunk_idx=i,
                total_num_chunks=num_chunks,
            ), self._target_chunk_size


class ParquetReader(FileReader, SupportsMetadata, SupportsSchema):
    """Reads Parquet files.

    This file reader implementation leverages PyArrow's `ParquetDataset` and
    `ParquetFileFragment.to_batches` APIs to efficiently read Parquet files. It first
    creates fragments from the given paths and then reads batches from each fragment
    using multiple threads.
    """

    _NUM_THREADS_PER_TASK = 16

    # NOTE: This is a mostly arbitrary number. We might get better performance by tuning
    # this value.
    _COUNT_ROWS_BATCH_SIZE = 16

    def __init__(
        self,
        *,
        schema: Optional["pyarrow.Schema"],
        dataset_kwargs: Dict[str, Any],
        batch_size: Optional[int],
        to_batches_kwargs: Dict[str, Any],
        block_udf: Optional[Callable[[Block], Block]],
        include_paths: bool,
        partitioning: Optional[Partitioning],
        target_block_size: Optional[int],
    ):
        """Initialize the ParquetReader.

        Args:
            schema: An explicit user-provided schema. If not provided, the schema is
                inferred from the data.
            dataset_kwargs: Additional keyword arguments to pass to `ParquetDataset`
                when this class creates fragments.
            batch_size: The number of rows to read per batch. If not provided, a default
                value is used.
            to_batches_kwargs: Additional keyword arguments to pass to
                `ParquetFileFragment.to_batches`.
            block_udf: A function that takes a `Block` and returns a `Block`. This
                argument is required for legacy reasons.
            include_paths: Whether to include the file path in the output.
            partitioning: The partitioning scheme to use when reading the data.
            target_block_size: The target block size to use for reading the data.
        """
        super().__init__()

        self._schema = schema
        self._dataset_kwargs = dataset_kwargs
        self._batch_size = batch_size
        self._to_batches_kwargs = to_batches_kwargs
        self._block_udf = block_udf
        self._include_paths = include_paths
        self._partitioning = partitioning
        self._target_block_size = target_block_size

        # Users should use the top-level 'partitioning' argument instead of passing it
        # through 'dataset_kwargs'.
        assert "partitioning" not in dataset_kwargs
        # This reader adds partitions at the Ray Data-level. To prevent PyArrow from
        # adding partitions, we set the 'partitioning' to 'None'.
        self._dataset_kwargs["partitioning"] = None

        ctx = DataContext.get_current()

        self._should_preserve_order = ctx.execution_options.preserve_order
        self._retried_io_errors = ctx.retried_io_errors
        self._sampled_batch_size = None

        # Check if ID generation is requested via checkpoint config
        self._generated_id_column = None
        if (
            ctx.checkpoint_config
            and ctx.checkpoint_config.generated_id_column
            and not ctx.checkpoint_enabled_override
        ):
            self._generated_id_column = ctx.checkpoint_config.generated_id_column

    def read_files(
        self,
        file_manifest: FileManifest,
        *,
        filter_expr: Optional[pyarrow.dataset.Expression] = None,
        columns: Optional[List[str]] = None,
        columns_rename: Optional[Dict[str, str]] = None,
        filesystem: pyarrow.fs.FileSystem,
        checkpoint_ids: Optional[Block] = None,
    ) -> Iterable[DataBatch]:
        if columns and columns_rename:
            assert set(columns_rename.keys()).issubset(columns), (
                f"All column rename keys must be a subset of the columns list. "
                f"Invalid keys: {set(columns_rename.keys()) - set(columns)}"
            )

        paths = list(file_manifest.paths)
        for path in paths:
            if not _has_file_extension(
                path, ParquetDatasource._FUTURE_FILE_EXTENSIONS
            ) and log_once("read_parquet_file_extensions_future_warning"):
                emit_file_extensions_future_warning(
                    ParquetDatasource._FUTURE_FILE_EXTENSIONS
                )
                break
        chunk_metadatas = list(file_manifest.file_chunk_metadatas)

        fragments = self._create_fragments(
            paths, chunk_metadatas, filesystem=filesystem
        )

        if len(fragments) == 0:
            return

        # Check for column name collision with generated_id_column
        if self._generated_id_column:
            # Check collision with columns_rename mapping
            if columns_rename is not None:
                if self._generated_id_column in columns_rename:
                    raise ValueError(
                        f"generated_id_column='{self._generated_id_column}' conflicts with a column "
                        f"that will be renamed (original name)"
                    )
                if self._generated_id_column in columns_rename.values():
                    raise ValueError(
                        f"generated_id_column='{self._generated_id_column}' conflicts with a renamed "
                        f"column (target name)"
                    )

            # Check collision with existing columns
            field_index = fragments[0].physical_schema.get_field_index(
                self._generated_id_column
            )
            if field_index >= 0:
                if columns is not None and self._generated_id_column in columns:
                    raise ValueError(
                        f"generated_id_column='{self._generated_id_column}' conflicts with a column in the columns list"
                    )
                else:
                    raise ValueError(
                        f"generated_id_column='{self._generated_id_column}' conflicts with an existing column"
                    )

        # Users can pass both data columns and partition columns in the 'columns'
        # argument. To prevent PyArrow from complaining about missing columns, we
        # separate the partition columns from the data columns. When we read the
        # fragments, we pass the data columns to PyArrow and add the partition
        # columns manually.
        data_columns = None
        partition_columns = None
        if columns is not None:
            data_columns = [
                column
                for column in columns
                if column in fragments[0].physical_schema.names
            ]
            if self._partitioning is not None:
                parse = PathPartitionParser(self._partitioning)
                partitions = parse(fragments[0].path)
                partition_columns = [
                    column for column in columns if column in partitions
                ]

        num_threads = self._get_num_threads(fragments)
        if num_threads > 0:
            yield from make_async_gen(
                iter(fragments),
                functools.partial(
                    self._read_fragments,
                    filter_expr=filter_expr,
                    schema=self._schema,
                    data_columns=data_columns,
                    partition_columns=partition_columns,
                    columns_rename=columns_rename,
                    checkpoint_ids=checkpoint_ids,
                ),
                # NOTE: It's crucial for the sequence to have preserved (deterministic)
                #       ordering so that that tasks could be safely retried (when
                #       reconstructing lost blocks)
                preserve_ordering=True,
                num_workers=num_threads,
            )
        else:
            yield from self._read_fragments(
                fragments,
                filter_expr=filter_expr,
                schema=self._schema,
                data_columns=data_columns,
                partition_columns=partition_columns,
                columns_rename=columns_rename,
                checkpoint_ids=checkpoint_ids,
            )

    def _calculate_row_group_range(
        self, chunk_idx: int, total_num_chunks: int, total_row_groups: int
    ) -> Optional[Tuple[int, int]]:
        """Calculate the range of row groups for a given chunk.

        Distributes row groups as evenly as possible across chunks. If row groups
        don't divide evenly, earlier chunks get the extra row groups.

        Example:
            - 10 row groups, 3 chunks -> [0:4), [4:7), [7:10)
            - 11 row groups, 3 chunks -> [0:4), [4:8), [8:11)

        Args:
            chunk_idx: Index of the current chunk (0-based)
            total_num_chunks: Total number of chunks
            total_row_groups: Total number of row groups to distribute

        Returns:
            Tuple of (start_row_group, end_row_group) where end is exclusive,
            or None if chunk_idx is out of range (indicating no work for this chunk)
        """
        assert (
            total_row_groups >= 0
        ), f"total_row_groups must be non-negative, got {total_row_groups}"
        assert (
            total_num_chunks > 0
        ), f"total_num_chunks must be positive, got {total_num_chunks}"
        assert (
            chunk_idx < total_num_chunks
        ), f"chunk_idx must be less than total_num_chunks, got {chunk_idx} and {total_num_chunks}"
        assert chunk_idx >= 0, f"chunk_idx must be non-negative, got {chunk_idx}"

        # Handle case where chunk_idx is beyond the actual number of chunks needed
        # This can happen when we overestimate the number of chunks during planning
        if chunk_idx >= total_row_groups:
            return None

        base_row_groups_per_chunk = total_row_groups // total_num_chunks
        remainder = total_row_groups % total_num_chunks

        # Chunks 0 through (remainder-1) get one extra row group
        if chunk_idx < remainder:
            row_groups_in_this_chunk = base_row_groups_per_chunk + 1
            start = chunk_idx * row_groups_in_this_chunk
        else:
            row_groups_in_this_chunk = base_row_groups_per_chunk
            start = (
                remainder * (base_row_groups_per_chunk + 1)
                + (chunk_idx - remainder) * base_row_groups_per_chunk
            )

        end = start + row_groups_in_this_chunk

        # Verify our calculation doesn't go out of bounds
        assert (
            0 <= start <= end <= total_row_groups
        ), f"Invalid range [{start}, {end}) for {total_row_groups} row groups"

        return start, end

    def _fragments_from_chunk_metadata(
        self,
        fragment: pyarrow.dataset.ParquetFileFragment,
        chunk_metadata: ParquetFileChunkMetadata,
    ) -> List[pyarrow.dataset.ParquetFileFragment]:
        chunk_idx = chunk_metadata["chunk_idx"]
        total_num_chunks = chunk_metadata["total_num_chunks"]
        fragment_metadata = fragment.metadata
        total_row_groups = fragment_metadata.num_row_groups

        row_group_range = self._calculate_row_group_range(
            chunk_idx, total_num_chunks, total_row_groups
        )

        # Skip this chunk if it's out of range (can happen with overestimated chunk counts)
        # This gracefully handles cases where we estimated more chunks than actually needed
        if row_group_range is None:
            return []

        start, end = row_group_range

        # create a new fragment for each row group, this will allow us to read the row groups in parallel with threading
        fragments = []
        for row_group_index in range(start, end):
            fragments.append(fragment.subset(row_group_ids=[row_group_index]))
        return fragments

    def _create_fragments(
        self,
        paths: List[str],
        chunk_metadatas: List[ParquetFileChunkMetadata],
        *,
        filesystem: pa.fs.FileSystem,
    ) -> List[pyarrow.dataset.ParquetFileFragment]:
        deduped_paths = list(set(paths))
        parquet_dataset = call_with_retry(
            lambda: get_parquet_dataset(
                deduped_paths, filesystem, self._dataset_kwargs
            ),
            "create ParquetDataset",
            match=self._retried_io_errors,
        )
        path_to_fragment = {}
        for fragment in parquet_dataset.fragments:
            path_to_fragment[fragment.path] = fragment
        fragments = []
        for path, chunk_metadata in zip(paths, chunk_metadatas):
            fragment = path_to_fragment[path]
            if chunk_metadata is None:
                if self._generated_id_column:
                    # For checkpointing, we need to create a fragment for each row group.
                    for row_group_index in range(fragment.metadata.num_row_groups):
                        fragments.append(
                            fragment.subset(row_group_ids=[row_group_index])
                        )
                else:
                    fragments.append(fragment)
            else:
                if self._generated_id_column:
                    for fragment_item in self._fragments_from_chunk_metadata(
                        fragment, chunk_metadata
                    ):
                        assert len(fragment_item.row_groups) == 1, (
                            f"Expected each fragment to have exactly 1 row group when "
                            f"generated_id_column is set, but found fragment with "
                            f"{len(fragment_item.row_groups)} row groups"
                        )
                        fragments.append(fragment_item)
                else:
                    fragments.extend(
                        self._fragments_from_chunk_metadata(fragment, chunk_metadata)
                    )

        check_for_legacy_tensor_type(parquet_dataset.schema)
        return fragments

    def _get_num_threads(
        self, fragments: List[pyarrow.dataset.ParquetFileFragment]
    ) -> int:
        num_threads = self._NUM_THREADS_PER_TASK
        num_threads = min(num_threads, len(fragments))

        # TODO: We should refactor the code so that we can get the results in order even
        # when using multiple threads.
        if self._should_preserve_order:
            num_threads = 0

        return num_threads

    def _read_fragments(
        self,
        fragments: List[pyarrow.dataset.ParquetFileFragment],
        filter_expr: pyarrow.dataset.Expression,
        schema: pyarrow.Schema,
        data_columns: Optional[List[str]] = None,
        partition_columns: Optional[List[str]] = None,
        columns_rename: Optional[Dict[str, str]] = None,
        checkpoint_ids: Optional[Block] = None,
    ) -> Iterable["pyarrow.Table"]:
        for fragment in fragments:
            for table in self._read_batches(
                fragment,
                schema=schema,
                data_columns=data_columns,
                partition_columns=partition_columns,
                filter_expr=filter_expr,
                checkpoint_ids=checkpoint_ids,
            ):
                if columns_rename is not None:
                    table = table.rename_columns(
                        [columns_rename.get(col, col) for col in table.schema.names]
                    )

                yield table

    def _calculate_batch_size(self, avg_row_size: float, num_rows: int) -> int:
        """Calculate optimal batch size based on average row size and target block size.

        Args:
            avg_row_size: Average size per row in bytes
            num_rows: Number of rows in the data

        Returns:
            Calculated batch size (at least 1)
        """
        if num_rows == 0 or avg_row_size == 0:
            return 1

        total_memory_bytes = avg_row_size * num_rows
        if (
            self._target_block_size is not None
            and total_memory_bytes
            > MAX_SAFE_BLOCK_SIZE_FACTOR * self._target_block_size
        ):
            # If memory usage is large, calculate batch size based on target block size
            return max(1, int(self._target_block_size / avg_row_size))
        else:
            # If total memory usage is small, or target block size is not set,
            # consider batch size as num_rows.
            return max(1, num_rows)

    def _get_batch_size(
        self,
        fragment: pyarrow.dataset.ParquetFileFragment,
        target_column_indices: List[int],
    ) -> Optional[int]:
        """Calculate an optimal batch size from the first row-group stats.

        If ``target_block_size`` is ``None`` (i.e. unlimited block size),
        the full first row-group is read in a single batch.
        """
        if (
            # Handle the cases where there are no row-groups or rows.
            fragment.metadata is None
            or fragment.metadata.num_row_groups == 0
            or
            # Handle the case when we're reading out empty projection
            len(target_column_indices) == 0
        ):
            # Fallback to default batch size
            return None

        row_group_meta = fragment.metadata.row_group(0)
        row_group_num_rows = row_group_meta.num_rows

        if row_group_num_rows == 0:
            # Row group has no rows
            return None

        # Calculate row group size in bytes for the projected columns
        row_group_size_bytes = sum(
            row_group_meta.column(col_idx).total_uncompressed_size
            for col_idx in target_column_indices
        )

        # Estimate the in-memory size of the row group
        estimated_in_memory_row_group_size = (
            row_group_size_bytes * PARQUET_ENCODING_RATIO_ESTIMATE_DEFAULT
        )

        avg_row_size = estimated_in_memory_row_group_size / row_group_num_rows
        return self._calculate_batch_size(avg_row_size, row_group_num_rows)

    def _get_sampled_batch_size(self, table: pyarrow.Table) -> int:
        """Estimate the batch size based on the table size and the target block size."""
        if not table:
            return 1

        avg_row_size = table.nbytes / table.num_rows
        return self._calculate_batch_size(avg_row_size, table.num_rows)

    def _read_batches(
        self,
        fragment: pyarrow.dataset.ParquetFileFragment,
        *,
        schema: pyarrow.Schema,
        data_columns: Optional[List[str]],
        partition_columns: Optional[List[str]],
        filter_expr: pyarrow.dataset.Expression,
        checkpoint_ids: Optional[Block] = None,
    ) -> Iterable[pyarrow.Table]:
        checkpointed_fragment_info: Optional[CheckpointedFragmentInfo] = None
        if self._generated_id_column and checkpoint_ids is not None:
            assert len(fragment.row_groups) == 1
            checkpointed_fragment_info = get_checkpointed_fragment_info(
                fragment=fragment,
                row_group_idx=fragment.row_groups[0].id,
                checkpointed_ids=checkpoint_ids,
            )
            if checkpointed_fragment_info.fully_checkpointed:
                # Skip batching the fragment if all rows are checkpointed
                logger.debug(
                    "Skipping reading fragment %s row group %d because all rows are checkpointed",
                    fragment.path,
                    fragment.row_groups[0].id,
                )
                return
            else:
                # If the fragment is not fully checkpointed, we need to exclude the checkpointed rows from the table
                logger.debug(
                    "Exclude checkpointed rows from fragment %s, row group %d, "
                    "row count %d, checkpointed row count %d",
                    fragment.path,
                    fragment.row_groups[0].id,
                    fragment.metadata.row_group(0).num_rows,
                    checkpointed_fragment_info.checkpointed_row_count,
                )

        if self._batch_size is not None:
            batch_size = self._batch_size
        elif self._sampled_batch_size is not None:
            batch_size = self._sampled_batch_size
        else:
            # Get column indices for projected columns
            target_column_indices = (
                [fragment.physical_schema.get_field_index(col) for col in data_columns]
                if data_columns is not None
                else list(range(len(fragment.physical_schema)))
            )

            # Estimate batch size from first row group stats
            # TODO replace w/ PDS estimation
            batch_size = self._get_batch_size(fragment, target_column_indices)

        # Track the current row offset for the row IDs
        current_row_offset: int = 0

        # S3 can raise transient errors during iteration, and PyArrow doesn't expose a
        # way to retry specific batches.
        for table in iterate_with_retry(
            lambda: _read_batches_from(
                fragment,
                schema=schema,
                data_columns=data_columns,
                partition_columns=partition_columns,
                partitioning=self._partitioning,
                filter_expr=filter_expr,
                batch_size=batch_size,
                include_path=self._include_paths,
                use_threads=True,
                to_batches_kwargs=self._to_batches_kwargs,
            ),
            "ParquetReader read batches",
            match=self._retried_io_errors,
        ):
            if table.num_rows == 0:
                continue

            if self._generated_id_column:
                # Add generated ID column to the table
                assert len(fragment.row_groups) == 1
                table = table.append_column(
                    self._generated_id_column,
                    get_generated_id_column(
                        path=fragment.path,
                        row_group_idx=fragment.row_groups[0].id,
                        total_num_rows=fragment.metadata.row_group(0).num_rows,
                        current_row_offset=current_row_offset,
                        current_num_rows=table.num_rows,
                    ),
                )

                # Store the original offset used for generating row IDs
                original_offset = current_row_offset

                # Update the current row offset for the row IDs before excluding checkpointed rows
                num_rows_before_exclude = table.num_rows
                current_row_offset += table.num_rows

                # Exclude checkpointed rows from the table using the original offset
                if checkpointed_fragment_info is not None:
                    assert (
                        not checkpointed_fragment_info.fully_checkpointed
                    ), "Table should not be fully checkpointed"
                    table = exclude_checkpointed_rows(
                        table=table,
                        checkpointed_fragment_info=checkpointed_fragment_info,
                        current_row_offset=original_offset,
                        current_num_rows=num_rows_before_exclude,
                    )

                if num_rows_before_exclude > table.num_rows:
                    logger.debug(
                        "Excluded %d checkpointed rows from fragment %s, row group %d, range [%d, %d]",
                        num_rows_before_exclude - table.num_rows,
                        fragment.path,
                        fragment.row_groups[0].id,
                        original_offset,
                        original_offset + num_rows_before_exclude - 1,
                    )

                # Even if this fragment is fully checkpointed, the table may be empty because
                # the all the rows for this batch were checkpointed.
                if table.num_rows == 0:
                    continue

            if self._block_udf is not None:
                table = self._block_udf(table)
            if self._sampled_batch_size is None:
                # Note: _sampled_batch_size is only updated locally in each read task,
                # which means for each read task, we'll always use the
                # PARQUET_ENCODING_RATIO_ESTIMATE_DEFAULT-based batch size to read the
                # first file. Only the remaining files will be read with the sampled
                # batch size.
                #
                # Ideally, we can propagate the sampled batch size back to
                # the executor and use it for future tasks.
                self._sampled_batch_size = self._get_sampled_batch_size(table)

            yield table

    def read_metadata(
        self,
        file_manifest: FileManifest,
        *,
        filesystem: pyarrow.fs.FileSystem,
    ) -> Iterator[BlockMetadata]:
        parquet_dataset = call_with_retry(
            lambda: get_parquet_dataset(
                file_manifest.paths.tolist(), filesystem, self._dataset_kwargs
            ),
            "open ParquetDataset",
            match=self._retried_io_errors,
        )

        def get_metadata_for_path(
            fragment: "pa.dataset.ParquetFileFragment",
        ) -> int:
            # Getting the metadata requires network calls, so it might fail with
            # transient errors.
            num_rows = call_with_retry(
                lambda: fragment.metadata.num_rows,
                "fragment num_rows",
                match=self._retried_io_errors,
            )

            return num_rows

        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(get_metadata_for_path, fragment)
                for fragment in parquet_dataset.fragments
            ]
            for future in as_completed(futures):
                num_rows = future.result()
                metadata = BlockMetadata(
                    num_rows=num_rows,
                    size_bytes=None,
                    exec_stats=None,
                    input_files=None,
                )
                yield metadata

    def read_schema(
        self,
        file_manifest: FileManifest,
        *,
        filesystem: pyarrow.fs.FileSystem,
        columns: Optional[List[str]],
    ) -> "Schema":
        schema = self._schema
        parquet_dataset = call_with_retry(
            lambda: get_parquet_dataset(
                file_manifest.paths.tolist(), filesystem, self._dataset_kwargs
            ),
            "open ParquetDataset",
            match=self._retried_io_errors,
        )

        if not schema:
            schema = _infer_schema(
                parquet_dataset,
                None,
                columns,
                self._partitioning,
                self._block_udf,
            )

        # Add row ID column to schema if requested
        if self._generated_id_column:
            row_id_field = pa.field(self._generated_id_column, GENERATED_ID_COLUMN_TYPE)
            schema = pa.schema(list(schema) + [row_id_field])

        return schema

    def available_metadata(self) -> Set[MetadataType]:
        available = set()
        if "filter" not in self._to_batches_kwargs:
            available.add(MetadataType.NUM_ROWS)
            available.add(MetadataType.NUM_BYTES)
        return available

    def get_target_metadata_batch_size(self) -> int:
        return self._COUNT_ROWS_BATCH_SIZE

    def supports_predicate_pushdown(self) -> bool:
        return True


class ParquetInMemorySizeEstimator(InMemorySizeEstimator):
    def estimate_in_memory_sizes(self, manifest: FileManifest) -> np.ndarray:
        # Reading a batch of Parquet data can be slow, even if you try to read a single
        # row. To avoid slow startup times, just return a constant value. For more
        # information, see https://github.com/anyscale/rayturbo/issues/924.
        return PARQUET_ENCODING_RATIO_ESTIMATE_DEFAULT * manifest.file_sizes
