import abc
import logging
import time
from typing import List, Optional

import numpy
import pyarrow
import pyarrow.compute as pc

import ray
from ray.anyscale.data.checkpoint.interfaces import CheckpointBackend, CheckpointConfig
from ray.anyscale.data.checkpoint.util import (
    GeneratedIdFieldIndex,
    get_struct_field_index,
    FILE_NAME_FIELD,
    GENERATED_ID_FIELD_MAPPING,
    PATH_PREFIX_FIELD,
    ROW_GROUP_FIELD,
    NUM_ROWS_FIELD,
    ROW_ID_FIELD,
    CHECKPOINTED_FRAGMENT_COLUMN_NAME,
    CHECKPOINTED_ROW_COUNT_COLUMN_NAME,
    CHECKPOINTED_ROW_IDS_COLUMN_NAME,
    CHECKPOINTED_GENERATED_ID_COLUMN_TABLE_SCHEMA,
)
from ray.data._internal.execution.interfaces.ref_bundle import RefBundle
from ray.data.block import Block, BlockAccessor, DataBatch, BlockMetadata, Schema
from ray.data.datasource.path_util import _unwrap_protocol
from ray.types import ObjectRef
from ray.data import DataContext
from ray.data.context import ShuffleStrategy
from ray.data.datasource import PathPartitionFilter


logger = logging.getLogger(__name__)


class CheckpointFilter(abc.ABC):
    """Abstract class which defines the interface for filtering checkpointed rows
    based on varying backends.
    """

    def __init__(self, config: CheckpointConfig):
        self.ckpt_config = config
        self.checkpoint_path = self.ckpt_config.checkpoint_path
        self.checkpoint_path_unwrapped = _unwrap_protocol(
            self.ckpt_config.checkpoint_path
        )
        self.id_column = self.ckpt_config.id_column
        self.generated_id_column = self.ckpt_config.generated_id_column
        self.filesystem = self.ckpt_config.filesystem
        self.filter_num_threads = self.ckpt_config.filter_num_threads


class RowBasedCheckpointFilter(CheckpointFilter):
    """CheckpointFiter for row-based backends."""

    @staticmethod
    def create(config: CheckpointConfig) -> "RowBasedCheckpointFilter":
        """Factory method to create a `RowBasedCheckpointFilter` based on the
        provided `CheckpointConfig`."""
        assert config.is_row_based()
        backend = config.backend
        if backend == CheckpointBackend.CLOUD_OBJECT_STORAGE_ROW:
            from ray.anyscale.data.checkpoint.checkpoint_cloud_object_storage_row import (
                RowBasedCloudObjectStorageCheckpointFilter,
            )

            return RowBasedCloudObjectStorageCheckpointFilter(config)
        if backend == CheckpointBackend.FILE_STORAGE_ROW:
            from ray.anyscale.data.checkpoint.checkpoint_file_storage_row import (
                RowBasedFileStorageCheckpointFilter,
            )

            return RowBasedFileStorageCheckpointFilter(config)

        raise NotImplementedError(f"Backend {backend} not implemented")

    @abc.abstractmethod
    def filter_rows_for_block(self, block: Block) -> Block:
        """For the given block, filter out rows that have already
        been checkpointed, and return the resulting block.

        Subclasses must implement this method.

        Args:
            block: The input block to filter.
        Returns:
            A new block with rows that have not been checkpointed.
        """
        ...

    def filter_rows_for_batch(self, batch: DataBatch) -> DataBatch:
        """For the given batch, filter out rows that have already
        been checkpointed, and return the resulting batch.

        Note that this method calls `filter_rows_for_block()` under the hood,
        so it is preferred to call that method directly if you already have a block.
        """
        arrow_block = BlockAccessor.batch_to_block(batch)
        filtered_block = self.filter_rows_for_block(arrow_block)
        filtered_batch = BlockAccessor.for_block(filtered_block).to_batch_format(None)
        return filtered_batch


@ray.remote(max_retries=-1)
def _combine_chunks(ckpt_block: pyarrow.Table) -> pyarrow.Table:
    """Combine chunks for the checkpoint block.

    Args:
        ckpt_block: The checkpoint block to combine chunks for

    Returns:
        The combined checkpoint block
    """
    from ray.data._internal.arrow_ops.transform_pyarrow import combine_chunks

    combined_ckpt_block = combine_chunks(ckpt_block)
    logger.debug(
        "Checkpoint block stats for id column checkpoint: Combined block: type=%s, %d rows, %d bytes",
        combined_ckpt_block.schema.to_string(),
        combined_ckpt_block.num_rows,
        combined_ckpt_block.nbytes,
    )

    return combined_ckpt_block


class CheckpointLoader:
    """Loading checkpoint data."""

    def __init__(
        self,
        checkpoint_path: str,
        filesystem: pyarrow.fs.FileSystem,
        id_column: str,
        checkpoint_path_partition_filter: Optional[PathPartitionFilter] = None,
    ):
        """Initialize the CheckpointLoader.

        Args:
            checkpoint_path: The path to the checkpoint
            filesystem: The filesystem to use
            id_column: The name of the ID column
            checkpoint_path_partition_filter: Filter for checkpoint files to load during
                restoration when reading from `checkpoint_path`.
        """
        self.checkpoint_path = checkpoint_path
        self.filesystem = filesystem
        self.id_column = id_column
        self.checkpoint_path_partition_filter = checkpoint_path_partition_filter

    def load_checkpoint(self) -> ObjectRef[Block]:
        """Loading checkpoint data.

        Returns:
            ObjectRef[Block]: ObjectRef to the checkpointed IDs block.
        """
        start_t = time.time()

        # Load the checkpoint data
        checkpoint_ds: ray.data.Dataset = ray.data.read_parquet(
            self.checkpoint_path,
            filesystem=self.filesystem,
            partition_filter=self.checkpoint_path_partition_filter,
        )

        # Pre-process data pipeline
        checkpoint_ds: ray.data.Dataset = self._preprocess_data_pipeline(checkpoint_ds)

        # Repartition to 1 block.
        checkpoint_ds = checkpoint_ds.repartition(num_blocks=1)

        # Get the block reference
        ref_bundles: List[RefBundle] = list(checkpoint_ds.iter_internal_ref_bundles())
        assert len(ref_bundles) == 1
        ref_bundle: RefBundle = ref_bundles[0]
        schema: Schema = ref_bundle.schema
        assert len(ref_bundle.blocks) == 1
        block_ref: ObjectRef[Block] = ref_bundle.blocks[0][0]
        metadata: BlockMetadata = ref_bundle.blocks[0][1]

        # Post-process the block
        checkpoint_block_ref: ObjectRef[Block] = self._postprocess_block(block_ref)

        if (
            isinstance(self, GeneratedIdColumnCheckpointLoader)
            and metadata.num_rows > 0
        ):
            assert (
                schema == CHECKPOINTED_GENERATED_ID_COLUMN_TABLE_SCHEMA
            ), f"Schema mismatch: {schema} != {CHECKPOINTED_GENERATED_ID_COLUMN_TABLE_SCHEMA}"

        logger.info(
            "Checkpoint loaded for %s in %.2f seconds. NumRows = %d, SizeBytes = %d, Schema = %s",
            type(self).__name__,
            time.time() - start_t,
            metadata.num_rows,
            metadata.size_bytes,
            schema.to_string(),
        )

        return checkpoint_block_ref

    @abc.abstractmethod
    def _preprocess_data_pipeline(
        self, checkpoint_ds: ray.data.Dataset
    ) -> ray.data.Dataset:
        """Pre-process the checkpoint dataset. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement this method")

    def _postprocess_block(self, block_ref: ObjectRef[Block]) -> ObjectRef[Block]:
        """Combine the block so it has fewer chunks."""
        return _combine_chunks.remote(block_ref)


class IdColumnCheckpointLoader(CheckpointLoader):
    """Loader for regular ID columns."""

    def _preprocess_data_pipeline(
        self, checkpoint_ds: ray.data.Dataset
    ) -> ray.data.Dataset:
        """In the pre-process data pipeline,
            - Sort by the IDs, as `filter_rows_for_block` will perform binary search on the
              checkpointed IDs during restore.

        Args:
            checkpoint_ds: The checkpoint dataset to pre-process

        Returns:
            The pre-processed checkpoint dataset
        """
        # Sort by the ID column.
        return checkpoint_ds.sort(self.id_column)


class GeneratedIdColumnCheckpointLoader(CheckpointLoader):
    """Loader for generated ID columns."""

    def _extract_grouping_fields(self, batch: pyarrow.Table) -> pyarrow.Table:
        """Extract the fields needed for grouping from the checkpoint data.

        Args:
            batch: PyArrow table containing checkpoint data with ID column

        Returns:
            PyArrow table with extracted grouping fields and ID column
        """
        id_col: pyarrow.ChunkedArray = batch[self.id_column]

        # Extract path_prefix
        path_prefix_idx: int = get_struct_field_index(id_col, PATH_PREFIX_FIELD)
        path_prefix = pc.struct_field(id_col, [path_prefix_idx])

        # Extract file_name
        file_name_idx: int = get_struct_field_index(id_col, FILE_NAME_FIELD)
        file_name = pc.struct_field(id_col, [file_name_idx])

        # Extract row_group
        row_group_idx: int = get_struct_field_index(id_col, ROW_GROUP_FIELD)
        row_group = pc.struct_field(id_col, [row_group_idx])

        return pyarrow.Table.from_arrays(
            [
                path_prefix.cast(pyarrow.large_string()),
                file_name.cast(pyarrow.large_string()),
                row_group.cast(pyarrow.int32()),
                batch[self.id_column],
            ],
            names=[
                PATH_PREFIX_FIELD,
                FILE_NAME_FIELD,
                ROW_GROUP_FIELD,
                self.id_column,
            ],
        )

    def _process_row_group(self, row_group_batch: pyarrow.Table) -> pyarrow.Table:
        """Process a single group (file fragment) to create a single row with the
        following columns:
        - path_prefix
        - file_name
        - row_group
        - num_rows
        - checkpointed_row_count
        - checkpointed_row_ids sorted list

        Args:
            row_group_batch: PyArrow table containing rows from a single group, i.e. file
            fragment, i.e. path_prefix/file_name/row_group=<row_group>

        Returns:
            PyArrow table with 1 row containing the specified columns
        """
        # Create CHECKPOINTED_FRAGMENT_COLUMN_NAME with format:
        # /path/to/file/row_group=<row_group>
        path_prefix = row_group_batch[PATH_PREFIX_FIELD][0].as_py()
        file_name = row_group_batch[FILE_NAME_FIELD][0].as_py()
        row_group = row_group_batch[ROW_GROUP_FIELD][0].as_py()
        checkpoint_fragment_col = pyarrow.array(
            [f"{path_prefix}/{file_name}/row_group={row_group}"]
        )

        id_column = row_group_batch[self.id_column]

        # Extract NUM_ROWS_FIELD from the generated ID column
        num_rows_field_idx = get_struct_field_index(id_column, NUM_ROWS_FIELD)
        num_rows = pc.struct_field(id_column, [num_rows_field_idx]).to_numpy()[0]

        # Extract ROW_ID_FIELD from the generated ID column and sort as list of ints.
        # Note that row IDs are not in sorted order, so we need to sort them.
        row_id_field_idx = get_struct_field_index(id_column, ROW_ID_FIELD)
        checkpointed_row_ids = pc.struct_field(
            id_column, [row_id_field_idx]
        ).combine_chunks()
        sort_indices = pc.sort_indices(checkpointed_row_ids)
        sorted_checkpointed_row_ids = pc.take(checkpointed_row_ids, sort_indices).cast(
            pyarrow.int32()
        )
        actual_count = len(sorted_checkpointed_row_ids)

        # Create boolean array where each position represents a row
        # True = checkpointed, False = not checkpointed
        if actual_count == num_rows:
            # All rows checkpointed - use empty list for efficiency
            # Wrap empty list in array to match expected length 1
            checkpointed_row_ids_col = pyarrow.array(
                [[]], type=pyarrow.list_(pyarrow.bool_())
            )
        else:
            # Create row indices array [0, 1, 2, ..., num_rows-1]
            row_indices = pyarrow.array(numpy.arange(num_rows), type=pyarrow.int32())

            # Use is_in to check which row indices are in the checkpointed row IDs
            # is_in returns True for values that exist in the second array
            boolean_array = pc.is_in(row_indices, sorted_checkpointed_row_ids)

            # Wrap the boolean array in a ListArray to match the expected schema
            # The schema expects a list<item: bool> column, so we need to wrap it.

            # Wrap that as a single List<bool>: offsets [0, N]
            offsets = pyarrow.array([0, len(boolean_array)], type=pyarrow.int64())
            checkpointed_row_ids_col = pyarrow.ListArray.from_arrays(
                offsets, boolean_array
            )

        logger.debug(
            "Group processing stats - Fragment: %s/%s/row_group=%d, Total rows: %d, Checkpointed: %d, BooleanArray: %s",
            path_prefix,
            file_name,
            row_group,
            num_rows,
            actual_count,
            (
                f"empty_list({checkpointed_row_ids_col.type})"
                if actual_count == num_rows
                else f"scattered[{actual_count}/{num_rows}]({checkpointed_row_ids_col.type})"
            ),
        )

        # Return exactly 1 row per group with the specified columns
        return pyarrow.Table.from_arrays(
            [
                # checkpointed_fragment
                checkpoint_fragment_col,
                # num_rows
                pyarrow.array([num_rows], type=pyarrow.int32()),
                # checkpointed_row_count
                pyarrow.array([actual_count], type=pyarrow.int32()),
                # checkpointed_row_ids
                checkpointed_row_ids_col,
            ],
            names=[
                CHECKPOINTED_FRAGMENT_COLUMN_NAME,
                NUM_ROWS_FIELD,
                CHECKPOINTED_ROW_COUNT_COLUMN_NAME,
                CHECKPOINTED_ROW_IDS_COLUMN_NAME,
            ],
        )

    def _preprocess_data_pipeline(
        self, checkpoint_ds: ray.data.Dataset
    ) -> ray.data.Dataset:
        """Pre-process generated ID columns with grouping and sorting.

        Args:
            checkpoint_ds: The checkpoint dataset to pre-process

        Returns:
            The pre-processed checkpoint dataset
        """
        # Map batches to extract grouping fields.
        checkpoint_ds = checkpoint_ds.map_batches(
            self._extract_grouping_fields,
            batch_format="pyarrow",
            batch_size=None,
        )
        # Set shuffle strategy to hash shuffle for groupby
        checkpoint_ds.context._shuffle_strategy = ShuffleStrategy.HASH_SHUFFLE

        # Group by path_prefix, file_name, and row_group
        checkpoint_ds = checkpoint_ds.groupby(
            [
                GENERATED_ID_FIELD_MAPPING[GeneratedIdFieldIndex.PATH_PREFIX],
                GENERATED_ID_FIELD_MAPPING[GeneratedIdFieldIndex.FILE_NAME],
                GENERATED_ID_FIELD_MAPPING[GeneratedIdFieldIndex.ROW_GROUP],
            ]
        )

        # Process each group and reduce to a single row per group
        checkpoint_ds = checkpoint_ds.map_groups(
            self._process_row_group, batch_format="pyarrow"
        )

        # Sort by CHECKPOINTED_FRAGMENT_COLUMN_NAME.
        checkpoint_ds = checkpoint_ds.sort(CHECKPOINTED_FRAGMENT_COLUMN_NAME)

        return checkpoint_ds


class BatchBasedCheckpointFilter(CheckpointFilter):
    """CheckpointFilter for batch-based backends."""

    def load_checkpoint(self) -> ObjectRef[Block]:
        """Load checkpointed ids as a sorted block.

        Returns:
            ObjectRef[Block]: ObjectRef to the checkpointed IDs block.
        """
        # Override checkpointing here since we are loading the checkpoint metadata and should not generate ID column.
        # TODO: Clean way to do this would be to introduce per Op config [https://github.com/ray-project/ray/issues/54520]
        data_context = DataContext.get_current()
        if self.generated_id_column:
            data_context.checkpoint_enabled_override = True

        if self.generated_id_column:
            loader = GeneratedIdColumnCheckpointLoader(
                checkpoint_path=self.checkpoint_path,
                filesystem=self.filesystem,
                id_column=self.id_column,
                checkpoint_path_partition_filter=self.ckpt_config.checkpoint_path_partition_filter,
            )
        else:
            loader = IdColumnCheckpointLoader(
                checkpoint_path=self.checkpoint_path,
                filesystem=self.filesystem,
                id_column=self.id_column,
                checkpoint_path_partition_filter=self.ckpt_config.checkpoint_path_partition_filter,
            )
        return loader.load_checkpoint()

    def delete_checkpoint(self) -> None:
        self.filesystem.delete_dir(self.checkpoint_path_unwrapped)

    def filter_rows_for_block(
        self,
        block: Block,
        checkpointed_ids: Block,
    ) -> Block:
        """For the given block, filter out rows that have already
        been checkpointed, and return the resulting block.

        Args:
            block: The input block to filter.
            checkpointed_ids: A block containing IDs of all rows that have
                been checkpointed.
        Returns:
            A new block with rows that have not been checkpointed.
        """

        if len(checkpointed_ids) == 0 or len(block) == 0:
            return block

        assert isinstance(block, pyarrow.Table)
        assert isinstance(checkpointed_ids, pyarrow.Table)

        if self.generated_id_column:
            # For generated id column, filter out rows that have already been checkpointed
            # in the read task.
            return block

        # The checkpointed_ids block is sorted (see load_checkpoint).
        # We'll use binary search to filter out processed rows.
        # And we process a single chunk at a time, otherwise `to_numpy` below
        # will copy the data from shared memory to worker's heap memory.

        import concurrent.futures

        # Get all chunks of the checkpointed ID column.
        ckpt_chunks = checkpointed_ids[self.id_column].chunks
        # Convert the block's ID column to a numpy array for fast processing.
        block_ids = block[self.id_column].to_numpy()

        def filter_with_ckpt_chunk(ckpt_chunk: pyarrow.ChunkedArray) -> numpy.ndarray:
            # Convert checkpoint chunk to numpy for fast search.
            ckpt_ids = ckpt_chunk.to_numpy(zero_copy_only=True)
            # Start with a mask of all True (keep all rows).
            mask = numpy.ones(len(block_ids), dtype=bool)
            # Use binary search to find where block_ids would be in ckpt_ids.
            sorted_indices = numpy.searchsorted(ckpt_ids, block_ids)
            # Only consider indices that are within bounds.
            valid_indices = sorted_indices < len(ckpt_ids)
            # For valid indices, check for exact matches.
            potential_matches = sorted_indices[valid_indices]
            matched = ckpt_ids[potential_matches] == block_ids[valid_indices]
            # Mark matched IDs as False (filter out these rows).
            mask[valid_indices] = ~matched
            # Delete the chunk to free memory.
            del ckpt_chunk
            return mask

        # Use ThreadPoolExecutor to process each checkpoint chunk in parallel.
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.filter_num_threads or None
        ) as executor:
            masks = list(executor.map(filter_with_ckpt_chunk, ckpt_chunks))

        # Combine all masks using logical AND (row must not be in any checkpoint chunk).
        final_mask = numpy.logical_and.reduce(masks)
        # Convert the final mask to a PyArrow array and filter the block.
        mask_array = pyarrow.array(final_mask)
        filtered_block = block.filter(mask_array)
        return filtered_block

    def filter_rows_for_batch(
        self,
        batch: DataBatch,
        checkpointed_ids: Block,
    ) -> DataBatch:
        """For the given batch, filter out rows that have already
        been checkpointed, and return the resulting batch.

        Note that this method calls `filter_rows_for_block()` under the hood,
        so it is preferred to call that method directly if you already have a block.
        """
        arrow_block = BlockAccessor.batch_to_block(batch)
        filtered_block = self.filter_rows_for_block(arrow_block, checkpointed_ids)
        filtered_batch = BlockAccessor.for_block(filtered_block).to_batch_format(None)
        return filtered_batch
