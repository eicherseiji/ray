import abc
import logging
import time
from typing import List

import numpy
import pyarrow

import ray
from ray.anyscale.data.checkpoint.interfaces import CheckpointBackend, CheckpointConfig
from ray.data._internal.execution.interfaces.ref_bundle import RefBundle
from ray.data.block import Block, BlockAccessor, DataBatch
from ray.data.datasource.path_util import _unwrap_protocol
from ray.types import ObjectRef
from ray.data import DataContext


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
    # Combine chunks for the checkpoint block.
    from ray.data._internal.arrow_ops.transform_pyarrow import combine_chunks

    return combine_chunks(ckpt_block)


class BatchBasedCheckpointFilter(CheckpointFilter):
    """CheckpointFilter for batch-based backends."""

    def load_checkpoint(self) -> ObjectRef[Block]:
        """Load checkpointed ids as a sorted block."""
        start_t = time.time()

        # Override checkpointing here since we are loading the checkpoint metadata and should not generate ID column.
        # TODO: Clean way to do this would be to introduce per Op config [https://github.com/ray-project/ray/issues/54520]
        data_context = DataContext.get_current()
        if self.generated_id_column:
            data_context.checkpoint_enabled_override = True

        checkpoint_ds = (
            ray.data.read_parquet(self.checkpoint_path, filesystem=self.filesystem)
            .sort(self.id_column)  # Sort the IDs, as filter will use binary search.
            .repartition(1)
        )

        ref_bundles: List[RefBundle] = list(checkpoint_ds.iter_internal_ref_bundles())
        assert len(ref_bundles) == 1
        ref_bundle = ref_bundles[0]
        assert len(ref_bundle.blocks) == 1

        block_ref = ref_bundle.blocks[0][0]
        metadata = ref_bundle.blocks[0][1]

        # Combine the block so it has fewer chunks.
        res = _combine_chunks.remote(block_ref)

        logger.info(
            "Checkpoint loaded in %.2f seconds. Num rows = %d, size bytes = %d.",
            time.time() - start_t,
            metadata.num_rows,
            metadata.size_bytes,
        )

        return res

    def delete_checkpoint(self):
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

        # The checkpointed_ids block is sorted (see load_checkpoint).
        # We'll use binary search to filter out processed rows.
        # And we process a single chunk at a time, otherwise `to_numpy` below
        # will copy the data from shared memory to worker's heap memory.

        import concurrent.futures

        # Get all chunks of the checkpointed ID column.
        ckpt_chunks = checkpointed_ids[self.id_column].chunks
        # Convert the block's ID column to a numpy array for fast processing.
        block_ids = block[self.id_column].to_numpy()

        def filter_with_ckpt_chunk(ckpt_chunk):
            # Convert checkpoint chunk to numpy for fast search.
            if not self.generated_id_column:
                ckpt_ids = ckpt_chunk.to_numpy(zero_copy_only=True)
            else:
                # Generated row IDs are GENERATED_ID_COLUMN_TYPE, not ints, so
                # cannot use zero_copy_only=True.
                ckpt_ids = ckpt_chunk.to_numpy(zero_copy_only=False)
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
