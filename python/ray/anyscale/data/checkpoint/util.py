from typing import Union, Iterable
import urllib.parse

import pyarrow as pa
import numpy as np

from ray.anyscale.data.checkpoint.checkpoint_filter import (
    BatchBasedCheckpointFilter,
    RowBasedCheckpointFilter,
)
from ray.anyscale.data.checkpoint.interfaces import (
    CheckpointConfig,
)
from ray.data._internal.execution.interfaces.task_context import TaskContext
from ray.data.block import Block, BlockAccessor, DataBatch

CHECKPOINTED_IDS_KWARG_NAME = "checkpointed_ids"

# Type for generated ID
GENERATED_ID_COLUMN_TYPE = pa.string()


def filter_checkpointed_rows_for_blocks(
    blocks: Iterable[Block],
    task_context: TaskContext,
    checkpoint_config: CheckpointConfig,
) -> Iterable[Block]:
    """For each block, filter rows that have already been checkpointed
    and yield the resulting block."""
    if checkpoint_config.is_batch_based():
        ckpt_filter = BatchBasedCheckpointFilter(checkpoint_config)
        checkpointed_ids = task_context.kwargs[CHECKPOINTED_IDS_KWARG_NAME]

        def filter_fn(block):
            return ckpt_filter.filter_rows_for_block(
                block,
                checkpointed_ids,
            )

    else:
        ckpt_filter = RowBasedCheckpointFilter.create(checkpoint_config)

        def filter_fn(block):
            return ckpt_filter.filter_rows_for_block(block)

    for block in blocks:
        filtered_block = filter_fn(block)
        ba = BlockAccessor.for_block(filtered_block)
        if ba.num_rows() > 0:
            yield filtered_block


def filter_checkpointed_rows_for_batches(
    batches: Iterable[DataBatch],
    task_context: TaskContext,
    checkpoint_config: CheckpointConfig,
) -> Iterable[DataBatch]:
    """For each batch, filter rows that have already been checkpointed
    and yield the resulting batches."""
    if checkpoint_config.is_batch_based():
        ckpt_filter = BatchBasedCheckpointFilter(checkpoint_config)
        checkpointed_ids = task_context.kwargs[CHECKPOINTED_IDS_KWARG_NAME]

        def filter_fn(batch):
            return ckpt_filter.filter_rows_for_batch(
                batch, checkpointed_ids=checkpointed_ids
            )

    else:
        ckpt_filter = RowBasedCheckpointFilter.create(checkpoint_config)

        def filter_fn(batch):
            return ckpt_filter.filter_rows_for_batch(batch)

    for batch in batches:
        filtered_batch = filter_fn(batch)
        yield filtered_batch


def get_generated_id_column(
    path: str, current_row_offset: int, num_rows: int
) -> pa.StringArray:
    """Helper function to get generated ID column string array from path
       information.

    Args:
        path: Full path to the file
        current_row_offset: Current row offset for sequential IDs
        num_rows: Number of rows in the current batch

    Returns:
        PyArrow StringArray with row ID strings
    """
    # Create string IDs in format: "/path/to/file/row_id"
    row_ids = np.arange(current_row_offset, current_row_offset + num_rows).astype(str)
    full_prefix = f"{path}/"
    row_id_strings = np.char.add(full_prefix, row_ids)
    return pa.array(row_id_strings, type=pa.string())


def normalize_id(id: Union[str, int]) -> str:
    """Normalize an ID for use as a filename.

    Args:
        id: The ID (string or int)

    Returns:
        A normalized string safe for use as a filename
    """
    if isinstance(id, int):
        return str(id)

    # For string IDs, URL encode to handle path separators and special characters
    return urllib.parse.quote(id, safe="")
