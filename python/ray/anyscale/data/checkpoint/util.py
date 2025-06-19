from typing import Iterable

from ray.anyscale.data.checkpoint.interfaces import (
    BatchBasedCheckpointFilter,
    CheckpointConfig,
    RowBasedCheckpointFilter,
)
from ray.data._internal.execution.interfaces.task_context import TaskContext
from ray.data.block import Block, BlockAccessor, DataBatch

CHECKPOINTED_IDS_KWARG_NAME = "checkpointed_ids"


def filter_checkpointed_rows_for_blocks(
    blocks: Iterable[Block],
    task_context: TaskContext,
    checkpoint_config: CheckpointConfig,
) -> Iterable[Block]:
    """For each block, filter rows that have already been checkpointed
    and yield the resulting block."""
    if checkpoint_config.is_batch_based():
        ckpt_filter = BatchBasedCheckpointFilter.create(checkpoint_config)
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
        ckpt_filter = BatchBasedCheckpointFilter.create(checkpoint_config)
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
