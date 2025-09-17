import functools
from typing import Callable, List, Optional

from ray.anyscale.data._internal.logical.operators.read_files_operator import ReadFiles
from ray.anyscale.data._internal.readers.parquet_reader import ParquetReader
from ray.anyscale.data._internal.planner.plan_read_files_op import plan_read_files_op
from ray.anyscale.data.checkpoint.util import (
    CHECKPOINTED_IDS_KWARG_NAME,
    filter_checkpointed_rows_for_batches,
)
from ray.data import DataContext
from ray.data._internal.execution.interfaces import PhysicalOperator
from ray.data._internal.execution.operators.map_transformer import (
    BatchMapTransformFn,
)
from ray import ObjectRef
from ray.data._internal.output_buffer import OutputBlockSizeOption


def plan_read_files_op_with_checkpoint_filter(
    op: ReadFiles,
    physical_children: List[PhysicalOperator],
    data_context: DataContext,
    load_checkpoint: Optional[Callable[[], ObjectRef]] = None,
) -> PhysicalOperator:
    if (
        data_context.checkpoint_config is not None
        and data_context.checkpoint_config.generated_id_column
    ):
        assert isinstance(op.reader, ParquetReader), (
            f"For checkpointing with `generated_id_column`, ReadFiles operator must use a "
            f"ParquetReader, but got {type(op.reader)}"
        )

    physical_op = plan_read_files_op(op, physical_children, data_context)

    # TODO avoid modifying in-place
    physical_op._map_transformer.add_transform_fns(
        [
            BatchMapTransformFn(
                functools.partial(
                    filter_checkpointed_rows_for_batches,
                    checkpoint_config=data_context.checkpoint_config,
                ),
                output_block_size_option=OutputBlockSizeOption.of(
                    target_max_block_size=data_context.target_max_block_size,
                ),
            )
        ]
    )

    if load_checkpoint is not None:
        # Checkpoint ObjectRef is resolved by Ray Core and the task is run only
        # after the object is loaded.
        physical_op.add_map_task_kwargs_fn(
            lambda: {CHECKPOINTED_IDS_KWARG_NAME: load_checkpoint()}
        )

    return physical_op
