from typing import Callable, List, Optional

from ray import ObjectRef
from ray.anyscale.data._internal.readers.parquet_reader import ParquetReader
from ray.data._internal.execution.interfaces import PhysicalOperator
from ray.data._internal.logical.operators.read_operator import Read
from ray.data._internal.planner.checkpoint import (
    plan_read_op_with_checkpoint_filter as _plan_read_op_with_checkpoint_filter,
)
from ray.data.context import DataContext


def plan_read_op_with_checkpoint_filter(
    op: Read,
    physical_children: List[PhysicalOperator],
    data_context: DataContext,
    load_checkpoint: Optional[Callable[[], ObjectRef]] = None,
) -> PhysicalOperator:
    # Anyscale-specific validation for generated_id_column
    if (
        data_context.checkpoint_config is not None
        and data_context.checkpoint_config.generated_id_column
    ):
        assert isinstance(op.datasource_or_legacy_reader, ParquetReader), (
            f"For checkpointing with `generated_id_column`, Read operator must use a "
            f"ParquetReader, but got {type(op.datasource_or_legacy_reader)}"
        )

    # Delegate to OSS implementation
    return _plan_read_op_with_checkpoint_filter(
        op, physical_children, data_context, load_checkpoint
    )
