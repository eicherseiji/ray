import functools
from typing import Callable, List, Optional

from ray.anyscale.data.checkpoint.util import (
    CHECKPOINTED_IDS_KWARG_NAME,
    filter_checkpointed_rows_for_blocks,
)
from ray.data._internal.execution.interfaces import PhysicalOperator
from ray.data._internal.execution.operators.input_data_buffer import InputDataBuffer
from ray.data._internal.execution.operators.map_operator import MapOperator
from ray.data._internal.execution.operators.map_transformer import (
    BlockMapTransformFn,
    BuildOutputBlocksMapTransformFn,
    MapTransformer,
)
from ray import ObjectRef
from ray.data._internal.logical.operators.from_operators import AbstractFrom
from ray.data.context import DataContext


def plan_from_op_with_checkpoint_filter(
    op: AbstractFrom,
    physical_children: List[PhysicalOperator],
    data_context: DataContext,
    load_checkpoint: Optional[Callable[[], ObjectRef]] = None,
) -> PhysicalOperator:
    assert len(physical_children) == 0

    input_operator = InputDataBuffer(data_context, op.input_data)
    transform_fns = [
        BlockMapTransformFn(
            functools.partial(
                filter_checkpointed_rows_for_blocks,
                checkpoint_config=data_context.checkpoint_config,
            )
        ),
        BuildOutputBlocksMapTransformFn.for_blocks(),
    ]
    map_transformer = MapTransformer(transform_fns, init_fn=None)
    map_operator = MapOperator.create(
        map_transformer=map_transformer,
        input_op=input_operator,
        data_context=data_context,
        name="FilterCheckpointedRows",
    )

    if load_checkpoint is not None:
        map_operator.add_map_task_kwargs_fn(
            lambda: {CHECKPOINTED_IDS_KWARG_NAME: load_checkpoint()}
        )

    return map_operator
