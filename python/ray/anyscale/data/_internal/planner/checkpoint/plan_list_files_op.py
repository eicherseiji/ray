from typing import Callable, List, Optional

from ray.anyscale.data._internal.logical.operators.list_files_operator import ListFiles
from ray.anyscale.data._internal.planner.plan_list_files_op import plan_list_files_op
from ray.data._internal.execution.interfaces import PhysicalOperator
from ray.data.context import DataContext
from ray.types import ObjectRef


def plan_list_files_op_with_checkpoint_filter(
    op: ListFiles,
    physical_children: List[PhysicalOperator],
    data_context: DataContext,
    load_checkpoint: Optional[Callable[[], ObjectRef]] = None,
) -> PhysicalOperator:
    # ListFiles operator doesn't have a reader attribute - it just lists files
    # The reader check should be done in the ReadFiles operator planner, not here

    physical_op = plan_list_files_op(
        op, physical_children, data_context, load_checkpoint
    )
    return physical_op
