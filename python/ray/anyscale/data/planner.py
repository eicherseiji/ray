import functools
import warnings
from typing import Callable, Dict, List, Type

from ray import ObjectRef
from ray.anyscale.data._internal.execution.operators.streaming_hash_aggregate import (
    StreamingHashAggregate,
)
from ray.anyscale.data._internal.logical.operators.list_files_operator import ListFiles
from ray.anyscale.data._internal.logical.operators.read_files_operator import ReadFiles
from ray.anyscale.data._internal.logical.operators.streaming_aggregate import (
    StreamingAggregate,
)
from ray.anyscale.data._internal.planner import (
    plan_list_files_op,
    plan_read_files_op,
)
from ray.anyscale.data._internal.planner.checkpoint import (
    plan_from_op_with_checkpoint_filter,
    plan_read_files_op_with_checkpoint_filter,
    plan_read_op_with_checkpoint_filter,
    plan_write_op_with_checkpoint_writer,
)
from ray.anyscale.data.checkpoint.load_checkpoint_callback import LoadCheckpointCallback
from ray.data._internal.execution.execution_callback import add_execution_callback
from ray.data._internal.execution.interfaces import PhysicalOperator
from ray.data._internal.logical.interfaces import (
    LogicalOperator,
    LogicalPlan,
    PhysicalPlan,
)
from ray.data._internal.logical.operators.from_operators import AbstractFrom
from ray.data._internal.logical.operators.read_operator import Read
from ray.data._internal.logical.operators.write_operator import Write
from ray.data._internal.planner.planner import (
    PlanLogicalOpFn,
    Planner,
    get_plan_logical_op_fns,
    plan_recursively,
    register_plan_logical_op_fn,
)
from ray.data.context import DataContext

_CHECKPOINT_FILTER_OPS = (Read, ReadFiles, AbstractFrom)


def _register_anyscale_plan_logical_op_fns():
    def plan_streaming_aggregate(
        logical_op: StreamingAggregate,
        physical_children: List[PhysicalOperator],
        data_context: DataContext,
    ) -> PhysicalOperator:
        assert len(physical_children) == 1
        return StreamingHashAggregate(
            input_op=physical_children[0],
            data_context=data_context,
            key=logical_op.key,
            agg_fn=logical_op.agg_fn,
            num_aggregators=logical_op.num_aggregators,
        )

    register_plan_logical_op_fn(StreamingAggregate, plan_streaming_aggregate)
    register_plan_logical_op_fn(ListFiles, plan_list_files_op)
    register_plan_logical_op_fn(ReadFiles, plan_read_files_op)


class AnyscalePlanner(Planner):
    def plan(self, logical_plan: LogicalPlan) -> PhysicalPlan:
        plan_fns = get_plan_logical_op_fns()

        checkpoint_config = logical_plan.context.checkpoint_config
        if checkpoint_config is not None and _supports_checkpointing(logical_plan):
            if checkpoint_config.is_batch_based():
                checkpoint_callback = LoadCheckpointCallback(checkpoint_config)
                add_execution_callback(checkpoint_callback, logical_plan.context)
                get_checkpoint_ref = checkpoint_callback.get_checkpoint_ref
            else:
                get_checkpoint_ref = None

            plan_fns.update(_get_plan_fns_for_checkpointing(get_checkpoint_ref))

        elif checkpoint_config is not None:
            assert not _supports_checkpointing(logical_plan)
            warnings.warn(
                "You've enabled checkpointing, but the logical plan doesn't support "
                "checkpointing. Checkpointing will be disabled."
            )

        physical_dag, op_map = plan_recursively(
            logical_plan.dag, plan_fns, logical_plan.context
        )
        return PhysicalPlan(physical_dag, op_map, logical_plan.context)


def _supports_checkpointing(logical_plan: LogicalPlan) -> bool:
    # TODO: Add useful warnings and error messages if we don't support checkpointing.
    if not isinstance(logical_plan.dag, Write):
        return False

    def _all_paths_contain_checkpoint_filter(op: LogicalOperator) -> bool:
        if isinstance(op, _CHECKPOINT_FILTER_OPS):
            return True

        return all(
            _all_paths_contain_checkpoint_filter(input_dep)
            for input_dep in op.input_dependencies
        )

    return _all_paths_contain_checkpoint_filter(logical_plan.dag)


def _get_plan_fns_for_checkpointing(
    get_checkpoint_ref: Callable[[], ObjectRef]
) -> Dict[Type[LogicalOperator], PlanLogicalOpFn]:
    plan_fns = {
        Read: functools.partial(
            plan_read_op_with_checkpoint_filter,
            get_checkpoint_ref=get_checkpoint_ref,
        ),
        ReadFiles: functools.partial(
            plan_read_files_op_with_checkpoint_filter,
            get_checkpoint_ref=get_checkpoint_ref,
        ),
        AbstractFrom: functools.partial(
            plan_from_op_with_checkpoint_filter,
            get_checkpoint_ref=get_checkpoint_ref,
        ),
        Write: plan_write_op_with_checkpoint_writer,
    }
    # Check that we have plan functions for all ops we claim to support.
    assert set(plan_fns) > set(_CHECKPOINT_FILTER_OPS)
    return plan_fns
