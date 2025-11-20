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
    plan_list_files_op_with_checkpoint_filter,
    plan_read_files_op_with_checkpoint_filter,
    plan_read_op_with_checkpoint_filter,
    plan_write_op_with_checkpoint_writer,
)
from ray.anyscale.data.checkpoint.load_checkpoint_callback import LoadCheckpointCallback
from ray.data._internal.execution.execution_callback import add_execution_callback
from ray.data._internal.execution.interfaces import PhysicalOperator
from ray.data._internal.execution.operators.join import JoinOperator
from ray.data._internal.logical.interfaces import (
    LogicalOperator,
    LogicalPlan,
    PhysicalPlan,
)
from ray.data._internal.logical.operators.from_operators import AbstractFrom
from ray.data._internal.logical.operators.join_operator import Join
from ray.data._internal.logical.operators.read_operator import Read
from ray.data._internal.logical.operators.streaming_split_operator import StreamingSplit
from ray.data._internal.logical.operators.write_operator import Write
from ray.data._internal.planner.planner import (
    PlanLogicalOpFn,
    Planner,
    find_plan_fn,
)
from ray.data.context import DataContext

_CHECKPOINT_FILTER_OPS = (Read, ReadFiles, AbstractFrom)


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


def plan_join_op(
    logical_op: Join,
    physical_children: List[PhysicalOperator],
    data_context: DataContext,
) -> PhysicalOperator:
    assert len(physical_children) == 2
    assert logical_op._num_outputs is not None

    if data_context.use_polars_join:
        from ray.anyscale.data._internal.execution.operators.join_operator import (
            AnyscaleJoinOperator,
        )

        # Validate GPU configuration if GPU joins are enabled
        if (
            hasattr(data_context, "use_polars_gpu_join")
            and data_context.use_polars_gpu_join
        ):
            if hasattr(data_context, "validate_polars_gpu_config"):
                data_context.validate_polars_gpu_config()

        return AnyscaleJoinOperator(
            data_context=data_context,
            left_input_op=physical_children[0],
            right_input_op=physical_children[1],
            join_type=logical_op._join_type,
            left_key_columns=logical_op._left_key_columns,
            right_key_columns=logical_op._right_key_columns,
            left_columns_suffix=logical_op._left_columns_suffix,
            right_columns_suffix=logical_op._right_columns_suffix,
            num_partitions=logical_op._num_outputs,
            partition_size_hint=logical_op._partition_size_hint,
            aggregator_ray_remote_args_override=logical_op._aggregator_ray_remote_args,
        )

    return JoinOperator(
        data_context=data_context,
        left_input_op=physical_children[0],
        right_input_op=physical_children[1],
        join_type=logical_op._join_type,
        left_key_columns=logical_op._left_key_columns,
        right_key_columns=logical_op._right_key_columns,
        left_columns_suffix=logical_op._left_columns_suffix,
        right_columns_suffix=logical_op._right_columns_suffix,
        num_partitions=logical_op._num_outputs,
        partition_size_hint=logical_op._partition_size_hint,
        aggregator_ray_remote_args_override=logical_op._aggregator_ray_remote_args,
    )


class RayTurboPlanner(Planner):
    _RAYTURBO_PLAN_FNS = {
        StreamingAggregate: plan_streaming_aggregate,
        ListFiles: plan_list_files_op,
        ReadFiles: plan_read_files_op,
        Join: plan_join_op,
    }

    def __init__(self):
        super().__init__()

        self._supports_checkpointing = False
        self._plan_fns_for_checkpointing = {}

    def plan(self, logical_plan: LogicalPlan) -> PhysicalPlan:
        checkpoint_config = logical_plan.context.checkpoint_config
        if checkpoint_config is not None and _supports_checkpointing(logical_plan):
            self._supports_checkpointing = True

            checkpoint_callback = LoadCheckpointCallback(checkpoint_config)
            add_execution_callback(checkpoint_callback, logical_plan.context)
            load_checkpoint = checkpoint_callback.load_checkpoint

            # Dynamically set the plan functions for checkpointing because they
            # need to a reference to the checkpoint ref.
            self._plan_fns_for_checkpointing = _get_plan_fns_for_checkpointing(
                load_checkpoint
            )

        elif checkpoint_config is not None:
            assert not _supports_checkpointing(logical_plan)
            warnings.warn(
                "You've enabled checkpointing, but the logical plan doesn't support "
                "checkpointing. Checkpointing will be disabled."
            )

        return super().plan(logical_plan)

    def get_plan_fn(self, logical_op: LogicalOperator) -> PlanLogicalOpFn:
        # Try checkpointing plan functions first (if enabled)
        if self._supports_checkpointing:
            assert self._plan_fns_for_checkpointing
            plan_fn = find_plan_fn(logical_op, self._plan_fns_for_checkpointing)
            if plan_fn is not None:
                return plan_fn

        # Try RayTurbo plan functions
        plan_fn = find_plan_fn(logical_op, self._RAYTURBO_PLAN_FNS)
        if plan_fn is not None:
            return plan_fn

        # Fall back to OSS plan functions
        return super().get_plan_fn(logical_op)


def _supports_checkpointing(logical_plan: LogicalPlan) -> bool:
    # TODO: Add useful warnings and error messages if we don't support checkpointing.
    if not isinstance(logical_plan.dag, (Write, StreamingSplit)):
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
    load_checkpoint: Callable[[], ObjectRef],
) -> Dict[Type[LogicalOperator], PlanLogicalOpFn]:
    plan_fns = {
        ListFiles: functools.partial(
            plan_list_files_op_with_checkpoint_filter,
            load_checkpoint=load_checkpoint,
        ),
        Read: functools.partial(
            plan_read_op_with_checkpoint_filter,
            load_checkpoint=load_checkpoint,
        ),
        ReadFiles: functools.partial(
            plan_read_files_op_with_checkpoint_filter,
            load_checkpoint=load_checkpoint,
        ),
        AbstractFrom: functools.partial(
            plan_from_op_with_checkpoint_filter,
            load_checkpoint=load_checkpoint,
        ),
        Write: plan_write_op_with_checkpoint_writer,
    }
    # Check that we have plan functions for all ops we claim to support.
    assert set(plan_fns) > set(_CHECKPOINT_FILTER_OPS)
    return plan_fns
