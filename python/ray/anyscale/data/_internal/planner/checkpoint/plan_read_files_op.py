import functools
from typing import Any, Callable, List, Optional

from ray.anyscale.data._internal.logical.operators.read_files_operator import ReadFiles
from ray.anyscale.data._internal.readers.parquet_reader import ParquetReader
from ray.anyscale.data._internal.planner.plan_read_files_op import plan_read_files_op
from ray.anyscale.data.checkpoint.util import (
    CHECKPOINTED_IDS_KWARG_NAME,
    filter_checkpointed_rows_for_batches,
)
from ray.data import DataContext
from ray.data._internal.execution.interfaces import PhysicalOperator
from ray.data._internal.execution.operators.map_operator import MapOperator
from ray.data._internal.execution.operators.map_transformer import (
    BatchMapTransformFn,
    MapTransformFn,
    MapTransformFnDataType,
)


def plan_read_files_op_with_checkpoint_filter(
    op: ReadFiles,
    physical_children: List[PhysicalOperator],
    data_context: DataContext,
    get_checkpoint_ref: Optional[Callable[[], Any]] = None,
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
    _insert_filter_transform_fn(physical_op, data_context, get_checkpoint_ref)
    return physical_op


def _insert_filter_transform_fn(
    physical_op: MapOperator,
    data_context: DataContext,
    get_checkpoint_ref: Optional[Callable[[], Any]],
) -> MapOperator:
    transform_fns: List[
        MapTransformFn
    ] = physical_op._map_transformer.get_transform_fns().copy()
    assert transform_fns[1].output_type == MapTransformFnDataType.Batch
    assert transform_fns[2].input_type == MapTransformFnDataType.Batch

    # Insert the MapTransform directly after read_paths transform:
    # BlocksToBatchesMapTransformFn()
    # -> BatchMapTransformFn(read_paths)
    # -> BatchMapTransformFn(filter_checkpointed_rows_for_batches) -> ...
    transform_fns.insert(
        2,
        BatchMapTransformFn(
            functools.partial(
                filter_checkpointed_rows_for_batches,
                checkpoint_config=data_context.checkpoint_config,
            )
        ),
    )
    physical_op._map_transformer.set_transform_fns(transform_fns)

    if get_checkpoint_ref is not None:
        physical_op.add_map_task_kwargs_fn(
            lambda: {CHECKPOINTED_IDS_KWARG_NAME: get_checkpoint_ref()}
        )
