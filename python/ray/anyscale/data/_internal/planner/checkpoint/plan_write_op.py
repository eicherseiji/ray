import itertools
from typing import Iterable, List

from ray.anyscale.data.checkpoint.checkpoint_writer import CheckpointWriter
from ray.anyscale.data.checkpoint.interfaces import (
    InvalidCheckpointingOperators,
)
from ray.data import DataContext
from ray.data._internal.execution.interfaces import PhysicalOperator
from ray.data._internal.execution.interfaces.task_context import TaskContext
from ray.data._internal.execution.operators.map_operator import MapOperator
from ray.data._internal.execution.operators.map_transformer import (
    BlockMapTransformFn,
    MapTransformFn,
    MapTransformFnDataType,
)
from ray.data._internal.logical.operators.write_operator import Write
from ray.data._internal.planner.plan_write_op import plan_write_op
from ray.data.block import Block, BlockAccessor
from ray.data.datasource.datasink import Datasink


def plan_write_op_with_checkpoint_writer(
    op: Write, physical_children: List[PhysicalOperator], data_context: DataContext
) -> PhysicalOperator:
    assert data_context.checkpoint_config is not None
    map_operator = plan_write_op(op, physical_children, data_context)
    _insert_write_checkpoint_transform_fn(op, map_operator, data_context)
    return map_operator


def _insert_write_checkpoint_transform_fn(
    logical_op: Write, physical_op: MapOperator, data_context: DataContext
) -> MapOperator:
    datasink = logical_op._datasink_or_legacy_datasource
    if not isinstance(datasink, Datasink):
        raise InvalidCheckpointingOperators(
            f"To enable row-based checkpointing, Write operation must use a "
            f"Datasink and not a legacy Datasource, but got: "
            f"{type(datasink)}"
        )

    checkpoint_writer = CheckpointWriter.create(data_context.checkpoint_config)

    # MapTransformFn for writing checkpoint files after write completes.
    def write_checkpoint_for_block(
        blocks: Iterable[Block], ctx: TaskContext
    ) -> Iterable[Block]:
        it1, it2 = itertools.tee(blocks, 2)
        for block in it1:
            ba = BlockAccessor.for_block(block)
            if ba.num_rows() > 0:
                if data_context.checkpoint_config.id_column not in ba.column_names():
                    raise ValueError(
                        f"ID column {data_context.checkpoint_config.id_column} is "
                        f"absent in the block to be written. Do not drop or rename "
                        f"this column."
                    )
            checkpoint_writer.write_block_checkpoint(ba)

        return list(it2)

    # Insert the MapTransformFn into the physical MapOperator
    # created from logical Write op.
    assert isinstance(physical_op, MapOperator), type(physical_op)
    transform_fns: List[
        MapTransformFn
    ] = physical_op._map_transformer.get_transform_fns().copy()

    # Check that `transform_fns` are compatible with `write_checkpoint_for_block`.
    assert len(transform_fns) >= 2, transform_fns
    assert transform_fns[0].output_type == MapTransformFnDataType.Block
    assert transform_fns[1].input_type == MapTransformFnDataType.Block

    # Insert the MapTransform directly after write transform:
    # BlockMapTransformFn(write_fn)
    # -> BlockMaptransformFn(write_checkpoint_for_block)
    # -> BlockMaptransformFn(write_stats_fn)
    transform_fns.insert(1, BlockMapTransformFn(write_checkpoint_for_block))
    physical_op._map_transformer.set_transform_fns(transform_fns)
