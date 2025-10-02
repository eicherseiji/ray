import logging
from typing import Dict, Iterable, List, Optional

from ray.anyscale.data._internal.logical.operators.list_files_operator import (
    FileManifest,
)
from ray.anyscale.data._internal.logical.operators.read_files_operator import ReadFiles
from ray.data._internal.execution.interfaces import PhysicalOperator
from ray.data._internal.execution.interfaces.task_context import TaskContext
from ray.data._internal.execution.operators.map_operator import MapOperator
from ray.data._internal.execution.operators.map_transformer import (
    MapTransformer,
    MapTransformFn,
    BlockMapTransformFn,
    BatchMapTransformFn,
)
from ray.data._internal.output_buffer import OutputBlockSizeOption
from ray.data._internal.table_block import TableBlockAccessor
from ray.data.block import Block, BlockType, DataBatch, BatchFormat
from ray.data.context import DataContext


logger = logging.getLogger(__name__)


def plan_read_files_op(
    op: ReadFiles,
    physical_children: List[PhysicalOperator],
    data_context: DataContext,
) -> PhysicalOperator:
    assert len(physical_children) == 1
    input_op = physical_children[0]

    #
    # NOTE: Avoid capturing operators in closures!
    #
    columns: Optional[List[str]] = op.columns
    columns_rename_map: Optional[Dict[str, str]] = op.columns_rename

    predicate_expr = op.predicate_expr

    fs = op.filesystem
    reader = op.reader

    def read_files(blocks: Iterable[Block], ctx: TaskContext) -> Iterable[DataBatch]:
        for block in blocks:
            file_manifest = FileManifest(block)
            # For some readers, we need to filter the rows in-memory.
            yield from reader.read_files(
                file_manifest,
                columns=columns,
                columns_rename=columns_rename_map,
                predicate_expr=predicate_expr,
                filesystem=fs,
            )

    transform_fns: List[MapTransformFn] = [
        BatchMapTransformFn(
            read_files,
            batch_size=None,
            batch_format=BatchFormat.ARROW,
            zero_copy_batch=True,
            output_block_size_option=OutputBlockSizeOption.of(
                target_max_block_size=data_context.target_max_block_size
            ),
        ),
    ]

    # Operator fusion *should* take care of the in-memory filtering
    # instead - but needs https://github.com/anyscale/rayturbo/pull/881
    if op.predicate_expr is not None and not op.reader.supports_predicate_pushdown():

        def _apply_predicate(
            blocks: Iterable[Block], ctx: TaskContext
        ) -> Iterable[Block]:
            for block in blocks:
                block = TableBlockAccessor.normalize_block_types(
                    [block], BlockType.ARROW
                )[0]

                from ray.data.expressions import Expr

                if isinstance(op.predicate_expr, Expr):
                    # Use ArrowBlockAccessor's filter method which handles Ray Data expressions
                    from ray.data._internal.arrow_block import ArrowBlockAccessor

                    block_accessor = ArrowBlockAccessor(block)
                    filtered_table = block_accessor.filter(op.predicate_expr)
                    yield filtered_table
                else:
                    # Use PyArrow filter directly for PyArrow expressions
                    yield block.filter(op.predicate_expr)

        transform_fns.append(
            BlockMapTransformFn(_apply_predicate)
        )  # Fixed indentation!

    map_transformer = MapTransformer(
        transform_fns,
        output_block_size_option_override=OutputBlockSizeOption.of(
            target_max_block_size=data_context.target_max_block_size,
        ),
    )

    map_operator = MapOperator.create(
        map_transformer,
        input_op,
        data_context,
        name="ReadFiles",
        compute_strategy=op._compute,
        supports_fusion=(
            # NOTE: By default fusion of the Read ops is turned off for now
            #       until we can reliably estimate whether fusing read op
            #       with subsequent operation will negatively affect parallelism
            #       level of the either (this would require listing of the dataset
            #       to be performed at the planning phase to accurately estimate
            #       parallelism level of the reading op)
            False
            if data_context._enable_read_files_fusion_override is None
            else data_context._enable_read_files_fusion_override
        ),
        ray_remote_args=op._ray_remote_args,
    )

    return map_operator
