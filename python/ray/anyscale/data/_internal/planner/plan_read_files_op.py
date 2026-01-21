import logging
from typing import Iterable, List

from ray.anyscale.data._internal.logical.operators.list_files_operator import (
    FileManifest,
)
from ray.anyscale.data._internal.logical.operators.read_files_operator import ReadFiles
from ray.data._internal.execution.interfaces import PhysicalOperator
from ray.data._internal.execution.interfaces.task_context import TaskContext
from ray.data._internal.execution.operators.map_operator import MapOperator
from ray.data._internal.execution.operators.map_transformer import (
    BatchMapTransformFn,
    MapTransformer,
    MapTransformFn,
)
from ray.data._internal.output_buffer import OutputBlockSizeOption
from ray.data.block import BatchFormat, Block, DataBatch
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
    fs = op.filesystem
    reader = op.reader

    # Apply pushdowns to reader using OSS pattern (instead of direct assignment)
    if op.predicate_expr is not None:
        reader = reader.apply_predicate(op.predicate_expr)

    if op.columns is not None or op.columns_rename is not None:
        projection_map = op.get_projection_map()
        reader = reader.apply_projection(projection_map)

    def read_files(blocks: Iterable[Block], ctx: TaskContext) -> Iterable[DataBatch]:
        for block in blocks:
            file_manifest = FileManifest(block)
            # Reader now has state from apply_projection/apply_predicate
            yield from reader.read_files(
                file_manifest,
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
