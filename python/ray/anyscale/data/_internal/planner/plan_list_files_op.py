import logging
from functools import partial
from typing import Callable, Iterable, List, Optional

import numpy as np
import pyarrow as pa
from pyarrow.fs import FileSystem

import ray
from ray.anyscale.data._internal.file_indexer import (
    FileIndexer,
    FileManifest,
)
from ray.anyscale.data._internal.logical.operators.list_files_operator import (
    PATH_COLUMN_NAME,
    ListFiles,
)
from ray.anyscale.data._internal.partitioners import FilePartitioner
from ray.anyscale.data.checkpoint.util import CHECKPOINTED_IDS_KWARG_NAME
from ray.data._internal.delegating_block_builder import DelegatingBlockBuilder
from ray.data._internal.execution.interfaces import PhysicalOperator, RefBundle
from ray.data._internal.execution.interfaces.task_context import TaskContext
from ray.data._internal.execution.operators.input_data_buffer import InputDataBuffer
from ray.data._internal.execution.operators.map_operator import MapOperator
from ray.data._internal.execution.operators.map_transformer import (
    BlockMapTransformFn,
    MapTransformer,
)
from ray.data.block import Block, BlockAccessor
from ray.data.context import DataContext
from ray.data.datasource import FileShuffleConfig, PathPartitionFilter
from ray.types import ObjectRef

logger = logging.getLogger(__name__)

# TODO(@bveeramani): 200 is arbitrary.
# This is the maximum total number of list files task that we launch. In practice, we'll
# usually only launch one list files task (i.e., in the case the user provides a single
# directory).
DEFAULT_MAX_NUM_LIST_FILES_TASKS = 200


def plan_list_files_op(
    op: ListFiles,
    physical_children: List[PhysicalOperator],
    data_context: DataContext,
    load_checkpoint: Optional[Callable[[], ObjectRef]] = None,
) -> MapOperator:
    assert len(physical_children) == 0

    #
    # NOTE: Avoid capturing operators in closures!
    #
    file_extensions = op.file_extensions
    partition_filter = op.partition_filter

    # Instantiate shuffle configuration (if any)
    shuffle_config = op.shuffle_config_factory()

    filesystem = op.filesystem
    indexer = op.file_indexer
    partitioner = op.file_partitioner

    transform_fns = [
        BlockMapTransformFn(
            partial(
                list_files_for_each_block,
                indexer=indexer,
                filesystem=filesystem,
                file_extensions=file_extensions,
                partition_filter=partition_filter,
                preserve_order=data_context.execution_options.preserve_order,
            ),
            # NOTE: Disable block-shaping to produce blocks as is
            disable_block_shaping=True,
        ),
    ]

    if shuffle_config is not None:
        transform_fns.append(
            BlockMapTransformFn(
                partial(
                    shuffle_files,
                    shuffle_config=shuffle_config,
                    execution_idx=data_context._execution_idx,
                ),
                # NOTE: Disable block-shaping to produce blocks as is
                disable_block_shaping=True,
            )
        )

    if partitioner is not None:
        transform_fns.append(
            BlockMapTransformFn(
                partial(partition_files, partitioner=partitioner),
                # NOTE: Disable block-shaping to produce blocks as is
                disable_block_shaping=True,
            )
        )

    map_transformer = MapTransformer(transform_fns)

    map_operator = MapOperator.create(
        map_transformer,
        create_input_data_buffer(
            op,
            data_context,
            # NOTE: If shuffling is requested we can't parallelize the listing
            #       as we need to collect all files in a single task for subsequent
            #       global shuffling
            should_parallelize=shuffle_config is None,
        ),
        data_context,
        name="ListFiles",
        ray_remote_args={
            # This is operator is extremely fast. If we don't unblock backpressure, this
            # operator gets bottlenecked by the Ray Data scheduler. This can prevent Ray
            # Data from launching enough read tasks.
            "_generator_backpressure_num_objects": -1,
        },
        # Avoid fuse ListFiles with the following ReadFiles.
        supports_fusion=False,
    )
    # ListFiles is extremely fast and should not be throttled by backpressure.
    map_operator.throttling_disabled = lambda: True

    if (
        load_checkpoint is not None
        and data_context.checkpoint_config.generated_id_column
    ):
        # Checkpoint restore is run as an execution callback, so the checkpoint block
        # object reference is not yet available. Instead we pass in load_checkpoint
        # function, so when the map task is executed, the checkpoint block is loaded
        # and passed to the map task.
        map_operator.add_map_task_kwargs_fn(
            lambda: {CHECKPOINTED_IDS_KWARG_NAME: load_checkpoint()}
        )

    return map_operator


def create_input_data_buffer(
    logical_op: ListFiles, data_context: DataContext, *, should_parallelize: bool
) -> InputDataBuffer:

    if should_parallelize:
        max_num_list_files_tasks = data_context.get_config(
            "max_num_list_files_tasks", DEFAULT_MAX_NUM_LIST_FILES_TASKS
        )
        path_splits = np.array_split(
            logical_op.paths, min(max_num_list_files_tasks, len(logical_op.paths))
        )
    else:
        path_splits = [logical_op.paths]

    input_data = []
    for path_split in path_splits:
        block = pa.Table.from_pydict({PATH_COLUMN_NAME: path_split})
        metadata = BlockAccessor.for_block(block).get_metadata(
            input_files=None, exec_stats=None
        )
        ref_bundle = RefBundle(
            [(ray.put(block), metadata)],
            # `owns_blocks` is False, because these refs are the root of the
            # DAG. We shouldn't eagerly free them. Otherwise, the DAG cannot
            # be reconstructed.
            owns_blocks=False,
            schema=BlockAccessor.for_block(block).schema(),
        )
        input_data.append(ref_bundle)
    return InputDataBuffer(data_context, input_data=input_data)


def list_files_for_each_block(
    blocks: Iterable[Block],
    ctx: TaskContext,
    *,
    indexer: FileIndexer,
    filesystem: FileSystem,
    file_extensions: Optional[List[str]],
    partition_filter: Optional[PathPartitionFilter],
    preserve_order: bool,
) -> Iterable[Block]:
    checkpoint_ids = None
    if CHECKPOINTED_IDS_KWARG_NAME in ctx.kwargs:
        checkpoint_ids = ctx.kwargs[CHECKPOINTED_IDS_KWARG_NAME]
    for block in blocks:
        for file_manifest in indexer.list_files(
            block[PATH_COLUMN_NAME],
            filesystem=filesystem,
            checkpoint_ids=checkpoint_ids,
            file_extensions=file_extensions,
            partition_filter=partition_filter,
            preserve_order=preserve_order,
        ):
            assert (
                len(file_manifest) > 0
            ), "list_files is guaranteed to not return an empty block."
            yield file_manifest.as_block()


def shuffle_files(
    blocks: Iterable[Block],
    _: TaskContext,
    shuffle_config: FileShuffleConfig,
    execution_idx: int,
) -> Iterable[Block]:
    builder = DelegatingBlockBuilder()

    # NOTE: This will block until file listing is complete!
    for block in blocks:
        builder.add_block(block)

    combined_block = builder.build()
    shuffled_block = BlockAccessor.for_block(combined_block).random_shuffle(
        shuffle_config.get_seed(execution_idx)
    )
    yield shuffled_block


def partition_files(
    blocks: Iterable[Block],
    _: TaskContext,
    partitioner: FilePartitioner,
) -> Iterable[Block]:
    for block in blocks:
        partitioner.add_input(FileManifest(block))
        while partitioner.has_partition():
            yield partitioner.next_partition().as_block()

    partitioner.finalize()
    while partitioner.has_partition():
        yield partitioner.next_partition().as_block()
