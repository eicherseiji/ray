import logging
from functools import partial
from typing import Iterable, List, Optional

import numpy as np
import pyarrow as pa
from pyarrow.fs import FileSystem

import ray
from ray.anyscale.data._internal.file_indexer import (
    FileIndexer,
    FileManifest,
    filter_file_manifest,
)
from ray.anyscale.data._internal.logical.operators.list_files_operator import (
    PATH_COLUMN_NAME,
    ListFiles,
)
from ray.anyscale.data._internal.partitioners import FilePartitioner
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
            ),
        ),
    ]

    if shuffle_config is not None:
        transform_fns.append(
            BlockMapTransformFn(partial(shuffle_files, shuffle_config=shuffle_config))
        )

    if partitioner is not None:
        transform_fns.append(
            BlockMapTransformFn(partial(partition_files, partitioner=partitioner))
        )

    map_transformer = MapTransformer(transform_fns)

    return MapOperator.create(
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
    _: TaskContext,
    *,
    indexer: FileIndexer,
    filesystem: FileSystem,
    file_extensions: Optional[List[str]],
    partition_filter: Optional[PathPartitionFilter],
) -> Iterable[Block]:
    for block in blocks:
        for file_manifest in indexer.list_files(
            block[PATH_COLUMN_NAME], filesystem=filesystem
        ):
            file_manifest = filter_file_manifest(
                file_manifest, file_extensions, partition_filter
            )

            # Don't yield empty manifests. This can happen if rows get filtered out for
            # `file_extensions` or `partition_filter`.
            if len(file_manifest) > 0:
                yield file_manifest.as_block()


def shuffle_files(
    blocks: Iterable[Block],
    _: TaskContext,
    shuffle_config: FileShuffleConfig,
) -> Iterable[Block]:
    builder = DelegatingBlockBuilder()

    # NOTE: This will block until file listing is complete!
    for block in blocks:
        builder.add_block(block)

    combined_block = builder.build()
    shuffled_block = BlockAccessor.for_block(combined_block).random_shuffle(
        shuffle_config.seed
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
