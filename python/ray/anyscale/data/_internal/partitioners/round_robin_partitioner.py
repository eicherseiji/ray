import collections
import logging
from typing import Tuple

from ray.anyscale.data._internal.logical.operators.list_files_operator import (
    FileManifest,
)
from ray.anyscale.data._internal.readers import InMemorySizeEstimator
from ray.anyscale.data.checkpoint.util import CheckpointFragmentsInfo

from .file_partitioner import FilePartitioner

logger = logging.getLogger(__name__)


class _FileBucket:
    """A bucket of paths."""

    def __init__(self):
        self._paths = []
        self._file_sizes = []
        self._file_chunk_metadatas = []
        self._checkpoint_fragments_info = []
        self._in_memory_size = 0

    @property
    def paths(self):
        return self._paths

    @property
    def in_memory_size(self):
        return self._in_memory_size

    @property
    def file_sizes(self):
        return self._file_sizes

    @property
    def file_chunk_metadatas(self):
        return self._file_chunk_metadatas

    @property
    def checkpoint_fragments_info(self):
        return self._checkpoint_fragments_info

    def add(
        self,
        path: str,
        file_size: int,
        file_chunk_metadata: Tuple[int, int],
        in_memory_size: int,
        checkpoint_fragments_info=None,
    ):
        self._paths.append(path)
        self._file_sizes.append(file_size)
        self._file_chunk_metadatas.append(file_chunk_metadata)
        self._checkpoint_fragments_info.append(checkpoint_fragments_info)
        self._in_memory_size += in_memory_size

    def clear(self):
        self._paths.clear()
        self._file_sizes.clear()
        self._file_chunk_metadatas.clear()
        self._checkpoint_fragments_info.clear()
        self._in_memory_size = 0


class RoundRobinPartitioner(FilePartitioner):
    """Partitions input paths into blocks based on the in-memory size of files.

    This partitioning ensures read tasks effectively utilize the cluster and
    produce appropriately-sized blocks

    **Steps:**
        1. Initialize empty buckets.
        2. Iterate through input blocks and add paths to buckets. For each path:
            - If the current bucket falls below `min_bucket_size`, add the path and don't move
              to the next bucket.
            - If the current bucket exceeds `min_bucket_size` but not `max_bucket_size`,
              add the path and move to the next bucket.
            - If the current bucket exceeds `max_bucket_size`, yield the paths as a block, clear
              the bucket, and move to the next bucket.
        3. Yield any remaining paths in the buckets as blocks.

    This algorithm ensures that each block contains [min_bucket_size, max_bucket_size]
    worth of files.  It's a deterministic algorithm, but it doesn't maintain the order
    of the input paths.
    """

    def __init__(
        self,
        in_memory_size_estimator: InMemorySizeEstimator,
        *,
        min_bucket_size: int,
        max_bucket_size: int,
        num_buckets: int,
    ):
        self._in_memory_size_estimator = in_memory_size_estimator
        self._num_buckets = num_buckets
        self._min_bucket_size = min_bucket_size
        self._max_bucket_size = max_bucket_size

        self._buckets = [_FileBucket() for _ in range(num_buckets)]
        self._current_bucket_index = 0
        self._output_queue: collections.deque[FileManifest] = collections.deque()

    def add_input(self, input: FileManifest):
        in_memory_size_estimates = (
            self._in_memory_size_estimator.estimate_in_memory_sizes(input)
        )
        for (
            file_path,
            file_size,
            file_chunk_metadata,
            in_memory_size_estimate,
            checkpoint_file_fragments,
        ) in zip(
            input.paths,
            input.file_sizes,
            input.file_chunk_metadatas,
            in_memory_size_estimates,
            input.file_fragments_checkpoint,
        ):
            if checkpoint_file_fragments is not None:
                checkpoint_fragments_info = CheckpointFragmentsInfo(
                    path=file_path,
                    checkpointed_file_fragments=checkpoint_file_fragments,
                )
            else:
                checkpoint_fragments_info = None
            current_bucket = self._buckets[self._current_bucket_index]

            # If an in-memory size estimate isn't available, add the file to the current
            # bucket and move to the next bucket. This has the effect of spreading the
            # files evenly across the buckets if no in-memory size estimates are
            # available.
            #
            # This is a special-case for file systems that don't provide file sizes
            # like HTTP-based file systems.
            if in_memory_size_estimate is None:
                current_bucket.add(
                    file_path,
                    file_size,
                    file_chunk_metadata,
                    0,
                    checkpoint_fragments_info,
                )
                self._current_bucket_index = (
                    self._current_bucket_index + 1
                ) % self._num_buckets
                continue

            current_bucket.add(
                file_path,
                file_size,
                file_chunk_metadata,
                in_memory_size_estimate,
                checkpoint_fragments_info,
            )
            if current_bucket.in_memory_size >= self._max_bucket_size:
                manifest = FileManifest.construct_manifest(
                    current_bucket.paths,
                    current_bucket.file_sizes,
                    current_bucket.file_chunk_metadatas,
                    current_bucket.checkpoint_fragments_info,
                )
                self._output_queue.append(manifest)
                self._current_bucket_index = (
                    self._current_bucket_index + 1
                ) % self._num_buckets
                current_bucket.clear()
            elif current_bucket.in_memory_size >= self._min_bucket_size:
                self._current_bucket_index = (
                    self._current_bucket_index + 1
                ) % self._num_buckets

    def has_partition(self) -> bool:
        return len(self._output_queue) > 0

    def next_partition(self) -> FileManifest:
        return self._output_queue.popleft()

    def finalize(self):
        for bucket in self._buckets:
            if bucket.paths:
                manifest = FileManifest.construct_manifest(
                    bucket.paths,
                    bucket.file_sizes,
                    bucket.file_chunk_metadatas,
                    bucket.checkpoint_fragments_info,
                )
                self._output_queue.append(manifest)
