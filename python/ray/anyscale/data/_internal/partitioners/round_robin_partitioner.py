import collections
import logging
from typing import List, Optional, Tuple

from ray.anyscale.data._internal.logical.operators.list_files_operator import (
    FileManifest,
)
from ray.anyscale.data._internal.readers import InMemorySizeEstimator

from .file_partitioner import FilePartitioner

logger = logging.getLogger(__name__)


class _FileBucket:
    """A bucket of paths."""

    def __init__(self) -> None:
        self._paths: List[str] = []
        self._file_sizes: List[int] = []
        self._file_chunk_metadatas: List[Tuple[int, int]] = []
        self._in_memory_size: int = 0
        self._file_start_row_counts: List[int] = []
        self._file_end_row_counts: List[int] = []

    @property
    def paths(self) -> List[str]:
        return self._paths

    @property
    def in_memory_size(self) -> int:
        return self._in_memory_size

    @property
    def file_sizes(self) -> List[int]:
        return self._file_sizes

    @property
    def file_chunk_metadatas(self) -> List[Tuple[int, int]]:
        return self._file_chunk_metadatas

    @property
    def file_start_row_counts(self) -> List[int]:
        return self._file_start_row_counts

    @property
    def file_end_row_counts(self) -> List[int]:
        return self._file_end_row_counts

    def add(
        self,
        path: str,
        file_size: int,
        file_chunk_metadata: Tuple[int, int],
        in_memory_size: int,
        file_start_row_count: int,
        file_end_row_count: int,
    ) -> None:
        self._paths.append(path)
        self._file_sizes.append(file_size)
        self._file_chunk_metadatas.append(file_chunk_metadata)
        self._in_memory_size += in_memory_size
        self._file_start_row_counts.append(file_start_row_count)
        self._file_end_row_counts.append(file_end_row_count)

    def clear(self) -> None:
        self._paths.clear()
        self._file_sizes.clear()
        self._file_chunk_metadatas.clear()
        self._in_memory_size = 0
        self._file_start_row_counts.clear()
        self._file_end_row_counts.clear()


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

    def _get_row_ranges_from_input(
        self, input: FileManifest
    ) -> Tuple[List[Optional[int]], List[Optional[int]]]:
        """Get start and end row counts from input, or None lists if not available."""
        if input.has_row_ranges():
            return input.file_start_row_counts, input.file_end_row_counts
        else:
            return [None] * len(input.paths), [None] * len(input.paths)

    def _has_bucket_row_ranges(self, bucket: _FileBucket) -> bool:
        """Check if a bucket has valid row range information."""
        return (
            len(bucket.file_start_row_counts) > 0
            and all(count is not None for count in bucket.file_start_row_counts)
            and all(count is not None for count in bucket.file_end_row_counts)
        )

    def _create_manifest_from_bucket(self, bucket: _FileBucket) -> FileManifest:
        """Create a FileManifest from a bucket, preserving row ranges if available."""
        if self._has_bucket_row_ranges(bucket):
            return FileManifest.construct_manifest(
                bucket.paths,
                bucket.file_sizes,
                bucket.file_chunk_metadatas,
                (bucket.file_start_row_counts, bucket.file_end_row_counts),
            )
        else:
            return FileManifest.construct_manifest(
                bucket.paths,
                bucket.file_sizes,
                bucket.file_chunk_metadatas,
            )

    def add_input(self, input: FileManifest):
        in_memory_size_estimates = (
            self._in_memory_size_estimator.estimate_in_memory_sizes(input)
        )

        start_row_counts, end_row_counts = self._get_row_ranges_from_input(input)

        for (
            file_path,
            file_size,
            file_chunk_metadata,
            in_memory_size_estimate,
            start_row_count,
            end_row_count,
        ) in zip(
            input.paths,
            input.file_sizes,
            input.file_chunk_metadatas,
            in_memory_size_estimates,
            start_row_counts,
            end_row_counts,
        ):
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
                    start_row_count,
                    end_row_count,
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
                start_row_count,
                end_row_count,
            )
            if current_bucket.in_memory_size >= self._max_bucket_size:
                manifest = self._create_manifest_from_bucket(current_bucket)
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
                manifest = self._create_manifest_from_bucket(bucket)
                self._output_queue.append(manifest)
