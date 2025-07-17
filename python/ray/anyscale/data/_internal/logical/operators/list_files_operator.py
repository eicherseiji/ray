from functools import cached_property
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, Union

import numpy as np
import pyarrow as pa

from ray.data import FileShuffleConfig
from ray.data._internal.logical.interfaces import LogicalOperator, SourceOperator
from ray.data.block import Block, BlockAccessor, BlockColumnAccessor
from ray.data.datasource import PathPartitionFilter

if TYPE_CHECKING:
    from ray.anyscale.data._internal.file_indexer import FileIndexer, ChunkMetadata
    from ray.anyscale.data._internal.partitioners import FilePartitioner
    from ray.data._internal.execution.interfaces.ref_bundle import RefBundle
    from ray.anyscale.data._internal.readers.file_reader import FileReader
    from pyarrow.fs import FileSystem

PATH_COLUMN_NAME = "__path"
FILE_SIZE_COLUMN_NAME = "__file_size"
FILE_CHUNK_METADATA_COLUMN_NAME = "__file_chunk_metadata"
FILE_START_ROW_COUNT_COLUMN_NAME = "__file_start_row_count"
FILE_END_ROW_COUNT_COLUMN_NAME = "__file_end_row_count"


class FileManifest:
    """Structured view over file paths and file sizes.

    A thin wrapper over `ListFiles` outputs that provides structured access to file
    paths and sizes. This avoids making implicit assumptions about block structure as
    data moves between file listing, partitioning, and reading stages.

    All extracted views (i.e., `paths`, `file_sizes`) share the same row order as the
    underlying block. Any transformation must preserve this.
    """

    def __init__(self, block: Block) -> None:
        """Create a new `FileManifest` from a block.

        Args:
            block: Block with `PATH_COLUMN_NAME` and `FILE_SIZE_COLUMN_NAME` columns.
                Any other columns are optional and treated as input data.
        """
        column_names = BlockAccessor.for_block(block).column_names()
        assert FILE_SIZE_COLUMN_NAME in column_names
        assert PATH_COLUMN_NAME in column_names
        assert FILE_CHUNK_METADATA_COLUMN_NAME in column_names

        self._block = block

        self._paths = block[PATH_COLUMN_NAME]
        self._file_sizes = block[FILE_SIZE_COLUMN_NAME]
        self._file_chunk_metadatas = block[FILE_CHUNK_METADATA_COLUMN_NAME]
        if (
            FILE_START_ROW_COUNT_COLUMN_NAME in column_names
            and FILE_END_ROW_COUNT_COLUMN_NAME in column_names
        ):
            self._file_start_row_counts = block[FILE_START_ROW_COUNT_COLUMN_NAME]
            self._file_end_row_counts = block[FILE_END_ROW_COUNT_COLUMN_NAME]
        else:
            assert (
                FILE_START_ROW_COUNT_COLUMN_NAME not in column_names
                and FILE_END_ROW_COUNT_COLUMN_NAME not in column_names
            )
            self._file_start_row_counts = None
            self._file_end_row_counts = None

    def __len__(self) -> int:
        return len(self._block)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} length={len(self._block)}>"

    @property
    def num_rows(self) -> int:
        return len(self._block)

    @cached_property
    def paths(self) -> np.ndarray:
        return BlockColumnAccessor.for_column(self._paths).to_numpy()

    @cached_property
    def file_sizes(self) -> np.ndarray:
        return BlockColumnAccessor.for_column(self._file_sizes).to_numpy()

    @cached_property
    def file_chunk_metadatas(self) -> np.ndarray:
        return BlockColumnAccessor.for_column(self._file_chunk_metadatas).to_numpy()

    @cached_property
    def file_start_row_counts(self) -> Optional[np.ndarray]:
        """Get start row counts for files, if available."""
        if self._file_start_row_counts is None:
            return None
        return BlockColumnAccessor.for_column(self._file_start_row_counts).to_numpy()

    @cached_property
    def file_end_row_counts(self) -> Optional[np.ndarray]:
        """Get end row counts for files, if available."""
        if self._file_end_row_counts is None:
            return None
        return BlockColumnAccessor.for_column(self._file_end_row_counts).to_numpy()

    def has_row_ranges(self) -> bool:
        """Check if this manifest contains row range information (start/end row counts)."""
        return (
            self._file_start_row_counts is not None
            and self._file_end_row_counts is not None
        )

    def as_block(self) -> Block:
        """Return the underlying block for the `FileManifest`.

        This doesn't make a copy of the underlying data.
        """
        return self._block

    @classmethod
    def get_row_ranges_from_metadata(
        cls,
        paths: List[str],
        sizes: List[int],
        chunk_metadatas: List[Optional["ChunkMetadata"]],
        reader: "FileReader",
        filesystem: "FileSystem",
        starting_row_count: int,
    ) -> Optional[Tuple[List[int], List[int]]]:
        """Extract row ranges from metadata if available."""
        from ray.anyscale.data._internal.readers.supports_metadata import (
            SupportsMetadata,
        )

        if not isinstance(reader, SupportsMetadata):
            return None

        temp_block = pa.table(
            {
                PATH_COLUMN_NAME: paths,
                FILE_SIZE_COLUMN_NAME: sizes,
                FILE_CHUNK_METADATA_COLUMN_NAME: chunk_metadatas,
            }
        )
        temp_manifest = cls(temp_block)
        metadata_iter = reader.read_metadata(temp_manifest, filesystem=filesystem)

        start_row_counts = []
        end_row_counts = []
        current_cumulative = starting_row_count

        for metadata in metadata_iter:
            if metadata and metadata.num_rows is not None:
                start_row_counts.append(current_cumulative)
                current_cumulative += metadata.num_rows
                end_row_counts.append(current_cumulative - 1)  # End is inclusive
            else:
                return None

        if len(start_row_counts) == len(paths) and all(
            count is not None for count in start_row_counts
        ):
            return (start_row_counts, end_row_counts)

        return None

    @classmethod
    def construct_manifest(
        cls,
        paths: List[str],
        sizes: List[int],
        chunk_metadatas: List[Optional["ChunkMetadata"]],
        row_ranges: Optional[Tuple[List[int], List[int]]] = None,
    ) -> "FileManifest":
        if row_ranges is not None:
            assert (
                len(paths)
                == len(sizes)
                == len(chunk_metadatas)
                == len(row_ranges[0])
                == len(row_ranges[1])
            )
            block = pa.table(
                {
                    PATH_COLUMN_NAME: paths,
                    FILE_SIZE_COLUMN_NAME: sizes,
                    FILE_CHUNK_METADATA_COLUMN_NAME: chunk_metadatas,
                    FILE_START_ROW_COUNT_COLUMN_NAME: row_ranges[0],
                    FILE_END_ROW_COUNT_COLUMN_NAME: row_ranges[1],
                }
            )
        else:
            assert len(paths) == len(sizes) == len(chunk_metadatas)
            block = pa.table(
                {
                    PATH_COLUMN_NAME: paths,
                    FILE_SIZE_COLUMN_NAME: sizes,
                    FILE_CHUNK_METADATA_COLUMN_NAME: chunk_metadatas,
                }
            )
        return cls(block)


class ListFiles(SourceOperator, LogicalOperator):
    """List files and get file sizes.

    If an input path is a directory, recursively list all files in the directory and
    their sizes. If an input path is a file, list the file and its size.

    Physical operators that implement this logical operator should output blocks that
    you can construct a `FileManifest` from.
    """

    def __init__(
        self,
        *,
        paths: Union[str, List[str]],
        file_indexer: "FileIndexer",
        file_partitioner: Optional["FilePartitioner"],
        filesystem,
        file_extensions: List[str],
        partition_filter: PathPartitionFilter,
        shuffle_config_factory: Optional[Callable[[], Optional[FileShuffleConfig]]],
    ):
        assert filesystem is not None

        super().__init__(name="ListFiles", input_dependencies=[])

        if isinstance(paths, str):
            paths = [paths]

        self.paths = paths
        self.file_indexer = file_indexer
        self.file_partitioner = file_partitioner
        self.filesystem = filesystem
        self.file_extensions = file_extensions
        self.partition_filter = partition_filter
        self.shuffle_config_factory = shuffle_config_factory

    def output_data(self) -> Optional[List["RefBundle"]]:
        """The output data of this operator if already known, or ``None``."""
        return None
