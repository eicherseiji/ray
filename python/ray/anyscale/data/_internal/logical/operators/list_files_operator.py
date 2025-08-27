from functools import cached_property
from typing import TYPE_CHECKING, Callable, List, Optional, Union

import numpy as np
import pyarrow as pa

from ray.data import FileShuffleConfig
from ray.data._internal.logical.interfaces import LogicalOperator, SourceOperator
from ray.data.block import Block, BlockAccessor, BlockColumnAccessor
from ray.data.datasource import PathPartitionFilter

if TYPE_CHECKING:
    from ray.anyscale.data._internal.file_indexer import FileIndexer
    from ray.anyscale.data._internal.partitioners import FilePartitioner
    from ray.data._internal.execution.interfaces.ref_bundle import RefBundle

PATH_COLUMN_NAME = "__path"
FILE_SIZE_COLUMN_NAME = "__file_size"
FILE_CHUNK_METADATA_COLUMN_NAME = "__file_chunk_metadata"


class FileManifest:
    """Structured view over file paths and file sizes.

    A thin wrapper over `ListFiles` outputs that provides structured access to file
    paths and sizes. This avoids making implicit assumptions about block structure as
    data moves between file listing, partitioning, and reading stages.

    All extracted views (i.e., `paths`, `file_sizes`) share the same row order as the
    underlying block. Any transformation must preserve this.
    """

    def __init__(self, block: Block):
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

    def __len__(self) -> int:
        return len(self._block)

    def __repr__(self):
        return f"<{self.__class__.__name__} length={len(self._block)}>"

    @cached_property
    def paths(self) -> np.ndarray:
        return BlockColumnAccessor.for_column(self._paths).to_numpy()

    @cached_property
    def file_sizes(self) -> np.ndarray:
        return BlockColumnAccessor.for_column(self._file_sizes).to_numpy()

    @cached_property
    def file_chunk_metadatas(self) -> np.ndarray:
        return BlockColumnAccessor.for_column(self._file_chunk_metadatas).to_numpy()

    def as_block(self) -> Block:
        """Return the underlying block for the `FileManifest`.

        This doesn't make a copy of the underlying data.
        """
        return self._block

    @classmethod
    def construct_manifest(cls, paths, sizes, chunk_metadatas) -> "FileManifest":
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
