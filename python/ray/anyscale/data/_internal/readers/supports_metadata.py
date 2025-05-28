import abc
from enum import Enum
from typing import Iterator, List, Optional, Set
import pyarrow.fs
from ray.data.block import BlockMetadata
from ray.anyscale.data._internal.logical.operators.list_files_operator import (
    FileManifest,
)
from .file_reader import FileReader


class MetadataType(Enum):
    NUM_ROWS = 0
    NUM_BYTES = 1
    SCHEMA = 2


class SupportsMetadata(abc.ABC):
    """A mix-in to implement `BlockMetadata` logic.

    The `PushdownCountFiles` rule uses this interface to optimize row counting.
    """

    @abc.abstractmethod
    def read_metadata(
        self: FileReader,
        file_manifest: FileManifest,
        *,
        filesystem: pyarrow.fs.FileSystem,
        columns: Optional[List[str]],
    ) -> Iterator[BlockMetadata]:
        """Count the number of rows in the files at the given paths.

        This method is used by the `PushdownCountFiles` rule to avoid reading the entire
        file when only the number of rows is needed.

        Args:
            path: A list of file paths to count rows from.
            filesystem: The filesystem to read from.

        Returns:
            The number of rows in the files.
        """
        ...

    @abc.abstractmethod
    def available_metadata(self: FileReader) -> Set[MetadataType]:
        """Return whether `count_rows` can be called on this reader instance."""
        ...

    @abc.abstractmethod
    def get_target_metadata_batch_size(self: FileReader) -> Optional[int]:
        """Return the number of paths to pass to `count_rows` at a time.

        Under-the-hood, the count pushdown rule uses the `MapBatches` logical operator.
        The semantics for the batch size are the same.
        """
        ...
