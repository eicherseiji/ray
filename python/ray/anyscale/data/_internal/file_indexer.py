import abc
import logging
from typing import Iterable, List, Optional, Tuple
from typing import TypedDict, Type, get_type_hints

import math
import pyarrow as pa
from pyarrow.fs import FileSelector, FileSystem, FileType

from ray.anyscale.data._internal.logical.operators.list_files_operator import (
    FileManifest,
)
from ray.data.block import BlockColumn, Block
from ray.data.datasource.file_meta_provider import _handle_read_os_error
from ray.data.datasource.partitioning import PathPartitionFilter
from ray.data.datasource.path_util import (
    _has_file_extension,
    _resolve_paths_and_filesystem,
)
from ray.anyscale.data._internal.util.compression import infer_compression
from ray._private.ray_constants import env_integer


logger = logging.getLogger(__name__)


class ChunkMetadata(TypedDict):
    """Base interface for chunk metadata types."""

    pass


def create_chunk_metadata(cls: Type[ChunkMetadata], **kwargs) -> ChunkMetadata:
    """Create a metadata instance with validation, ensure the keys are correct."""
    # Automatically get required keys from the class annotations
    required_keys = list(get_type_hints(cls).keys())

    # Check that all required keys are present
    missing_keys = [key for key in required_keys if key not in kwargs]
    if missing_keys:
        raise ValueError(f"Missing required keys: {missing_keys}")

    # Check that no extra keys are provided
    extra_keys = [key for key in kwargs if key not in required_keys]
    if extra_keys:
        raise ValueError(f"Unexpected keys: {extra_keys}")

    return kwargs


class LineDelimitedFileChunkMetadata(ChunkMetadata):
    """Metadata for line-delimited file chunks."""

    chunk_byte_start_idx: int
    chunk_byte_end_idx: int


class FileChunker(abc.ABC):
    """Abstract base class for chunking files into smaller pieces for parallel processing.

    File chunkers determine how large files should be split into chunks that can be
    processed in parallel. Different file formats may require different chunking strategies.

    For example:
    - Line-delimited files (JSONL, CSV) can be chunked by byte ranges
    - Parquet files can be chunked by row groups
    """

    @abc.abstractmethod
    def generate_chunk_metadatas(
        self, path: str, file_size: int
    ) -> Iterable[Tuple[Optional[ChunkMetadata], int]]:
        """Generate metadata for file chunks.

        Args:
            path: The file path being chunked.
            file_size: The total size in bytes of the file to be chunked.

        Returns:
            An iterable of tuples containing (metadata, chunk_size) where metadata
            describes the chunk and chunk_size is the size of the chunk in bytes.
            Metadata can be None for chunks that don't require metadata
            (e.g., whole file processing).
        """
        ...


class WholeFileChunker(FileChunker):
    """File chunker that treats the whole file as a single chunk.

    This chunker is used when files should be processed as a single unit,
    typically for smaller files or when the file format doesn't support
    efficient chunking (e.g., compressed files).

    Yields a single chunk with no metadata, indicating the entire file
    should be processed as one unit.
    """

    def generate_chunk_metadatas(
        self, path: str, file_size: int
    ) -> Iterable[Tuple[Optional[ChunkMetadata], int]]:
        yield None, file_size


class LineDelimitedFileChunker(FileChunker):
    """File chunker for line-delimited files (JSONL, CSV, TSV, etc.).

    This chunker splits files into fixed-size byte chunks (default: 256MB)
    and provides metadata about the byte ranges for each chunk. The actual
    line boundaries are handled by the reader to ensure complete records.
    """

    # TODO(mowen): This should probably be a parameter or pulled from the DataContext.
    _CHUNK_BYTE_SIZE = 256 * 1024 * 1024  # 256MB

    def generate_chunk_metadatas(
        self, path: str, file_size: int
    ) -> Iterable[Tuple[Optional[ChunkMetadata], int]]:
        compression = infer_compression(path)
        if compression is not None:
            # For compressed files, use whole-file chunking
            yield None, file_size
        else:
            num_chunks = math.ceil(file_size / self._CHUNK_BYTE_SIZE)
            for chunk_idx in range(num_chunks):
                chunk_start = self._CHUNK_BYTE_SIZE * chunk_idx
                chunk_end = min(self._CHUNK_BYTE_SIZE * (chunk_idx + 1), file_size)
                chunk_size = chunk_end - chunk_start
                yield create_chunk_metadata(
                    LineDelimitedFileChunkMetadata,
                    chunk_byte_start_idx=chunk_start,
                    chunk_byte_end_idx=chunk_end,
                ), chunk_size


class FileIndexer(abc.ABC):
    @abc.abstractmethod
    def list_files(
        self,
        paths: "BlockColumn",
        *,
        filesystem: "FileSystem",
        file_extensions: Optional[List[str]] = None,
        partition_filter: Optional[PathPartitionFilter] = None,
        checkpoint_ids: Optional[Block] = None,
    ) -> Iterable[FileManifest]:
        """List files and their on-disk sizes for the given path.

        Args:
            paths: A column of paths pointing to files or directories.
            filesystem: A PyArrow filesystem object.
            file_extensions: A list of file extensions to filter by.
            partition_filter: A partition filter to filter by.
            checkpoint_ids: A block of checkpointed IDs.

        Returns:
            An iterator of `FileManifest` objects, each of which contains a file path
            and the on-disk size of the file in bytes.
        """
        ...


class NonSamplingFileIndexer(FileIndexer):
    """A file indexer that exhaustively lists files.

    This implementation works with paths that point to files or directories, although
    it's slow if you try to list lots of paths pointing to files rather than a single
    directory.
    """

    # This number was chosen because it's the maximum number of paths returned by S3
    # per page when listing a single directory.
    _MAX_PATHS_PER_LIST_FILES_OUTPUT = env_integer(
        "RAY_DATA_MAX_PATHS_PER_LIST_FILES_OUTPUT", 1000
    )

    def __init__(
        self,
        *,
        ignore_missing_paths: bool,
        file_chunker: Optional[FileChunker] = None,
    ):
        self._ignore_missing_paths = ignore_missing_paths
        self._file_chunker = (
            file_chunker if file_chunker is not None else WholeFileChunker()
        )

    def list_files(
        self,
        paths: "BlockColumn",
        *,
        filesystem: "FileSystem",
        file_extensions: Optional[List[str]] = None,
        partition_filter: Optional[PathPartitionFilter] = None,
        checkpoint_ids: Optional[Block] = None,
    ) -> Iterable[FileManifest]:
        from ray.anyscale.data.checkpoint.util import (
            get_checkpoint_fragments_info_for_file,
            index_checkpointed_fragments,
            is_file_fragments_fully_checkpointed,
        )

        if checkpoint_ids is not None:
            # Index the checkpointed fragments by file path
            checkpointed_fragments_by_path: dict[
                str, int
            ] = index_checkpointed_fragments(checkpoint_ids)
        else:
            checkpointed_fragments_by_path = {}

        running_paths = []
        running_file_sizes = []
        running_file_chunk_metadatas = []
        running_file_fragments_checkpoint = []
        manifests_count = 0
        filtered_paths_count = 0
        file_chunks_count = 0
        for input_path in paths.to_pylist():
            resolved_paths, _ = _resolve_paths_and_filesystem(input_path, filesystem)
            assert len(resolved_paths) == 1
            for path, file_size in _get_file_infos(
                resolved_paths[0], filesystem, self._ignore_missing_paths
            ):
                # Some filesystems (e.g., HTTP) return `None` for file size,
                # so we explicitly check for zero-byte files rather than checking for
                # a falsey file size.
                if file_size == 0:
                    logger.warning(f"Skipping zero-size file: {path!r}")
                    continue

                # Skip if path doesn't match file extensions or partition filter
                if filter_file_path(path, file_extensions, partition_filter):
                    filtered_paths_count += 1
                    continue

                checkpoint_fragments_info = None
                if checkpoint_ids is not None:
                    # Get checkpoint file fragments for this file
                    checkpoint_fragments_info = get_checkpoint_fragments_info_for_file(
                        checkpoint_ids, path, checkpointed_fragments_by_path
                    )

                    # Check if the file is fully checkpointed
                    if (
                        checkpoint_fragments_info.checkpointed_file_fragments
                        is not None
                        and is_file_fragments_fully_checkpointed(
                            checkpoint_fragments_info
                        )
                    ):
                        logger.debug(
                            f"list_files: Skipping fully checkpointed file: {path!r}"
                        )
                        continue

                for chunk_metadata, size in self._file_chunker.generate_chunk_metadatas(
                    path, file_size
                ):
                    running_paths.append(path)
                    running_file_sizes.append(size)
                    running_file_chunk_metadatas.append(chunk_metadata)
                    running_file_fragments_checkpoint.append(checkpoint_fragments_info)
                    file_chunks_count += 1
                    if len(running_paths) >= self._MAX_PATHS_PER_LIST_FILES_OUTPUT:
                        manifests_count += 1
                        yield FileManifest.construct_manifest(
                            running_paths,
                            running_file_sizes,
                            running_file_chunk_metadatas,
                            running_file_fragments_checkpoint,
                        )
                        running_paths = []
                        running_file_sizes = []
                        running_file_chunk_metadatas = []
                        running_file_fragments_checkpoint = []
        if running_paths:
            manifests_count += 1
            yield FileManifest.construct_manifest(
                running_paths,
                running_file_sizes,
                running_file_chunk_metadatas,
                running_file_fragments_checkpoint,
            )
        logger.debug(
            f"Listing files: filtered {filtered_paths_count} paths, constructed manifests {manifests_count} with {file_chunks_count} file chunks"
        )


def _get_file_infos(
    path: str, filesystem: pa.fs.FileSystem, ignore_missing_path: bool
) -> Iterable[Tuple[str, Optional[int]]]:
    from pyarrow.fs import FileType

    try:
        file_info = filesystem.get_file_info(path)
    except OSError as e:
        _handle_read_os_error(e, path)

    if file_info.type == FileType.Directory:
        yield from _expand_directory(path, filesystem, ignore_missing_path)
    elif file_info.type == FileType.File:
        yield (path, file_info.size)
    elif file_info.type == FileType.NotFound and ignore_missing_path:
        pass
    else:
        raise FileNotFoundError(path)


def _expand_directory(
    base_path: str, filesystem: pa.fs.FileSystem, ignore_missing_path: bool
) -> Iterable[Tuple[str, Optional[int]]]:
    exclude_prefixes = [".", "_"]
    selector = FileSelector(
        base_path, recursive=False, allow_not_found=ignore_missing_path
    )
    files = filesystem.get_file_info(selector)

    # Lineage reconstruction doesn't work if tasks aren't deterministic, and
    # `filesystem.get_file_info` might return files in a non-deterministic order. So, we
    # sort the files.
    assert isinstance(files, list), type(files)
    files.sort(key=lambda file_: file_.path)

    for file_ in files:
        if not file_.path.startswith(base_path):
            continue

        relative = file_.path[len(base_path) :]
        if any(relative.startswith(prefix) for prefix in exclude_prefixes):
            continue

        if file_.type == FileType.File:
            yield (file_.path, file_.size)
        elif file_.type == FileType.Directory:
            yield from _expand_directory(file_.path, filesystem, ignore_missing_path)
        elif file_.type == FileType.UNKNOWN:
            logger.warning(f"Discovered file with unknown type: '{file_.path}'")
            continue
        else:
            assert file_.type == FileType.NotFound
            raise FileNotFoundError(file_.path)


def filter_file_path(
    path: str,
    file_extensions: Optional[List[str]],
    partition_filter: Optional[PathPartitionFilter],
) -> bool:
    """Checks if a file should be filtered out by file extensions or partition filter.
    Args:
        path: The path of the file to check.
        file_extensions: The file extensions to filter by.
        partition_filter: The partition filter to filter by.

    Returns:
        True if the file should be filtered out, False otherwise.
    """
    if file_extensions is not None and not _has_file_extension(path, file_extensions):
        return True

    if partition_filter is not None and not partition_filter([path]):
        return True
    return False
