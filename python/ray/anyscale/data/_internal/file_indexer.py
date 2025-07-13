import abc
import logging
from typing import Callable, Iterable, List, Optional, Tuple

import pyarrow as pa
from pyarrow.fs import FileSelector, FileSystem, FileType

from ray.anyscale.data._internal.logical.operators.list_files_operator import (
    FileManifest,
)
from ray.data.block import BlockAccessor, BlockColumn
from ray.data.datasource.file_meta_provider import _handle_read_os_error
from ray.data.datasource.partitioning import PathPartitionFilter
from ray.data.datasource.path_util import (
    _has_file_extension,
    _resolve_paths_and_filesystem,
)

logger = logging.getLogger(__name__)


class FileIndexer(abc.ABC):
    @abc.abstractmethod
    def list_files(
        self, paths: "BlockColumn", *, filesystem: "FileSystem"
    ) -> Iterable[FileManifest]:
        """List files and their on-disk sizes for the given path.

        Args:
            paths: A column of paths pointing to files or directories.
            filesystem: A PyArrow filesystem object.

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
    _MAX_PATHS_PER_LIST_FILES_OUTPUT = 1000

    def __init__(self, *, ignore_missing_paths: bool):
        self._ignore_missing_paths = ignore_missing_paths

    def list_files(
        self, paths: "BlockColumn", *, filesystem: "FileSystem"
    ) -> Iterable[FileManifest]:
        running_paths = []
        running_file_sizes = []
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

                running_paths.append(path)
                running_file_sizes.append(file_size)
                if len(running_paths) >= self._MAX_PATHS_PER_LIST_FILES_OUTPUT:
                    yield FileManifest.from_paths_and_sizes(
                        running_paths, running_file_sizes
                    )
                    running_paths = []
                    running_file_sizes = []

        if running_paths:
            yield FileManifest.from_paths_and_sizes(running_paths, running_file_sizes)


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


# TODO: Maybe push these down to the `FileIndexer` interface so that `FileIndexer`
#       implementations can more efficiently filter paths.
def filter_paths(
    manifest: FileManifest, filter_fn: Callable[[str], bool]
) -> FileManifest:
    """Return a new manifest with only the paths that match the filter.

    Args:
        manifest: The manifest to filter.
        filter_fn: A function that takes a path and returns `True` if the path should be
            included in the new manifest.

    Returns:
        A new manifest with only the paths that match the filter.
    """
    indices = []
    for i, path in enumerate(manifest.paths):
        if filter_fn(path):
            indices.append(i)

    if not indices:
        # `Table.take` doesn't work if `indices` is empty. So, we explicitly return an
        # empty manifest.
        return FileManifest.from_paths_and_sizes([], [])
    else:
        filtered_block = BlockAccessor.for_block(manifest.as_block()).take(indices)
        return FileManifest(filtered_block)


def filter_file_manifest(
    file_manifest: FileManifest,
    file_extensions: Optional[List[str]],
    partition_filter: Optional[PathPartitionFilter],
) -> FileManifest:
    # Apply `file_extensions` parameter.
    if file_extensions is not None:
        file_manifest = filter_paths(
            file_manifest,
            lambda path: _has_file_extension(path, file_extensions),
        )

    # Apply `partition_filter` parameter.
    if partition_filter is not None:
        file_manifest = filter_paths(
            file_manifest, lambda path: partition_filter([path])
        )
    return file_manifest
