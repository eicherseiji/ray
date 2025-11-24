import abc
import io
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import pyarrow


from .file_reader import FileReader
from ray.anyscale.data._internal.logical.operators.list_files_operator import (
    FileManifest,
)
from ray.data._internal.util import (
    RetryingPyFileSystem,
    iterate_with_retry,
    make_async_gen,
)
from ray.data.block import DataBatch
from ray.data.context import DataContext
from ray.data.datasource import Partitioning, PathPartitionParser
from ray.anyscale.data._internal.file_indexer import ChunkMetadata
from ray.data._internal.util import infer_compression


class NativeFileReader(FileReader):
    """Base class for reading a stream of bytes.

    Implementations of this interface should implement the `read_stream` method to read
    data from a stream of bytes and return data batches.
    """

    _NUM_THREADS_PER_TASK = 8

    def __init__(
        self,
        *,
        include_paths: bool = False,
        partitioning: Optional[Partitioning] = None,
        open_args: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        if open_args is None:
            open_args = {}

        # Initialize projection map for mixin (None = all columns, no renames)
        self._projection_map: Optional[Dict[str, str]] = None

        self._include_paths = include_paths
        self._partitioning = partitioning
        self._open_args = open_args
        self._data_context = DataContext.get_current()

    @property
    def data_context(self) -> DataContext:
        return self._data_context

    @abc.abstractmethod
    def read_stream(
        self,
        file: "pyarrow.NativeFile",
        path: str,
        metadata: Optional[ChunkMetadata] = None,
    ) -> Iterable[DataBatch]:
        ...

    def read_files(
        self,
        file_manifest: FileManifest,
        *,
        filesystem: "pyarrow.fs.FileSystem",
    ) -> Iterable[DataBatch]:
        # Get columns and column renames from stored projection state (OSS pattern)
        columns = self._get_data_columns()
        columns_rename = self.get_column_renames()

        paths = file_manifest.paths
        file_chunk_metadatas = file_manifest.file_chunk_metadatas
        num_threads = self._NUM_THREADS_PER_TASK
        if len(paths) < num_threads:
            num_threads = len(paths)

        # TODO: We should refactor the code so that we can get the results in order even
        # when using multiple threads.
        if self._data_context.execution_options.preserve_order:
            num_threads = 0

        if columns and columns_rename:
            assert set(columns_rename.keys()).issubset(columns), (
                f"All column rename keys must be a subset of the columns list. "
                f"Invalid keys: {set(columns_rename.keys()) - set(columns)}"
            )

        def _read_paths(path_info: List[Tuple[str, ChunkMetadata]]):
            for path, file_chunk_metadata in path_info:
                partitions = {}
                if self._partitioning is not None:
                    parse = PathPartitionParser(self._partitioning)
                    partitions = parse(path)

                file = self.open_input_source(path, filesystem=filesystem)

                for batch in iterate_with_retry(
                    lambda: self.read_stream(file, path, file_chunk_metadata),
                    description="read stream iteratively",
                    match=self._data_context.retried_io_errors,
                ):
                    if self._include_paths:
                        batch = _add_column_to_batch(batch, "path", path)
                    for partition, value in partitions.items():
                        if not columns or partition in columns:
                            batch = _add_column_to_batch(batch, partition, value)
                    if columns:
                        batch = _filter_columns(batch, columns)
                    if columns_rename:
                        batch = _rename_columns(batch, columns_rename)
                    yield batch

        if num_threads > 0:
            yield from make_async_gen(
                iter(zip(paths, file_chunk_metadatas)),
                _read_paths,
                # NOTE: It's crucial for the sequence to have preserved (deterministic)
                #       ordering so that that tasks could be safely retried (when
                #       reconstructing lost blocks)
                preserve_ordering=True,
                num_workers=num_threads,
            )
        else:
            yield from _read_paths(zip(paths, file_chunk_metadatas))

    def resolve_compression(
        self, path: str, open_args: Dict[str, Any]
    ) -> Optional[str]:
        """Resolves the compression format for a stream.

        Args:
            path: The file path to resolve compression for.
            open_args: Keyword arguments passed to
                `pyarrow.fs.FileSystem.open_input_stream <https://arrow.apache.org/docs/python/generated/pyarrow.fs.FileSystem.html#pyarrow.fs.FileSystem.open_input_stream>`_
                when opening input files to read.

        Returns:
            The compression format (e.g., "gzip", "snappy", "bz2") or None if
            no compression is detected or specified.
        """
        compression = open_args.get("compression", None)
        if compression is None:
            compression = infer_compression(path)
        return compression

    def _resolve_buffer_size(self, open_args: Dict[str, Any]) -> Optional[int]:
        buffer_size = open_args.pop("buffer_size", None)
        if buffer_size is None:
            buffer_size = self._data_context.streaming_read_buffer_size
        return buffer_size

    def _file_to_snappy_stream(
        self,
        file: "pyarrow.NativeFile",
        filesystem: "RetryingPyFileSystem",
    ) -> "pyarrow.PythonFile":
        import pyarrow as pa
        import snappy
        from pyarrow.fs import HadoopFileSystem

        stream = io.BytesIO()
        if isinstance(filesystem.unwrap(), HadoopFileSystem):
            snappy.hadoop_snappy.stream_decompress(src=file, dst=stream)
        else:
            snappy.stream_decompress(src=file, dst=stream)
        stream.seek(0)

        return pa.PythonFile(stream, mode="r")

    def open_input_source(
        self,
        path: str,
        *,
        filesystem: "RetryingPyFileSystem",
    ) -> "pyarrow.NativeFile":
        """Opens a source path and returns a file-like object that can be read from.

        This implementation opens the source path as a sequential input stream, using
        `DataContext.streaming_read_buffer_size` as the buffer size if none is given by
        the caller.
        """
        open_args = self._open_args.copy()
        compression = self.resolve_compression(path, open_args)
        buffer_size = self._resolve_buffer_size(open_args)

        if compression == "snappy":
            # Arrow doesn't support streaming Snappy decompression since the canonical
            # C++ Snappy library doesn't natively support streaming decompression. We
            # works around this by manually decompressing the file with python-snappy.
            open_args["compression"] = None
            file = filesystem.open_input_stream(
                path, buffer_size=buffer_size, **open_args
            )
            return self._file_to_snappy_stream(file, filesystem)

        open_args["compression"] = compression
        return filesystem.open_input_stream(path, buffer_size=buffer_size, **open_args)


def _rename_columns(batch: DataBatch, columns_rename: Dict[str, str]) -> DataBatch:
    assert isinstance(batch, (pd.DataFrame, dict, pyarrow.Table)), batch

    if isinstance(batch, pd.DataFrame):
        batch = batch.rename(columns=columns_rename)
    elif isinstance(batch, dict):
        batch = {columns_rename.get(k, k): v for k, v in batch.items()}
    elif isinstance(batch, pyarrow.Table):
        batch = batch.rename_columns(
            [columns_rename.get(col, col) for col in batch.schema.names]
        )

    return batch


def _filter_columns(batch: DataBatch, columns: List[str]) -> DataBatch:
    assert isinstance(batch, (pd.DataFrame, dict, pyarrow.Table)), batch

    if isinstance(batch, pd.DataFrame):
        batch = batch[columns]
    elif isinstance(batch, dict):
        batch = {column: batch[column] for column in columns}
    elif isinstance(batch, pyarrow.Table):
        batch = batch.select(columns)

    return batch


def _add_column_to_batch(batch: DataBatch, column: str, value: Any) -> DataBatch:
    assert isinstance(batch, (pd.DataFrame, dict, pyarrow.Table)), batch

    if isinstance(batch, pd.DataFrame) and column not in batch.columns:
        batch[column] = value
    elif isinstance(batch, dict) and column not in batch:
        batch_size = len(batch[next(iter(batch.keys()))])
        batch[column] = [value] * batch_size
    elif isinstance(batch, pyarrow.Table) and column not in batch.column_names:
        batch = batch.append_column(column, pyarrow.array([value] * len(batch)))

    return batch
