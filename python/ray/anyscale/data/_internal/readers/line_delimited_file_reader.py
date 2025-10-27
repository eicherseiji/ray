from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Optional

import pyarrow as pa
from ray.anyscale.data._internal.file_indexer import ChunkMetadata
from ray.data.datasource.partitioning import Partitioning
from ray.data._internal.output_buffer import (
    BlockOutputBuffer,
    OutputBlockSizeOption,
)
from ray.data.block import DataBatch
from ray.data.context import DataContext

from .native_file_reader import NativeFileReader
from ray.data._internal.util import RetryingPyFileSystem


def _read_complete_file(
    f: pa.NativeFile,
    decode_fn: Callable[[bytes], Dict[str, Any]],
    buffer_size: int,
    delimiter: bytes,
    drop_empty_lines: bool = False,
) -> Iterable[DataBatch]:
    """Read the entire file as a stream. This is a fallback for when the file is not seekable."""
    ctx = DataContext.get_current()
    builder = BlockOutputBuffer(
        OutputBlockSizeOption.of(target_max_block_size=ctx.target_max_block_size)
    )

    buffer = bytearray()
    while True:
        chunk: bytes = f.read(buffer_size)
        if not chunk:
            break

        buffer.extend(chunk)

        lines = bytes(buffer).split(delimiter)
        # We don't process the last line because it might be incomplete.
        for line in lines[:-1]:
            if drop_empty_lines and not line.strip():
                continue

            builder.add(decode_fn(line))
            if builder.has_next():
                yield builder.next()

        buffer = bytearray(lines[-1])

    if buffer and not (drop_empty_lines and not buffer.strip()):
        line = bytes(buffer)
        builder.add(decode_fn(line))

    builder.finalize()
    if builder.has_next():
        yield builder.next()


def _newline_aligned(
    f: pa.NativeFile, chunk_byte_start_idx: int, delimiter: bytes
) -> bool:
    """
    Check if the given byte index in the file is aligned with the delimiter. This may change
    the file position, you should be sure to call f.seek() or _fast_forward_to_newline() after
    calling this function.

    Args:
        f: The file object (must be seekable).
        chunk_byte_start_idx: The byte index in the file to check for alignment.
        delimiter: The delimiter bytes (e.g., b'\n').

    Returns:
        True if the byte at chunk_byte_start_idx is aligned with the delimiter,
        i.e., if it is at the start of the file or immediately follows a delimiter.
        False otherwise.
    """
    if chunk_byte_start_idx == 0:
        # no bytes before, we're aligned
        return True
    elif chunk_byte_start_idx >= len(delimiter):
        # more than len(delimiter) bytes before, check if the preceding bytes are a delimiter
        f.seek(chunk_byte_start_idx - len(delimiter))
        prev_bytes = f.read(len(delimiter))
        return prev_bytes == delimiter
    else:
        # fewer than delimiter bytes before, can't be aligned
        return False


def _read_file_chunk(
    f: pa.NativeFile,
    metadata: Optional[ChunkMetadata],
    decode_fn: Callable[[bytes], dict],
    buffer_size: int,
    delimiter: bytes,
    drop_empty_lines: bool = False,
) -> Iterable[DataBatch]:
    """
    Read a chunk of a line-delimited file.

    The start and end are not necessarily aligned with, newline
    boundaries, so we also will ensure that all lines are read in full. Any record that starts in the
    file chunk byte range [start, end) will be read in full. This means that if the file chunk starts in the middle
    of a partial record, we will not include that partial record in the output. It also means that if a
    file chunk ends in the middle of a record, we will read past the end of the file chunk to ensure that
    the entire record is read in full.

    Args:
        f: The file to read from, this should be a seeakable file.
        metadata: The metadata for the file including the start and end positions.
        decode_fn: The function to decode the bytes to a dict.
        buffer_size: The size of the buffer to read.
        delimiter: The delimiter to use to split the lines.
        drop_empty_lines: Whether to drop empty lines.
    """
    assert f.seekable(), "File must be seekable to be read in chunks."
    ctx = DataContext.get_current()
    buf = BlockOutputBuffer(
        OutputBlockSizeOption.of(target_max_block_size=ctx.target_max_block_size)
    )

    chunk_byte_start_idx, chunk_byte_end_idx = (
        metadata["chunk_byte_start_idx"],
        metadata["chunk_byte_end_idx"],
    )

    if not _newline_aligned(f, chunk_byte_start_idx, delimiter):
        _fast_forward_to_newline(f, chunk_byte_end_idx, buffer_size, delimiter)
    else:
        # newline aligned, seek to the start of the chunk
        f.seek(chunk_byte_start_idx)

    if f.tell() >= chunk_byte_end_idx:
        # Advancing the new line has moved us past the end of the file chunk, do not read any records.
        return

    tail = b""

    # Read the file until we reach the end of the file chunk.
    while f.tell() < chunk_byte_end_idx:
        chunk_start = f.tell()
        # Read up to buffer_size bytes, or until we reach the end of the file chunk.
        chunk = f.read(min(buffer_size, chunk_byte_end_idx - chunk_start))
        if not chunk:
            break

        # If the record does not end in this chunk, full will be [].
        # TODO(mowen): Might be a way to make this more memory efficient?
        lines = (tail + chunk).split(delimiter)
        tail, full = lines[-1], lines[:-1]

        for ln in full:
            if drop_empty_lines and not ln.strip():
                continue

            buf.add(decode_fn(ln))
            if buf.has_next():
                yield buf.next()

    # Ensure that any record starting in the chunk is read in full, even if this means reading
    # past the end of the file chunk.
    if tail:
        while True:
            chunk = f.read(buffer_size)
            if not chunk:
                # EOF reached, partial record is complete.
                buf.add(decode_fn(tail))
                if buf.has_next():
                    yield buf.next()
                break
            lines = chunk.split(delimiter)
            if len(lines) > 1:
                # Lines was split into more than one line which means we found the end of the record
                # lines[0] is the remaining partial line so add it to tail and add it to the buffer.
                buf.add(decode_fn(tail + lines[0]))
                if buf.has_next():
                    yield buf.next()
                break
            else:
                # Have not found the end of the record yet, continue reading.
                tail += chunk

    buf.finalize()
    if buf.has_next():
        yield buf.next()


def _fast_forward_to_newline(
    f: pa.NativeFile, end: int, buf_size: int, delimiter: bytes
):
    """
    Advance `f` to just after the next newline, unless at EOF.
    Do not read well beyond the end of the file chunk.
    """
    while f.tell() < end:
        buf = f.read(buf_size)
        if not buf:
            # EOF reached / no bytes read, cannot fast forward any more
            return
        nl = buf.find(delimiter)
        # if nl != -1 we found the delimiter
        if nl != -1:
            # advance pointer to just after the delimiter: current position of file- length of buffer already read
            # + position of delimiter within the buffer + 1
            f.seek(f.tell() - len(buf) + nl + 1)
            return


class LineDelimitedFileReader(NativeFileReader):
    """
    Read *line-delimited* files (JSONL, CSV, TSV, plain-text logs …).

    Use metadata to allow for parallel file reading.
    """

    DEFAULT_BUFFER_SIZE = 8 << 20

    def __init__(
        self,
        *,
        include_paths: bool = False,
        partitioning: Optional[Partitioning] = None,
        open_args: Optional[Dict[str, Any]] = None,
        decode_fn: Optional[Callable[[bytes], dict]] = None,
        drop_empty_lines: bool = False,
        buffer_size: Optional[int] = None,
        delimiter: Optional[bytes] = None,
    ):
        super().__init__(
            include_paths=include_paths, partitioning=partitioning, open_args=open_args
        )
        # Default to reading the lines as delimited plaintext.
        self._decode_fn = (
            decode_fn
            if decode_fn is not None
            else lambda b: {"text": b.decode("utf-8", "replace")}
        )
        # Default buffer size is 8MB, but can be configured
        self._buffer_size = (
            buffer_size if buffer_size is not None else self.DEFAULT_BUFFER_SIZE
        )
        self._delimiter = delimiter if delimiter is not None else b"\n"
        self._drop_empty_lines = drop_empty_lines

    def read_stream(
        self,
        file: "pa.NativeFile",
        path: str,
        metadata: Optional[ChunkMetadata] = None,
    ) -> Iterable[DataBatch]:
        if metadata is not None:
            yield from _read_file_chunk(
                file,
                metadata,
                self._decode_fn,
                self._buffer_size,
                self._delimiter,
                self._drop_empty_lines,
            )
        else:
            yield from _read_complete_file(
                file,
                self._decode_fn,
                self._buffer_size,
                self._delimiter,
                self._drop_empty_lines,
            )

    def open_input_source(
        self,
        path: str,
        *,
        filesystem: "RetryingPyFileSystem",
    ) -> "pa.NativeFile":
        """Opens a source path and returns a file-like object.

        For uncompressed files, returns a seekable file for efficient chunking.
        For compressed files, falls back to super's streaming reads since seeking is not supported.
        """
        compression = self.resolve_compression(path, self._open_args)

        # For compressed files, use the parent's streaming implementation
        if compression:
            return super().open_input_source(path, filesystem=filesystem)
        else:
            # For uncompressed files, use seekable file access
            return filesystem.open_input_file(path)
