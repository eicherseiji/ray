import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, Optional

from .native_file_reader import NativeFileReader
from ray.data._internal.output_buffer import (
    BlockOutputBuffer,
    OutputBlockSizeOption,
)
from ray.data.block import DataBatch
from ray.data.context import DataContext

if TYPE_CHECKING:
    import pyarrow

logger = logging.getLogger(__name__)


class TextReader(NativeFileReader):
    def __init__(
        self,
        *,
        drop_empty_lines: bool = False,
        encoding: str = "utf-8",
        chunk_size: int = 8 * 1024**2,
        decode_fn: Optional[Callable[[bytes], Dict[str, Any]]] = None,
        **file_reader_kwargs,
    ):
        assert chunk_size > 0, "`chunk_size` must be greater than 0"

        super().__init__(**file_reader_kwargs)

        if decode_fn is None:

            def decode_fn(data: bytes) -> Dict[str, Any]:
                return {"text": data.decode(encoding)}

        self._drop_empty_lines = drop_empty_lines
        self._decode_fn = decode_fn
        self._chunk_size = chunk_size

        self._target_max_block_size = DataContext.get_current().target_max_block_size
        self._newline_bytes = "\n".encode(encoding)

    def read_stream(
        self,
        f: "pyarrow.NativeFile",
        path: str,
        metadata: Any = None,
    ) -> Iterable[DataBatch]:
        builder = BlockOutputBuffer(
            OutputBlockSizeOption.of(target_max_block_size=self._target_max_block_size)
        )

        buffer = bytearray()
        while True:
            chunk: bytes = f.read(self._chunk_size)
            if not chunk:
                break

            buffer.extend(chunk)

            lines = bytes(buffer).split(self._newline_bytes)
            # We don't process the last line because it might be incomplete.
            for line in lines[:-1]:
                if self._drop_empty_lines and not line.strip():
                    continue

                builder.add(self._decode_fn(line))
                if builder.has_next():
                    yield builder.next()

            buffer = bytearray(lines[-1])

        if buffer and not (self._drop_empty_lines and not buffer.strip()):
            line = bytes(buffer)
            builder.add(self._decode_fn(line))

        builder.finalize()
        if builder.has_next():
            yield builder.next()
