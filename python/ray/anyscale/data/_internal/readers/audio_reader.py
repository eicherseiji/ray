import io
from typing import Iterable, Optional

import pyarrow

from .native_file_reader import NativeFileReader
from ray.data._internal.util import _check_import
from ray.data.block import DataBatch
from ray.anyscale.data._internal.file_indexer import ChunkMetadata


class AudioReader(NativeFileReader):
    def __init__(
        self,
        **file_reader_kwargs,
    ):
        super().__init__(**file_reader_kwargs)

        _check_import(self, module="soundfile", package="soundfile")

    def read_stream(
        self,
        file: "pyarrow.NativeFile",
        path: str,
        metadata: Optional[ChunkMetadata] = None,
    ) -> Iterable[DataBatch]:
        import soundfile

        # `soundfile` doesn't support reading from a `pyarrow.NativeFile` directly, so
        # we need to read the file into memory first.
        stream = io.BytesIO(file.read())
        amplitude, sample_rate = soundfile.read(stream, always_2d=True, dtype="float32")

        # (amplitude, channels) -> (channels, amplitude)
        amplitude = amplitude.transpose((1, 0))

        batch = {"amplitude": [amplitude], "sample_rate": [sample_rate]}
        yield batch

    def produces_one_row_per_file(self) -> bool:
        """Each audio file produces 1 row"""
        return True
