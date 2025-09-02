from typing import Iterable, Optional

import pyarrow

from .native_file_reader import NativeFileReader
from ray.data._internal.util import _check_import
from ray.data.block import DataBatch
from ray.anyscale.data._internal.file_indexer import ChunkMetadata


class VideoReader(NativeFileReader):
    def __init__(
        self,
        **file_reader_kwargs,
    ):
        super().__init__(**file_reader_kwargs)

        _check_import(self, module="decord", package="decord")

    def read_stream(
        self,
        file: "pyarrow.NativeFile",
        path: str,
        metadata: Optional[ChunkMetadata] = None,
    ) -> Iterable[DataBatch]:
        from decord import VideoReader

        reader = VideoReader(file)

        for frame_index, frame in enumerate(reader):
            yield {"frame": [frame.asnumpy()], "frame_index": [frame_index]}
