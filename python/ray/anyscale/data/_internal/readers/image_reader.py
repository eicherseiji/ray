import io
from typing import Iterator, Iterable, Optional, Set, Tuple

import numpy as np
import pyarrow

from ray.anyscale.data._internal.logical.operators.list_files_operator import (
    FileManifest,
)
from ray.data._internal.util import _check_import
from ray.data.block import BlockMetadata, DataBatch
from ray.anyscale.data._internal.file_indexer import ChunkMetadata

from .native_file_reader import NativeFileReader
from .supports_metadata import MetadataType, SupportsMetadata


class ImageReader(NativeFileReader, SupportsMetadata):
    def __init__(
        self,
        size: Optional[Tuple[int, int]] = None,
        mode: Optional[str] = None,
        **file_reader_kwargs,
    ):
        super().__init__(**file_reader_kwargs)

        _check_import(self, module="PIL", package="Pillow")

        if size is not None and len(size) != 2:
            raise ValueError(
                "Expected `size` to contain two integers for height and width, "
                f"but got {len(size)} integers instead."
            )

        if size is not None and (size[0] < 0 or size[1] < 0):
            raise ValueError(
                f"Expected `size` to contain positive integers, but got {size} instead."
            )

        self.size = size
        self.mode = mode

    def read_stream(
        self,
        file: "pyarrow.NativeFile",
        path: str,
        metadata: Optional[ChunkMetadata] = None,
    ) -> Iterable[DataBatch]:
        from PIL import Image, UnidentifiedImageError

        data = file.readall()

        try:
            image = Image.open(io.BytesIO(data))
        except UnidentifiedImageError as e:
            raise ValueError(f"PIL couldn't load image file at path '{path}'.") from e

        if self.size is not None and image.size != tuple(reversed(self.size)):
            height, width = self.size
            image = image.resize((width, height), resample=Image.BILINEAR)
        if self.mode is not None and image.mode != self.mode:
            image = image.convert(self.mode)

        batch = np.expand_dims(np.asarray(image), axis=0)
        yield {"image": batch}

    def read_metadata(
        self,
        file_manifest: FileManifest,
        *,
        filesystem,
    ) -> Iterator[BlockMetadata]:
        yield BlockMetadata(
            num_rows=len(file_manifest.paths),
            size_bytes=None,
            exec_stats=None,
            input_files=file_manifest.paths,
        )

    def available_metadata(self) -> Set[MetadataType]:
        return {MetadataType.NUM_ROWS}

    def get_target_metadata_batch_size(self) -> Optional[int]:
        # Since we just return the number of paths, we don't need to batch.
        return None

    def produces_one_row_per_file(self) -> bool:
        """Each image file produces 1 row"""
        return True
