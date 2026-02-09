from typing import TYPE_CHECKING, Iterable, Iterator, Optional, Set

import numpy as np

from .in_memory_size_estimator import InMemorySizeEstimator
from .native_file_reader import NativeFileReader
from .supports_metadata import MetadataType, SupportsMetadata
from ray.anyscale.data._internal.file_indexer import ChunkMetadata
from ray.anyscale.data._internal.logical.operators.list_files_operator import (
    FileManifest,
)
from ray.data.block import BlockMetadata, DataBatch

if TYPE_CHECKING:
    import pyarrow


class BinaryReader(NativeFileReader, SupportsMetadata):
    def read_stream(
        self,
        file: "pyarrow.NativeFile",
        path: str,
        metadata: Optional[ChunkMetadata] = None,
    ) -> Iterable[DataBatch]:
        yield {"bytes": [file.readall()]}

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
            input_files=None,
        )

    def available_metadata(self) -> Set[MetadataType]:
        return {MetadataType.NUM_ROWS}

    def get_target_metadata_batch_size(self) -> Optional[int]:
        # Since we just return the number of paths, we don't need to batch.
        return None

    def produces_one_row_per_file(self) -> bool:
        """Each binary file produces 1 row"""
        return True


class BinaryInMemorySizeEstimator(InMemorySizeEstimator):
    def estimate_in_memory_sizes(self, manifest: FileManifest) -> np.array:
        # NOTE: This method assumes that the file isn't compressed.
        return manifest.file_sizes
