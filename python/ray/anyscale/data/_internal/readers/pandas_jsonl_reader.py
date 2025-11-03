import logging
from typing import TYPE_CHECKING, Iterable, Optional

import pandas as pd
from .native_file_reader import NativeFileReader
from ray.anyscale.data._internal.file_indexer import ChunkMetadata
from ray.data._internal.datasource.json_datasource import (
    StrictBufferedReader,
    _cast_range_index_to_string,
)
from ray.data._internal.pandas_block import PandasBlockAccessor
from ray.data.block import DataBatch
from ray.data.context import DataContext

if TYPE_CHECKING:
    import pyarrow

logger = logging.getLogger(__name__)


class PandasJSONLReader(NativeFileReader):

    # Buffer size in bytes for reading files. Default is 1MB.
    #
    # pandas reads data in small chunks (~8 KiB), which leads to many costly
    # small read requests when accessing cloud storage. To reduce overhead and
    # improve performance, we wrap the file in a larger buffered reader that
    # reads bigger blocks at once.
    _BUFFER_SIZE = 1024**2

    # In the case of zipped json files, we cannot infer the chunk_size.
    _DEFAULT_CHUNK_SIZE = 10000

    # Target output size in bytes, if the target max block size isn't set.
    _DEFAULT_TARGET_OUTPUT_SIZE_BYTES = 128 * 1024**2

    def __init__(
        self,
        **native_file_reader_kwargs,
    ):
        super().__init__(**native_file_reader_kwargs)

        self._target_output_size_bytes = DataContext.get_current().target_max_block_size
        if self._target_output_size_bytes is None:
            self._target_output_size_bytes = self._DEFAULT_TARGET_OUTPUT_SIZE_BYTES

    def open_input_source(
        self,
        path: str,
        *,
        filesystem,
    ) -> "pyarrow.NativeFile":
        open_args = self._open_args.copy()
        compression = self.resolve_compression(path, open_args)

        if compression is None:
            # We use a seekable file to estimate chunksize.
            return filesystem.open_input_file(path, **open_args)

        return super().open_input_source(path, filesystem=filesystem)

    def read_stream(
        self,
        f: "pyarrow.NativeFile",
        path: str,
        metadata: Optional[ChunkMetadata] = None,
    ) -> Iterable[DataBatch]:
        assert metadata is None, "Metadata isn't supported"

        chunksize = self._estimate_chunksize(f)

        stream = StrictBufferedReader(f, buffer_size=self._BUFFER_SIZE)

        with pd.read_json(stream, chunksize=chunksize, lines=True) as reader:
            for df in reader:
                df = _cast_range_index_to_string(df)
                # pandas uses NaN for both nulls and invalid arithmetic operations. To
                # preserve the distinction, we convert to NA.
                yield df.convert_dtypes().fillna(pd.NA)

    def _estimate_chunksize(self, f: "pyarrow.NativeFile") -> Optional[int]:
        """Estimate the chunksize by sampling the first row.

        This is necessary to avoid OOMs while reading the file.
        """
        if not f.seekable():
            return self._DEFAULT_CHUNK_SIZE
        assert f.tell() == 0, "File pointer must be at the beginning"
        stream = StrictBufferedReader(f, buffer_size=self._BUFFER_SIZE)
        with pd.read_json(stream, chunksize=1, lines=True) as reader:
            try:
                df = _cast_range_index_to_string(next(reader))
            except StopIteration:
                return 1

        block_accessor = PandasBlockAccessor.for_block(df)
        if block_accessor.num_rows() == 0:
            chunksize = 1
        else:
            bytes_per_row = block_accessor.size_bytes() / block_accessor.num_rows()
            chunksize = max(round(self._target_output_size_bytes / bytes_per_row), 1)

        # Reset file pointer to the beginning.
        f.seek(0)

        return chunksize
