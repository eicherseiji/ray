import logging
from io import BytesIO
from typing import TYPE_CHECKING, Iterable

import pandas as pd

from .native_file_reader import NativeFileReader
from ray.data.block import DataBatch

if TYPE_CHECKING:
    import pyarrow

logger = logging.getLogger(__name__)

# TODO(rliaw): Arbitrarily chosen. Make this configurable
_JSONL_ROWS_PER_CHUNK = 10000


class PandasJSONLReader(NativeFileReader):
    def read_stream(self, f: "pyarrow.NativeFile", path: str) -> Iterable[DataBatch]:
        buffer: "pyarrow.lib.Buffer" = f.read_buffer()

        # Check if the buffer is empty
        if buffer.size == 0:
            return

        reader = pd.read_json(
            BytesIO(buffer), chunksize=_JSONL_ROWS_PER_CHUNK, lines=True
        )
        for df in reader:
            # Note: PandasBlockAccessor doesn't support RangeIndex, so we need to convert
            # to string.
            if isinstance(df.columns, pd.RangeIndex):
                df.columns = df.columns.astype(str)

            # pandas uses NaN for both nulls and invalid arithmetic operations. To
            # preserve the distinction, we convert to NA.
            yield df.convert_dtypes().fillna(pd.NA)
