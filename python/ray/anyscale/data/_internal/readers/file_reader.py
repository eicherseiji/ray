import abc
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Union

from ray.anyscale.data._internal.logical.operators.list_files_operator import (
    FileManifest,
)
from ray.data.block import DataBatch

if TYPE_CHECKING:
    import pyarrow
    import pyarrow.dataset as pd
    from ray.data.expressions import Expr


# TODO(@bveeramani): Consolidate this with `FileBasedDatasource` so that there aren't
# two divergent code paths.
class FileReader(abc.ABC):
    """Base class for reading files.

    The `ReadFiles` operator uses implementations of this interface to read data from
    files.
    """

    @abc.abstractmethod
    def read_files(
        self,
        file_manifest: FileManifest,
        *,
        predicate_expr: Optional[Union["Expr", "pd.Expression"]] = None,
        columns: Optional[List[str]],
        columns_rename: Optional[Dict[str, str]],
        filesystem: "pyarrow.fs.FileSystem",
    ) -> Iterable[DataBatch]:
        """Read batches of data from the given file paths.

        Args:
            file_manifest: A manifest containing the paths and on-disk sizes of the
                files.
            predicate_expr: Ray Data expression or PyArrow expression for predicate pushdown.
            columns: The columns that will be read. If None, all columns will be read.
            columns_rename: Mapping to rename columns.
            filesystem: The filesystem to read from.

        Returns:
            An iterable of data batches. Batches can be any size.
        """
        ...

    def supports_predicate_pushdown(self) -> bool:
        """Whether expressions can be handled upon reading"""
        return False

    def produces_one_row_per_file(self) -> bool:
        """Whether each file produces 1 row"""
        return False
