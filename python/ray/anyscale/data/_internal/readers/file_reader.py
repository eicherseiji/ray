import abc
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional

from ray.anyscale.data._internal.logical.operators.list_files_operator import (
    FileManifest,
)
from ray.data.block import DataBatch
from ray.data.datasource.datasource import (
    _DatasourceProjectionPushdownMixin,
    _DatasourcePredicatePushdownMixin,
)

if TYPE_CHECKING:
    import pyarrow

# TODO(@bveeramani): Consolidate this with `FileBasedDatasource` so that there aren't
# two divergent code paths.
class FileReader(
    _DatasourceProjectionPushdownMixin,
    _DatasourcePredicatePushdownMixin,
    abc.ABC,
):
    """Base class for reading files.

    The `ReadFiles` operator uses implementations of this interface to read data from
    files.
    """

    def __init__(self):
        """Initialize the datasource and its mixins."""
        _DatasourcePredicatePushdownMixin.__init__(self)

    @abc.abstractmethod
    def read_files(
        self,
        file_manifest: FileManifest,
        *,
        columns: Optional[List[str]],
        columns_rename: Optional[Dict[str, str]],
        filesystem: "pyarrow.fs.FileSystem",
    ) -> Iterable[DataBatch]:
        """Read batches of data from the given file paths.

        Args:
            file_manifest: A manifest containing the paths and on-disk sizes of the
                files.
            columns: The columns that will be read. If None, all columns will be read.
            columns_rename: Mapping to rename columns.
            filesystem: The filesystem to read from.

        Returns:
            An iterable of data batches. Batches can be any size.
        """
        ...

    def produces_one_row_per_file(self) -> bool:
        """Whether each file produces 1 row"""
        return False
