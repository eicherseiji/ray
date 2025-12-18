import abc
from typing import TYPE_CHECKING, Iterable

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
        _DatasourceProjectionPushdownMixin.__init__(self)

    @abc.abstractmethod
    def read_files(
        self,
        file_manifest: FileManifest,
        *,
        filesystem: "pyarrow.fs.FileSystem",
    ) -> Iterable[DataBatch]:
        """Read batches of data from the given file paths.

        The reader should use its stored projection and predicate state
        (from apply_projection/apply_predicate calls) to determine what
        columns to read and how to filter the data.

        Args:
            file_manifest: A manifest containing the paths and on-disk sizes of the
                files.
            filesystem: The filesystem to read from.

        Returns:
            An iterable of data batches. Batches can be any size.
        """
        ...

    def produces_one_row_per_file(self) -> bool:
        """Whether each file produces 1 row"""
        return False
