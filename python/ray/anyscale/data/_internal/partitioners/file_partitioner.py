import abc

from ray.anyscale.data._internal.logical.operators.list_files_operator import (
    FileManifest,
)


class FilePartitioner(abc.ABC):
    """Abstract base class for partitioning file manifests.

    A ``FilePartitioner`` groups file paths and their associated metadata into new
    file manifests based on a specific partitioning strategy.

    Implementations must be deterministic to ensure consistent partitioning across
    retries.
    """

    @abc.abstractmethod
    def add_input(self, manifest: FileManifest):
        """Add a file manifest to be partitioned.

        Args:
            manifest: A ``FileManifest`` containing paths and metadata to partition.
        """
        ...

    @abc.abstractmethod
    def has_partition(self) -> bool:
        """Check if there are any partitions available.

        Returns:
            ``True`` if there are partitions ready to be retrieved via
            ``next_partition()``, ``False`` otherwise.
        """
        ...

    @abc.abstractmethod
    def next_partition(self) -> FileManifest:
        """Get the next available partition.

        Returns:
            A ``FileManifest`` containing the paths and metadata for the next partition.
        """
        ...

    @abc.abstractmethod
    def finalize(self):
        """Process any remaining files and complete the partitioning.

        This method is called after all inputs have been added via ``add_input()`` to
        ensure any buffered files are properly partitioned.
        """
        ...
