from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, Dict, Union

from ray.anyscale.data.checkpoint.interfaces import CheckpointBackend, CheckpointConfig
from ray.anyscale.data.issue_detection.issue_detector_configuration import (
    IssueDetectorsConfiguration,
)
from ray.data.context import (
    DEFAULT_TARGET_MAX_BLOCK_SIZE,
    DEFAULT_TARGET_MIN_BLOCK_SIZE,
    env_bool,
)

if TYPE_CHECKING:
    from ray.anyscale.data.issue_detection.issue_detector_configuration import (
        IssueDetectorsConfiguration,
    )

DEFAULT_NUM_BLOCKS_PER_READ_TASK = 8

DEFAULT_DISABLE_LARGE_FILE_CHUNKING = env_bool(
    "RAY_TURBO_DISABLE_LARGE_FILE_CHUNKING", False
)


def _issue_detectors_config_factory() -> "IssueDetectorsConfiguration":
    # Lazily import to avoid circular dependencies.
    from ray.anyscale.data.issue_detection.issue_detector_configuration import (
        IssueDetectorsConfiguration,
    )

    return IssueDetectorsConfiguration()


@dataclass
class DataContextMixin:
    """A mix-in class that allows adding Anyscale proprietary
    attributes and methods to :class:`~ray.data.DataContext`."""

    # Configuration for Ray Data checkpointing.
    # If None, checkpointing is disabled.
    _checkpoint_config: Optional[CheckpointConfig] = None

    # Configuration for Issue Detection
    issue_detectors_config: "IssueDetectorsConfiguration" = field(
        default_factory=_issue_detectors_config_factory
    )

    # Overrides viability of fusion for file reading ops
    _enable_read_files_fusion_override: Optional[bool] = None

    min_read_partition_size: int = DEFAULT_TARGET_MIN_BLOCK_SIZE
    # To amortize overheads associated with launching Ray tasks and using multi-
    # threading, produce multiple blocks in each read task. This doesn't change the
    # size of the blocks, but it does change the number of blocks produced by each
    # task.
    max_read_partition_size: int = (
        DEFAULT_TARGET_MAX_BLOCK_SIZE * DEFAULT_NUM_BLOCKS_PER_READ_TASK
    )

    use_polars_join: bool = True

    # Controls whether large file chunking is disabled
    # When True, uses WholeFileChunker instead of more granular chunking strategies
    disable_large_file_chunking: bool = DEFAULT_DISABLE_LARGE_FILE_CHUNKING

    @property
    def checkpoint_config(self) -> Optional[CheckpointConfig]:
        """Get the checkpoint configuration."""
        return self._checkpoint_config

    @checkpoint_config.setter
    def checkpoint_config(
        self, value: Optional[Union[CheckpointConfig, Dict[str, Any]]]
    ) -> None:
        """Set the checkpoint configuration."""
        if value is None:
            self._checkpoint_config = None
        elif isinstance(value, dict):
            if "override_backend" in value:
                if not isinstance(value["override_backend"], str):
                    raise TypeError(
                        "Expected 'override_backend' to be a string,"
                        f" but got {type(value['override_backend'])}."
                    )
                value["override_backend"] = CheckpointBackend[value["override_backend"]]
            self._checkpoint_config = CheckpointConfig(**value)
        elif isinstance(value, CheckpointConfig):
            self._checkpoint_config = value
        else:
            raise TypeError(
                "checkpoint_config must be a CheckpointConfig instance, a dict, or None."
            )

    # Controls whether checkpointing state is overridden
    checkpoint_enabled_override: Optional[bool] = None
