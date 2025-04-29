from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from ray.anyscale.data.checkpoint.interfaces import CheckpointConfig
from ray.anyscale.data.issue_detection.issue_detector_configuration import (
    IssueDetectorsConfiguration,
)
from ray.data.context import (
    DEFAULT_TARGET_MAX_BLOCK_SIZE,
    DEFAULT_TARGET_MIN_BLOCK_SIZE,
)

if TYPE_CHECKING:
    from ray.anyscale.data.issue_detection.issue_detector_configuration import (
        IssueDetectorsConfiguration,
    )

DEFAULT_NUM_BLOCKS_PER_READ_TASK = 8


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
    checkpoint_config: Optional["CheckpointConfig"] = None

    # Configuration for Issue Detection
    issue_detectors_config: "IssueDetectorsConfiguration" = field(
        default_factory=_issue_detectors_config_factory
    )

    min_read_partition_size: int = DEFAULT_TARGET_MIN_BLOCK_SIZE
    # To amortize overheads associated with launching Ray tasks and using multi-
    # threading, produce multiple blocks in each read task. This doesn't change the
    # size of the blocks, but it does change the number of blocks produced by each
    # task.
    max_read_partition_size: int = (
        DEFAULT_TARGET_MAX_BLOCK_SIZE * DEFAULT_NUM_BLOCKS_PER_READ_TASK
    )
