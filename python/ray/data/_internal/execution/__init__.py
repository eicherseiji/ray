import enum
import os
from typing import TYPE_CHECKING, Optional

from .ranker import DefaultRanker, Ranker

if TYPE_CHECKING:
    from ... import DataContext
    from .resource_manager import OpResourceAllocator, ResourceManager


DEFAULT_USE_OP_RESOURCE_ALLOCATOR_VERSION = os.environ.get(
    "RAY_DATA_USE_OP_RESOURCE_ALLOCATOR_VERSION", "V2"
)


class OpResourceAllocatorVersion(str, enum.Enum):
    V1 = "V1"  # ReservationOpResourceAllocator
    V2 = "V2"  # ThroughputBasedResourceAllocator


def create_resource_allocator(
    resource_manager: "ResourceManager",
    data_context: "DataContext",
) -> Optional["OpResourceAllocator"]:
    """Creates ``OpResourceAllocator`` instances.``"""

    if not data_context.op_resource_reservation_enabled:
        # This is a historical kill-switch to disable resource allocator, that
        # will be soon deprecated and removed.
        return None

    if DEFAULT_USE_OP_RESOURCE_ALLOCATOR_VERSION == OpResourceAllocatorVersion.V2:
        from .throughput_based_resource_allocator import (
            ThroughputBasedResourceAllocator,
        )

        return ThroughputBasedResourceAllocator(resource_manager)

    elif DEFAULT_USE_OP_RESOURCE_ALLOCATOR_VERSION == OpResourceAllocatorVersion.V1:
        from .resource_manager import ReservationOpResourceAllocator

        return ReservationOpResourceAllocator(
            resource_manager,
            reservation_ratio=data_context.op_resource_reservation_ratio,
        )

    else:
        raise ValueError(
            f"Resource allocator version of '{DEFAULT_USE_OP_RESOURCE_ALLOCATOR_VERSION}' is not supported"
        )


def create_ranker() -> Ranker:
    """Create a ranker instance based on environment and configuration."""
    from ray._private.ray_constants import env_bool

    # Check if RayTurbo ranker should be used
    if env_bool("RAY_DATA_USE_TURBO_RANKER", True):
        from ray.anyscale.data._internal.execution.ranker import LocationAwareRanker

        return LocationAwareRanker()
    else:
        return DefaultRanker()
