from typing import TYPE_CHECKING

from .base_cluster_autoscaler import ClusterAutoscaler
from .default_cluster_autoscaler import DefaultClusterAutoscaler

if TYPE_CHECKING:
    from ray.data._internal.execution.resource_manager import ResourceManager
    from ray.data._internal.execution.streaming_executor_state import Topology


def create_cluster_autoscaler(
    topology: "Topology", resource_manager: "ResourceManager", *, execution_id: str
) -> ClusterAutoscaler:
    from ray._private.ray_constants import env_bool

    if env_bool("RAY_DATA_ENABLE_RATE_BASED_CLUSTER_AUTOSCALER", False):
        from ray.anyscale.data._internal.cluster_autoscaler import (
            RateBasedClusterAutoscaler,
        )

        return RateBasedClusterAutoscaler.create(
            topology, resource_manager, execution_id
        )
    elif env_bool("RAY_DATA_ENABLE_RAYTURBO_CLUSTER_AUTOSCALER", True):
        from ray.anyscale.data._internal.cluster_autoscaler import (
            RayTurboClusterAutoscaler,
        )

        return RayTurboClusterAutoscaler(
            topology,
            resource_manager,
            execution_id=execution_id,
        )
    else:
        return DefaultClusterAutoscaler(
            topology,
            resource_manager,
            execution_id=execution_id,
        )


__all__ = ["ClusterAutoscaler"]
