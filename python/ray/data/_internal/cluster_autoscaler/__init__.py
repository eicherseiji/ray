from typing import TYPE_CHECKING

from .base_cluster_autoscaler import ClusterAutoscaler
from .default_cluster_autoscaler import DefaultClusterAutoscaler

if TYPE_CHECKING:
    from ray.data._internal.execution.streaming_executor import StreamingExecutor


def create_cluster_autoscaler(
    executor: "StreamingExecutor",
) -> ClusterAutoscaler:
    from ray._private.ray_constants import env_bool

    if env_bool("RAY_DATA_ENABLE_RATE_BASED_CLUSTER_AUTOSCALER", False):
        from ray.anyscale.data._internal.cluster_autoscaler import (
            RateBasedClusterAutoscaler,
        )

        return RateBasedClusterAutoscaler.for_executor(executor)
    elif env_bool("RAY_DATA_ENABLE_RAYTURBO_CLUSTER_AUTOSCALER", True):
        from ray.anyscale.data._internal.cluster_autoscaler import (
            RayTurboClusterAutoscaler,
        )

        return RayTurboClusterAutoscaler(
            executor._topology,
            executor._resource_manager,
            execution_id=executor._dataset_id,
        )
    else:
        return DefaultClusterAutoscaler(
            executor._topology,
            executor._resource_manager,
            execution_id=executor._dataset_id,
        )


__all__ = ["ClusterAutoscaler"]
