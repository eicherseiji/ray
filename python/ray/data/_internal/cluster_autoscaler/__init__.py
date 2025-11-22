import os
from enum import Enum
from typing import TYPE_CHECKING

from .base_cluster_autoscaler import ClusterAutoscaler
from .default_cluster_autoscaler import DefaultClusterAutoscaler

if TYPE_CHECKING:
    from ray.data._internal.execution.resource_manager import ResourceManager
    from ray.data._internal.execution.streaming_executor_state import Topology


CLUSTER_AUTOSCALER_ENV_KEY = "RAY_DATA_CLUSTER_AUTOSCALER"
CLUSTER_AUTOSCALER_ENV_DEFAULT_VALUE = "RAYTURBO"


class ClusterAutoscalerVersion(Enum):
    RAYTURBO = "RAYTURBO"
    RAYTURBO_LEGACY = "RAYTURBO_LEGACY"
    OSS = "OSS"


def create_cluster_autoscaler(
    topology: "Topology", resource_manager: "ResourceManager", *, execution_id: str
) -> ClusterAutoscaler:
    selected_autoscaler = _get_cluster_autoscaler_version()
    if selected_autoscaler == ClusterAutoscalerVersion.RAYTURBO:
        from ray.anyscale.data._internal.cluster_autoscaler import (
            RateBasedClusterAutoscaler,
        )

        return RateBasedClusterAutoscaler.create(
            topology, resource_manager, execution_id=execution_id
        )
    elif selected_autoscaler == ClusterAutoscalerVersion.RAYTURBO_LEGACY:
        from ray.anyscale.data._internal.cluster_autoscaler import (
            LegacyRayTurboClusterAutoscaler,
        )

        return LegacyRayTurboClusterAutoscaler(
            topology,
            resource_manager,
            execution_id=execution_id,
        )
    elif selected_autoscaler == ClusterAutoscalerVersion.OSS:
        return DefaultClusterAutoscaler(
            topology,
            resource_manager,
            execution_id=execution_id,
        )

    assert False, "Invalid cluster autoscaler option"


def _get_cluster_autoscaler_version() -> ClusterAutoscalerVersion:
    cluster_autoscaler_env_value = os.getenv(
        CLUSTER_AUTOSCALER_ENV_KEY, CLUSTER_AUTOSCALER_ENV_DEFAULT_VALUE
    )
    try:
        return ClusterAutoscalerVersion(cluster_autoscaler_env_value)
    except ValueError:
        valid_values = [version.value for version in ClusterAutoscalerVersion]
        raise ValueError(
            f"{cluster_autoscaler_env_value} isn't a valid option for "
            f"{CLUSTER_AUTOSCALER_ENV_KEY}. Valid options are: {valid_values}."
        )


__all__ = ["ClusterAutoscaler"]
