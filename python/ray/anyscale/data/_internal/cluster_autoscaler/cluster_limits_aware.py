from logging import Logger
from typing import TYPE_CHECKING, List

from ray.data._internal.execution.interfaces import ExecutionResources

if TYPE_CHECKING:
    from ray.anyscale.data._internal.cluster_autoscaler.rate_based_cluster_autoscaler import (
        NodeType,
    )

logger = Logger(__name__)


def clamp_resource_limits(
    node_type_request: List["NodeType"],
    max_cluster_limits: ExecutionResources,
) -> List["NodeType"]:
    """Clamp resource requests to respect cluster limits.

    Args:
        node_type_request: List of node types to be requested.
        max_cluster_limits: Maximum cluster resource limits to respect.

    Returns:
        Clamped list of resource requests that respect cluster limits.
    """

    # We employ a greedy approach to request until we can satisfy the cluster limits.
    clamped: List["NodeType"] = []
    total = ExecutionResources.zero()
    for node_type in node_type_request:
        total = total.add(node_type._resources)

        if not total.satisfies_limit(max_cluster_limits):
            logger.debug(
                f"Clamping cluster to current cluster limits: {max_cluster_limits}"
            )
            break

        clamped.append(node_type)

    return clamped
