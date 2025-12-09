import math
import time
from collections import defaultdict
from logging import getLogger
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Set

import ray
from .bottleneck_detector import (
    NormalizedThroughputBottleneckDetector,
    BottleneckDetector,
)
from .supports_cluster_autoscaling import SupportsClusterAutoscaling
from ray._private.ray_constants import env_float, env_integer
from ray.anyscale.air._internal.autoscaling_coordinator import (
    AutoscalingCoordinator,
    DefaultAutoscalingCoordinator,
)
from ray.data._internal.cluster_autoscaler import (
    ClusterAutoscaler,
)
from ray.data._internal.execution.interfaces import PhysicalOperator
from ray.data._internal.execution.interfaces.execution_options import ExecutionResources
from ray.data._internal.execution.operators.base_physical_operator import (
    AllToAllOperator,
)
from ray.data._internal.execution.operators.hash_shuffle import (
    HashShufflingOperatorBase,
)

if TYPE_CHECKING:
    from ray.data._internal.execution.resource_manager import ResourceManager
    from ray.data._internal.execution.streaming_executor_state import Topology

logger = getLogger(__name__)


class NodeType:
    """Represents a node type available in the cluster."""

    def __init__(self, resource_dict: Dict[str, float]):
        # Remove object store memory because the autoscaler SDK doesn't work correctly
        # with it.
        self._resources = ExecutionResources.from_resource_dict(resource_dict).copy(
            object_store_memory=0
        )

    def to_bundle(self):
        return {k: v for k, v in self._resources.to_resource_dict().items() if v > 0}

    def __hash__(self):
        return hash(self._resources)

    def __eq__(self, other):
        return self._resources == other._resources

    def __repr__(self):
        return f"<NodeType {self._resources}>"

    def can_schedule(self, res: ExecutionResources) -> bool:
        """Check if this node can schedule the given resources."""
        return res.satisfies_limit(self._resources)


def _get_node_types_and_counts() -> Dict[NodeType, int]:
    """Get the unique worker node types and their counts in the cluster."""
    # TODO(hchen): Use the new API to get cluster scaling config
    # when https://github.com/anyscale/rayturbo/issues/577 is done.

    # Filter out the head node because we can't scale it.
    # NOTE: Even though we filter the head node here, the streaming executor can
    # usually still schedule tasks on a head-node-only cluster because we request all
    # remaining resources when we submit a request to the autoscaling coordinator.
    node_resources = [
        node["Resources"]
        for node in ray.nodes()
        if node["Alive"] and "node:__internal_head__" not in node["Resources"]
    ]

    node_type_counts = defaultdict(int)
    for r in node_resources:
        node_type = NodeType(r)
        node_type_counts[node_type] += 1

    return node_type_counts


class RateBasedClusterAutoscaler(ClusterAutoscaler):
    """Autoscaler that only scales up the bottleneck operator.

    This autoscaler identifies the bottleneck operator using the provided
    `BottleneckDetector`. It then doubles the count of all node types that can
    schedule the bottleneck operator.

    This autoscaler only scales up the cluster. It relies on idle termination to scale
    down
    """

    # Default scaling up factor for cluster autoscaling.
    DEFAULT_CLUSTER_SCALING_UP_FACTOR: float = env_float(
        "RAY_DATA_DEFAULT_CLUSTER_SCALING_UP_FACTOR", 2.0
    )
    # Min number of seconds between two autoscaling requests.
    MIN_GAP_BETWEEN_AUTOSCALING_REQUESTS_S = env_integer(
        "RAY_DATA_MIN_GAP_BETWEEN_AUTOSCALING_REQUESTS", 10
    )
    # The time in seconds after which an autoscaling request will expire.
    AUTOSCALING_REQUEST_EXPIRE_TIME_S = 180

    def __init__(
        self,
        ops: List[SupportsClusterAutoscaling],
        execution_id: str,
        bottleneck_detector: BottleneckDetector,
        *,
        autoscaling_coordinator: Optional["AutoscalingCoordinator"] = None,
        get_node_counts: Callable[[], Dict[NodeType, int]] = _get_node_types_and_counts,
        cluster_scaling_up_factor: float = DEFAULT_CLUSTER_SCALING_UP_FACTOR,
        min_gap_between_autoscaling_requests_s: int = MIN_GAP_BETWEEN_AUTOSCALING_REQUESTS_S,
        autoscaling_request_expire_time_s: int = AUTOSCALING_REQUEST_EXPIRE_TIME_S,
    ):
        """Initialize the cluster autoscaler.

        Args:
            ops: The operators to autoscale.
            execution_id: The execution ID of the dataset. This is used to identify the
                dataset when requesting resources.
            bottleneck_detector: The detector to identify the bottleneck operator.
            autoscaling_coordinator: The `AutoscalingCoordinator` to request resources
                from. This is exposed as a seam for testing. If not provided, this uses
                the default coordinator.
            get_node_counts: A function to get the number of nodes of each type. This
                is exposed as a seam for testing.
            cluster_scaling_up_factor: The factor to scale up the cluster.
            min_gap_between_autoscaling_requests_s: The minimum gap between two
                autoscaling requests. This is exposed as a seam for testing.
            autoscaling_request_expire_time_s: The number of seconds before requested
                resources expire. This is exposed as a seam for testing.
        """
        assert all(isinstance(op, SupportsClusterAutoscaling) for op in ops)

        if autoscaling_coordinator is None:
            autoscaling_coordinator = DefaultAutoscalingCoordinator()

        self._ops = ops
        self._execution_id = execution_id
        self._bottleneck_detector = bottleneck_detector
        self._autoscaling_coordinator = autoscaling_coordinator
        self._get_node_counts = get_node_counts
        self._cluster_scaling_up_factor = cluster_scaling_up_factor
        self._min_gap_between_autoscaling_requests = (
            min_gap_between_autoscaling_requests_s
        )
        self._autoscaling_request_expire_time_s = autoscaling_request_expire_time_s

        self._last_request_time = 0
        self._requester_id = f"data-{execution_id}"

        # Send an empty request to register ourselves as soon as possible,
        # so the first `get_total_resources` call can get the allocated resources.
        self._send_resource_request([])

    @classmethod
    def create(
        cls,
        topology: "Topology",
        resource_manager: "ResourceManager",
        *,
        execution_id: str,
    ) -> "RateBasedClusterAutoscaler":
        """Create a cluster autoscaler.

        This logic is defined here to minimize the risk of merge conflicts in the
        streaming executor, and keep the `ray.data._internal.cluster_autoscaler`
        `__init__` file small.
        """
        scalable_ops = [op for op in topology if cls._is_eligible_for_scaling(op)]
        requester_id = f"data-{execution_id}"
        bottleneck_detector = NormalizedThroughputBottleneckDetector(
            resource_manager, requester_id
        )
        return RateBasedClusterAutoscaler(
            scalable_ops,
            execution_id=execution_id,
            bottleneck_detector=bottleneck_detector,
        )

    @classmethod
    def _is_eligible_for_scaling(cls, op: PhysicalOperator) -> bool:
        """Returns true if the operator is eligible for cluster autoscaling."""
        return (
            isinstance(op, SupportsClusterAutoscaling)
            # There's no point in autoscaling if the operator doesn't require any
            # resources.
            and op.min_scheduling_resources() != ExecutionResources.zero()
            # We explicitly exempt shuffle operators from autoscaling. Unlike other
            # operators, shuffle operators don't respect Ray Data's resource
            # allocations. They launch more tasks than the cluster can handle, and the
            # Ray autoscaler will add nodes to the cluster.
            and not isinstance(op, (AllToAllOperator, HashShufflingOperatorBase))
        )

    def try_trigger_scaling(self):
        # Limit the frequency of autoscaling requests.
        now = time.time()
        if now - self._last_request_time < self._min_gap_between_autoscaling_requests:
            return

        # Identify the bottleneck operator.
        bottleneck_op = self._bottleneck_detector.get_bottleneck(self._ops)

        # Log the bottleneck.
        if bottleneck_op:
            logger.debug(f"Bottleneck operator: {bottleneck_op}")

        # Still send an empty request if we couldn't identify a bottleneck. This is
        # necessary to renew our registration with the autoscaling coordinator and
        # ensure we request the remaining resources for operators that don't support
        # autoscaling.
        if not bottleneck_op:
            self._send_resource_request([])
            return []

        needed_node_types = self._find_needed_node_types(bottleneck_op)
        requested_resources = self._scale_up_cluster(needed_node_types)
        return requested_resources

    def _find_needed_node_types(self, op: SupportsClusterAutoscaling) -> Set[NodeType]:
        """Find all the worker node types that can schedule the given operator."""
        valid_worker_node_types: Set[NodeType] = set()

        resource_requirement = op.min_scheduling_resources().copy(object_store_memory=0)
        worker_node_type_counts = self._get_node_counts()
        for worker_node_type in worker_node_type_counts:
            if worker_node_type.can_schedule(resource_requirement):
                valid_worker_node_types.add(worker_node_type)

        is_head_node_only = len(worker_node_type_counts) == 0
        if (
            not valid_worker_node_types
            # Don't emit a warning if the compute config doesn't contain worker nodes,
            # because head-node-only clusters are common for small scale testing.
            and not is_head_node_only
        ):
            # Convert the `ExecutionResources` object to a resource dict because that's
            # the abstraction people use when configuring clusters, and filter out
            # falsey values to improve readability.
            min_scheduling_resources_dict = {
                key: value
                for key, value in op.min_scheduling_resources()
                .to_resource_dict()
                .items()
                if value
            }
            logger.warning(
                "The Ray Data autoscaler couldn't find any worker node types that "
                f"can execute tasks for {op}. This can happen if you misconfigure your "
                "cluster. To fix the warning, add a worker node type that satisfies "
                f"these resource requirements: {min_scheduling_resources_dict}.",
            )

        return valid_worker_node_types

    def _scale_up_cluster(
        self, needed_node_types: Set[NodeType]
    ) -> List[ExecutionResources]:
        """Scale up the cluster by requesting resources for the needed node types."""
        if not needed_node_types:
            # Still send an empty request when upscaling is not needed, to renew our
            # registration on AutoscalingCoordinator.
            self._send_resource_request([])
            return []

        resource_request = []
        debug_msg = "Scaling up cluster. Requesting resources:"
        node_type_counts = self._get_node_counts()
        for node_type, count in node_type_counts.items():
            bundle = node_type.to_bundle()

            if node_type in needed_node_types:
                num_to_request = int(math.ceil(count * self._cluster_scaling_up_factor))
                debug_msg += f" [{bundle}: {count} -> {num_to_request}]"
            else:
                num_to_request = count

            resource_request.extend([bundle] * num_to_request)

        logger.debug(debug_msg)
        self._send_resource_request(resource_request)

        return [ExecutionResources.from_resource_dict(r) for r in resource_request]

    def _send_resource_request(self, resource_request):
        self._autoscaling_coordinator.request_resources(
            requester_id=self._requester_id,
            resources=resource_request,
            expire_after_s=self._autoscaling_request_expire_time_s,
            request_remaining=True,
        )
        self._last_request_time = time.time()

    def on_executor_shutdown(self):
        # Cancel the resource request when the executor is shutting down.
        try:
            self._autoscaling_coordinator.cancel_request(self._requester_id)
        except Exception:
            msg = (
                f"Failed to cancel resource request for {self._requester_id}."
                " The request will still expire after the timeout of"
                f" {self.MIN_GAP_BETWEEN_AUTOSCALING_REQUESTS_S} seconds."
            )
            logger.warning(msg, exc_info=True)

    def get_total_resources(self) -> ExecutionResources:
        resources = self._autoscaling_coordinator.get_allocated_resources(
            self._requester_id
        )
        total = ExecutionResources.zero()
        for res in resources:
            total = total.add(ExecutionResources.from_resource_dict(res))
        return total
