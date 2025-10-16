import math
import time
from collections import defaultdict
from logging import getLogger
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Set

import ray
from .productivity_calculator import (
    NormalizedThroughputCalculator,
    ProductivityCalculator,
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
from ray.util.metrics import Gauge

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

    This autoscaler calculates a productivity score for each operator using the provided
    `ProductivityCalculator`. It then doubles the count of all node types that can
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
        productivity_calculator: ProductivityCalculator,
        *,
        autoscaling_coordinator: Optional["AutoscalingCoordinator"] = None,
        get_node_counts: Callable[[], Dict[NodeType, int]] = _get_node_types_and_counts,
        cluster_scaling_up_factor: float = DEFAULT_CLUSTER_SCALING_UP_FACTOR,
        min_gap_between_autoscaling_requests_s: int = MIN_GAP_BETWEEN_AUTOSCALING_REQUESTS_S,
    ):
        """Initialize the cluster autoscaler.

        Args:
            ops: The operators to autoscale.
            execution_id: The execution ID of the dataset. This is used to identify the
                dataset when requesting resources.
            productivity_calculator: The calculator to compute the productivity of the
                operators.
            autoscaling_coordinator: The `AutoscalingCoordinator` to request resources
                from. This is exposed as a seam for testing. If not provided, this uses
                the default coordinator.
            get_node_counts: A function to get the number of nodes of each type. This
                is exposed as a seam for testing.
            cluster_scaling_up_factor: The factor to scale up the cluster.
            min_gap_between_autoscaling_requests_s: The minimum gap between two
                autoscaling requests. This is exposed as a seam for testing.
        """
        assert all(isinstance(op, SupportsClusterAutoscaling) for op in ops)

        if autoscaling_coordinator is None:
            autoscaling_coordinator = DefaultAutoscalingCoordinator()

        self._ops = ops
        self._execution_id = execution_id
        self._productivity_calculator = productivity_calculator
        self._autoscaling_coordinator = autoscaling_coordinator
        self._get_node_counts = get_node_counts
        self._cluster_scaling_up_factor = cluster_scaling_up_factor
        self._min_gap_between_autoscaling_requests = (
            min_gap_between_autoscaling_requests_s
        )

        self._last_request_time = 0
        self._requester_id = f"data-{execution_id}"

        self._productivity_gauge: Gauge = Gauge(
            "data_productivity",
            "Productivity per operator",
            tag_keys=("requester", "operator"),
        )

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
        productivity_calculator = NormalizedThroughputCalculator(
            scalable_ops,
            resource_manager,
        )
        return RateBasedClusterAutoscaler(
            scalable_ops,
            execution_id=execution_id,
            productivity_calculator=productivity_calculator,
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
        # Calculate the productivity of each operator. We run this even if the
        # autoscaling interval hasn't elapsed, because the values are useful for
        # debugging and monitoring.
        productivities: Dict[SupportsClusterAutoscaling, float] = {
            op: score
            for op in self._ops
            if (score := self._productivity_calculator.get_productivity(op)) is not None
        }

        # Update Prometheus metrics.
        for op, score in productivities.items():
            self._productivity_gauge.set(
                score,
                tags={"requester": self._requester_id, "operator": repr(op)},
            )

        if not productivities:
            return

        # Limit the frequency of autoscaling requests.
        now = time.time()
        if now - self._last_request_time < self._min_gap_between_autoscaling_requests:
            return

        # Log metrics. We don't log these every call because they can be spammy.
        logger.debug(f"Operator productivities: {productivities}")

        bottleneck_op = min(productivities, key=productivities.get)
        needed_node_types = self._find_needed_node_types(bottleneck_op)
        if not needed_node_types:
            logger.warning("No existing node types can schedule %s.", bottleneck_op)

        requested_resources = self._scale_up_cluster(needed_node_types)
        return requested_resources

    def _find_needed_node_types(self, op: SupportsClusterAutoscaling) -> Set[NodeType]:
        """Find all the node types that can schedule the given operator."""
        node_types: Set[NodeType] = set()

        resource_requirement = op.min_scheduling_resources().copy(object_store_memory=0)
        node_type_counts = self._get_node_counts()
        for node_type in node_type_counts:
            if node_type.can_schedule(resource_requirement):
                node_types.add(node_type)

        return node_types

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
            expire_after_s=self.AUTOSCALING_REQUEST_EXPIRE_TIME_S,
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
