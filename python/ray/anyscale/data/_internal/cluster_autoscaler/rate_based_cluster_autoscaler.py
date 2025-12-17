import math
import time
from collections import defaultdict
from logging import getLogger
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Set

import ray
from ray.anyscale.data._internal.cluster_autoscaler.cluster_limits_aware import (
    clamp_resource_limits,
)
from ray.data._internal.cluster_autoscaler.base_cluster_autoscaler import (
    ClusterAutoscaler,
)
from ray.data._internal.cluster_autoscaler.resource_utilization_gauge import (
    ResourceUtilizationGauge,
    RollingLogicalUtilizationGauge,
)
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
from ray.data._internal.execution.interfaces import ExecutionOptions, PhysicalOperator
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
        self._resources = ExecutionResources.from_resource_dict(resource_dict)

    def to_bundle(self, include_obj_store: bool) -> Dict[str, float]:
        resources = self._resources
        if not include_obj_store:
            resources = resources.copy(object_store_memory=0)
        return {k: v for k, v in resources.to_resource_dict().items() if v > 0}

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
    # Default scaling max delta (# of nodes) for cluster autoscaling.
    # 32 was chosen because it's not too low so that scaling by 2 is worth it
    # for smaller clusters and not too high to prevent scaling too many nodes
    # at a time. In english, this means "no more than 32 nodes can be provisioned
    # of a node type at a time"
    DEFAULT_CLUSTER_SCALING_UP_MAX_DELTA: float = env_float(
        "RAY_DATA_DEFAULT_CLUSTER_SCALING_UP_MAX_DELTA", 32.0
    )
    # Min number of seconds between two autoscaling requests.
    MIN_GAP_BETWEEN_AUTOSCALING_REQUESTS_S = env_integer(
        "RAY_DATA_MIN_GAP_BETWEEN_AUTOSCALING_REQUESTS", 10
    )
    # The time in seconds after which an autoscaling request will expire.
    AUTOSCALING_REQUEST_EXPIRE_TIME_S = 180

    # Default time window in seconds to calculate the average of cluster utilization.
    DEFAULT_CLUSTER_UTIL_AVG_WINDOW_S: int = 10

    # TODO(Justin): Make this env variables so that users can override
    # Default cluster utilization threshold to trigger scaling up.
    DEFAULT_CLUSTER_SCALING_UP_UTIL_THRESHOLD: float = 0.75

    # TODO(Justin): Make this env variables so that users can override
    # Default cluster GPU utilization threshold to trigger scaling up.
    DEFAULT_CLUSTER_GPU_SCALING_UP_UTIL_THRESHOLD: float = 0.75

    def __init__(
        self,
        ops: List[SupportsClusterAutoscaling],
        resource_manager: "ResourceManager",
        execution_id: str,
        bottleneck_detector: BottleneckDetector,
        *,
        max_cluster_limits: ExecutionResources,
        utility_calculator: ResourceUtilizationGauge,
        autoscaling_coordinator: Optional["AutoscalingCoordinator"] = None,
        get_node_counts: Callable[[], Dict[NodeType, int]] = _get_node_types_and_counts,
        cluster_scaling_up_util_threshold: float = DEFAULT_CLUSTER_SCALING_UP_UTIL_THRESHOLD,  # noqa: E501
        cluster_scaling_up_gpu_threshold: float = DEFAULT_CLUSTER_GPU_SCALING_UP_UTIL_THRESHOLD,  # noqa: E501
        cluster_scaling_up_factor: float = DEFAULT_CLUSTER_SCALING_UP_FACTOR,
        cluster_scaling_up_max_delta: float = DEFAULT_CLUSTER_SCALING_UP_MAX_DELTA,
        min_gap_between_autoscaling_requests_s: int = MIN_GAP_BETWEEN_AUTOSCALING_REQUESTS_S,
        autoscaling_request_expire_time_s: int = AUTOSCALING_REQUEST_EXPIRE_TIME_S,
    ):
        """Initialize the cluster autoscaler.

        Args:
            ops: The operators to autoscale.
            resource_manager: The resource manager.
            execution_id: The execution ID of the dataset. This is used to identify the
                dataset when requesting resources.
            bottleneck_detector: The detector to identify the bottleneck operator.
            max_cluster_limits: Maximum cluster resource limits. Used to clamp resource
                requests to ensure we don't exceed the maximum cluster capacity.
            utility_calculator: The calculator to track and compute cluster resource
                utilization (CPU, GPU, object store memory). Used to determine if cluster
                utilization is high enough to trigger scaling up.
            autoscaling_coordinator: The `AutoscalingCoordinator` to request resources
                from. This is exposed as a seam for testing. If not provided, this uses
                the default coordinator.
            get_node_counts: A function to get the number of nodes of each type. This
                is exposed as a seam for testing.
            cluster_scaling_up_util_threshold: The cluster utilization threshold that
                must be exceeded before scaling up. If average CPU or memory utilization
                is below this threshold, the autoscaler will not scale up even if there
                is a bottleneck. Defaults to 0.75 (75%).
            cluster_scaling_up_gpu_threshold: The GPU cluster utilization threshold that
                must be exceeded before scaling up. If average GPU utilization
                is below this threshold, the autoscaler will not scale up even if there
                is a bottleneck. Defaults to 0.75 (75%).
            cluster_scaling_up_factor: The factor to scale up the cluster.
            cluster_scaling_up_max_delta: Maximum absolute increase in number of nodes
                when scaling up. By default, because we scale the number of nodes by 2 every time,
                the cluster size can experience unbounded growth, which is bad for cost savings
                and resource management. By limiting the # of nodes added to the cluster, the resource
                manager has time to accordingly adjust to the new cluster size.
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
        self._resource_manager = resource_manager
        self._bottleneck_detector = bottleneck_detector
        self._max_cluster_limits = max_cluster_limits
        self._utility_calculator = utility_calculator
        self._autoscaling_coordinator = autoscaling_coordinator
        self._get_node_counts = get_node_counts
        self._cluster_scaling_up_factor = cluster_scaling_up_factor
        self._cluster_scaling_up_max_delta = cluster_scaling_up_max_delta
        self._min_gap_between_autoscaling_requests = (
            min_gap_between_autoscaling_requests_s
        )
        self._autoscaling_request_expire_time_s = autoscaling_request_expire_time_s
        self._cluster_scaling_up_util_threshold = cluster_scaling_up_util_threshold
        self._cluster_scaling_up_gpu_util_threshold = cluster_scaling_up_gpu_threshold
        self._last_request_time = 0
        self._requester_id = f"data-{execution_id}"

        # Send an empty request to register ourselves as soon as possible,
        # so the first `get_total_resources` call can get the allocated resources.
        self._send_resource_request([])

    @classmethod
    def create(
        cls,
        topology: "Topology",
        execution_options: ExecutionOptions,
        resource_manager: "ResourceManager",
        *,
        execution_id: str,
    ) -> "RateBasedClusterAutoscaler":
        """Create a cluster autoscaler.

        This logic is defined here to minimize the risk of merge conflicts in the
        streaming executor, and keep the `ray.data._internal.cluster_autoscaler`
        `__init__` file small.
        """
        # `SupportsClusterAutoscaling` defines the subset of `PhysicalOperator` methods
        # required for this implementation. We use a protocol rather than
        # `PhyiscalOperator`s directly because the  `PhysicalOperator` interface is
        # wide and hard to explicitly stub for testing.
        assert all(isinstance(op, SupportsClusterAutoscaling) for op in topology)

        scalable_ops = [op for op in topology if cls._is_eligible_for_scaling(op)]
        requester_id = f"data-{execution_id}"
        bottleneck_detector = NormalizedThroughputBottleneckDetector(
            resource_manager, requester_id
        )
        # This is the amount of resources we can only scale up to.
        max_cluster_limits = execution_options.max_cluster_limits()
        return RateBasedClusterAutoscaler(
            scalable_ops,
            execution_id=execution_id,
            resource_manager=resource_manager,
            bottleneck_detector=bottleneck_detector,
            max_cluster_limits=max_cluster_limits,
            utility_calculator=RollingLogicalUtilizationGauge(resource_manager),
        )

    @classmethod
    def _is_eligible_for_scaling(cls, op: PhysicalOperator) -> bool:
        """Returns true if the operator is eligible for cluster autoscaling."""
        return (
            # There's no point in autoscaling if the operator doesn't require any
            # resources.
            op.min_scheduling_resources() != ExecutionResources.zero()
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

        # 1. update and report cluster utilization based on usages / limits
        self._utility_calculator.observe()

        # 2. Identify the bottleneck operator.
        bottleneck_op = self._bottleneck_detector.get_bottleneck(self._ops)

        # 3. Send an empty request if we couldn't identify a bottleneck. This is
        # necessary to renew our registration with the autoscaling coordinator and
        # ensure we request the remaining resources for operators that don't support
        # autoscaling.
        if not bottleneck_op:
            logger.debug(
                "Bottleneck Operator not identified yet -- skipping cluster autoscaling."
            )
            self._send_resource_request([])
            return []

        min_scheduling_resources = bottleneck_op.min_scheduling_resources()
        logger.debug(
            f"Bottleneck operator: {bottleneck_op} requires {min_scheduling_resources} per task/actor"
        )

        # 4. Calculate the average utilization across CPU, GPU and Object Store
        util = self._utility_calculator.get()

        logger.debug(
            f"Average cluster util: (cpu={util.cpu}, "
            f"gpu={util.gpu}, "
            f"object_store={util.object_store_memory}) "
            f"threshold={self._cluster_scaling_up_util_threshold}"
            f"gpu_threshold={self._cluster_scaling_up_gpu_util_threshold}"
        )

        gpu_util_high = util.gpu >= self._cluster_scaling_up_gpu_util_threshold
        cpu_util_high = util.cpu >= self._cluster_scaling_up_util_threshold
        obj_store_util_high = (
            util.object_store_memory >= self._cluster_scaling_up_util_threshold
        )

        if not gpu_util_high and not cpu_util_high and not obj_store_util_high:
            # We need utilization to high enough for GPU, CPU, or Object Store
            logger.debug("Cluster utilization is low -- skipping cluster autoscaling.")
            self._send_resource_request([])
            return []

        # 5. Check bottleneck operator is at scheduling capacity.
        op_usage = self._resource_manager.get_op_usage(bottleneck_op)
        max_op_limits = bottleneck_op.min_max_resource_requirements()[1]
        op_usage_with_additional_task = op_usage.add(min_scheduling_resources)
        under_op_max_limit = op_usage_with_additional_task.satisfies_limit(
            max_op_limits
        )
        if not under_op_max_limit:
            # If the operator were to schedule an additional task, and it goes above the
            # max op resource limits, then a larger cluster configuration would not benefit
            # the bottlenecked operator because it's already at its max limits.
            logger.debug(
                f"{bottleneck_op} at max resource usage: {op_usage}, "
                f"-- skipping cluster autoscaling."
            )
            self._send_resource_request([])
            return []

        # 6. Find the nodes that will benefit the bottlenecked operator.
        needed_node_types = self._find_needed_node_types(op=bottleneck_op)

        if not needed_node_types:
            logger.warning(f"No existing node types can schedule {bottleneck_op}.")
            # Still send an empty request when upscaling is not needed, to renew our
            # registration on AutoscalingCoordinator.
            self._send_resource_request([])
            return []

        node_type_request = self._create_node_type_request(needed_node_types)
        # 6. Clamp resources to respect user resource_limits.
        resource_request = clamp_resource_limits(
            node_type_request=node_type_request,
            max_cluster_limits=self._max_cluster_limits,
        )

        # 7. Exclude Object Store becaue Autoscaler SDK doesn't work with obj store.
        resource_request = [
            r.to_bundle(include_obj_store=False) for r in resource_request
        ]
        self._send_resource_request(resource_request)

        return [ExecutionResources.from_resource_dict(r) for r in resource_request]

    def _find_needed_node_types(
        self,
        op: SupportsClusterAutoscaling,
    ) -> List[NodeType]:
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

    def _create_node_type_request(
        self, needed_node_types: Set[NodeType]
    ) -> List[NodeType]:
        """Scale up the cluster by requesting resources for the needed node types."""

        node_type_request: List[Dict[str, float]] = []
        debug_msg = "Scaling up cluster. Requesting resources:"
        node_type_counts = self._get_node_counts()
        for node_type, count in node_type_counts.items():

            if node_type in needed_node_types:
                num_to_request = int(
                    min(
                        math.ceil(count * self._cluster_scaling_up_factor),
                        self._cluster_scaling_up_max_delta,
                    )
                )
                debug_msg += f" [{node_type.to_bundle(include_obj_store=True)}: {count} -> {num_to_request}]"
            else:
                num_to_request = count

            node_type_request.extend([node_type] * num_to_request)

        logger.debug(debug_msg)

        # NOTE: We sort the resource request by str to get a deterministic ordering of nodes.
        # This is important for clamping resources to limits.
        node_type_request.sort(key=lambda x: str(x))
        return node_type_request

    def _send_resource_request(self, resource_request: List[Dict[str, float]]):
        logger.debug(f"Sending Resource Request: {resource_request}")
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
