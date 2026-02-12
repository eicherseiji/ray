import logging
import math
import time
from collections import Counter
from typing import TYPE_CHECKING, Dict, List, Optional

from .bottleneck_detector import (
    BottleneckDetector,
    NormalizedThroughputBottleneckDetector,
)
from .supports_cluster_autoscaling import SupportsClusterAutoscaling
from ray._private.ray_constants import env_float, env_integer
from ray.data._internal.cluster_autoscaler import (
    AutoscalingCoordinator,
    DefaultAutoscalingCoordinator,
)
from ray.data._internal.cluster_autoscaler.base_cluster_autoscaler import (
    ClusterAutoscaler,
)
from ray.data._internal.cluster_autoscaler.resource_utilization_gauge import (
    ResourceUtilizationGauge,
    RollingLogicalUtilizationGauge,
)
from ray.data._internal.execution.interfaces import ExecutionOptions, PhysicalOperator
from ray.data._internal.execution.interfaces.execution_options import ExecutionResources
from ray.data._internal.execution.operators.base_physical_operator import (
    AllToAllOperator,
)
from ray.data._internal.execution.operators.hash_shuffle import (
    HashShufflingOperatorBase,
)
from ray.data._internal.util import get_max_task_capacity

if TYPE_CHECKING:
    from ray.data._internal.execution.resource_manager import ResourceManager
    from ray.data._internal.execution.streaming_executor_state import Topology


logger = logging.getLogger(__name__)


def _to_resource_bundle(resources: ExecutionResources) -> Dict[str, float]:
    """Convert ExecutionResources to a resource bundle dict for the autoscaler.

    Excludes object_store_memory and filters out zero values.
    """
    resource_dict = resources.copy(object_store_memory=0).to_resource_dict()
    return {k: v for k, v in resource_dict.items() if v > 0}


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
    DEFAULT_CLUSTER_SCALING_UP_MAX_CPU_RESOURCE_DELTA: float = env_float(
        "RAY_DATA_DEFAULT_CLUSTER_SCALING_UP_MAX_CPU_RESOURCE_DELTA", 256.0
    )
    DEFAULT_CLUSTER_SCALING_UP_MAX_GPU_RESOURCE_DELTA: float = env_float(
        "RAY_DATA_DEFAULT_CLUSTER_SCALING_UP_MAX_GPU_RESOURCE_DELTA", 32.0
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
        topology: "Topology",
        bottleneck_detector: BottleneckDetector,
        *,
        max_cluster_limits: ExecutionResources,
        utility_calculator: ResourceUtilizationGauge,
        autoscaling_coordinator: Optional["AutoscalingCoordinator"] = None,
        cluster_scaling_up_util_threshold: float = DEFAULT_CLUSTER_SCALING_UP_UTIL_THRESHOLD,  # noqa: E501
        cluster_scaling_up_gpu_threshold: float = DEFAULT_CLUSTER_GPU_SCALING_UP_UTIL_THRESHOLD,  # noqa: E501
        cluster_scaling_up_factor: float = DEFAULT_CLUSTER_SCALING_UP_FACTOR,
        cluster_scaling_up_max_cpu_resource_delta: float = DEFAULT_CLUSTER_SCALING_UP_MAX_CPU_RESOURCE_DELTA,
        cluster_scaling_up_max_gpu_resource_delta: float = DEFAULT_CLUSTER_SCALING_UP_MAX_GPU_RESOURCE_DELTA,
        min_gap_between_autoscaling_requests_s: int = MIN_GAP_BETWEEN_AUTOSCALING_REQUESTS_S,
        autoscaling_request_expire_time_s: int = AUTOSCALING_REQUEST_EXPIRE_TIME_S,
    ):
        """Initialize the cluster autoscaler.

        Args:
            ops: The operators to autoscale.
            resource_manager: The resource manager.
            execution_id: The execution ID of the dataset. This is used to identify the
                dataset when requesting resources.
            topology: The topology of the operators.
            bottleneck_detector: The detector to identify the bottleneck operator.
            max_cluster_limits: Maximum cluster resource limits. Used to clamp resource
                requests to ensure we don't exceed the maximum cluster capacity.
            utility_calculator: The calculator to track and compute cluster resource
                utilization (CPU, GPU, object store memory). Used to determine if cluster
                utilization is high enough to trigger scaling up.
            autoscaling_coordinator: The `AutoscalingCoordinator` to request resources
                from. This is exposed as a seam for testing. If not provided, this uses
                the default coordinator.
            cluster_scaling_up_util_threshold: The cluster utilization threshold that
                must be exceeded before scaling up. If average CPU or memory utilization
                is below this threshold, the autoscaler will not scale up even if there
                is a bottleneck. Defaults to 0.75 (75%).
            cluster_scaling_up_gpu_threshold: The GPU cluster utilization threshold that
                must be exceeded before scaling up. If average GPU utilization
                is below this threshold, the autoscaler will not scale up even if there
                is a bottleneck. Defaults to 0.75 (75%).
            cluster_scaling_up_factor: The factor to scale up the cluster.
            cluster_scaling_up_max_cpu_resource_delta: Maximum absolute increase in CPU resource
                when scaling up.
            cluster_scaling_up_max_gpu_resource_delta: Maximum absolute increase in GPU resource
                when scaling up.
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
        self._topology = topology
        self._resource_manager = resource_manager
        self._bottleneck_detector = bottleneck_detector
        self._max_cluster_limits = max_cluster_limits
        self._utility_calculator = utility_calculator
        self._autoscaling_coordinator = autoscaling_coordinator
        self._cluster_scaling_up_factor = cluster_scaling_up_factor
        self._cluster_scaling_up_max_cpu_resource_delta = (
            cluster_scaling_up_max_cpu_resource_delta
        )
        self._cluster_scaling_up_max_gpu_resource_delta = (
            cluster_scaling_up_max_gpu_resource_delta
        )
        self._min_gap_between_autoscaling_requests = (
            min_gap_between_autoscaling_requests_s
        )
        self._autoscaling_request_expire_time_s = autoscaling_request_expire_time_s
        self._cluster_scaling_up_util_threshold = cluster_scaling_up_util_threshold
        self._cluster_scaling_up_gpu_util_threshold = cluster_scaling_up_gpu_threshold
        self._last_request_time = 0
        self._requester_id = f"data-{execution_id}"
        self._last_resource_request = []

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
        # required for this implementation. We depend on a protocol rather than
        # `PhyiscalOperator` directly because the `PhysicalOperator` interface is
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
            topology=topology,
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

    def _get_allocated_resource_bundles(
        self, topology: "Topology"
    ) -> List[Dict[str, float]]:
        """Get the current cluster resources as a list of resource bundles.

        This iterates over all operators in the topology (including those not eligible
        for autoscaling) and calculates their current resource allocation in terms
        of bundles.
        """
        resource_request = []

        for op in topology:
            # Skip operators that don't require resources (e.g. InputDataBuffer)
            min_resources = op.min_scheduling_resources()
            if min_resources == ExecutionResources.zero():
                continue

            # 1. Determine current bundle count for this op
            # Try to get the specific allocation target from the allocator
            allocation = self._resource_manager.op_resource_allocator.get_allocation(op)

            # If the allocator doesn't track this op (e.g. Shuffle), fallback to current usage
            if allocation is None:
                allocation = self._resource_manager.get_op_usage(op)

            # Convert allocation (Total Resources) -> Bundle Count
            current_bundles = get_max_task_capacity(allocation, min_resources)

            if current_bundles > 0:
                bundle = _to_resource_bundle(min_resources)
                resource_request.extend([bundle] * current_bundles)

        return resource_request

    def try_trigger_scaling(self):
        # If there are no operators to scale, send the previous resource request
        # to renew our registration with the autoscaling coordinator.
        if not self._ops:
            logger.debug("No operators to scale -- skipping cluster autoscaling.")
            self._send_resource_request(self._last_resource_request)
            return self._last_resource_request
        # Limit the frequency of autoscaling requests.
        now = time.time()
        if now - self._last_request_time < self._min_gap_between_autoscaling_requests:
            return

        # 1. update and report cluster utilization based on usages / limits
        self._utility_calculator.observe()

        # 2. Identify the bottleneck operator.
        bottleneck_op = self._bottleneck_detector.get_bottleneck(self._ops)

        if not bottleneck_op:
            logger.debug(
                "Bottleneck Operator not identified yet -- skipping cluster autoscaling."
            )
            self._send_resource_request(self._last_resource_request)
            return self._last_resource_request

        min_scheduling_resources = bottleneck_op.min_scheduling_resources()
        logger.debug(
            f"Bottleneck operator: {bottleneck_op} requires {min_scheduling_resources} per task/actor"
        )

        # 3. Calculate the average utilization across CPU, GPU and Object Store
        util = self._utility_calculator.get()

        logger.debug(
            f"Average cluster util: (cpu={util.cpu}, "
            f"gpu={util.gpu}, "
            f"object_store={util.object_store_memory}), "
            f"threshold={self._cluster_scaling_up_util_threshold}, "
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
            self._send_resource_request(self._last_resource_request)
            return self._last_resource_request

        # 4. Check bottleneck operator is at scheduling capacity.
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
            self._send_resource_request(self._last_resource_request)
            return self._last_resource_request

        # 5. Calculate the additional task num needed to scale up.
        allocated_resources = (
            self._resource_manager.op_resource_allocator.get_allocation(bottleneck_op)
        )
        maximum_task_capacity = get_max_task_capacity(
            allocated_resources, min_scheduling_resources
        )
        target_task_num = math.ceil(
            maximum_task_capacity * self._cluster_scaling_up_factor
        )
        additional_task_num = target_task_num - maximum_task_capacity

        # 6. Cap the additional bundles based on max resource deltas
        max_additional_by_cpu = (
            math.floor(
                self._cluster_scaling_up_max_cpu_resource_delta
                / min_scheduling_resources.cpu
            )
            if min_scheduling_resources.cpu > 0
            else float("inf")
        )
        max_additional_by_gpu = (
            math.floor(
                self._cluster_scaling_up_max_gpu_resource_delta
                / min_scheduling_resources.gpu
            )
            if min_scheduling_resources.gpu > 0
            else float("inf")
        )
        capped_additional = int(
            min(additional_task_num, max_additional_by_cpu, max_additional_by_gpu)
        )
        # Ensure at least 1 additional bundle if we have a bottleneck
        capped_additional = max(capped_additional, 1)

        # 7. Calculate total bundle count (current + capped additional)
        # AutoscalingCoordinator expects total resources, not incremental.
        # We need to request resources for ALL operators, not just the bottleneck one.
        resource_request = self._get_allocated_resource_bundles(self._topology)

        # Add the additional bundles for the bottleneck operator
        resource_bundle = _to_resource_bundle(min_scheduling_resources)

        if capped_additional > 0:
            resource_request.extend([resource_bundle] * capped_additional)

        total_bundle_count = maximum_task_capacity + capped_additional

        logger.debug(
            f"Scaling bottleneck {bottleneck_op}: maximum_task_capacity={maximum_task_capacity}, "
            f"additional_tasks={capped_additional}, total_bundle_count={total_bundle_count}, "
            f"resource_bundle={resource_bundle}"
        )

        self._log_resource_request(resource_request)
        self._send_resource_request(resource_request)

        return resource_request

    @staticmethod
    def _log_resource_request(resource_request: List[Dict[str, float]]) -> None:
        """Log the resource request with identical bundles grouped.

        This method is static so it's easier to unit test.
        """
        if not logger.isEnabledFor(logging.DEBUG):
            return

        hashable_bundles = [
            tuple(sorted(bundle.items())) for bundle in resource_request
        ]
        hashable_bundle_counts = Counter(hashable_bundles)
        formatted_bundles = [
            f"[{dict(bundle)}] * {count}"
            for bundle, count in hashable_bundle_counts.items()
        ]
        logger.debug(f"Sending resource request: {', '.join(formatted_bundles)}")

    def _send_resource_request(self, resource_request: List[Dict[str, float]]):
        self._last_resource_request = [r.copy() for r in resource_request]
        self._autoscaling_coordinator.request_resources(
            requester_id=self._requester_id,
            resources=[r.copy() for r in resource_request],
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
