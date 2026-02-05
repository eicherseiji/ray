from dataclasses import dataclass, field
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from ray.anyscale.data._internal.cluster_autoscaler import (
    BottleneckDetector,
    ClusterAutoscalingMetrics,
    NormalizedThroughputBottleneckDetector,
    RateBasedClusterAutoscaler,
    SupportsClusterAutoscaling,
)
from ray.anyscale.data._internal.cluster_autoscaler.rate_based_cluster_autoscaler import (
    _to_resource_bundle,
)
from ray.data._internal.cluster_autoscaler import (
    DefaultClusterAutoscaler,
    DefaultClusterAutoscalerV2,
    create_cluster_autoscaler,
)
from ray.data._internal.cluster_autoscaler.fake_autoscaling_coordinator import (
    FakeAutoscalingCoordinator,
)
from ray.data._internal.cluster_autoscaler.resource_utilization_gauge import (
    ResourceUtilizationGauge,
)
from ray.data._internal.execution.interfaces import PhysicalOperator
from ray.data._internal.execution.interfaces.execution_options import (
    ExecutionOptions,
    ExecutionResources,
)
from ray.data._internal.execution.resource_manager import ResourceManager
from ray.data.context import DataContext
from ray.data.tests.conftest import propagate_logs  # noqa


class StubBottleneckDetector(BottleneckDetector):
    """A stub implementation for testing."""

    def __init__(self, bottleneck: Optional[SupportsClusterAutoscaling] = None):
        self._bottleneck = bottleneck

    def get_bottleneck(
        self, ops: List[SupportsClusterAutoscaling]
    ) -> Optional[SupportsClusterAutoscaling]:
        return self._bottleneck


class StubUtilizationGauge(ResourceUtilizationGauge):
    def __init__(self, utilization: Optional[ExecutionResources] = None):
        if utilization is None:
            utilization = ExecutionResources(
                cpu=1, gpu=1, object_store_memory=1, memory=1
            )
        self._utilization = utilization

    def observe(self):
        pass

    def get(self):
        return self._utilization


@dataclass(frozen=True)
class StubClusterAutoscalingMetrics(ClusterAutoscalingMetrics):
    """A stub `OpRuntimeMetrics` implementation for testing."""

    average_num_inputs_per_task: Optional[float] = None
    average_num_outputs_per_task: Optional[float] = None
    num_output_blocks_per_task_s: Optional[float] = None


@dataclass(eq=False)
class StubClusterAutoscalingOperator(SupportsClusterAutoscaling):
    """A stub implementation for testing."""

    # These fields define the attributes required by the interface.
    metrics: StubClusterAutoscalingMetrics = field(
        default_factory=StubClusterAutoscalingMetrics
    )
    output_dependencies: List[PhysicalOperator] = field(default_factory=list)

    # These fields define return values for the methods required by the interface.
    _per_task_resource_allocation: ExecutionResources = ExecutionResources.zero()
    _get_max_concurrency_limit: Optional[int] = None
    _min_scheduling_resources: ExecutionResources = ExecutionResources.zero()
    _completed: bool = False
    _min_resource_requirements: ExecutionResources = field(
        default_factory=lambda: ExecutionResources.zero()
    )
    _max_resource_requirements: ExecutionResources = field(
        default_factory=lambda: ExecutionResources.for_limits()
    )

    def per_task_resource_allocation(self) -> ExecutionResources:
        return self._per_task_resource_allocation

    def get_max_concurrency_limit(self) -> Optional[int]:
        return self._get_max_concurrency_limit

    def min_max_resource_requirements(self):
        return (self._min_resource_requirements, self._max_resource_requirements)

    def min_scheduling_resources(self) -> ExecutionResources:
        return self._min_scheduling_resources

    def has_completed(self) -> bool:
        return self._completed


def test_autoscaler_requests_resources_if_no_scalable_ops():
    """Test the autoscaler requests resources even if no ops support cluster
    autoscaling.

    Some operators don't support cluster autoscaling. If a DAG only contains these
    operators, the autoscaler should still request the remaining resources. Otherwise,
    the operators won't get any resources and the pipeline won't run.
    """
    time = 0
    resource_manager = StubResourceManager()
    autoscaler = RateBasedClusterAutoscaler(
        ops=[],
        resource_manager=resource_manager,
        execution_id="test",
        topology=resource_manager._topology,
        bottleneck_detector=StubBottleneckDetector(bottleneck=None),
        max_cluster_limits=ExecutionResources.for_limits(),
        utility_calculator=StubUtilizationGauge(),
        autoscaling_coordinator=FakeAutoscalingCoordinator(
            get_time=lambda: time, remaining=[{"CPU": 1}]
        ),
        min_gap_between_autoscaling_requests_s=0,
        autoscaling_request_expire_time_s=1,
    )

    # The autoscaler should immediately request the remaining resources.
    assert autoscaler.get_total_resources() == ExecutionResources(cpu=1)

    # After the specified `autoscaling_request_expire_time_s` has passed, the autoscaler
    # shouldn't get any resources.
    time += 2
    assert autoscaler.get_total_resources() == ExecutionResources()

    # Calling `try_trigger_scaling` should re-request the remaining resources, even if
    # there aren't any scalable ops.
    autoscaler.try_trigger_scaling()
    assert autoscaler.get_total_resources() == ExecutionResources(cpu=1)


class StubOpResourceAllocator:
    """Stub for the op_resource_allocator used by RateBasedClusterAutoscaler."""

    def __init__(
        self, allocations: Dict[SupportsClusterAutoscaling, ExecutionResources]
    ):
        self._allocations = allocations

    def get_allocation(
        self, op: SupportsClusterAutoscaling
    ) -> Optional[ExecutionResources]:
        return self._allocations.get(op)


class StubResourceManager:
    def __init__(
        self,
        global_limits: ExecutionResources = None,
        op_usage: Dict[PhysicalOperator, ExecutionResources] = None,
        op_allocations: Dict[SupportsClusterAutoscaling, ExecutionResources] = None,
        topology: Dict[PhysicalOperator, any] = None,
    ):
        if global_limits is None:
            global_limits = ExecutionResources.for_limits()
        if op_usage is None:
            op_usage = {}
        if op_allocations is None:
            op_allocations = {}
        if topology is None:
            # If topology not provided, infer from op_allocations keys as a simple list
            # The real topology is a dict, but RateBasedClusterAutoscaler iterates over it
            # so a list or dict keys view works fine for iteration.
            # Using dict keys to mimic topology iteration.
            topology = list(op_allocations.keys())

        self.global_limits = global_limits
        self._op_usage = op_usage
        self.op_resource_allocator = StubOpResourceAllocator(op_allocations)
        self._topology = topology

    def get_global_limits(self) -> ExecutionResources:
        return self.global_limits

    def get_op_usage(self, op: PhysicalOperator) -> ExecutionResources:
        return self._op_usage.get(op, ExecutionResources.zero())


class TestNormalizedThroughputBottleneckDetector:
    def test_get_bottleneck_completed_operator(self):
        op = StubClusterAutoscalingOperator(_completed=True)
        detector = NormalizedThroughputBottleneckDetector(
            StubResourceManager(), "test-requester"
        )

        bottleneck = detector.get_bottleneck([op])

        assert bottleneck is None

    def test_get_bottleneck_inf_global_limits(self):
        op = StubClusterAutoscalingOperator()
        resource_manager = StubResourceManager(
            global_limits=ExecutionResources.for_limits()
        )
        detector = NormalizedThroughputBottleneckDetector(
            resource_manager, "test-requester"
        )

        bottleneck = detector.get_bottleneck([op])

        assert bottleneck is None

    def test_get_bottleneck_zero_global_limits(self):
        op = StubClusterAutoscalingOperator(
            _per_task_resource_allocation=ExecutionResources(cpu=1),
            metrics=StubClusterAutoscalingMetrics(
                num_output_blocks_per_task_s=1,
            ),
        )
        resource_manager = StubResourceManager(global_limits=ExecutionResources.zero())
        detector = NormalizedThroughputBottleneckDetector(
            resource_manager, "test-requester"
        )

        bottleneck = detector.get_bottleneck([op])

        assert bottleneck is None

    def test_get_bottleneck_single_operator(self):
        op = StubClusterAutoscalingOperator(
            _per_task_resource_allocation=ExecutionResources(cpu=1),
            metrics=StubClusterAutoscalingMetrics(
                num_output_blocks_per_task_s=1,
            ),
        )
        resource_manager = StubResourceManager(global_limits=ExecutionResources(cpu=2))
        detector = NormalizedThroughputBottleneckDetector(
            resource_manager, "test-requester"
        )

        bottleneck = detector.get_bottleneck([op])

        # With a single operator, that operator is the bottleneck.
        assert bottleneck == op

    def test_returns_first_running_operator_when_scores_approximately_equal(self):
        op3 = StubClusterAutoscalingOperator(
            _per_task_resource_allocation=ExecutionResources(cpu=1),
            metrics=StubClusterAutoscalingMetrics(
                num_output_blocks_per_task_s=1,
            ),
        )
        op2 = StubClusterAutoscalingOperator(
            output_dependencies=[op3],
            _per_task_resource_allocation=ExecutionResources(cpu=1),
            metrics=StubClusterAutoscalingMetrics(
                num_output_blocks_per_task_s=1,
            ),
        )
        op1 = StubClusterAutoscalingOperator(
            output_dependencies=[op2],
            _per_task_resource_allocation=ExecutionResources(cpu=1),
            metrics=StubClusterAutoscalingMetrics(
                num_output_blocks_per_task_s=1,
            ),
            _completed=True,
        )
        resource_manager = StubResourceManager(global_limits=ExecutionResources(cpu=2))
        detector = NormalizedThroughputBottleneckDetector(
            resource_manager, "test-requester"
        )

        bottleneck = detector.get_bottleneck([op1, op2, op3])

        # When the scores are approximately equal, it's ambiguous which operator should
        # be the bottleneck. In this case, we expect the detector to return the most
        # upstream operator that is running rather than an arbitrary operator.
        #
        # In this example, op1 has completed, so op2 is the most upstream operator that
        # is running. Therefore, op2 is the bottleneck.
        assert bottleneck == op2

    def test_get_bottleneck_identifies_slower_operator(self):
        # op2 is fast (produces 12 blocks/s per task)
        # Normalization factor = 1.0 (sink)
        # Normalized rate = 12 * 1.0 = 12.0
        op2 = StubClusterAutoscalingOperator(
            _per_task_resource_allocation=ExecutionResources(cpu=1),
            metrics=StubClusterAutoscalingMetrics(
                average_num_inputs_per_task=2,
                average_num_outputs_per_task=6,
                num_output_blocks_per_task_s=12,
            ),
        )
        # op1 is slower (produces 2 blocks/s per task, which become 6 sink outputs/s)
        # Normalization factor = 6/2 = 3.0 (from op2 ratio)
        # Normalized rate = 2 * 3.0 = 6.0
        #
        # Optimal Allocation (Total CPU=2):
        # - Op1 needs 2x more CPU than Op2 to match rates (6.0 vs 12.0).
        # - Allocation: Op1=1.33 CPU, Op2=0.67 CPU.
        #
        # Scores:
        # - Op2: 0.67 tasks * 12.0 = 8.0
        # - Op1 (with limit=1): min(1.33, 1.0) tasks * 6.0 = 6.0
        # Result: 6.0 < 8.0 -> Op1 is bottleneck.
        op1 = StubClusterAutoscalingOperator(
            output_dependencies=[op2],
            _per_task_resource_allocation=ExecutionResources(cpu=1),
            metrics=StubClusterAutoscalingMetrics(
                num_output_blocks_per_task_s=2,
            ),
            # Set a concurrency limit so op1 is a real bottleneck; otherwise
            # 1.33 * 6.0 ≈ 8 and the scores are nearly equal, so no bottleneck
            # would be detected.
            _get_max_concurrency_limit=1,
        )
        resource_manager = StubResourceManager(global_limits=ExecutionResources(cpu=2))
        detector = NormalizedThroughputBottleneckDetector(
            resource_manager, "test-requester"
        )

        bottleneck = detector.get_bottleneck([op1, op2])

        # op1 is the bottleneck because it's slower
        assert bottleneck == op1


@patch(
    "ray.data._internal.cluster_autoscaler.DEFAULT_CLUSTER_AUTOSCALER_VERSION",
    "invalid",
)
def test_invalid_cluster_autoscaler_env_value_raises_value_error():
    with pytest.raises(ValueError):
        create_cluster_autoscaler(
            topology={},
            data_context=DataContext(execution_options=ExecutionOptions()),
            resource_manager=MagicMock(spec=ResourceManager),
            execution_id="test",
        )


@pytest.mark.parametrize(
    "cluster_autoscaler_env_value, expected_autoscaler_type",
    [
        ("RAYTURBO", RateBasedClusterAutoscaler),
        ("V2", DefaultClusterAutoscalerV2),
        ("V1", DefaultClusterAutoscaler),
    ],
)
def test_cluster_autoscaler_env_value_creates_correct_autoscaler(
    cluster_autoscaler_env_value, expected_autoscaler_type
):
    with patch(
        "ray.data._internal.cluster_autoscaler.DEFAULT_CLUSTER_AUTOSCALER_VERSION",
        cluster_autoscaler_env_value,
    ):
        autoscaler = create_cluster_autoscaler(
            topology={},
            data_context=DataContext(execution_options=ExecutionOptions()),
            resource_manager=MagicMock(spec=ResourceManager),
            execution_id="test",
        )

        assert isinstance(autoscaler, expected_autoscaler_type)


@pytest.mark.parametrize("cpu_usage", [0.25, 0.9])
@pytest.mark.parametrize("gpu_usage", [0.25, 0.9])
def test_autoscaler_utilization_threshold(cpu_usage, gpu_usage):
    """Test autoscaler scaling behavior based on cluster utilization thresholds.

    Tests all combinations of cpu and gpu utilization values.
    The autoscaler should scale up if CPU or GPU utilization exceeds the 0.75 threshold.
    """
    threshold = 0.75

    cpu_op = StubClusterAutoscalingOperator(
        _min_scheduling_resources=ExecutionResources(cpu=1),
    )

    execution_resources = ExecutionResources(cpu=cpu_usage, gpu=gpu_usage)

    # Allocate some resources to the operator so it can calculate current task num
    resource_manager = StubResourceManager(
        op_allocations={cpu_op: ExecutionResources(cpu=4)}
    )
    autoscaler = RateBasedClusterAutoscaler(
        ops=[cpu_op],
        resource_manager=resource_manager,
        execution_id="test",
        topology=resource_manager._topology,
        bottleneck_detector=StubBottleneckDetector(bottleneck=cpu_op),
        max_cluster_limits=ExecutionResources.for_limits(),
        utility_calculator=StubUtilizationGauge(execution_resources),
        autoscaling_coordinator=FakeAutoscalingCoordinator(),
        min_gap_between_autoscaling_requests_s=0,
        cluster_scaling_up_util_threshold=threshold,  # 75% threshold
        cluster_scaling_up_gpu_threshold=threshold,  # 75% threshold for GPU
    )

    result = autoscaler.try_trigger_scaling()

    over_threshold = cpu_usage >= threshold or gpu_usage >= threshold
    if over_threshold:
        # Should return non-empty list of resource bundles
        assert result is not None and len(result) > 0
    else:
        # Should return empty list when under threshold
        assert result == []


@pytest.mark.parametrize(
    "min_scheduling_resources,current_allocation,max_cpu_delta,max_gpu_delta,expected_total_bundle_count",
    [
        # CPU-only operator: 4 tasks allocated, scaling factor 2x = 4 additional tasks
        # Max CPU delta 256 / 1 CPU per task = 256 bundles allowed, so no capping
        # Total = current (4) + additional (4) = 8
        (
            ExecutionResources(cpu=1),
            ExecutionResources(cpu=4),
            256.0,
            32.0,
            8,  # 4 current + 4 additional
        ),
        # CPU-only operator: 100 tasks allocated, scaling factor 2x = 100 additional
        # Max CPU delta 50 / 1 CPU per task = 50 bundles allowed (capped)
        # Total = current (100) + additional (50 capped) = 150
        (
            ExecutionResources(cpu=1),
            ExecutionResources(cpu=100),
            50.0,
            32.0,
            150,  # 100 current + 50 additional (capped by max_cpu_delta)
        ),
        # GPU operator: 8 GPUs allocated, 2 GPU per task = 4 tasks
        # scaling factor 2x = 4 additional tasks
        # Max GPU delta 32 / 2 GPU per task = 16 bundles allowed, so no capping
        # Total = current (4) + additional (4) = 8
        (
            ExecutionResources(gpu=2),
            ExecutionResources(gpu=8),
            256.0,
            32.0,
            8,  # 4 current + 4 additional
        ),
        # GPU operator: Max GPU delta 4 / 2 GPU per task = 2 bundles allowed (capped)
        # Total = current (4) + additional (2 capped) = 6
        (
            ExecutionResources(gpu=2),
            ExecutionResources(gpu=8),
            256.0,
            4.0,
            6,  # 4 current + 2 additional (capped by max_gpu_delta)
        ),
        # Mixed CPU+GPU operator: GPU is the limiting factor for both task count and delta
        # Total = current (4) + additional (2 capped) = 6
        (
            ExecutionResources(cpu=4, gpu=1),
            ExecutionResources(cpu=16, gpu=4),  # 4 tasks based on GPU
            256.0,
            2.0,  # Allows only 2 additional bundles
            6,  # 4 current + 2 additional (capped by GPU delta)
        ),
    ],
)
def test_autoscaler_requests_correct_bundle_count(
    min_scheduling_resources: ExecutionResources,
    current_allocation: ExecutionResources,
    max_cpu_delta: float,
    max_gpu_delta: float,
    expected_total_bundle_count: int,
):
    """Test that autoscaler requests total bundles (current + capped additional) and respects delta caps."""
    op = StubClusterAutoscalingOperator(
        _min_scheduling_resources=min_scheduling_resources,
    )
    resource_manager = StubResourceManager(
        op_allocations={op: current_allocation},
    )
    autoscaler = RateBasedClusterAutoscaler(
        ops=[op],
        resource_manager=resource_manager,
        execution_id="test",
        topology=resource_manager._topology,
        bottleneck_detector=StubBottleneckDetector(bottleneck=op),
        max_cluster_limits=ExecutionResources.for_limits(),
        utility_calculator=StubUtilizationGauge(),
        autoscaling_coordinator=FakeAutoscalingCoordinator(),
        min_gap_between_autoscaling_requests_s=0,
        cluster_scaling_up_max_cpu_resource_delta=max_cpu_delta,
        cluster_scaling_up_max_gpu_resource_delta=max_gpu_delta,
    )

    result = autoscaler.try_trigger_scaling()

    assert len(result) == expected_total_bundle_count
    # Each bundle should match the min_scheduling_resources (excluding object_store_memory and zeros)
    expected_bundle = _to_resource_bundle(min_scheduling_resources)
    for bundle in result:
        assert bundle == expected_bundle

    # Trigger scaling with low utilization. The cluster autoscaler should re-request the previous resources.
    autoscaler._utility_calculator = StubUtilizationGauge(ExecutionResources(cpu=0.1))
    requested_resources_low_util = autoscaler.try_trigger_scaling()
    assert requested_resources_low_util == result


@pytest.mark.parametrize(
    "max_resource_requirements,current_op_usage,min_scheduling_resources,should_scale",
    [
        # Case 1: Current usage (4) + min_scheduling (1) = 5 > max (4) -> don't scale
        (
            ExecutionResources(cpu=4),
            ExecutionResources(cpu=4),
            ExecutionResources(cpu=1),
            False,
        ),
        # Case 2: Current usage (3) + min_scheduling (1) = 4 <= max (4) -> scale
        (
            ExecutionResources(cpu=4),
            ExecutionResources(cpu=3),
            ExecutionResources(cpu=1),
            True,
        ),
        # Case 3: Heterogeneous - CPU at limit (4) but GPU below limit (2)
        # Adding one more task: CPU 4+1=5 > 4 (exceeds), GPU 2+1=3 <= 4 (within limit)
        # Should not scale because CPU exceeds limit
        (
            ExecutionResources(cpu=4, gpu=4),
            ExecutionResources(cpu=4, gpu=2),
            ExecutionResources(cpu=1, gpu=1),
            False,
        ),
    ],
)
def test_autoscaler_skips_scaling_when_at_max_schedulable_tasks(
    max_resource_requirements: ExecutionResources,
    current_op_usage: ExecutionResources,
    min_scheduling_resources: ExecutionResources,
    should_scale: bool,
):
    """Test that autoscaler skips scaling when bottleneck operator would exceed max resource limits."""

    # Set up operator with min_scheduling_resources
    op = StubClusterAutoscalingOperator(
        _min_scheduling_resources=min_scheduling_resources,
        _max_resource_requirements=max_resource_requirements,
    )
    resource_manager = StubResourceManager(
        op_usage={op: current_op_usage},
        op_allocations={
            op: current_op_usage
        },  # Also set allocations for task_num calculation
    )
    autoscaler = RateBasedClusterAutoscaler(
        ops=[op],
        resource_manager=resource_manager,
        execution_id="test",
        topology=resource_manager._topology,
        bottleneck_detector=StubBottleneckDetector(bottleneck=op),
        max_cluster_limits=ExecutionResources.for_limits(),
        utility_calculator=StubUtilizationGauge(
            ExecutionResources(cpu=0.9, gpu=0, object_store_memory=0, memory=0)
        ),
        autoscaling_coordinator=FakeAutoscalingCoordinator(),
        min_gap_between_autoscaling_requests_s=0,
    )

    requested_resources = autoscaler.try_trigger_scaling()

    if should_scale:
        # Should scale - return non-empty list
        assert len(requested_resources) > 0
    else:
        # Should not scale - return empty list
        assert requested_resources == []


def test_autoscaler_requests_at_least_one_bundle_when_no_allocation():
    """Test that autoscaler requests at least 1 bundle even when current allocation is 0."""
    op = StubClusterAutoscalingOperator(
        _min_scheduling_resources=ExecutionResources(cpu=2, gpu=1),
    )
    # No allocations yet
    resource_manager = StubResourceManager(op_allocations={})
    autoscaler = RateBasedClusterAutoscaler(
        ops=[op],
        resource_manager=resource_manager,
        execution_id="test",
        topology=resource_manager._topology,
        bottleneck_detector=StubBottleneckDetector(bottleneck=op),
        max_cluster_limits=ExecutionResources.for_limits(),
        utility_calculator=StubUtilizationGauge(),
        autoscaling_coordinator=FakeAutoscalingCoordinator(),
        min_gap_between_autoscaling_requests_s=0,
    )

    result = autoscaler.try_trigger_scaling()

    # Should still request at least 1 bundle
    assert len(result) >= 1
    # Bundle should have CPU=2, GPU=1 (no object_store_memory or zero values)
    assert result[0] == {"CPU": 2, "GPU": 1}


def test_get_allocated_resource_bundles_aggregates_correctly():
    """Test that _get_allocated_resource_bundles correctly sums up resources from all operators."""
    # Op1: 2 CPUs per task, 3 tasks allocated -> 6 CPUs total
    op1 = StubClusterAutoscalingOperator(
        _min_scheduling_resources=ExecutionResources(cpu=2),
    )
    # Op2: 1 GPU per task, 2 tasks allocated -> 2 GPUs total
    op2 = StubClusterAutoscalingOperator(
        _min_scheduling_resources=ExecutionResources(gpu=1),
    )
    # Op3: Non-scalable op (or just skipped by autoscaler for scaling purposes),
    # but still consumes resources. 1 CPU per task, 4 tasks -> 4 CPUs total.
    op3 = StubClusterAutoscalingOperator(
        _min_scheduling_resources=ExecutionResources(cpu=1),
    )

    op_allocations = {
        op1: ExecutionResources(cpu=6),
        op2: ExecutionResources(gpu=2),
        op3: ExecutionResources(cpu=4),
    }

    resource_manager = StubResourceManager(
        op_allocations=op_allocations,
        # Ensure all ops are in the topology
        topology={op1: None, op2: None, op3: None},
    )

    autoscaler = RateBasedClusterAutoscaler(
        ops=[op1, op2],  # op3 might not be in the scalable ops list
        resource_manager=resource_manager,
        execution_id="test",
        topology=resource_manager._topology,
        bottleneck_detector=StubBottleneckDetector(bottleneck=op1),
        max_cluster_limits=ExecutionResources.for_limits(),
        utility_calculator=StubUtilizationGauge(),
        autoscaling_coordinator=FakeAutoscalingCoordinator(),
        min_gap_between_autoscaling_requests_s=0,
    )

    current_resources = autoscaler._get_allocated_resource_bundles(
        resource_manager._topology
    )

    # Expected:
    # Op1: 3 bundles of {CPU: 2}
    # Op2: 2 bundles of {GPU: 1}
    # Op3: 4 bundles of {CPU: 1}
    # Order doesn't strictly matter for correctness of the sum, but let's check counts.

    cpu2_bundles = [r for r in current_resources if r.get("CPU") == 2]
    gpu1_bundles = [r for r in current_resources if r.get("GPU") == 1]
    cpu1_bundles = [r for r in current_resources if r.get("CPU") == 1]

    assert len(cpu2_bundles) == 3
    assert len(gpu1_bundles) == 2
    assert len(cpu1_bundles) == 4
    assert len(current_resources) == 9


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-sv", __file__]))
