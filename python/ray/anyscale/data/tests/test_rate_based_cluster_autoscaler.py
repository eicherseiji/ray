from collections import defaultdict
from dataclasses import dataclass, field
from typing import DefaultDict, List, Optional, Dict
from unittest.mock import MagicMock

import logging
import pytest

from ray.anyscale.air._internal.autoscaling_coordinator import (
    FakeAutoscalingCoordinator,
)
from ray.anyscale.data._internal.cluster_autoscaler import (
    BottleneckDetector,
    ClusterAutoscalingMetrics,
    LegacyRayTurboClusterAutoscaler,
    NodeType,
    NormalizedThroughputBottleneckDetector,
    RateBasedClusterAutoscaler,
    SupportsClusterAutoscaling,
)
from ray.data._internal.cluster_autoscaler import (
    CLUSTER_AUTOSCALER_ENV_KEY,
    DefaultClusterAutoscaler,
    DefaultClusterAutoscalerV2,
    create_cluster_autoscaler,
)
from ray.data._internal.cluster_autoscaler.resource_utilization_gauge import (
    ResourceUtilizationGauge,
)
from ray.data._internal.execution.interfaces import PhysicalOperator
from ray.data._internal.execution.interfaces.execution_options import ExecutionOptions
from ray.data._internal.execution.interfaces.execution_options import ExecutionResources
from ray.data._internal.execution.resource_manager import ResourceManager
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

    def completed(self) -> bool:
        return self._completed


def test_autoscaler_doubles_nodes_for_bottleneck_op():
    cpu_op = StubClusterAutoscalingOperator(
        _min_scheduling_resources=ExecutionResources(cpu=1),
    )
    gpu_op = StubClusterAutoscalingOperator(
        _min_scheduling_resources=ExecutionResources(gpu=1),
    )
    cpu_node_type, gpu_node_type = NodeType({"CPU": 1}), NodeType({"GPU": 1})
    resource_manager = StubResourceManager()
    autoscaler = RateBasedClusterAutoscaler(
        ops=[cpu_op, gpu_op],
        resource_manager=resource_manager,
        execution_id="test",
        bottleneck_detector=StubBottleneckDetector(bottleneck=gpu_op),
        max_cluster_limits=ExecutionResources.for_limits(),
        utility_calculator=StubUtilizationGauge(),
        autoscaling_coordinator=FakeAutoscalingCoordinator(),
        get_node_counts=lambda: {cpu_node_type: 1, gpu_node_type: 1},
        min_gap_between_autoscaling_requests_s=0,
    )

    autoscaler.try_trigger_scaling()

    # The requested resources include the existing nodes and the new nodes. Since the
    # bottleneck is the GPU operator, it should double the GPU nodes.
    assert autoscaler.get_total_resources().gpu == 2


def test_autoscaler_logs_warning_if_no_valid_node_types(
    # We need the `propagate_logs` fixture to propagate Ray Data logs to the root
    # logger. This is necessary for the `caplog` pytest fixture to work correctly.
    propagate_logs,  # noqa
    caplog,
):
    # This test sets up a GPU operator but only provides CPU node types. The autoscaler
    # should detect this mismatch and warn the user to add a compatible worker node
    # type.
    gpu_op = StubClusterAutoscalingOperator(
        _min_scheduling_resources=ExecutionResources(gpu=1),
    )
    cpu_node_type = NodeType({"CPU": 1})
    resource_manager = StubResourceManager()
    autoscaler = RateBasedClusterAutoscaler(
        ops=[gpu_op],
        resource_manager=resource_manager,
        execution_id="test",
        bottleneck_detector=StubBottleneckDetector(gpu_op),
        max_cluster_limits=ExecutionResources.for_limits(),
        utility_calculator=StubUtilizationGauge(),
        autoscaling_coordinator=FakeAutoscalingCoordinator(),
        get_node_counts=lambda: {cpu_node_type: 1},
        min_gap_between_autoscaling_requests_s=0,
    )

    with caplog.at_level(logging.WARNING):
        autoscaler.try_trigger_scaling()

    assert (
        "add a worker node type that satisfies these resource requirements"
        in caplog.text
    )


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
        bottleneck_detector=StubBottleneckDetector(bottleneck=None),
        max_cluster_limits=ExecutionResources.for_limits(),
        utility_calculator=StubUtilizationGauge(),
        autoscaling_coordinator=FakeAutoscalingCoordinator(
            get_time=lambda: time, remaining=[{"CPU": 1}]
        ),
        get_node_counts=lambda: {NodeType({"CPU": 1}): 1},
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


class StubResourceManager:
    def __init__(
        self,
        global_limits: ExecutionResources = None,
        op_usage: Dict[PhysicalOperator, ExecutionResources] = None,
    ):
        if global_limits is None:
            global_limits = ExecutionResources.for_limits()
        if op_usage is None:
            op_usage = {}
        self.global_limits = global_limits
        self._op_usage = op_usage

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

    def test_get_bottleneck_identifies_slower_operator(self):
        # op2 is fast (produces 12 blocks/s per task)
        op2 = StubClusterAutoscalingOperator(
            _per_task_resource_allocation=ExecutionResources(cpu=1),
            metrics=StubClusterAutoscalingMetrics(
                average_num_inputs_per_task=2,
                average_num_outputs_per_task=6,
                num_output_blocks_per_task_s=12,
            ),
        )
        # op1 is slower (produces 2 blocks/s per task, which become 6 sink outputs/s)
        op1 = StubClusterAutoscalingOperator(
            output_dependencies=[op2],
            _per_task_resource_allocation=ExecutionResources(cpu=1),
            metrics=StubClusterAutoscalingMetrics(
                num_output_blocks_per_task_s=2,
            ),
        )
        resource_manager = StubResourceManager(global_limits=ExecutionResources(cpu=2))
        detector = NormalizedThroughputBottleneckDetector(
            resource_manager, "test-requester"
        )

        bottleneck = detector.get_bottleneck([op1, op2])

        # op1 is the bottleneck because it's slower
        assert bottleneck == op1


def test_invalid_cluster_autoscaler_env_value_raises_value_error(monkeypatch):
    monkeypatch.setenv(CLUSTER_AUTOSCALER_ENV_KEY, "invalid")

    with pytest.raises(ValueError):
        create_cluster_autoscaler(
            topology={},
            execution_options=ExecutionOptions(),
            resource_manager=MagicMock(spec=ResourceManager),
            execution_id="test",
        )


@pytest.mark.parametrize(
    "cluster_autoscaler_env_value, expected_autoscaler_type",
    [
        ("RAYTURBO", RateBasedClusterAutoscaler),
        ("RAYTURBO_LEGACY", LegacyRayTurboClusterAutoscaler),
        ("V2", DefaultClusterAutoscalerV2),
        ("V1", DefaultClusterAutoscaler),
    ],
)
def test_cluster_autoscaler_env_value_creates_correct_autoscaler(
    monkeypatch, cluster_autoscaler_env_value, expected_autoscaler_type
):
    monkeypatch.setenv(CLUSTER_AUTOSCALER_ENV_KEY, cluster_autoscaler_env_value)

    autoscaler = create_cluster_autoscaler(
        topology={},
        execution_options=ExecutionOptions(),
        resource_manager=MagicMock(spec=ResourceManager),
        execution_id="test",
    )

    assert isinstance(autoscaler, expected_autoscaler_type)


@pytest.mark.parametrize("cpu_usage", [0.25, 0.9])
@pytest.mark.parametrize("gpu_usage", [0.25, 0.9])
@pytest.mark.parametrize("memory_usage", [0.25, 0.9])
def test_autoscaler_utilization_threshold(cpu_usage, gpu_usage, memory_usage):
    """Test autoscaler scaling behavior based on cluster utilization thresholds.

    Tests all combinations of cpu, gpu, and memory utilization values.
    The autoscaler should scale up if any utilization exceeds the 0.75 threshold.
    """
    # Calculate expected resources: scale up if any utilization >= 0.75
    threshold = 0.75

    cpu_op = StubClusterAutoscalingOperator(
        _min_scheduling_resources=ExecutionResources(cpu=1),
    )
    cpu_node_type = NodeType({"CPU": 4, "memory": 1000})

    execution_resources = ExecutionResources(
        cpu=cpu_usage, gpu=gpu_usage, object_store_memory=memory_usage
    )

    resource_manager = StubResourceManager()
    autoscaler = RateBasedClusterAutoscaler(
        ops=[cpu_op],
        resource_manager=resource_manager,
        execution_id="test",
        bottleneck_detector=StubBottleneckDetector(bottleneck=cpu_op),
        max_cluster_limits=ExecutionResources.for_limits(),
        utility_calculator=StubUtilizationGauge(execution_resources),
        autoscaling_coordinator=FakeAutoscalingCoordinator(),
        get_node_counts=lambda: {cpu_node_type: 1},
        min_gap_between_autoscaling_requests_s=0,
        cluster_scaling_up_util_threshold=threshold,  # 75% threshold
        cluster_scaling_up_gpu_threshold=threshold,  # 75% threshold for GPU
    )

    autoscaler.try_trigger_scaling()

    over_threshold = (
        cpu_usage >= threshold or gpu_usage >= threshold or memory_usage >= threshold
    )
    if over_threshold:
        assert autoscaler.get_total_resources().cpu > 4
    else:
        assert autoscaler.get_total_resources().cpu <= 4


@pytest.mark.parametrize(
    "max_cluster_limits,resource_request,expected_clamped",
    [
        # Test with requests that don't exceed the limit - all should be included
        (
            ExecutionResources(cpu=8, gpu=0, memory=0),
            [{"CPU": 2}, {"CPU": 2}],
            [{"CPU": 2}, {"CPU": 2}],
        ),
        # Test with requests that exceed the cluster limit
        (
            ExecutionResources(cpu=8, gpu=0, memory=0),
            [{"CPU": 3}, {"CPU": 3}, {"CPU": 3}, {"CPU": 3}],
            [{"CPU": 3}, {"CPU": 3}],
        ),
        # Test with heterogeneous cluster (both CPU and GPU resources, not evenly divisible)
        (
            ExecutionResources(cpu=7, gpu=2, memory=0),
            [{"CPU": 3, "GPU": 1}, {"CPU": 3, "GPU": 1}, {"CPU": 3, "GPU": 1}],
            [{"CPU": 3, "GPU": 1}, {"CPU": 3, "GPU": 1}],
        ),
        # Test with object store memory limits
        # Note: object_store_memory is stripped from the final result because Autoscaler SDK
        # doesn't work with obj store, but it still affects clamping logic
        (
            ExecutionResources(cpu=8, gpu=0, memory=0, object_store_memory=1000),
            [
                {"CPU": 2, "object_store_memory": 400},
                {"CPU": 2, "object_store_memory": 400},
                {"CPU": 2, "object_store_memory": 400},
            ],
            [{"CPU": 2}, {"CPU": 2}],  # object_store_memory stripped in final result
        ),
    ],
)
def test_resource_limits_clamper(
    max_cluster_limits: ExecutionResources,
    resource_request: List[Dict[str, float]],
    expected_clamped: List[Dict[str, float]],
):
    """Test that clamp_resource_limits respects cluster limits through RateBasedClusterAutoscaler."""
    # Create a minimal operator for the autoscaler
    cpu_op = StubClusterAutoscalingOperator(
        _min_scheduling_resources=ExecutionResources(cpu=1),
    )
    node_count: DefaultDict[NodeType, int] = defaultdict(int)
    for r in resource_request:
        node_count[NodeType(r)] += 1
    resource_manager = StubResourceManager()
    autoscaler = RateBasedClusterAutoscaler(
        ops=[cpu_op],
        resource_manager=resource_manager,
        execution_id="test",
        bottleneck_detector=StubBottleneckDetector(bottleneck=cpu_op),
        max_cluster_limits=max_cluster_limits,
        utility_calculator=StubUtilizationGauge(),
        autoscaling_coordinator=FakeAutoscalingCoordinator(),
        get_node_counts=lambda: node_count,
        min_gap_between_autoscaling_requests_s=0,
        # Setting this to 1.0 so that we can directly compare against
        # the clamped value.
        cluster_scaling_up_factor=1.0,
    )

    # Test clamping through the autoscaler
    requested_resources = autoscaler.try_trigger_scaling()

    expected_clamped = [
        ExecutionResources.from_resource_dict(er) for er in expected_clamped
    ]
    assert len(requested_resources) == len(expected_clamped)
    assert requested_resources == expected_clamped


@pytest.mark.parametrize(
    "max_resource_requirements,current_op_usage,min_scheduling_resources,resource_type,should_scale",
    [
        # Case 1: Current usage (4) + min_scheduling (1) = 5 > max (4) -> don't scale
        (
            ExecutionResources(cpu=4),
            ExecutionResources(cpu=4),
            ExecutionResources(cpu=1),
            "cpu",
            False,
        ),
        # Case 2: Current usage (3) + min_scheduling (1) = 4 <= max (4) -> scale
        (
            ExecutionResources(cpu=4),
            ExecutionResources(cpu=3),
            ExecutionResources(cpu=1),
            "cpu",
            True,
        ),
        # Case 3: Heterogeneous - CPU at limit (4) but memory below limit (500)
        # Adding one more task: CPU 4+1=5 > 4 (exceeds), memory 500+100=600 <= 1000 (within limit)
        # Should not scale because CPU exceeds limit
        (
            ExecutionResources(cpu=4, memory=1000),
            ExecutionResources(cpu=4, memory=500),
            ExecutionResources(cpu=1, memory=100),
            "cpu",
            False,
        ),
    ],
)
def test_autoscaler_skips_scaling_when_at_max_schedulable_tasks(
    max_resource_requirements: ExecutionResources,
    current_op_usage: ExecutionResources,
    min_scheduling_resources: ExecutionResources,
    resource_type: str,
    should_scale: bool,
):
    """Test that autoscaler skips scaling when bottleneck operator would exceed max resource limits."""

    # Set up operator with min_scheduling_resources
    cpu_op = StubClusterAutoscalingOperator(
        _min_scheduling_resources=min_scheduling_resources,
        _max_resource_requirements=max_resource_requirements,
    )
    node_type = NodeType({resource_type.upper(): 1})
    resource_manager = StubResourceManager(op_usage={cpu_op: current_op_usage})
    autoscaler = RateBasedClusterAutoscaler(
        ops=[cpu_op],
        resource_manager=resource_manager,
        execution_id="test",
        bottleneck_detector=StubBottleneckDetector(bottleneck=cpu_op),
        max_cluster_limits=ExecutionResources.for_limits(),
        utility_calculator=StubUtilizationGauge(
            ExecutionResources(cpu=0.9, gpu=0, object_store_memory=0, memory=0)
        ),
        autoscaling_coordinator=FakeAutoscalingCoordinator(),
        get_node_counts=lambda: {node_type: 1},
        min_gap_between_autoscaling_requests_s=0,
    )

    requested_resources = autoscaler.try_trigger_scaling()

    if should_scale:
        # Should scale - return non-empty list
        assert len(requested_resources) > 0
    else:
        # Should not scale - return empty list because adding one more task would exceed max_op_limits
        assert requested_resources == []


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-sv", __file__]))
