from dataclasses import dataclass, field
from typing import List, Optional
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
from ray.data._internal.execution.interfaces import PhysicalOperator
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
    _max_task_concurrency: Optional[int] = None
    _min_scheduling_resources: ExecutionResources = ExecutionResources.zero()
    _completed: bool = False

    def per_task_resource_allocation(self) -> ExecutionResources:
        return self._per_task_resource_allocation

    def max_task_concurrency(self) -> Optional[int]:
        return self._max_task_concurrency

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
    autoscaler = RateBasedClusterAutoscaler(
        ops=[cpu_op, gpu_op],
        execution_id="test",
        bottleneck_detector=StubBottleneckDetector(bottleneck=gpu_op),
        autoscaling_coordinator=FakeAutoscalingCoordinator(),
        get_node_counts=lambda: {cpu_node_type: 1, gpu_node_type: 1},
        min_gap_between_autoscaling_requests_s=0,
    )

    requested_resources = autoscaler.try_trigger_scaling()

    # The requested resources include the existing nodes and the new nodes. Since the
    # bottleneck is the GPU operator, it should double the GPU nodes.
    assert requested_resources == [
        ExecutionResources(cpu=1),
        ExecutionResources(gpu=1),
        ExecutionResources(gpu=1),
    ]


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
    autoscaler = RateBasedClusterAutoscaler(
        ops=[gpu_op],
        execution_id="test",
        bottleneck_detector=StubBottleneckDetector(gpu_op),
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
    autoscaler = RateBasedClusterAutoscaler(
        ops=[],
        execution_id="test",
        bottleneck_detector=StubBottleneckDetector(bottleneck=None),
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


@dataclass
class StubResourceManager:
    global_limits: ExecutionResources = field(
        default_factory=ExecutionResources.for_limits
    )

    def get_global_limits(self) -> ExecutionResources:
        return self.global_limits


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
        resource_manager=MagicMock(spec=ResourceManager),
        execution_id="test",
    )

    assert isinstance(autoscaler, expected_autoscaler_type)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-sv", __file__]))
