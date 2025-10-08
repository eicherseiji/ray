from dataclasses import dataclass, field
from typing import Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from ray.anyscale.air._internal.autoscaling_coordinator import (
    FakeAutoscalingCoordinator,
)
from ray.anyscale.data._internal.cluster_autoscaler import (
    ClusterAutoscalingMetrics,
    LegacyRayTurboClusterAutoscaler,
    NodeType,
    NormalizedThroughputCalculator,
    ProductivityCalculator,
    RateBasedClusterAutoscaler,
    SupportsClusterAutoscaling,
)
from ray.data._internal.cluster_autoscaler import (
    CLUSTER_AUTOSCALER_ENV_KEY,
    DefaultClusterAutoscaler,
    create_cluster_autoscaler,
)
from ray.data._internal.execution.interfaces import PhysicalOperator
from ray.data._internal.execution.interfaces.execution_options import ExecutionResources
from ray.data._internal.execution.resource_manager import ResourceManager


class StubProductivityCalculator(ProductivityCalculator):
    """A stub implementation for testing."""

    def __init__(self, productivity: Dict[SupportsClusterAutoscaling, float]):
        self._productivity = productivity

    def get_productivity(self, op: SupportsClusterAutoscaling) -> Optional[float]:
        return self._productivity.get(op)


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
        productivity_calculator=StubProductivityCalculator({cpu_op: 1, gpu_op: 0}),
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


@dataclass
class StubResourceManager:
    global_limits: Optional[ExecutionResources] = None

    def get_global_limits(self) -> Optional[ExecutionResources]:
        return self.global_limits


class TestNormalizedThroughputCalculator:
    def test_get_productivity_completed_operator(self):
        op = StubClusterAutoscalingOperator(_completed=True)
        calculator = NormalizedThroughputCalculator([op], StubResourceManager())

        productivity = calculator.get_productivity(op)

        assert productivity is None

    def test_get_productivity_no_global_limits(self):
        op = StubClusterAutoscalingOperator()
        resource_manager = StubResourceManager()  # No global limits set
        calculator = NormalizedThroughputCalculator([op], resource_manager)

        productivity = calculator.get_productivity(op)

        assert productivity is None

    def test_get_productivity_single_operator(self):
        op = StubClusterAutoscalingOperator(
            _per_task_resource_allocation=ExecutionResources(cpu=1),
            metrics=StubClusterAutoscalingMetrics(
                num_output_blocks_per_task_s=1,
            ),
        )
        global_limits = ExecutionResources(cpu=2)
        resource_manager = StubResourceManager(global_limits=global_limits)
        calculator = NormalizedThroughputCalculator([op], resource_manager)

        productivity = calculator.get_productivity(op)

        # The operator uses 1 CPU per task, and the global limits are 2 CPUs. So, the
        # operator can launch 2 tasks producing a total of 2 outputs per second.
        assert productivity == 2

    def test_get_productivity_with_downstream_operator(self):
        op2 = StubClusterAutoscalingOperator(
            _per_task_resource_allocation=ExecutionResources(cpu=1),
            metrics=StubClusterAutoscalingMetrics(
                average_num_inputs_per_task=2,
                average_num_outputs_per_task=6,
                num_output_blocks_per_task_s=12,
            ),
        )
        op1 = StubClusterAutoscalingOperator(
            output_dependencies=[op2],
            _per_task_resource_allocation=ExecutionResources(cpu=1),
            metrics=StubClusterAutoscalingMetrics(
                num_output_blocks_per_task_s=4,
            ),
        )
        resource_manager = StubResourceManager(global_limits=ExecutionResources(cpu=2))
        calculator = NormalizedThroughputCalculator([op1, op2], resource_manager)

        productivity1 = calculator.get_productivity(op1)
        productivity2 = calculator.get_productivity(op2)

        # Each op1 output is worth 6 / 2 = 3 op2 outputs. So, op1 produces 4 * 3 = 12
        # op2 outputs per second. op2 produces 12 outputs per second. Since there are
        # 2 CPUs available, if op1 launches 1 tasks and op2 launches 1 task, each
        # operator can produce 12 outputs per second and the pipeline will be optimally
        # balanced.
        assert productivity1 == 12 and productivity2 == 12, (
            productivity1,
            productivity2,
        )


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
        ("OSS", DefaultClusterAutoscaler),
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
