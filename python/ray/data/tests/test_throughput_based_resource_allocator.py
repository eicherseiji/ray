import math
import sys
from typing import Optional
from unittest.mock import MagicMock, Mock, patch

import pytest

from ray.data import ExecutionResources
from ray.data._internal.execution.interfaces.execution_options import ExecutionOptions
from ray.data._internal.execution.operators.base_physical_operator import (
    AllToAllOperator,
)
from ray.data._internal.execution.operators.hash_shuffle import (
    HashShuffleOperator,
)
from ray.data._internal.execution.operators.input_data_buffer import InputDataBuffer
from ray.data._internal.execution.operators.limit_operator import LimitOperator
from ray.data._internal.execution.operators.map_operator import MapOperator
from ray.data._internal.execution.resource_manager import ResourceManager
from ray.data._internal.execution.streaming_executor_state import (
    build_streaming_topology,
)
from ray.data._internal.execution.throughput_based_resource_allocator import (
    ThroughputBasedResourceAllocator,
    _estimate_byte_expansion_ratio,
)
from ray.data._internal.execution.util import make_ref_bundles
from ray.data._internal.util import GiB, MiB
from ray.data.context import DataContext
from ray.data.tests.conftest import *  # noqa
from ray.data.tests.conftest import mock_all_to_all_op
from ray.data.tests.test_resource_manager import mock_map_op, mock_union_op


def mock_shuffle_op(input_op, name="MockShuffle"):
    """Create a mock HashShuffleOperator for testing.

    Creates a HashShuffleOperator which is eligible for resource allocation
    (throttling_disabled=False) and is a blocking materializing operator.
    """
    mock_logical_op = MagicMock()
    mock_logical_op.estimated_num_outputs.return_value = 10
    mock_logical_op.infer_metadata.return_value.size_bytes = 1000
    input_op._logical_operators = [mock_logical_op]

    with patch.object(HashShuffleOperator, "start"):
        with patch(
            "ray.data._internal.execution.operators.hash_shuffle._get_total_cluster_resources"
        ) as mock:
            mock.return_value = ExecutionResources(cpu=1)
            op = HashShuffleOperator(
                input_op=input_op,
                data_context=DataContext.get_current(),
                key_columns=("key",),
                num_partitions=1,
            )

    return op


def create_mock_resource_manager(topology=None):
    """Create a ResourceManager with minimal dependencies for testing."""
    return ResourceManager(
        topology=topology if topology is not None else {},
        options=ExecutionOptions(),
        get_total_resources=MagicMock(),
        data_context=DataContext.get_current(),
    )


def create_mock_operator(
    name: str, op_class, max_concurrency: Optional[int] = None, metrics=None
):
    """Create a mock operator using unittest.Mock that inherits from the specified operator class"""
    mock_op = Mock(spec=op_class)
    # Make isinstance checks work correctly
    mock_op.__class__ = op_class

    # Only override what's absolutely necessary - let the test mock what it needs
    mock_op.__str__ = Mock(return_value=name)
    mock_op.name = name

    # Set provided parameters
    mock_op.get_max_concurrency_limit.return_value = max_concurrency
    mock_op.metrics = metrics

    # Bind real class methods for _is_blocking_materializing_op to work correctly
    mock_op.output_dependencies = []
    mock_op.throttling_disabled = lambda: op_class.throttling_disabled(mock_op)
    mock_op.has_execution_finished.return_value = False

    return mock_op


class TestThroughputBasedResourceAllocator:
    def test_allocate_water_filling_basic_scenario(self):
        """Test basic water filling allocation with fixed inputs and exact outputs"""
        allocator = ThroughputBasedResourceAllocator(create_mock_resource_manager())

        # Standard operator configuration
        metrics = Mock()
        metrics.average_bytes_inputs_per_task = 1000
        metrics.average_rows_inputs_per_task = 10
        metrics.average_bytes_outputs_per_task = 2000
        metrics.average_rows_outputs_per_task = 20

        # Create mock operators (all non-shuffle)
        read_op = create_mock_operator(
            "ReadFiles", op_class=MapOperator, max_concurrency=None, metrics=metrics
        )
        map_op = create_mock_operator(
            "MapBatches", op_class=MapOperator, max_concurrency=None, metrics=metrics
        )
        write_op = create_mock_operator(
            "WriteFiles", op_class=MapOperator, max_concurrency=None, metrics=metrics
        )

        eligible_ops = [read_op, map_op, write_op]

        # Compute byte expansion ratios (required by _allocate_water_filling)
        op_byte_expansion_ratios = {
            op: _estimate_byte_expansion_ratio(op) for op in eligible_ops
        }

        # Fixed productivity rates: CPU-s/MiB, GPU-s/MiB, Memory/MiB
        # These rates sum to 0.1, giving clean integer allocations
        productivity_rates = {
            read_op: (0.02, 0.0, 0.0),  # CPU-only operator -> 2.0 CPU
            map_op: (0.05, 0.0, 0.0),  # CPU-only operator -> 5.0 CPU
            write_op: (0.03, 0.0, 0.0),  # CPU-only operator -> 3.0 CPU
        }

        # Fixed resource limits
        limits = ExecutionResources(
            cpu=10.0, gpu=2.0, object_store_memory=1 * GiB, memory=4 * GiB
        )

        # Call the method under test
        (
            allocatable_limits,
            target_rate,
            op_allocations,
        ) = allocator._allocate_water_filling(
            eligible_ops, productivity_rates, op_byte_expansion_ratios, limits
        )

        # Assert exact whole outputs
        assert allocatable_limits == ExecutionResources(
            cpu=10.0, gpu=2.0, object_store_memory=1 * GiB, memory=4 * GiB
        )

        assert target_rate == 100.0

        # Calculate expected object store allocation using new cumulative algorithm
        # All ops have same input -> output (1000 -> 2000, 2x expansion)
        # Base = 1000
        # read_op:  1000 * 2 = 2000
        # map_op:   2000 * 2 = 4000
        # write_op: 4000 * 2 = 8000
        # Total weights: 2000 + 4000 + 8000 = 14000
        # Ratios: 1:2:4
        total_obj_store = 1 * GiB
        total_weights = 2000.0 + 4000.0 + 8000.0

        expected_read_obj_store = math.ceil(total_obj_store * 2000.0 / total_weights)
        expected_map_obj_store = math.ceil(total_obj_store * 4000.0 / total_weights)
        expected_write_obj_store = math.ceil(total_obj_store * 8000.0 / total_weights)

        # Assert exact allocations for each operator
        assert op_allocations == {
            read_op: ExecutionResources(
                cpu=2.0,
                gpu=0.0,
                memory=float("inf"),
                object_store_memory=expected_read_obj_store,
            ),
            map_op: ExecutionResources(
                cpu=5.0,
                gpu=0.0,
                memory=float("inf"),
                object_store_memory=expected_map_obj_store,
            ),
            write_op: ExecutionResources(
                cpu=3.0,
                gpu=0.0,
                memory=float("inf"),
                object_store_memory=expected_write_obj_store,
            ),
        }

    def test_allocate_water_filling_gpu_operator(self):
        """Test allocation for GPU-only operators with exact outputs"""
        allocator = ThroughputBasedResourceAllocator(create_mock_resource_manager())

        # Standard operator configuration
        metrics = Mock()
        metrics.average_bytes_inputs_per_task = 1000
        metrics.average_rows_inputs_per_task = 10
        metrics.average_bytes_outputs_per_task = 2000
        metrics.average_rows_outputs_per_task = 20

        # Create operators including a GPU-only one (both non-shuffle)
        cpu_op = create_mock_operator(
            "Read", op_class=MapOperator, max_concurrency=None, metrics=metrics
        )
        gpu_op = create_mock_operator(
            "Inference", op_class=MapOperator, max_concurrency=None, metrics=metrics
        )

        eligible_ops = [cpu_op, gpu_op]

        # Compute expansion ratios (required by _allocate_water_filling)
        op_expansion_ratios = {
            op: _estimate_byte_expansion_ratio(op) for op in eligible_ops
        }

        # Fixed productivity rates - GPU operator has zero CPU productivity
        productivity_rates = {
            cpu_op: (0.01, 0.0, 0.0),  # CPU-only
            gpu_op: (0.0, 0.02, 0.0),  # GPU-only
        }

        limits = ExecutionResources(
            cpu=10.0, gpu=4.0, object_store_memory=1 * GiB, memory=4 * GiB
        )

        (
            allocatable_limits,
            target_rate,
            op_allocations,
        ) = allocator._allocate_water_filling(
            eligible_ops, productivity_rates, op_expansion_ratios, limits
        )

        # Assert exact whole outputs
        assert allocatable_limits == ExecutionResources(
            cpu=10.0, gpu=4.0, object_store_memory=1 * GiB, memory=4 * GiB
        )

        # With GPU rate limiting enabled:
        # CPU-limited rate: 10.0 / 0.01 = 1000.0 MiB/s
        # GPU-limited rate: 4.0 / 0.02 = 200.0 MiB/s
        # Target rate = min(1000.0, 200.0) = 200.0 MiB/s
        assert target_rate == 200.0

        # Calculate expected object store allocation using new cumulative algorithm
        # Both ops have same input -> output (1000 -> 2000, 2x expansion)
        # Base = 1000
        # cpu_op: 1000 * 2 = 2000
        # gpu_op: 2000 * 2 = 4000
        # Total weights: 2000 + 4000 = 6000
        # Ratios: 1:2
        total_obj_store = 1 * GiB
        total_weights = 2000.0 + 4000.0

        expected_cpu_obj_store = math.ceil(total_obj_store * 2000.0 / total_weights)
        expected_gpu_obj_store = math.ceil(total_obj_store * 4000.0 / total_weights)

        assert op_allocations == {
            cpu_op: ExecutionResources(
                cpu=2.0,  # 200.0 * 0.01 = 2.0 (GPU bottleneck limits CPU allocation)
                gpu=0.0,
                memory=float("inf"),
                object_store_memory=expected_cpu_obj_store,
            ),
            gpu_op: ExecutionResources(
                cpu=0.0,
                gpu=4.0,  # 200.0 * 0.02 = 4.0 (all available GPU)
                memory=float("inf"),
                object_store_memory=expected_gpu_obj_store,
            ),
        }

    def test_allocate_water_filling_with_pending_shuffle_operator(self):
        """Test water-filling allocation with 1 running and 1 pending operators."""
        allocator = ThroughputBasedResourceAllocator(create_mock_resource_manager())

        # Standard operator configuration
        metrics = Mock()
        metrics.average_bytes_inputs_per_task = 1000
        metrics.average_rows_inputs_per_task = 10
        metrics.average_bytes_outputs_per_task = 2000
        metrics.average_rows_outputs_per_task = 20

        # Create operators with shuffle at the end
        map_op = create_mock_operator(
            "MapBatches", op_class=MapOperator, max_concurrency=None, metrics=metrics
        )
        shuffle_op = create_mock_operator(
            "Shuffle", op_class=AllToAllOperator, max_concurrency=None, metrics=metrics
        )

        eligible_ops = [map_op, shuffle_op]

        # Compute expansion ratios (required by _allocate_water_filling)
        op_expansion_ratios = {
            op: _estimate_byte_expansion_ratio(op) for op in eligible_ops
        }

        productivity_rates = {
            map_op: (0.01, 0.0, 0.0),
            shuffle_op: (0.0, 0.0, 0.0),  # Shuffle ops have zero productivity
        }

        limits = ExecutionResources(
            cpu=10.0, gpu=2.0, object_store_memory=1 * GiB, memory=4 * GiB
        )

        (
            allocatable_limits,
            target_rate,
            op_allocations,
        ) = allocator._allocate_water_filling(
            eligible_ops, productivity_rates, op_expansion_ratios, limits
        )

        # With 2 ops and baseline ratios [0.5, 0.5]:
        # - Only map_op has productivity > 0, so running_ops = [map_op]
        # - total_allocatable_fraction = 0.5 (first baseline ratio)
        # - allocatable_limits = limits * 0.5
        assert allocatable_limits == ExecutionResources(
            cpu=5.0, gpu=1.0, object_store_memory=0.5 * GiB, memory=2 * GiB
        )

        # target_rate = 5.0 CPU / 0.01 = 500 MiB/s
        assert target_rate == 500.0

        # map_op gets productivity-based allocation from allocatable resources
        # shuffle_op gets baseline reservation from remaining resources (limits * 0.5)
        assert op_allocations == {
            map_op: ExecutionResources(
                cpu=5.0,  # 500 * 0.01 = 5.0
                gpu=0.0,
                memory=float("inf"),
                object_store_memory=0.5 * GiB,  # All allocatable obj store
            ),
            shuffle_op: ExecutionResources(
                cpu=5.0,  # Reserved: limits.cpu * 0.5
                gpu=1.0,  # Reserved: limits.gpu * 0.5
                memory=float("inf"),
                # Blocking materializing ops get unlimited OS budget
                object_store_memory=float("inf"),
            ),
        }

    def test_allocate_water_filling_no_eligible_ops(self):
        """Test allocation with empty operator list - exact outputs"""
        allocator = ThroughputBasedResourceAllocator(create_mock_resource_manager())

        eligible_ops = []
        productivity_rates = {}
        op_expansion_ratios = {}
        limits = ExecutionResources(
            cpu=10.0, gpu=2.0, object_store_memory=1 * GiB, memory=4 * GiB
        )

        (
            allocatable_limits,
            target_rate,
            op_allocations,
        ) = allocator._allocate_water_filling(
            eligible_ops, productivity_rates, op_expansion_ratios, limits
        )

        # Assert exact whole result
        assert (allocatable_limits, target_rate, op_allocations) == (limits, 0, {})

    def test_allocate_water_filling_with_non_started_ops(self):
        """Test allocation when operators have mixed productivity with exact outputs"""
        allocator = ThroughputBasedResourceAllocator(create_mock_resource_manager())

        # Producing op metrics
        producing_op_metrics = Mock()
        producing_op_metrics.average_bytes_inputs_per_task = 1000
        producing_op_metrics.average_rows_inputs_per_task = 10
        producing_op_metrics.average_bytes_outputs_per_task = 2000
        producing_op_metrics.average_rows_outputs_per_task = 20
        # Pending op metrics
        pending_op_metrics = Mock()
        pending_op_metrics.average_bytes_inputs_per_task = None
        pending_op_metrics.average_rows_inputs_per_task = None
        pending_op_metrics.average_bytes_outputs_per_task = None
        pending_op_metrics.average_rows_outputs_per_task = None

        # Create operators with mixed states (some running, some not yet started)
        op1 = create_mock_operator(
            "RunningOperator",
            op_class=MapOperator,
            max_concurrency=None,
            metrics=producing_op_metrics,
        )
        op2 = create_mock_operator(
            "NotStartedOperator1",
            op_class=MapOperator,
            max_concurrency=None,
            metrics=pending_op_metrics,
        )
        op3 = create_mock_operator(
            "NotStartedOperator2",
            op_class=MapOperator,
            max_concurrency=None,
            metrics=pending_op_metrics,
        )

        eligible_ops = [op1, op2, op3]

        # Compute expansion ratios (required by _allocate_water_filling)
        op_expansion_ratios = {
            op: _estimate_byte_expansion_ratio(op) for op in eligible_ops
        }

        # Mixed productivity: one operator running, two not yet started
        productivity_rates = {
            op1: (0.02, 0.0, 0.0),  # Running operator with CPU productivity
            op2: (0.0, 0.0, 0.0),  # Not started yet
            op3: (0.0, 0.0, 0.0),  # Not started yet
        }

        limits = ExecutionResources(
            cpu=10.0, gpu=2.0, object_store_memory=1 * GiB, memory=4 * GiB
        )

        (
            allocatable_limits,
            target_rate,
            op_allocations,
        ) = allocator._allocate_water_filling(
            eligible_ops, productivity_rates, op_expansion_ratios, limits
        )

        # Only op1 has positive productivity, so total_allocatable_fraction = 0.5 (baseline ratio for first op)
        assert allocatable_limits == ExecutionResources(
            cpu=5.0,  # 10.0 * 0.5 (only first op is running)
            gpu=1.0,  # 2.0 * 0.5
            object_store_memory=0.5 * GiB,  # 512MB allocatable
            memory=2 * GiB,
        )

        assert target_rate == 250.0  # 5.0 / 0.02

        # Object store allocation behavior:
        # - Running operator (op1) gets most of the allocatable object store memory
        # - Non-started operators (op2, op3) get minimal allocation (1 byte for liveness)
        assert op_allocations == {
            # Running operator gets allocation based on productivity
            op1: ExecutionResources(
                cpu=5.0,  # 250.0 * 0.02 = 5.0 (all available CPU)
                gpu=0.0,
                memory=float("inf"),
                object_store_memory=0.5 * GiB,  # All allocatable object store memory
            ),
            # Not started operators get baseline reservations from reserved resources
            op2: ExecutionResources(
                cpu=2.5,  # Reserved: 5.0 * 0.5 (half of reserved resources)
                gpu=0.5,  # Reserved: 1.0 * 0.5
                memory=float("inf"),
                object_store_memory=0.5 * GiB / 2,  # half of reserved
            ),
            op3: ExecutionResources(
                cpu=2.5,  # Reserved: 5.0 * 0.5 (half of reserved resources)
                gpu=0.5,  # Reserved: 1.0 * 0.5
                memory=float("inf"),
                object_store_memory=0.5 * GiB / 2,  # half of reserved
            ),
        }

    def test_allocate_water_filling_with_max_rate_cap(self):
        """Test _allocate_water_filling with operator rate caps (finite and infinite).

        Tests _get_operator_max_rate_cap behavior:
        - Returns inf when max_concurrency is None
        - Returns inf when alpha <= 0 (even with finite max_concurrency)
        - Returns finite cap = max_concurrency * per_task_cpu / alpha when both are set
        """
        allocator = ThroughputBasedResourceAllocator(create_mock_resource_manager())

        metrics = Mock()
        metrics.average_bytes_inputs_per_task = 1000
        metrics.average_rows_inputs_per_task = 10
        metrics.average_bytes_outputs_per_task = 2000
        metrics.average_rows_outputs_per_task = 20

        # Op1 (finite max_concurrency)
        op1 = create_mock_operator(
            "Op1", MapOperator, max_concurrency=10, metrics=metrics
        )
        op1.per_task_resource_allocation.return_value = ExecutionResources(cpu=1.0)

        # Op2 (finite max_concurrency)
        op2 = create_mock_operator(
            "Op2", MapOperator, max_concurrency=5, metrics=metrics
        )
        op2.per_task_resource_allocation.return_value = ExecutionResources(cpu=1.5)

        # Verify _get_operator_max_rate_cap directly for different scenarios
        # Case 1: alpha = 0 with finite max_concurrency -> inf
        zero_alpha = (0.0, 0.0, 0.0)
        assert allocator._get_operator_max_rate_cap(op1, zero_alpha) == float("inf")
        assert allocator._get_operator_max_rate_cap(op2, zero_alpha) == float("inf")

        # Case 2: alpha > 0 with finite max_concurrency -> finite cap
        assert (
            allocator._get_operator_max_rate_cap(op1, (0.02, 0.0, 0.0)) == 500.0
        )  # 10 * 1.0 / 0.02
        assert (
            allocator._get_operator_max_rate_cap(op2, (0.03, 0.0, 0.0)) == 250.0
        )  # 5 * 1.5 / 0.03

        # Case 3: max_concurrency = None -> inf (tested via create_mock_operator default)
        op_no_limit = create_mock_operator(
            "OpNoLimit", MapOperator, max_concurrency=None, metrics=metrics
        )
        op_no_limit.per_task_resource_allocation.return_value = ExecutionResources(
            cpu=1.0
        )

        assert allocator._get_operator_max_rate_cap(
            op_no_limit, (0.01, 0.0, 0.0)
        ) == float("inf")

        # Now test that water-filling respects rate caps
        eligible_ops = [op1, op2]
        op_expansion_ratios = {
            op: _estimate_byte_expansion_ratio(op) for op in eligible_ops
        }
        op_productivity_rates = {
            op1: (0.02, 0.0, 0.0),
            op2: (0.03, 0.0, 0.0),
        }

        limits = ExecutionResources(
            cpu=50.0, gpu=0.0, object_store_memory=1 * GiB, memory=8 * GiB
        )

        (
            allocatable_limits,
            target_rate,
            op_allocations,
        ) = allocator._allocate_water_filling(
            eligible_ops, op_productivity_rates, op_expansion_ratios, limits
        )

        # CPU constraint: 50.0 / (0.02 + 0.03) = 1000.0 MiB/s
        # Rate cap constraint: min(500.0, 250.0) = 250.0 MiB/s
        # Final target rate = min(1000.0, 250.0) = 250.0 MiB/s
        assert target_rate == 250.0

        # CPU allocations = target_rate * alpha
        assert op_allocations[op1].cpu == 5.0  # 250.0 * 0.02
        assert op_allocations[op2].cpu == 7.5  # 250.0 * 0.03

    def test_allocate_water_filling_object_store_allocation(self):
        """Test object store allocation with different data expansion ratios between operators"""
        allocator = ThroughputBasedResourceAllocator(create_mock_resource_manager())

        # Create operators with cleaner expansion ratios for easier verification
        # Compressor: reduces data size by half (1000 bytes in -> 500 bytes out)
        compress_metrics = Mock()
        compress_metrics.average_bytes_inputs_per_task = 1000
        compress_metrics.average_rows_inputs_per_task = 10
        compress_metrics.average_bytes_outputs_per_task = 500  # 0.5x expansion
        compress_metrics.average_rows_outputs_per_task = 10

        # Neutral: keeps data size the same (1000 bytes in -> 1000 bytes out)
        neutral_metrics = Mock()
        neutral_metrics.average_bytes_inputs_per_task = 1000
        neutral_metrics.average_rows_inputs_per_task = 10
        neutral_metrics.average_bytes_outputs_per_task = 1000  # 1.0x expansion
        neutral_metrics.average_rows_outputs_per_task = 10

        # Expander: doubles data size (1000 bytes in -> 2000 bytes out)
        expand_metrics = Mock()
        expand_metrics.average_bytes_inputs_per_task = 1000
        expand_metrics.average_rows_inputs_per_task = 10
        expand_metrics.average_bytes_outputs_per_task = 2000  # 2.0x expansion
        expand_metrics.average_rows_outputs_per_task = 10

        compress_op = create_mock_operator(
            "Compress",
            op_class=MapOperator,
            max_concurrency=None,
            metrics=compress_metrics,
        )
        neutral_op = create_mock_operator(
            "Neutral",
            op_class=MapOperator,
            max_concurrency=None,
            metrics=neutral_metrics,
        )
        expand_op = create_mock_operator(
            "Expand", op_class=MapOperator, max_concurrency=None, metrics=expand_metrics
        )

        eligible_ops = [compress_op, neutral_op, expand_op]

        # Compute expansion ratios (required by _allocate_water_filling)
        op_expansion_ratios = {
            op: _estimate_byte_expansion_ratio(op) for op in eligible_ops
        }

        # All operators have same CPU productivity for clean comparison
        productivity_rates = {
            compress_op: (0.01, 0.0, 0.0),  # CPU-only
            neutral_op: (0.01, 0.0, 0.0),  # CPU-only
            expand_op: (0.01, 0.0, 0.0),  # CPU-only
        }

        # Use clean resource limits
        limits = ExecutionResources(
            cpu=6.0, gpu=0.0, object_store_memory=4 * GiB, memory=8 * GiB
        )

        (
            allocatable_limits,
            target_rate,
            op_allocations,
        ) = allocator._allocate_water_filling(
            eligible_ops, productivity_rates, op_expansion_ratios, limits
        )

        # All operators have positive productivity, so full resources are allocatable
        assert allocatable_limits == ExecutionResources(
            cpu=6.0, gpu=0.0, object_store_memory=4 * GiB, memory=8 * GiB
        )

        assert target_rate == 200.0  # 6.0 / 0.03 (sum of all productivities)

        # Calculate expected object store allocation using new cumulative algorithm
        # All ops have same input (1000 bytes)
        # Compress: 1000 -> 500 (0.5x), Neutral: 1000 -> 1000 (1.0x), Expand: 1000 -> 2000 (2.0x)
        # Base = 1000
        # Compress: 1000 * (500/1000) = 500
        # Neutral:  500 * (1000/1000) = 500
        # Expand:   500 * (2000/1000) = 1000
        # Total weights: 500 + 500 + 1000 = 2000
        total_obj_store = 4 * GiB
        total_weights = 500.0 + 500.0 + 1000.0  # 2000

        compress_obj_store = math.ceil(total_obj_store * 500.0 / total_weights)
        neutral_obj_store = math.ceil(total_obj_store * 500.0 / total_weights)
        expand_obj_store = math.ceil(total_obj_store * 1000.0 / total_weights)

        # Object store allocation should be proportional to expansion ratios
        assert op_allocations == {
            compress_op: ExecutionResources(
                cpu=2.0,  # 200.0 * 0.01 = 2.0, equally distributed
                gpu=0.0,
                memory=float("inf"),
                object_store_memory=compress_obj_store,  # 1/7 of total (0.5x expansion)
            ),
            neutral_op: ExecutionResources(
                cpu=2.0,  # 200.0 * 0.01 = 2.0, equally distributed
                gpu=0.0,
                memory=float("inf"),
                object_store_memory=neutral_obj_store,  # 2/7 of total (1.0x expansion)
            ),
            expand_op: ExecutionResources(
                cpu=2.0,  # 200.0 * 0.01 = 2.0, equally distributed
                gpu=0.0,
                memory=float("inf"),
                object_store_memory=expand_obj_store,  # 4/7 of total (2.0x expansion)
            ),
        }

    def test_update_productivity_coefficients_with_normalized_yield(self):
        """Test that _update_productivity_coefficients uses normalized yield based on downstream expansion ratios"""
        # Create a pipeline: A > B > C
        # A outputs 100 rows, B contracts to 50 rows (2:1), C contracts to 25 rows (2:1)

        allocator = ThroughputBasedResourceAllocator(create_mock_resource_manager())

        # Operator A
        metrics_a = Mock()
        metrics_a.average_total_task_completion_time_s = 10.0
        metrics_a.average_rows_inputs_per_task = 100
        metrics_a.average_rows_outputs_per_task = 100
        metrics_a.average_bytes_inputs_per_task = 1000 * MiB
        metrics_a.average_bytes_outputs_per_task = 1000 * MiB

        op_a = create_mock_operator("OpA", MapOperator, metrics=metrics_a)
        op_a.per_task_resource_allocation.return_value = ExecutionResources(
            cpu=2.0, gpu=0.0
        )
        op_a.incremental_resource_usage.return_value = ExecutionResources(memory=1000)

        # Operator B (contracts 2:1)
        metrics_b = Mock()
        metrics_b.average_total_task_completion_time_s = 5.0
        metrics_b.average_rows_inputs_per_task = 100
        metrics_b.average_rows_outputs_per_task = 50
        metrics_b.average_bytes_inputs_per_task = 1000 * MiB
        metrics_b.average_bytes_outputs_per_task = 500 * MiB

        op_b = create_mock_operator("OpB", MapOperator, metrics=metrics_b)
        op_b.per_task_resource_allocation.return_value = ExecutionResources(
            cpu=1.0, gpu=0.0
        )
        op_b.incremental_resource_usage.return_value = ExecutionResources(memory=500)

        # Operator C (contracts 2:1)
        metrics_c = Mock()
        metrics_c.average_total_task_completion_time_s = 2.5
        metrics_c.average_rows_inputs_per_task = 50
        metrics_c.average_rows_outputs_per_task = 25
        metrics_c.average_bytes_inputs_per_task = 500 * MiB
        metrics_c.average_bytes_outputs_per_task = 250 * MiB

        op_c = create_mock_operator("OpC", MapOperator, metrics=metrics_c)
        op_c.per_task_resource_allocation.return_value = ExecutionResources(
            cpu=0.0, gpu=1.0
        )
        op_c.incremental_resource_usage.return_value = ExecutionResources(memory=250)

        eligible_ops = [op_a, op_b, op_c]

        # Compute expansion ratios (required by _update_productivity_coefficients)
        op_expansion_ratios = {
            op: _estimate_byte_expansion_ratio(op) for op in eligible_ops
        }

        # Call _update_productivity_coefficients
        allocator._update_productivity_coefficients(eligible_ops, op_expansion_ratios)

        # Verify productivity coefficients were calculated with normalized yield
        # Expected for A: normalized_bytes = 1000 MiB * 0.5 (B ratio) * 0.5 (C ratio) = 250 MiB
        #   temporal_prod = 10.0s / 250 MiB = 0.04 s/byte
        #   alpha = 0.04 s/byte * 2.0 CPU = 0.08 CPU-s/byte
        alpha_a, beta_a, gamma_a = allocator._op_productivity[op_a]
        assert alpha_a == pytest.approx(0.08, rel=1e-6)
        assert beta_a == 0.0

        # Expected for B: normalized_bytes = 500 MiB * 0.5 (C ratio) = 250 MiB
        #   temporal_prod = 5.0s / 250 bytes = 0.02 s/byte
        #   alpha = 0.02 s/byte * 1.0 CPU = 0.02 CPU-s/byte
        alpha_b, beta_b, gamma_b = allocator._op_productivity[op_b]
        assert alpha_b == pytest.approx(0.02, rel=1e-6)
        assert beta_b == 0.0

        # Expected for C: normalized_bytes = 250 bytes * 1.0 (no downstream) = 250 bytes
        #   temporal_prod = 2.5s / 250 bytes = 0.01 s/byte
        #   beta = 0.01 s/byte * 1.0 GPU = 0.01 GPU-s/byte
        alpha_c, beta_c, gamma_c = allocator._op_productivity[op_c]
        assert alpha_c == 0.0
        assert beta_c == pytest.approx(0.01, rel=1e-6)

    def test_update_productivity_coefficients_with_expansion(self):
        """Test _update_productivity_coefficients with expansion (filter -> map that expands data)"""
        # Pipeline: Filter (10:1 contraction) > Expand (1:2 expansion)
        # Filter outputs 10 rows, Expand takes 10 input rows and produces 20 output rows

        allocator = ThroughputBasedResourceAllocator(create_mock_resource_manager())

        # Filter operator - contracts 10:1
        metrics_filter = Mock()
        metrics_filter.average_total_task_completion_time_s = 8.0
        metrics_filter.average_rows_inputs_per_task = 100
        metrics_filter.average_rows_outputs_per_task = 10
        metrics_filter.average_bytes_inputs_per_task = 1000 * MiB
        metrics_filter.average_bytes_outputs_per_task = 100 * MiB

        op_filter = create_mock_operator("Filter", MapOperator, metrics=metrics_filter)
        op_filter.per_task_resource_allocation.return_value = ExecutionResources(
            cpu=1.0, gpu=0.0
        )
        op_filter.incremental_resource_usage.return_value = ExecutionResources(
            memory=100
        )

        # Expand operator - expands 1:2
        metrics_expand = Mock()
        metrics_expand.average_total_task_completion_time_s = 4.0
        metrics_expand.average_rows_inputs_per_task = 10
        metrics_expand.average_rows_outputs_per_task = 20
        metrics_expand.average_bytes_inputs_per_task = 100 * MiB
        metrics_expand.average_bytes_outputs_per_task = 200 * MiB

        op_expand = create_mock_operator("Expand", MapOperator, metrics=metrics_expand)
        op_expand.per_task_resource_allocation.return_value = ExecutionResources(
            cpu=1.0, gpu=0.0
        )
        op_expand.incremental_resource_usage.return_value = ExecutionResources(
            memory=200
        )

        eligible_ops = [op_filter, op_expand]

        # Compute expansion ratios (required by _update_productivity_coefficients)
        op_expansion_ratios = {
            op: _estimate_byte_expansion_ratio(op) for op in eligible_ops
        }

        # Call _update_productivity_coefficients
        allocator._update_productivity_coefficients(eligible_ops, op_expansion_ratios)

        # Expected for Filter: normalized_bytes = 100 MiB * 2.0 (Expand ratio) = 200 MiB
        #   temporal_prod = 8.0s / 200 MiB = 0.04 s/byte
        #   alpha = 0.04 s/byte * 1.0 CPU = 0.04 CPU-s/byte
        alpha_filter, beta_filter, gamma_filter = allocator._op_productivity[op_filter]
        assert alpha_filter == pytest.approx(0.04, rel=1e-6)
        assert beta_filter == 0.0

        # Expected for Expand: normalized_bytes = 200 MiB * 1.0 (no downstream) = 200 MiB
        #   temporal_prod = 4.0s / 200 MiB = 0.02 s/byte
        #   alpha = 0.02 s/byte * 1.0 CPU = 0.02 CPU-s/byte
        alpha_expand, beta_expand, gamma_expand = allocator._op_productivity[op_expand]
        assert alpha_expand == pytest.approx(0.02, rel=1e-6)
        assert beta_expand == 0.0

    def test_update_productivity_coefficients_with_null_norm_factor(self):
        """Regression test for:

            - Pipeline: Read -> Map
            - Read has finished tasks with valid metrics
            - Map has submitted tasks but none have finished yet (metrics are None)
            - Read should gets alpha (productivity) computed correctly

        Before the fix

            - Bytes expansion ratio for Map would be null (with 0 being fallback)
            - Normalization factor derived from downstreams' bytes expansion
            - Upstream's productivity (alpha) multiplied by normalization factor
             becomes 0 even though it has valid metrics from finished tasks, and
             can compute productivity appropriately.
        """
        allocator = ThroughputBasedResourceAllocator(create_mock_resource_manager())

        # ReadFiles: has finished tasks with valid metrics
        metrics_read = Mock()
        metrics_read.average_total_task_completion_time_s = 4.0
        metrics_read.average_rows_inputs_per_task = 10
        metrics_read.average_rows_outputs_per_task = 1000
        metrics_read.average_bytes_inputs_per_task = 500  # Small input (file listing)
        metrics_read.average_bytes_outputs_per_task = 200 * MiB  # Large output (data)

        op_read = create_mock_operator("ReadFiles", MapOperator, metrics=metrics_read)
        op_read.per_task_resource_allocation.return_value = ExecutionResources(
            cpu=1.0, gpu=0.0
        )

        # Map: tasks submitted but none finished yet (all metrics are None)
        metrics_map = Mock()
        metrics_map.average_total_task_completion_time_s = None
        metrics_map.average_rows_inputs_per_task = None
        metrics_map.average_rows_outputs_per_task = None
        metrics_map.average_bytes_inputs_per_task = None
        metrics_map.average_bytes_outputs_per_task = None

        op_map = create_mock_operator("Map", MapOperator, metrics=metrics_map)
        op_map.per_task_resource_allocation.return_value = ExecutionResources(
            cpu=1.0, gpu=0.0
        )

        eligible_ops = [op_read, op_map]

        # Compute expansion ratios
        # ReadFiles: 200 MiB / 500 bytes = large expansion
        # Map: None / None -> returns None (no finished tasks)
        op_expansion_ratios = {
            op: _estimate_byte_expansion_ratio(op) for op in eligible_ops
        }

        # Verify Map's expansion ratio is None (no finished tasks)
        assert op_expansion_ratios[op_map] is None

        # Call _update_productivity_coefficients
        allocator._update_productivity_coefficients(eligible_ops, op_expansion_ratios)

        # Key assertion: ReadFiles should have non-zero alpha despite Map having
        # no finished tasks. The fix uses 1.0 as default expansion ratio for Map,
        # allowing ReadFiles to compute productivity based on its own metrics.
        alpha_read, beta_read, gamma_read = allocator._op_productivity[op_read]

        # Expected for ReadFiles:
        #   Map's expansion ratio defaults to 1.0 (not 0) since it has no data
        #   normalized_output = 200 MiB * 1.0 = 200 MiB
        #   temporal_prod = 4.0s / 200 MiB = 0.02 s/MiB
        #   alpha = 0.02 * 1.0 CPU = 0.02 CPU-s/MiB
        assert alpha_read > 0, (
            "ReadFiles should have non-zero productivity even when downstream "
            "operator (Map) has no finished tasks"
        )
        assert alpha_read == pytest.approx(0.02, rel=1e-6)
        assert beta_read == 0.0

        # Map should have zero productivity (no finished tasks to measure)
        alpha_map, beta_map, gamma_map = allocator._op_productivity[op_map]
        assert alpha_map == 0.0
        assert beta_map == 0.0

    def test_update_productivity_coefficients_with_shuffle(self):
        """Test that _update_productivity_coefficients calculates shuffle productivity
        appropriately.

        Note: Operators downstream from a shuffle cannot run until the shuffle
        completes, so only ops up to and including shuffle are eligible.
        """
        # Pipeline: A > Shuffle (> B not eligible until shuffle completes)

        allocator = ThroughputBasedResourceAllocator(create_mock_resource_manager())

        metrics_a = Mock()
        metrics_a.average_total_task_completion_time_s = 10.0
        metrics_a.average_rows_inputs_per_task = 100
        metrics_a.average_rows_outputs_per_task = 100
        metrics_a.average_bytes_inputs_per_task = 1000 * MiB
        metrics_a.average_bytes_outputs_per_task = 1000 * MiB

        op_a = create_mock_operator("OpA", MapOperator, metrics=metrics_a)
        op_a.per_task_resource_allocation.return_value = ExecutionResources(
            cpu=2.0, gpu=0.0
        )
        op_a.incremental_resource_usage.return_value = ExecutionResources(memory=1000)

        # Shuffle operator with real metrics (non-zero task time and CPU allocation)
        metrics_shuffle = Mock()
        metrics_shuffle.average_total_task_completion_time_s = 8.0
        metrics_shuffle.average_rows_inputs_per_task = 100
        metrics_shuffle.average_rows_outputs_per_task = 100
        metrics_shuffle.average_bytes_inputs_per_task = 1000 * MiB
        metrics_shuffle.average_bytes_outputs_per_task = 1000 * MiB

        op_shuffle = create_mock_operator(
            "Shuffle", AllToAllOperator, metrics=metrics_shuffle
        )
        op_shuffle.per_task_resource_allocation.return_value = ExecutionResources(
            cpu=1.0, gpu=0.0
        )
        op_shuffle.incremental_resource_usage.return_value = ExecutionResources(
            memory=500
        )

        # Operator B - downstream from shuffle, cannot run until shuffle completes
        metrics_b = Mock()
        metrics_b.average_total_task_completion_time_s = 5.0
        metrics_b.average_rows_inputs_per_task = 100
        metrics_b.average_rows_outputs_per_task = 50
        metrics_b.average_bytes_inputs_per_task = 1000 * MiB
        metrics_b.average_bytes_outputs_per_task = 500 * MiB

        op_b = create_mock_operator("OpB", MapOperator, metrics=metrics_b)
        op_b.per_task_resource_allocation.return_value = ExecutionResources(
            cpu=1.0, gpu=0.0
        )
        op_b.incremental_resource_usage.return_value = ExecutionResources(memory=500)

        # Only ops up to and including shuffle are eligible while shuffle is running.
        # B cannot run until shuffle completes.
        eligible_ops = [op_a, op_shuffle]

        # Compute expansion ratios (required by _update_productivity_coefficients)
        op_expansion_ratios = {
            op: _estimate_byte_expansion_ratio(op) for op in eligible_ops
        }

        # Call _update_productivity_coefficients
        allocator._update_productivity_coefficients(eligible_ops, op_expansion_ratios)

        # With no downstream ops beyond shuffle:
        # - Shuffle expansion ratio: 1000/1000 = 1.0
        # - A expansion ratio: 1000/1000 = 1.0
        #
        # Expected for A: normalized_bytes = 1000 MiB * 1.0 (shuffle ratio) = 1000 MiB
        #   temporal_prod = 10.0s / 1000 MiB = 0.01 s/MiB
        #   alpha = 0.01 s/MiB * 2.0 CPU = 0.02 CPU-s/MiB
        alpha_a, beta_a, gamma_a = allocator._op_productivity[op_a]
        assert alpha_a == pytest.approx(0.02, rel=1e-6)
        assert beta_a == 0.0

        # Shuffle gets non-zero productivity:
        # normalized_bytes = 1000 MiB * 1.0 (no downstream) = 1000 MiB
        # temporal_prod = 8.0s / 1000 MiB = 0.008 s/MiB
        # alpha = 0.008 s/MiB * 1.0 CPU = 0.008 CPU-s/MiB
        alpha_shuffle, beta_shuffle, gamma_shuffle = allocator._op_productivity[
            op_shuffle
        ]
        assert alpha_shuffle == pytest.approx(0.008, rel=1e-6)
        assert beta_shuffle == 0.0

        # B is not in eligible_ops (downstream from pending shuffle),
        # so it should have 0 productivity
        assert op_b not in allocator._op_productivity

    @pytest.mark.parametrize("shuffle_alpha", [0, 0.01])
    def test_shuffle_only_stage_allocation(self, shuffle_alpha, restore_data_context):
        """Test _allocate_water_filling when only a shuffle op remains (all map ops completed).

        When only shuffle operators remain in the eligible list (all map operators completed),
        the shuffle gets full baseline reservation since it has no productivity-based allocation.
        """
        # Create a realistic pipeline: InputDataBuffer -> MapOp -> MapOp
        o1 = InputDataBuffer(DataContext.get_current(), [])
        o2 = mock_map_op(o1, incremental_resource_usage=ExecutionResources(1, 0, 10))
        o3 = mock_map_op(o2, incremental_resource_usage=ExecutionResources(1, 0, 10))

        topo = build_streaming_topology(o3, ExecutionOptions())

        resource_manager = ResourceManager(
            topo, ExecutionOptions(), MagicMock(), DataContext.get_current()
        )
        resource_manager.get_op_usage = MagicMock(
            return_value=ExecutionResources.zero()
        )

        allocator = resource_manager._op_resource_allocator
        assert isinstance(allocator, ThroughputBasedResourceAllocator)

        # Test the scenario where only shuffle ops remain in the eligible list
        # This happens when all map operators have completed, leaving only shuffles
        metrics = Mock()
        metrics.average_bytes_inputs_per_task = 1000
        metrics.average_bytes_outputs_per_task = 1000
        metrics.average_rows_inputs_per_task = 10
        metrics.average_rows_outputs_per_task = 10

        shuffle_op = create_mock_operator("Shuffle", AllToAllOperator, metrics=metrics)

        eligible_ops = [shuffle_op]
        op_expansion_ratios = {shuffle_op: 1.0}
        productivity_rates = {shuffle_op: (shuffle_alpha, 0.0, 0.0)}
        limits = ExecutionResources(cpu=10, object_store_memory=1000)

        (
            allocatable_limits,
            target_rate,
            op_allocations,
        ) = allocator._allocate_water_filling(
            eligible_ops, productivity_rates, op_expansion_ratios, limits
        )

        # If productivity is 0
        #   - No running ops (all have zero productivity), so total_allocatable_fraction = 0
        #       - allocatable_limits = limits.scale(0) = 0 resources
        #       - target_rate = 0
        #   - Baseline reservation is limits.scale(1.0) = full limits
        if shuffle_alpha == 0.0:
            expected_target_rate = 0
            expected_allocatable_limits = ExecutionResources(
                cpu=0.0, gpu=0.0, object_store_memory=0, memory=0
            )
        else:
            expected_target_rate = 1000
            expected_allocatable_limits = limits

        assert allocatable_limits == expected_allocatable_limits
        assert target_rate == expected_target_rate

        # Blocking materializing ops get unlimited OS budget
        assert op_allocations[shuffle_op] == ExecutionResources(
            cpu=10.0,
            gpu=0.0,
            object_store_memory=float("inf"),
            memory=float("inf"),
        )

    def test_allocate_water_filling_running_shuffle_operator(
        self, restore_data_context
    ):
        """Test that shuffle operators get productivity-based allocation."""
        # Create a minimal ResourceManager to get an allocator instance
        o1 = InputDataBuffer(DataContext.get_current(), [])
        o2 = mock_map_op(o1, incremental_resource_usage=ExecutionResources(1, 0, 10))
        topo = build_streaming_topology(o2, ExecutionOptions())

        resource_manager = ResourceManager(
            topo, ExecutionOptions(), MagicMock(), DataContext.get_current()
        )
        allocator = resource_manager._op_resource_allocator

        map_metrics = Mock()
        map_metrics.average_bytes_inputs_per_task = 1000
        map_metrics.average_rows_inputs_per_task = 10
        map_metrics.average_bytes_outputs_per_task = 2000  # 2x expansion
        map_metrics.average_rows_outputs_per_task = 20

        # Shuffle doesn't expand data - it just redistributes
        shuffle_metrics = Mock()
        shuffle_metrics.average_bytes_inputs_per_task = 2000
        shuffle_metrics.average_rows_inputs_per_task = 20
        shuffle_metrics.average_bytes_outputs_per_task = 2000  # 1x (no expansion)
        shuffle_metrics.average_rows_outputs_per_task = 20

        map_op = create_mock_operator("Map", MapOperator, metrics=map_metrics)
        shuffle = create_mock_operator(
            "Shuffle", AllToAllOperator, metrics=shuffle_metrics
        )

        eligible_ops = [map_op, shuffle]
        op_expansion_ratios = {
            op: _estimate_byte_expansion_ratio(op) for op in eligible_ops
        }
        # Both operators have non-zero productivity (shuffle now reports metrics)
        productivity_rates = {
            map_op: (0.01, 0.0, 0.0),
            shuffle: (0.02, 0.0, 0.0),  # Shuffle with non-zero productivity
        }
        limits = ExecutionResources(cpu=10, object_store_memory=1000)

        (
            allocatable_limits,
            target_rate,
            op_allocations,
        ) = allocator._allocate_water_filling(
            eligible_ops, productivity_rates, op_expansion_ratios, limits
        )

        # With both ops having productivity, all resources are allocatable
        assert allocatable_limits == ExecutionResources(
            cpu=10.0, gpu=0.0, object_store_memory=1000, memory=0
        )

        # Target rate = 10.0 CPU / (0.01 + 0.02) = 333.33 MiB/s
        assert target_rate == pytest.approx(333.33, rel=0.01)

        # CPU allocation: map_op gets 333.33 * 0.01 = 3.33 CPU
        #                 shuffle gets 333.33 * 0.02 = 6.67 CPU
        # Object store allocation using cumulative algorithm:
        # map_op has 2x expansion (1000 -> 2000), shuffle has 1x (2000 -> 2000)
        # Cumulative weights: map_op = 1000 * 2 = 2000, shuffle = 2000 * 1 = 2000
        # Total weights = 4000
        total_obj_store = 1000
        total_weights = 2000.0 + 2000.0

        assert op_allocations[map_op] == ExecutionResources(
            cpu=target_rate * productivity_rates[map_op][0],
            gpu=0.0,
            object_store_memory=math.ceil(total_obj_store * 2000.0 / total_weights),
            memory=float("inf"),
        )

        # Blocking materializing ops get unlimited OS budget
        assert op_allocations[shuffle] == ExecutionResources(
            cpu=target_rate * productivity_rates[shuffle][0],
            gpu=0.0,
            object_store_memory=float("inf"),
            memory=float("inf"),
        )

    def test_shuffle_boundary_excludes_downstream_ops(self, restore_data_context):
        """Test that operators after a pending shuffle are excluded from budget allocation.

        The shuffle boundary logic ensures that operators downstream of a pending shuffle
        don't receive resource allocation until the shuffle completes. This prevents
        resource hoarding by upstream ops when shuffle is the bottleneck.

        Pipeline: InputDataBuffer -> MapBefore -> ShuffleOp -> MapAfter
        - When shuffle is pending: only MapBefore and ShuffleOp are eligible
        - After shuffle completes: MapAfter becomes eligible
        """
        # Build pipeline: InputDataBuffer -> MapBefore -> ShuffleOp -> MapAfter
        o1 = InputDataBuffer(DataContext.get_current(), [])
        map_before = mock_map_op(o1, name="MapBefore")
        shuffle_op = mock_shuffle_op(map_before, name="Shuffle")
        map_after = mock_map_op(shuffle_op, name="MapAfter")

        topo = build_streaming_topology(map_after, ExecutionOptions())

        resource_manager = ResourceManager(
            topo, ExecutionOptions(), MagicMock(), DataContext.get_current()
        )

        allocator = resource_manager._op_resource_allocator
        assert isinstance(allocator, ThroughputBasedResourceAllocator)

        # Test 1: With pending shuffle (completed() returns False),
        # only ops at or before shuffle should be eligible
        assert not shuffle_op.has_completed()  # Shuffle is pending by default

        eligible_ops = allocator._get_eligible_ops()

        # map_after should NOT be in eligible ops (it's after the pending shuffle)
        assert [map_before, shuffle_op] == eligible_ops

        # Test 2: After shuffle completes, downstream ops should become eligible
        # Mark shuffle as completed (removes the boundary)
        shuffle_op.has_completed = MagicMock(return_value=True)
        # Mark upstream ops as finished execution (they're no longer eligible)
        map_before.has_execution_finished = MagicMock(return_value=True)
        shuffle_op.has_execution_finished = MagicMock(return_value=True)

        eligible_ops_after = allocator._get_eligible_ops()

        # Only map_after should be eligible now
        assert [map_after] == eligible_ops_after

    def test_allocate_object_store(self):
        """Test _allocate_object_store method directly with new cumulative algorithm"""
        allocator = ThroughputBasedResourceAllocator(create_mock_resource_manager())

        # Test case 1: Three operators with different byte expansion ratios
        #
        # Pipeline:
        #   Compress (1000 > 500) ->
        #   Neutral (1000 > 1000) ->
        #   Expand (1000 > 2000)
        #
        # Base = first_op.average_bytes_inputs_per_task = 1000
        #
        # Compress: weight = 1000 * (500/1000) = 500
        # Neutral:  weight = 500 * (1000/500) = 1000
        # Expand:   weight = 1000 * (2000/1000) = 2000

        compress_metrics = Mock()
        compress_metrics.average_bytes_inputs_per_task = 1000
        compress_metrics.average_rows_inputs_per_task = 10
        compress_metrics.average_bytes_outputs_per_task = 500  # Contracts to 0.5x
        compress_metrics.average_rows_outputs_per_task = 10

        neutral_metrics = Mock()
        neutral_metrics.average_bytes_inputs_per_task = (
            1000  # Takes input from compress
        )
        neutral_metrics.average_rows_inputs_per_task = 10
        neutral_metrics.average_bytes_outputs_per_task = 1000  # Expands to 2x
        neutral_metrics.average_rows_outputs_per_task = 10

        expand_metrics = Mock()
        expand_metrics.average_bytes_inputs_per_task = 1000  # Takes input from neutral
        expand_metrics.average_rows_inputs_per_task = 10
        expand_metrics.average_bytes_outputs_per_task = 2000  # Expands to 2x
        expand_metrics.average_rows_outputs_per_task = 10

        compress_op = create_mock_operator(
            "Compress", MapOperator, metrics=compress_metrics
        )
        neutral_op = create_mock_operator(
            "Neutral", MapOperator, metrics=neutral_metrics
        )
        expand_op = create_mock_operator("Expand", MapOperator, metrics=expand_metrics)

        allocatable_ops = [compress_op, neutral_op, expand_op]
        total_object_store = 4 * GiB

        # Compute expansion ratios (required by _allocate_object_store)
        op_expansion_ratios = {
            op: _estimate_byte_expansion_ratio(op) for op in allocatable_ops
        }

        # Call _allocate_object_store
        allocations = allocator._allocate_object_store(
            allocatable_ops, total_object_store, op_expansion_ratios
        )

        # Verify allocations using new cumulative algorithm
        # Base = 1000
        # Compress: 1000 * (500/1000) = 500
        # Neutral:  500 * (1000/1000) = 500
        # Expand:   500 * (2000/1000) = 1000
        # Total weights: 500 + 500 + 1000 = 2000

        total_weights = 500.0 + 500.0 + 1000.0

        expected_compress = math.ceil(total_object_store * 500.0 / total_weights)
        expected_neutral = math.ceil(total_object_store * 500.0 / total_weights)
        expected_expand = math.ceil(total_object_store * 1000.0 / total_weights)

        assert len(allocations) == 3
        assert allocations[compress_op] == expected_compress
        assert allocations[neutral_op] == expected_neutral
        assert allocations[expand_op] == expected_expand

        # Verify proportional relationship
        # expand:neutral:compress = 1000:500:500 = 2:1:1
        assert (
            allocations[expand_op] > allocations[neutral_op] >= allocations[compress_op]
        )

        # Test case 2: Empty operators list
        empty_allocations = allocator._allocate_object_store([], total_object_store, {})
        assert empty_allocations == {}

        # Test case 3: Operators with no output yet (fallback to fair split)
        no_output_metrics = Mock()
        no_output_metrics.average_bytes_inputs_per_task = None
        no_output_metrics.average_rows_inputs_per_task = None
        no_output_metrics.average_bytes_outputs_per_task = None
        no_output_metrics.average_rows_outputs_per_task = None

        op1 = create_mock_operator("Op1", MapOperator, metrics=no_output_metrics)
        op2 = create_mock_operator("Op2", MapOperator, metrics=no_output_metrics)
        op3 = create_mock_operator("Op3", MapOperator, metrics=no_output_metrics)

        no_output_ops = [op1, op2, op3]

        # Compute expansion ratios for fallback test
        op_expansion_ratios = {
            op: _estimate_byte_expansion_ratio(op) for op in no_output_ops
        }

        fallback_allocations = allocator._allocate_object_store(
            no_output_ops, total_object_store, op_expansion_ratios
        )

        # With no outputs, weights should be 0, triggering fair split fallback
        # Each operator gets total_object_store / 3
        expected_fair_share = math.ceil(total_object_store / 3)

        assert len(fallback_allocations) == 3
        assert fallback_allocations[op1] == expected_fair_share
        assert fallback_allocations[op2] == expected_fair_share
        assert fallback_allocations[op3] == expected_fair_share

        # Test case 4: Single operator (1000 input -> 500 output, 0.5x compression)
        single_metrics = Mock()
        single_metrics.average_bytes_inputs_per_task = 1000
        single_metrics.average_rows_inputs_per_task = 10
        single_metrics.average_bytes_outputs_per_task = 500
        single_metrics.average_rows_outputs_per_task = 10

        single_op = create_mock_operator("Single", MapOperator, metrics=single_metrics)

        # Compute expansion ratios for single operator test
        op_expansion_ratios = {single_op: _estimate_byte_expansion_ratio(single_op)}

        single_allocations = allocator._allocate_object_store(
            [single_op], total_object_store, op_expansion_ratios
        )

        # Single operator: base = 1000, weight = 1000 * (500/1000) = 500
        # Gets total_object_store * (500/500) = total_object_store
        assert len(single_allocations) == 1
        assert single_allocations[single_op] == total_object_store

        # Test case 5: Verify cumulative multiplication with all expansions
        # Pipeline: A (100 -> 200) -> B (200 -> 600) -> C (600 -> 1200)
        a_metrics = Mock()
        a_metrics.average_bytes_inputs_per_task = 100
        a_metrics.average_bytes_outputs_per_task = 200  # 2x expansion
        a_metrics.average_rows_inputs_per_task = 10
        a_metrics.average_rows_outputs_per_task = 10

        b_metrics = Mock()
        b_metrics.average_bytes_inputs_per_task = 200
        b_metrics.average_bytes_outputs_per_task = 600  # 3x expansion
        b_metrics.average_rows_inputs_per_task = 10
        b_metrics.average_rows_outputs_per_task = 10

        c_metrics = Mock()
        c_metrics.average_bytes_inputs_per_task = 600
        c_metrics.average_bytes_outputs_per_task = 1200  # 2x expansion
        c_metrics.average_rows_inputs_per_task = 10
        c_metrics.average_rows_outputs_per_task = 10

        op_a = create_mock_operator("OpA", MapOperator, metrics=a_metrics)
        op_b = create_mock_operator("OpB", MapOperator, metrics=b_metrics)
        op_c = create_mock_operator("OpC", MapOperator, metrics=c_metrics)

        expansion_ops = [op_a, op_b, op_c]

        # Compute expansion ratios for expansion test
        op_expansion_ratios = {
            op: _estimate_byte_expansion_ratio(op) for op in expansion_ops
        }

        expansion_allocations = allocator._allocate_object_store(
            expansion_ops, total_object_store, op_expansion_ratios
        )

        # Base = 100
        # A: 100 * (200/100) = 200
        # B: 200 * (600/200) = 600
        # C: 600 * (1200/600) = 1200
        # Total: 200 + 600 + 1200 = 2000
        total_exp_weights = 200.0 + 600.0 + 1200.0

        expected_a = math.ceil(total_object_store * 200.0 / total_exp_weights)
        expected_b = math.ceil(total_object_store * 600.0 / total_exp_weights)
        expected_c = math.ceil(total_object_store * 1200.0 / total_exp_weights)

        assert expansion_allocations[op_a] == expected_a
        assert expansion_allocations[op_b] == expected_b
        assert expansion_allocations[op_c] == expected_c

        # Verify ratio: C:B:A = 1200:600:200 = 6:3:1
        assert (
            expansion_allocations[op_c]
            > expansion_allocations[op_b]
            > expansion_allocations[op_a]
        )

    def test_get_baseline_allocatable_ratios(self):
        """Test _get_baseline_allocatable_ratios for various operator counts.

        Baseline ratios are exponentially decreasing: [0.5, 0.25, 0.25] for 3 ops.
        This ensures upstream operators don't hog all resources before downstream starts.
        """
        from ray.data._internal.execution.throughput_based_resource_allocator import (
            _get_baseline_allocatable_ratios,
        )

        metrics = Mock()

        # Empty list returns [1.0]
        assert _get_baseline_allocatable_ratios([]) == [1.0]

        # Single operator gets 100%
        op1 = create_mock_operator("Op1", MapOperator, metrics=metrics)
        assert _get_baseline_allocatable_ratios([op1]) == [1.0]

        # Two operators: [0.5, 0.5] - first gets 50%, second gets 50%
        op2 = create_mock_operator("Op2", MapOperator, metrics=metrics)
        ratios_2 = _get_baseline_allocatable_ratios([op1, op2])
        assert ratios_2 == [0.5, 0.5]
        assert sum(ratios_2) == 1.0

        # Three operators: [0.5, 0.25, 0.25]
        op3 = create_mock_operator("Op3", MapOperator, metrics=metrics)
        ratios_3 = _get_baseline_allocatable_ratios([op1, op2, op3])
        assert ratios_3 == [0.5, 0.25, 0.25]
        assert sum(ratios_3) == 1.0

        # Five operators: [0.5, 0.25, 0.125, 0.0625, 0.0625]
        op4 = create_mock_operator("Op4", MapOperator, metrics=metrics)
        op5 = create_mock_operator("Op5", MapOperator, metrics=metrics)
        ratios_5 = _get_baseline_allocatable_ratios([op1, op2, op3, op4, op5])

        assert ratios_5 == [0.5, 0.25, 0.125, 0.0625, 0.0625]
        assert sum(ratios_5) == 1.0


class TestThroughputBasedResourceAllocatorE2E:
    """Integration tests for ``ThroughputBasedResourceAllocator`` with ``ResourceManager``."""

    @pytest.fixture(scope="function", autouse=True)
    def enable_throughput_based_resource_allocator(self, restore_data_context):
        DataContext.get_current().op_resource_reservation_enabled = True

        with patch(
            "ray.data._internal.execution.DEFAULT_USE_OP_RESOURCE_ALLOCATOR_VERSION",
            new="V2",
        ):
            yield

    def test_basic_allocation(self, restore_data_context):
        """Tests basic integration with ResourceManager"""
        o1 = InputDataBuffer(DataContext.get_current(), [])
        o2 = mock_map_op(o1, incremental_resource_usage=ExecutionResources(1, 0, 15))
        o3 = mock_map_op(o2, incremental_resource_usage=ExecutionResources(1, 0, 10))
        o4 = LimitOperator(1, o3, DataContext.get_current())

        # Set up mock metrics to get deterministic allocations
        # o2: 10s task time, 1000 bytes in, 2000 bytes out (2x expansion)
        o2_metrics = Mock()
        o2_metrics.average_total_task_completion_time_s = 10.0
        o2_metrics.average_bytes_inputs_per_task = 1000
        o2_metrics.average_rows_inputs_per_task = 10
        o2_metrics.average_bytes_outputs_per_task = 2000
        o2_metrics.average_rows_outputs_per_task = 20
        o2_metrics.num_tasks_submitted = 1
        o2._metrics = o2_metrics

        # o3: 5s task time, 2000 bytes in, 2000 bytes out (1x expansion)
        o3_metrics = Mock()
        o3_metrics.average_total_task_completion_time_s = 5.0
        o3_metrics.average_bytes_inputs_per_task = 2000
        o3_metrics.average_rows_inputs_per_task = 20
        o3_metrics.average_bytes_outputs_per_task = 2000
        o3_metrics.average_rows_outputs_per_task = 20
        o3_metrics.num_tasks_submitted = 1
        o3._metrics = o3_metrics

        topo = build_streaming_topology(o4, ExecutionOptions())

        global_limits = ExecutionResources(cpu=10, gpu=0, object_store_memory=1000)

        resource_manager = ResourceManager(
            topo, ExecutionOptions(), MagicMock(), DataContext.get_current()
        )
        resource_manager.get_op_usage = MagicMock(
            side_effect=lambda op, include_ineligible_downstream=False: ExecutionResources.zero()
        )

        assert resource_manager.op_resource_allocator_enabled()
        allocator = resource_manager._op_resource_allocator
        assert isinstance(allocator, ThroughputBasedResourceAllocator)

        allocator.update_budgets(limits=global_limits)

        # o1 (InputDataBuffer) and o4 (LimitOperator) should not be in allocations
        assert allocator.get_allocation(o1) is None
        assert allocator.get_allocation(o4) is None

        # Verify exact allocation values
        # o2: expansion_ratio = 2000/1000 = 2.0, normalized_output = 2000 * 1.0 (o3 ratio) = 2000
        #     temporal_prod = 10.0s / 2000 = 0.005 s/byte
        #     per_task_resource = 1 CPU (from incremental_resource_usage)
        #     alpha = 0.005 * 1 = 0.005 CPU-s/byte
        # o3: expansion_ratio = 2000/2000 = 1.0, normalized_output = 2000 * 1.0 = 2000
        #     temporal_prod = 5.0s / 2000 = 0.0025 s/byte
        #     alpha = 0.0025 * 1 = 0.0025 CPU-s/byte
        # Total alpha = 0.005 + 0.0025 = 0.0075
        # Target rate = 10.0 CPU / 0.0075 = 1333.33 bytes/s
        # o2 CPU allocation = 1333.33 * 0.005 = 6.67 CPU
        # o3 CPU allocation = 1333.33 * 0.0025 = 3.33 CPU

        o2_alloc = allocator.get_allocation(o2)
        o3_alloc = allocator.get_allocation(o3)

        # CPU allocations should sum to approximately total CPU
        total_cpu_alloc = o2_alloc.cpu + o3_alloc.cpu
        assert total_cpu_alloc == pytest.approx(10.0, rel=0.01)

        # o2 should get more CPU than o3 (ratio 2:1 based on productivity)
        assert o2_alloc.cpu == pytest.approx(2 * o3_alloc.cpu, rel=0.01)

        # Object store: o2 expansion=2.0, o3 expansion=1.0
        # Cumulative weights: o2=1000*2=2000, o3=2000*1=2000
        # Equal split expected
        assert o2_alloc.object_store_memory == pytest.approx(
            o3_alloc.object_store_memory, rel=0.01
        )

        # Budget = Allocation - Usage (usage is zero)
        o2_budget = allocator.get_budget(o2)
        o3_budget = allocator.get_budget(o3)

        assert o2_budget == o2_alloc
        assert o3_budget == o3_alloc

    def test_budget_and_task_submission(self, restore_data_context):
        """Tests
        - Budget calculation (allocation - usage)
        - That can_submit_new_task/get_output_budget act based on accounted budget
        """
        o1 = InputDataBuffer(DataContext.get_current(), [])
        o2 = mock_map_op(o1, incremental_resource_usage=ExecutionResources(2, 0, 100))

        # Set up mock metrics for deterministic allocations
        o2_metrics = Mock()
        o2_metrics.average_total_task_completion_time_s = 10.0
        o2_metrics.average_bytes_inputs_per_task = 1000
        o2_metrics.average_rows_inputs_per_task = 10
        o2_metrics.average_bytes_outputs_per_task = 2000
        o2_metrics.average_rows_outputs_per_task = 20
        o2_metrics.num_tasks_submitted = 0
        o2._metrics = o2_metrics

        topo = build_streaming_topology(o2, ExecutionOptions())
        global_limits = ExecutionResources(cpu=10, object_store_memory=1000)

        resource_manager = ResourceManager(
            topo, ExecutionOptions(), MagicMock(), DataContext.get_current()
        )

        resource_manager.get_op_usage = MagicMock(
            return_value=ExecutionResources.zero()
        )

        allocator = resource_manager._op_resource_allocator
        assert isinstance(allocator, ThroughputBasedResourceAllocator)

        allocator.update_budgets(limits=global_limits)

        # Before any tasks submitted, should be allowed (due to num_tasks_submitted == 0 check)
        assert allocator.can_submit_new_task(o2)

        # Single operator gets all resources
        allocation = allocator.get_allocation(o2)
        assert allocation == ExecutionResources(
            cpu=10.0,
            gpu=0.0,
            object_store_memory=1000,
            memory=float("inf"),
        )

        # Simulate tasks being submitted
        o2_metrics.num_tasks_submitted = 1

        # Budget == Allocation when usage is zero
        assert allocator.get_budget(o2) == allocation
        assert allocator.can_submit_new_task(o2)

        # get_output_budget returns object_store_memory from budget
        assert allocator.get_output_budget(o2) == allocation.object_store_memory
        assert allocator.get_output_budget(o1) is None  # InputDataBuffer has no budget

        # Verify proper budget calculation
        usage = ExecutionResources(cpu=5, object_store_memory=100)
        resource_manager.get_op_usage = MagicMock(
            side_effect=lambda op, include_ineligible_downstream=False: usage
            if op == o2
            else ExecutionResources.zero()
        )

        allocator.update_budgets(limits=global_limits)

        expected_budget = allocation.subtract(usage)

        assert allocator.get_budget(o2) == expected_budget
        assert allocator.can_submit_new_task(o2)

        # Budget clamped to 0 when usage exhausts allocation
        full_usage = ExecutionResources(cpu=10, object_store_memory=1000)
        resource_manager.get_op_usage = MagicMock(
            side_effect=lambda op, include_ineligible_downstream=False: full_usage
            if op == o2
            else ExecutionResources.zero()
        )

        allocator.update_budgets(limits=global_limits)

        # Budget = allocation (1000) - usage (1000) = 0
        assert allocator.get_budget(o2) == ExecutionResources(
            cpu=0, gpu=0, object_store_memory=0, memory=float("inf")
        )

        assert not allocator.can_submit_new_task(o2)

    def test_can_submit_new_task_under_allocation(self, restore_data_context):
        """Test can_submit_new_task edge case when allocation < incremental_usage.

        Scenario: Cluster has 10 CPUs, operator needs 6 CPUs per task.
        Due to baseline ratios (50% when downstream hasn't started), operator
        only gets 5 CPU allocation - less than the 6 needed per task.

        Despite under-allocation, the operator should be allowed to submit single task
        so long as it has a non-zero allocation AND it has non-zero budget remaining.

        This prevents deadlock during ramp-up.
        """
        o1 = InputDataBuffer(DataContext.get_current(), [])
        # Operator requires 6 CPU per task - more than its 50% baseline share
        o2 = mock_map_op(o1, incremental_resource_usage=ExecutionResources(6, 0, 100))
        o3 = mock_map_op(o2, incremental_resource_usage=ExecutionResources(6, 0, 100))

        # Set up mock metrics - only o2 is running (has outputs), o3 hasn't started
        o2_metrics = Mock()
        o2_metrics.average_total_task_completion_time_s = 10.0
        o2_metrics.average_bytes_inputs_per_task = 1000
        o2_metrics.average_rows_inputs_per_task = 10
        o2_metrics.average_bytes_outputs_per_task = 2000
        o2_metrics.average_rows_outputs_per_task = 20
        o2_metrics.num_tasks_submitted = 1
        o2._metrics = o2_metrics

        # o3 hasn't started yet (no outputs)
        o3_metrics = Mock()
        o3_metrics.average_total_task_completion_time_s = None
        o3_metrics.average_bytes_inputs_per_task = None
        o3_metrics.average_rows_inputs_per_task = None
        o3_metrics.average_bytes_outputs_per_task = None
        o3_metrics.average_rows_outputs_per_task = None
        o3_metrics.num_tasks_submitted = 1  # Submitted but no outputs yet
        o3._metrics = o3_metrics

        topo = build_streaming_topology(o3, ExecutionOptions())
        # Cluster has 10 CPUs - enough for one 6-CPU task, but baseline splits it
        global_limits = ExecutionResources(cpu=10, object_store_memory=1000)

        resource_manager = ResourceManager(
            topo, ExecutionOptions(), MagicMock(), DataContext.get_current()
        )
        resource_manager.get_op_usage = MagicMock(
            return_value=ExecutionResources.zero()
        )

        allocator = resource_manager._op_resource_allocator
        assert isinstance(allocator, ThroughputBasedResourceAllocator)

        allocator.update_budgets(limits=global_limits)

        # o2 gets 50% baseline allocation (5 CPU) since o3 hasn't started
        # o2's incremental_usage is 6 CPU, but allocation is only 5 CPU
        o2_allocation = allocator.get_allocation(o2)
        o2_budget = allocator.get_budget(o2)

        # Verify under-allocation scenario: allocation (5) < incremental_usage (6)
        assert o2_allocation.cpu == 5.0  # 50% of 10 CPUs
        assert o2_allocation.cpu < 6.0  # Less than per-task requirement

        # Despite under-allocation, can_submit_new_task should return True
        # because: allocation < incremental AND budget > 0
        assert o2_budget.cpu == 5.0  # Full allocation available as budget
        assert allocator.can_submit_new_task(o2)

        # Now simulate that o2 has used its entire allocation running a task
        resource_manager.get_op_usage = MagicMock(
            side_effect=lambda op, include_ineligible_downstream=False: ExecutionResources(
                cpu=5, object_store_memory=100
            )
            if op == o2
            else ExecutionResources.zero()
        )
        allocator.update_budgets(limits=global_limits)

        # After using budget, o2 should have 0 CPU budget remaining
        o2_budget_after = allocator.get_budget(o2)
        assert o2_budget_after.cpu == 0

        # Now can_submit_new_task should return False (no budget remaining)
        assert not allocator.can_submit_new_task(o2)

    def test_gpu_operator_allocation(self, restore_data_context):
        """Tests allocation for GPU operators."""
        o1 = InputDataBuffer(DataContext.get_current(), [])

        # Non-GPU operator (CPU only)
        o2 = mock_map_op(o1, incremental_resource_usage=ExecutionResources(1, 0, 10))

        # Set up mock metrics for o2
        o2_metrics = Mock()
        o2_metrics.average_total_task_completion_time_s = 10.0
        o2_metrics.average_bytes_inputs_per_task = 1000
        o2_metrics.average_rows_inputs_per_task = 10
        o2_metrics.average_bytes_outputs_per_task = 2000
        o2_metrics.average_rows_outputs_per_task = 20
        o2_metrics.num_tasks_submitted = 1
        o2._metrics = o2_metrics

        # GPU operator
        o3 = mock_map_op(
            o2,
            ray_remote_args={"num_gpus": 1},
            incremental_resource_usage=ExecutionResources(0, 1, 10),
        )

        # Set up mock metrics for o3 (GPU operator)
        o3_metrics = Mock()
        o3_metrics.average_total_task_completion_time_s = 5.0
        o3_metrics.average_bytes_inputs_per_task = 2000
        o3_metrics.average_rows_inputs_per_task = 20
        o3_metrics.average_bytes_outputs_per_task = 2000
        o3_metrics.average_rows_outputs_per_task = 20
        o3_metrics.num_tasks_submitted = 1
        o3._metrics = o3_metrics

        topo = build_streaming_topology(o3, ExecutionOptions())

        global_limits = ExecutionResources(cpu=8, gpu=4, object_store_memory=1000)
        op_usages = {
            o1: ExecutionResources.zero(),
            o2: ExecutionResources(cpu=1),
            o3: ExecutionResources(gpu=1),  # GPU op using 1 GPU
        }

        resource_manager = ResourceManager(
            topo, ExecutionOptions(), MagicMock(), DataContext.get_current()
        )
        resource_manager.get_op_usage = MagicMock(
            side_effect=lambda op, include_ineligible_downstream=False: op_usages[op]
        )

        allocator = resource_manager._op_resource_allocator
        allocator.update_budgets(limits=global_limits)

        # Both o2 and o3 should have allocations
        o2_allocation = allocator.get_allocation(o2)
        o3_allocation = allocator.get_allocation(o3)

        assert o2_allocation is not None
        assert o3_allocation is not None

        # o2 is CPU-only, o3 is GPU operator - verify they have positive allocations
        assert o2_allocation.cpu > 0
        assert o3_allocation.gpu >= 0

        # Verify budget = allocation - usage for o3
        expected_o3_budget = ExecutionResources(
            cpu=max(0.0, o3_allocation.cpu - op_usages[o3].cpu),
            gpu=max(0.0, o3_allocation.gpu - op_usages[o3].gpu),
            object_store_memory=max(
                0, o3_allocation.object_store_memory - op_usages[o3].object_store_memory
            ),
            memory=float("inf"),
        )

        assert allocator.get_budget(o3) == expected_o3_budget

    def test_only_handle_eligible_ops(self, restore_data_context):
        """Test that only non-completed map ops are handled."""

        input_refs = make_ref_bundles([[x] for x in range(1)])
        o1 = InputDataBuffer(DataContext.get_current(), input_refs)
        o2 = mock_map_op(o1)
        o3 = LimitOperator(1, o2, DataContext.get_current())
        topo = build_streaming_topology(o3, ExecutionOptions())

        resource_manager = ResourceManager(
            topo, ExecutionOptions(), MagicMock(), DataContext.get_current()
        )
        resource_manager.get_op_usage = MagicMock(
            return_value=ExecutionResources.zero()
        )

        assert resource_manager.op_resource_allocator_enabled()
        allocator = resource_manager._op_resource_allocator
        assert isinstance(allocator, ThroughputBasedResourceAllocator)

        global_limits = ExecutionResources.zero()
        allocator.update_budgets(limits=global_limits)

        # InputDataBuffer and LimitOperator should not be in allocations
        assert o1 not in allocator._op_allocations
        assert o3 not in allocator._op_allocations
        # MapOperator should have allocation
        assert o2 in allocator._op_allocations

        # Mark o2 as completed
        o2.mark_execution_finished()
        allocator.update_budgets(limits=global_limits)

        # After completion, o2 should no longer be in allocations
        # (allocations dict is replaced entirely on each update)
        assert o2 not in allocator._op_allocations

    def test_max_task_output_bytes_to_read(self, restore_data_context):
        """Test max_task_output_bytes_to_read returns correct values."""
        o1 = InputDataBuffer(DataContext.get_current(), [])
        o2 = mock_map_op(o1, incremental_resource_usage=ExecutionResources(1, 0, 10))
        # Shuffle operator gets infinite object store budget
        o3 = mock_shuffle_op(o2, name="Shuffle")

        topo = build_streaming_topology(o3, ExecutionOptions())

        resource_manager = ResourceManager(
            topo, ExecutionOptions(), MagicMock(), DataContext.get_current()
        )
        resource_manager.get_op_usage = MagicMock(
            return_value=ExecutionResources.zero()
        )

        allocator = resource_manager._op_resource_allocator

        global_limits = ExecutionResources(cpu=10, object_store_memory=1000)

        allocator.update_budgets(limits=global_limits)

        # For operators not in budgets (InputDataBuffer), should return None
        assert allocator.max_task_output_bytes_to_read(o1) is None

        # o2's downstream hash-shuffle is *eligible*, hence it gets finite budget
        assert allocator.max_task_output_bytes_to_read(o2) == 500

        # Shuffle operator itself is blocking materializing, gets infinite budget
        topo[o3].output_queue._num_blocks = 5
        assert allocator.max_task_output_bytes_to_read(o3) == sys.maxsize

    def test_all_to_all_downstream_gets_infinite_budget(self, restore_data_context):
        """Test that operators before an AllToAllOperator get infinite object store budget.

        AllToAllOperator has throttling_disabled=True, making it ineligible and
        since it's also a blocking, materializing op, means that first eligible
        upstream operator gets infinite object store budget.
        """
        o1 = InputDataBuffer(DataContext.get_current(), [])
        o2 = mock_map_op(o1, incremental_resource_usage=ExecutionResources(1, 0, 10))
        o3 = mock_all_to_all_op(o2, name="Sort")
        o4 = mock_map_op(o3, incremental_resource_usage=ExecutionResources(1, 0, 10))

        topo = build_streaming_topology(o4, ExecutionOptions())

        resource_manager = ResourceManager(
            topo, ExecutionOptions(), MagicMock(), DataContext.get_current()
        )
        resource_manager.get_op_usage = MagicMock(
            return_value=ExecutionResources.zero()
        )

        allocator = resource_manager._op_resource_allocator

        global_limits = ExecutionResources(cpu=10, object_store_memory=1000)
        allocator.update_budgets(limits=global_limits)

        # o2 has downstream ineligible `AllToAllOperator`
        # so it gets infinite object store budget
        topo[o2].output_queue._num_blocks = 5
        assert allocator.max_task_output_bytes_to_read(o2) == sys.maxsize

        # o3 (AllToAllOperator) is ineligible (throttling_disabled=True),
        # so it's not in the budgets and returns None
        assert allocator.max_task_output_bytes_to_read(o3) is None

        # o4 is excluded from eligible ops (after pending blocking materializing op)
        assert allocator.max_task_output_bytes_to_read(o4) is None

    def test_complex_graph_union(self, restore_data_context):
        """Test allocator with union operator (multiple inputs)."""
        o1 = InputDataBuffer(DataContext.get_current(), [])
        o2 = mock_map_op(o1, incremental_resource_usage=ExecutionResources(1, 0, 10))

        o3 = InputDataBuffer(DataContext.get_current(), [])
        o4 = mock_map_op(o3, incremental_resource_usage=ExecutionResources(1, 0, 10))

        o5 = mock_union_op(
            [o2, o4], incremental_resource_usage=ExecutionResources(1, 0, 20)
        )

        topo = build_streaming_topology(o5, ExecutionOptions())

        resource_manager = ResourceManager(
            topo, ExecutionOptions(), MagicMock(), DataContext.get_current()
        )

        op_usages = {op: ExecutionResources.zero() for op in [o1, o2, o3, o4, o5]}
        resource_manager.get_op_usage = MagicMock(
            side_effect=lambda op, include_ineligible_downstream=False: op_usages[op]
        )

        allocator = resource_manager._op_resource_allocator
        assert isinstance(allocator, ThroughputBasedResourceAllocator)

        global_limits = ExecutionResources(cpu=16, object_store_memory=2000)
        allocator.update_budgets(limits=global_limits)

        # Verify eligible operators have budgets
        # InputDataBuffers should not have budgets
        assert allocator.get_budget(o1) is None
        assert allocator.get_budget(o3) is None

        # Map and Union operators should have budgets
        assert allocator.get_budget(o2) is not None
        assert allocator.get_budget(o4) is not None
        assert allocator.get_budget(o5) is not None

    def test_completed_ops_are_excluded(self, restore_data_context):
        """Test that completed operators are properly excluded from allocation."""
        o1 = InputDataBuffer(DataContext.get_current(), [])
        o2 = mock_map_op(o1, incremental_resource_usage=ExecutionResources(1, 0, 10))
        o3 = mock_map_op(o2, incremental_resource_usage=ExecutionResources(1, 0, 10))
        o4 = mock_map_op(o3, incremental_resource_usage=ExecutionResources(1, 0, 10))

        topo = build_streaming_topology(o4, ExecutionOptions())

        op_usages = {
            o1: ExecutionResources.zero(),
            o2: ExecutionResources(
                cpu=2, object_store_memory=50
            ),  # Still using resources
            o3: ExecutionResources.zero(),
            o4: ExecutionResources.zero(),
        }

        resource_manager = ResourceManager(
            topo, ExecutionOptions(), MagicMock(), DataContext.get_current()
        )
        resource_manager.get_op_usage = MagicMock(
            side_effect=lambda op, include_ineligible_downstream=False: op_usages[op]
        )

        allocator = resource_manager._op_resource_allocator

        global_limits = ExecutionResources(cpu=10, object_store_memory=500)

        allocator.update_budgets(limits=global_limits)

        # InputDataBuffer should not have a budget (not eligible)
        assert allocator.get_budget(o1) is None

        # Active operators should have budgets
        assert allocator.get_budget(o2) is not None
        assert allocator.get_budget(o3) is not None
        assert allocator.get_budget(o4) is not None

        # Mark o2 as completed
        o2.mark_execution_finished()

        allocator.update_budgets(limits=global_limits)

        # Completed operator should no longer have a budget
        assert allocator.get_budget(o2) is None
        # Still-active operators should still have budgets
        assert allocator.get_budget(o3) is not None
        assert allocator.get_budget(o4) is not None


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
