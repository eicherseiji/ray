import itertools
import sys
from unittest.mock import MagicMock, patch

import pyarrow as pa
import pytest
from pyarrow import compute as pac

import ray
from ray._private.arrow_utils import get_pyarrow_version
from ray.anyscale.data._internal.block import OptimizedTableBlockMixin
from ray.anyscale.data.aggregate_vectorized import (
    MIN_PYARROW_VERSION_VECTORIZED_AGGREGATIONS,
)
from ray.core.generated import autoscaler_pb2
from ray.data._internal.execution.interfaces import ExecutionResources, PhysicalOperator
from ray.data._internal.execution.operators.hash_aggregate import (
    HashAggregateOperator,
)
from ray.data._internal.logical.interfaces import LogicalOperator
from ray.data._internal.tensor_extensions.arrow import _convert_to_pyarrow_native_array
from ray.data.aggregate import (
    AbsMax,
    Count,
    Max,
    Mean,
    Min,
    Quantile,
    Std,
    Sum,
    Unique,
)
from ray.data.context import DataContext
from ray.tests.conftest import *  # noqa  # noqa


@pytest.mark.parametrize(
    "agg_cls",
    [
        Count,
        Min,
        Max,
        Sum,
        Mean,
        Std,
        Quantile,
        AbsMax,
        Unique,
    ],
)
@pytest.mark.parametrize("ignore_nulls", [True, False])
def test_null_safe_aggregation_protocol(agg_cls, ignore_nulls):
    """This test verifies that all aggregation implementations
    properly implement aggregation protocol
    """

    col = pa.array([0, 1, 2, None])
    t = pa.table([col], names=["A"])

    pac_method = _map_to_pa_compute_method(agg_cls)

    expected = pac_method(col, ignore_nulls)

    agg_kwargs = {"ignore_nulls": ignore_nulls}

    agg = agg_cls(on="A", **agg_kwargs)

    # Step 1: Initialize accumulator
    # Step 2: Partially aggregate individual rows
    accumulators = [
        agg.accumulate_block(agg.init(None), t.slice(i, 1)) for i in range(t.num_rows)
    ]

    # NOTE: This test intentionally permutes all accumulators to verify
    #       that combination is associative
    for permuted_accumulators in itertools.permutations(accumulators):
        # Compose accumulators column
        accumulators_column = _convert_to_pyarrow_native_array(
            list(permuted_accumulators),
            "column",
        )

        # Step 3: Combine accumulators holding partial aggregations
        #         into final result
        combined = OptimizedTableBlockMixin._combine_column(agg, accumulators_column)
        # Step 4: Finalize
        res = agg.finalize(combined)

        # Assert that combining aggregations is an associative operation,
        # ie invariant of the order of combining partial aggregations
        if isinstance(agg, Unique):
            if get_pyarrow_version() >= MIN_PYARROW_VERSION_VECTORIZED_AGGREGATIONS:
                res_set = set(res.to_pylist())
            else:
                # NOTE: Default aggregations return set as is
                res_set = res

            # NOTE: Pyarrow's `unique` method doesn't provide `ignore_nulls`
            #       param hence we have to filter out manually
            if ignore_nulls:
                expected = expected.drop_null()

            assert set(expected.to_pylist()) == res_set, permuted_accumulators
        else:
            assert expected == res, permuted_accumulators


def _map_to_pa_compute_method(agg_cls: type):
    _map = {
        Count: lambda col, ignore_nulls: pac.count(
            col, mode=("only_valid" if ignore_nulls else "all")
        ).as_py(),
        Min: lambda col, ignore_nulls: pac.min(col, skip_nulls=ignore_nulls).as_py(),
        Max: lambda col, ignore_nulls: pac.max(col, skip_nulls=ignore_nulls).as_py(),
        Sum: lambda col, ignore_nulls: pac.sum(col, skip_nulls=ignore_nulls).as_py(),
        Mean: lambda col, ignore_nulls: pac.mean(col, skip_nulls=ignore_nulls).as_py(),
        Std: lambda col, ignore_nulls: pac.stddev(
            col, ddof=1, skip_nulls=ignore_nulls
        ).as_py(),
        Quantile: lambda col, ignore_nulls: pac.quantile(
            col, q=0.5, skip_nulls=ignore_nulls
        )[0].as_py(),
        AbsMax: lambda col, ignore_nulls: pac.max(
            pac.abs(col), skip_nulls=ignore_nulls
        ).as_py(),
        Unique: lambda col, ignore_nulls: pac.unique(col),
    }

    return _map[agg_cls]


def test_default_shuffle_aggregator_args(ray_start_regular_shared):
    from ray.data.block import BlockMetadata

    # Create a mock logical operator with proper infer_metadata method
    logical_op_mock = MagicMock(LogicalOperator)
    logical_op_mock.estimated_num_outputs.return_value = 1
    # Add infer_metadata method that returns BlockMetadata with size_bytes
    logical_op_mock.infer_metadata.return_value = BlockMetadata(
        num_rows=None,
        size_bytes=1024 * 1024,  # 1MB for testing
        exec_stats=None,
        input_files=None,
    )

    parent_op_mock = MagicMock(PhysicalOperator)
    parent_op_mock._output_dependencies = []
    parent_op_mock._logical_operators = [logical_op_mock]

    with patch.object(
        ray._private.state.state,
        "get_cluster_config",
        return_value=autoscaler_pb2.ClusterConfig(),
    ):
        op = HashAggregateOperator(
            input_op=parent_op_mock,
            data_context=DataContext.get_current(),
            key_columns=("id",),
            aggregation_fns=tuple(),
            num_partitions=16,
        )

        # - 1 partition per aggregator
        # - No partition size hint
        args = op._get_default_aggregator_ray_remote_args(
            num_partitions=16,
            num_aggregators=16,
            total_available_cluster_resources=ExecutionResources(
                cpu=8.0, memory=16 * 1024 * 1024 * 1024
            ),
            estimated_dataset_bytes=1024 * 1024,
        )

        assert {
            "num_cpus": 0.0,
            "memory": 131072,
            "scheduling_strategy": "SPREAD",
            "allow_out_of_order_execution": True,
        } == args

        # - 4 partitions per aggregator
        # - Larger dataset
        args = op._get_default_aggregator_ray_remote_args(
            num_partitions=64,
            num_aggregators=16,
            total_available_cluster_resources=ExecutionResources(
                cpu=8.0, memory=16 * 1024 * 1024 * 1024
            ),
            estimated_dataset_bytes=1024 * 1024 * 1024,  # 1GB for testing
        )

        # We'll update this assertion once we see what the actual value is
        assert {
            "num_cpus": 0.02,
            "memory": 83886080,
            "scheduling_strategy": "SPREAD",
            "allow_out_of_order_execution": True,
        } == args


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
