from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

import ray
from ray.anyscale.data._internal.execution.operators.actor_pool_map_operator import (
    ActorPoolMapOperator,
)
from ray.data._internal.execution.interfaces import ExecutionResources
from ray.data._internal.execution.operators.input_data_buffer import InputDataBuffer
from ray.data.tests.conftest import *  # noqa: F403


@dataclass(frozen=True)
class MinMaxResourceUsageBoundsTestCase:
    min_size: int
    max_size: int
    obj_store_mem_max_pending_output_per_task: int
    expected_min_resource_usage_bound: ExecutionResources
    expected_max_resource_usage_bound: ExecutionResources
    max_tasks_in_flight: int = 4
    max_concurrency: int = 1


@pytest.mark.parametrize(
    "case",
    [
        # Fixed-size pool.
        MinMaxResourceUsageBoundsTestCase(
            min_size=2,
            max_size=2,
            obj_store_mem_max_pending_output_per_task=1,
            expected_min_resource_usage_bound=ExecutionResources(
                cpu=2, object_store_memory=2
            ),
            expected_max_resource_usage_bound=ExecutionResources(
                cpu=2, object_store_memory=2
            ),
        ),
        # Autoscaling pool.
        MinMaxResourceUsageBoundsTestCase(
            min_size=1,
            max_size=2,
            obj_store_mem_max_pending_output_per_task=1,
            expected_min_resource_usage_bound=ExecutionResources(
                cpu=1, object_store_memory=1
            ),
            expected_max_resource_usage_bound=ExecutionResources(
                cpu=2, object_store_memory=2
            ),
        ),
        # Unbounded pool.
        MinMaxResourceUsageBoundsTestCase(
            min_size=1,
            max_size=None,
            obj_store_mem_max_pending_output_per_task=1,
            expected_min_resource_usage_bound=ExecutionResources(
                cpu=1, object_store_memory=1
            ),
            expected_max_resource_usage_bound=ExecutionResources.for_limits(),
        ),
        # Multi-threaded pool.
        MinMaxResourceUsageBoundsTestCase(
            min_size=1,
            max_size=1,
            obj_store_mem_max_pending_output_per_task=1,
            max_concurrency=2,
            max_tasks_in_flight=4,
            expected_min_resource_usage_bound=ExecutionResources(
                cpu=1, object_store_memory=1
            ),
            expected_max_resource_usage_bound=ExecutionResources(
                cpu=1, object_store_memory=1 * 2
            ),
        ),
    ],
    ids=[
        "fixed-size-pool",
        "autoscaling-pool",
        "unbounded-pool",
        "multi-threaded-pool",
    ],
)
def test_min_max_resource_requirements(
    case, ray_start_regular_shared, restore_data_context
):
    data_context = ray.data.DataContext.get_current()
    op = ActorPoolMapOperator(
        map_transformer=MagicMock(),
        input_op=InputDataBuffer(data_context, input_data=MagicMock()),
        data_context=data_context,
        target_max_block_size=None,
        compute_strategy=ray.data.ActorPoolStrategy(
            min_size=case.min_size,
            max_size=case.max_size,
            max_tasks_in_flight_per_actor=case.max_tasks_in_flight,
        ),
        ray_remote_args={
            "num_cpus": 1,
            "max_concurrency": case.max_concurrency,
        },
    )
    op._metrics = MagicMock(
        obj_store_mem_max_pending_output_per_task=case.obj_store_mem_max_pending_output_per_task
    )

    (
        min_resource_usage_bound,
        max_resource_usage_bound,
    ) = op.min_max_resource_requirements()

    assert (
        min_resource_usage_bound == case.expected_min_resource_usage_bound
        and max_resource_usage_bound == case.expected_max_resource_usage_bound
    )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
