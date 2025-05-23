import asyncio
import threading
import time

import pyarrow as pa
import pytest

import ray
from ray.data._internal.compute import (
    ActorPoolStrategy,
    ComputeStrategy,
    TaskPoolStrategy,
)
from ray.data._internal.execution.interfaces import ExecutionResources, RefBundle
from ray.data._internal.execution.operators.input_data_buffer import InputDataBuffer
from ray.data._internal.execution.operators.map_operator import MapOperator
from ray.data._internal.execution.operators.map_transformer import (
    BlockMapTransformFn,
    MapTransformer,
)
from ray.data._internal.execution.streaming_executor import StreamingExecutor
from ray.data.block import BlockAccessor
from ray.tests.conftest import *  # noqa  # noqa
from ray.tests.conftest import wait_for_condition


def test_removed_nodes_not_added_back(ray_start_cluster):
    """Test that a dataset with actor pools can finish, when some
    nodes in the cluster are removed and not added back."""
    cluster = ray_start_cluster
    cluster.add_node(num_cpus=0)
    ray.init()

    @ray.remote(num_cpus=0)
    class Signal:
        def __init__(self):
            self._num_alive_actors = 0
            self._nodes_removed = False

        async def notify_actor_alive(self):
            self._num_alive_actors += 1

        async def wait_for_actors_alive(self, value):
            while self._num_alive_actors != value:
                await asyncio.sleep(0.01)

        async def notify_nodes_removed(self):
            self._nodes_removed = True

        async def wait_for_nodes_removed(self):
            while not self._nodes_removed:
                await asyncio.sleep(0.01)

    # Create the signal actor on the head node.
    signal_actor = Signal.remote()

    num_nodes = 4
    nodes = []
    for _ in range(num_nodes):
        nodes.append(cluster.add_node(num_cpus=10, num_gpus=1))
    cluster.wait_for_nodes()

    num_items = 100

    class MyUDF:
        def __init__(self, signal_actor):
            self._signal_actor = signal_actor
            self._signal_sent = False

        def __call__(self, batch):
            if not self._signal_sent:
                self._signal_actor.notify_actor_alive.remote()
                # Wait for the driver to remove nodes. This makes sure all
                # actors are running tasks when removing nodes.
                ray.get(self._signal_actor.wait_for_nodes_removed.remote())
                self._signal_sent = True
            time.sleep(0.01)
            return batch

    res = []

    def run_dataset():
        nonlocal res

        ds = ray.data.range(num_items, override_num_blocks=num_items)
        ds = ds.map_batches(
            MyUDF,
            fn_constructor_args=[signal_actor],
            concurrency=num_nodes,
            batch_size=1,
            num_gpus=1,
        )
        res = ds.take_all()

    thread = threading.Thread(target=run_dataset)
    thread.start()

    # Wait for all actors to start, then remove some nodes.
    ray.get(signal_actor.wait_for_actors_alive.remote(num_nodes))
    print("Removing nodes")
    nodes_to_remove = nodes[-num_nodes // 2 :]
    for node in nodes_to_remove:
        cluster.remove_node(node)
    ray.get(signal_actor.notify_nodes_removed.remote())

    thread.join()
    assert sorted(res, key=lambda x: x["id"]) == [{"id": i} for i in range(num_items)]


@pytest.mark.parametrize(
    "compute", [TaskPoolStrategy(), ActorPoolStrategy(size=1)], ids=["tasks", "actors"]
)
def test_map_operator_counts_lineage_reconstruction_tasks(
    ray_start_cluster_enabled, compute: ComputeStrategy
):
    data_context = ray.data.DataContext.get_current()

    # Create a cluster with a head node and a single worker node.
    cluster = ray_start_cluster_enabled
    cluster.add_node(resources={"head": 1})
    ray.init(address=cluster.address)
    worker = cluster.add_node(resources={"worker": 1})

    # Create an input data operator with a single block as input.
    block = pa.Table.from_pylist([{"data": "\x00" * 128 * 1024 * 1024}])
    block_ref = ray.put(block)
    metadata = BlockAccessor.for_block(block).get_metadata()
    bundle = RefBundle([(block_ref, metadata)], owns_blocks=False)
    input_op = InputDataBuffer(data_context, [bundle])

    # Create a signal actor so the map only finishes when we want it to.
    @ray.remote(num_cpus=0, resources={"head": 1})
    class Signal:
        def __init__(self, is_map_blocked: bool):
            self._is_map_blocked = is_map_blocked

        def block_map(self):
            print("Blocking map function")
            self._is_map_blocked = True

        def unblock_map(self):
            print("Unblocking map function")
            self._is_map_blocked = False

        def is_map_blocked(self) -> bool:
            return self._is_map_blocked

    # Start with the transform function unblocked.
    signal = Signal.remote(False)

    def block_fn(block, _):
        print("Entering block function")

        while ray.get(signal.is_map_blocked.remote()):
            print("Waiting for map to be unblocked")
            time.sleep(0.1)

        print("Exiting block function")
        return block

    transform_fns = [BlockMapTransformFn(block_fn)]
    map_transformer = MapTransformer(transform_fns)
    map_op = MapOperator.create(
        map_transformer,
        input_op,
        data_context,
        compute_strategy=compute,
        ray_remote_args={"resources": {"worker": 1, "num_cpus": 0}},
    )

    output_bundles = []
    executor = StreamingExecutor(data_context)
    for bundle in executor.execute(map_op):
        output_bundles.append(bundle)

    assert map_op.extra_resource_usage() == ExecutionResources.zero()

    # Remove the node to trigger lineage reconstruction.
    ray.get(signal.block_map.remote())
    cluster.remove_node(worker)
    wait_for_condition(lambda: map_op.extra_resource_usage().cpu == 1, timeout=10)

    # Re-add the node and unblock the map function.
    ray.get(signal.unblock_map.remote())
    cluster.add_node(resources={"worker": 1})
    wait_for_condition(
        lambda: map_op.extra_resource_usage() == ExecutionResources.zero(), timeout=10
    )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
