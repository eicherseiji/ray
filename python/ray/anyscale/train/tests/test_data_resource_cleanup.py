import sys
from unittest.mock import MagicMock, create_autospec

import pytest

import ray.data
from ray.data import DataContext
from ray.data._internal.iterator.stream_split_iterator import (
    SplitCoordinator,
    _DatasetWrapper,
)

from ray.anyscale.train._internal.callbacks.datasets import (
    AnyscaleDatasetShardProvider,
    DatasetsSetupCallback,
)
from ray.anyscale.train._internal.data_integration.dataset_manager import DatasetManager
from ray.anyscale.train.tests.test_dataset_manager import (
    get_dataset_shard_for_all_workers,
)
from ray.train.v2.tests.test_worker_group import _default_worker_group_context
from ray.train.v2.tests.util import (
    create_dummy_run_context,
)


def test_after_worker_group_shutdown():
    """Test that the DatasetsSetupCallback calls cleanup on the dataset shard provider on after_worker_group_shutdown"""
    callback = DatasetsSetupCallback(train_run_context=create_dummy_run_context())
    shard_provider = create_autospec(AnyscaleDatasetShardProvider)
    callback._dataset_shard_provider = shard_provider

    callback.after_worker_group_shutdown(
        worker_group_context=_default_worker_group_context()
    )
    shard_provider.shutdown_data_executors.assert_called_once()


def test_after_worker_group_abort():
    """Test that the DatasetsSetupCallback calls cleanup on the dataset shard provider on after_worker_group_abort"""
    callback = DatasetsSetupCallback(train_run_context=create_dummy_run_context())
    shard_provider = create_autospec(AnyscaleDatasetShardProvider)

    callback._dataset_shard_provider = shard_provider

    callback.after_worker_group_abort(
        worker_group_context=_default_worker_group_context()
    )

    shard_provider.shutdown_data_executors.assert_called_once()


@pytest.mark.asyncio
async def test_dataset_manager_shutdown_multiple_datasets(ray_start_4_cpus):
    """
    Test that the DatasetManager is able to collect SplitCoordinator actors for multiple datasets and triggers
    the shutdown of their executors.
    """

    # Create datasets
    NUM_ROWS = 100
    NUM_TRAIN_WORKERS = 2

    sharded_ds_1 = ray.data.range(NUM_ROWS)
    sharded_ds_2 = ray.data.range(NUM_ROWS)
    unsharded_ds = ray.data.range(NUM_ROWS)

    # Create a DatasetManager
    dataset_manager = DatasetManager(
        datasets={
            "sharded_1": sharded_ds_1,
            "sharded_2": sharded_ds_2,
            "unsharded": unsharded_ds,
        },
        data_config=ray.train.DataConfig(datasets_to_split=["sharded_1", "sharded_2"]),
        data_context=DataContext.get_current(),
        world_size=NUM_TRAIN_WORKERS,
        worker_node_ids=None,
    )

    # Get the first dataset shard for all workers
    await get_dataset_shard_for_all_workers(
        dataset_manager, "sharded_1", NUM_TRAIN_WORKERS
    )

    assert len(dataset_manager._coordinator_actors) == 1
    assert isinstance(dataset_manager._coordinator_actors[0], ray.actor.ActorHandle)

    # Get the second dataset shard for all workers
    await get_dataset_shard_for_all_workers(
        dataset_manager, "sharded_2", NUM_TRAIN_WORKERS
    )

    assert len(dataset_manager._coordinator_actors) == 2
    assert isinstance(dataset_manager._coordinator_actors[1], ray.actor.ActorHandle)

    # Get the third unsharded dataset for all workers
    await get_dataset_shard_for_all_workers(
        dataset_manager, "unsharded", NUM_TRAIN_WORKERS
    )

    # The unsharded dataset should not have a SplitCoordinator actor
    assert len(dataset_manager._coordinator_actors) == 2

    # Replace the two SplitCoordinator actors with MagicMocks
    mocks = [MagicMock() for i in range(2)]
    remote_mocks = [mock.shutdown_executor.remote for mock in mocks]

    dataset_manager._coordinator_actors = mocks

    dataset_manager.shutdown_data_executors()

    for remote_mock in remote_mocks:
        remote_mock.assert_called_once()


def test_data_executor_shutdown():
    """Test that calling shutdown_executor on the SplitCoordinator actor triggers the shutdown of the executor"""

    from ray.data._internal.execution import streaming_executor

    NUM_SPLITS = 1
    dataset = ray.data.range(100)
    coordinator = SplitCoordinator.options(name="test_split_coordinator").remote(
        _DatasetWrapper(dataset), NUM_SPLITS, None
    )

    # Trigger executor creation and resource allocation
    ray.get(coordinator.start_epoch.remote(0))

    num_shutdown_calls = ray.get(
        coordinator.__ray_call__.remote(lambda _, x=streaming_executor: x._num_shutdown)
    )
    assert num_shutdown_calls == 0

    ray.get(coordinator.shutdown_executor.remote())

    # Shutdown called on the executor
    num_shutdown_calls = ray.get(
        coordinator.__ray_call__.remote(lambda _, x=streaming_executor: x._num_shutdown)
    )
    assert num_shutdown_calls == 1


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", "-x", __file__]))
