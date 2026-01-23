import asyncio

import pytest

import ray.data
from ray.anyscale.train._internal.data_integration.dataset_manager import (
    DatasetManager,
    DatasetShardMetadata,
)
from ray.data import DataContext
from ray.data._internal.iterator.stream_split_iterator import StreamSplitDataIterator


async def get_dataset_shard_for_worker(
    dataset_manager: DatasetManager,
    dataset_name: str,
    worker_rank: int,
):
    return await asyncio.create_task(
        dataset_manager.get_dataset_shard(
            DatasetShardMetadata(dataset_name=dataset_name, world_rank=worker_rank)
        )
    )


async def get_dataset_shard_for_all_workers(
    dataset_manager: DatasetManager,
    dataset_name: str,
    num_workers: int,
):
    return await asyncio.gather(
        *[
            get_dataset_shard_for_worker(dataset_manager, dataset_name, i)
            for i in range(num_workers)
        ]
    )


@pytest.mark.asyncio
async def test_get_multiple_datasets_serially(ray_start_4_cpus):
    """Tests DatasetManager.get_dataset_shard for multiple datasets,
    called serially by each worker. This is the typical case.
    Workers 0, 1:
    ray.train.get_dataset_shard("sharded_1")
    ray.train.get_dataset_shard("sharded_2")
    ray.train.get_dataset_shard("unsharded")
    """

    NUM_ROWS = 100
    NUM_TRAIN_WORKERS = 2

    sharded_ds_1 = ray.data.range(NUM_ROWS)
    sharded_ds_2 = ray.data.range(NUM_ROWS)
    unsharded_ds = ray.data.range(NUM_ROWS)

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

    shards = await get_dataset_shard_for_all_workers(
        dataset_manager, "sharded_1", NUM_TRAIN_WORKERS
    )
    assert all(isinstance(shard, StreamSplitDataIterator) for shard in shards)
    assert [shard._base_dataset.name for shard in shards] == [
        "sharded_1"
    ] * NUM_TRAIN_WORKERS

    shards = await get_dataset_shard_for_all_workers(
        dataset_manager, "sharded_2", NUM_TRAIN_WORKERS
    )
    assert all(isinstance(shard, StreamSplitDataIterator) for shard in shards)
    assert [shard._base_dataset.name for shard in shards] == [
        "sharded_2"
    ] * NUM_TRAIN_WORKERS

    shards = await get_dataset_shard_for_all_workers(
        dataset_manager, "unsharded", NUM_TRAIN_WORKERS
    )
    assert not any(isinstance(shard, StreamSplitDataIterator) for shard in shards)
    assert [shard._base_dataset.name for shard in shards] == [
        "unsharded"
    ] * NUM_TRAIN_WORKERS


@pytest.mark.asyncio
async def test_get_multiple_datasets_interleaved(ray_start_4_cpus):
    """Tests DatasetManager.get_dataset_shard for multiple datasets,
    called in an interleaved order by workers.
    Worker 0:
    ray.train.get_dataset_shard("train")
    ray.train.get_dataset_shard("valid")
    Worker 1:
    ray.train.get_dataset_shard("valid")
    ray.train.get_dataset_shard("train")
    """

    NUM_ROWS = 100
    NUM_TRAIN_WORKERS = 2

    train_ds = ray.data.range(NUM_ROWS)
    valid_ds = ray.data.range(NUM_ROWS)

    dataset_manager = DatasetManager(
        datasets={"train": train_ds, "valid": valid_ds},
        data_config=ray.train.DataConfig(datasets_to_split="all"),
        data_context=DataContext.get_current(),
        world_size=NUM_TRAIN_WORKERS,
        worker_node_ids=None,
    )

    tasks = [
        get_dataset_shard_for_worker(dataset_manager, "train", 0),
        get_dataset_shard_for_worker(dataset_manager, "valid", 1),
        get_dataset_shard_for_worker(dataset_manager, "train", 1),
        get_dataset_shard_for_worker(dataset_manager, "valid", 0),
    ]
    iterators = await asyncio.gather(*tasks)
    assert all(isinstance(iterator, StreamSplitDataIterator) for iterator in iterators)
    assert [iterator._base_dataset.name for iterator in iterators] == [
        "train",
        "valid",
        "train",
        "valid",
    ]


@pytest.mark.asyncio
async def test_get_multiple_datasets_rank_specific(ray_start_4_cpus):
    """Tests rank-specific DatasetManager.get_dataset_shard calls.
    # Epoch 1
    ray.train.get_dataset_shard("train")
    # Validation, which only happens on worker 0.
    if world_rank == 0:
        ray.train.get_dataset_shard("valid")
    # Epoch 2
    ray.train.get_dataset_shard("train")
    """

    NUM_ROWS = 100
    NUM_TRAIN_WORKERS = 2

    train_ds = ray.data.range(NUM_ROWS)
    valid_ds = ray.data.range(NUM_ROWS)

    dataset_manager = DatasetManager(
        datasets={"train": train_ds, "valid": valid_ds},
        data_config=ray.train.DataConfig(datasets_to_split=["train"]),
        data_context=DataContext.get_current(),
        world_size=NUM_TRAIN_WORKERS,
        worker_node_ids=None,
    )

    # ray.train.get_dataset_shard("train")
    iterators = await get_dataset_shard_for_all_workers(
        dataset_manager, "train", NUM_TRAIN_WORKERS
    )
    assert all(isinstance(iterator, StreamSplitDataIterator) for iterator in iterators)
    assert [iterator._base_dataset.name for iterator in iterators] == [
        "train"
    ] * NUM_TRAIN_WORKERS

    # if world_rank == 0:
    #     ray.train.get_dataset_shard("valid")
    iterator = await get_dataset_shard_for_worker(dataset_manager, "valid", 0)
    assert not isinstance(iterator, StreamSplitDataIterator)
    assert iterator._base_dataset.name == "valid"

    # ray.train.get_dataset_shard("train")
    iterators = await get_dataset_shard_for_all_workers(
        dataset_manager, "train", NUM_TRAIN_WORKERS
    )
    assert all(isinstance(iterator, StreamSplitDataIterator) for iterator in iterators)
    assert [iterator._base_dataset.name for iterator in iterators] == [
        "train"
    ] * NUM_TRAIN_WORKERS


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", "-x", __file__]))
