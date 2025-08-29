import asyncio
from typing import Dict, List

import ray
from ray.anyscale.data.checkpoint.data_iterator_checkpointer import (
    RowIDBasedDataIteratorCheckpointer,
)
from ray.anyscale.train._internal.data_integration.interfaces import (
    DatasetShardMetadata,
)
from ray.data import Dataset, DataContext, DataIterator, NodeIdStr
from ray.train.v2._internal.data_integration.interfaces import GenDataset


class DatasetManager:
    """Manages the dataset shards for datasets configured in the trainer."""

    def __init__(
        self,
        datasets: Dict[str, GenDataset],
        data_config: ray.train.DataConfig,
        data_context: DataContext,
        world_size: int,
        worker_node_ids: List[NodeIdStr],
    ):
        self._datasets = {k: v() if callable(v) else v for k, v in datasets.items()}
        self._data_config = data_config
        self._datasets_to_split = (
            set(self._datasets.keys())
            if data_config._datasets_to_split == "all"
            else set(data_config._datasets_to_split)
        )
        self._world_size = world_size
        self._worker_node_ids = worker_node_ids

        # Maps dataset name to a list of cached `DataIterator`s corresponding to
        # Train worker ranks.
        self._dataset_iterators: Dict[str, List[DataIterator]] = {}

        # A condition variable to synchronize the calls to the async `get_dataset_shard` method.
        self._condition = asyncio.Condition()

        DataContext._set_current(data_context)

    def _configure_checkpoint_save_on_iterators(
        self,
        dataset_info: DatasetShardMetadata,
        dataset_iterators: List[DataIterator],
    ) -> None:
        """Configures the dataset iterators per rank to save checkpoint state.

        Args:
            dataset_info: The metadata of the dataset shard,
                which can contain a `state_dict` passed in by the user.
            dataset_iterators: The dataset iterators for each rank.
        """
        if dataset_info.dataset_name not in self._datasets_to_split:
            return

        if (
            dataset_info.dataset_name
            not in self._data_config.dataset_checkpoint_configs
        ):
            return

        for rank, dataset_iterator in enumerate(dataset_iterators):
            checkpointer = RowIDBasedDataIteratorCheckpointer(
                checkpoint_config=self._data_config.dataset_checkpoint_configs[
                    dataset_info.dataset_name
                ],
                world_rank=rank,
                world_size=len(dataset_iterators),
            )
            # Checkpointing methods should have been patched onto the DataIterator.
            assert hasattr(dataset_iterator, "_enable_checkpointing")
            dataset_iterator._enable_checkpointing(checkpointer)

    def _create_dataset_iterators(
        self, dataset_info: DatasetShardMetadata, base_dataset: Dataset
    ) -> List[DataIterator]:
        dataset_name = dataset_info.dataset_name

        iterators_per_rank = self._data_config.configure(
            datasets={dataset_name: base_dataset},
            world_size=self._world_size,
            worker_handles=None,
            worker_node_ids=self._worker_node_ids,
        )
        assert len(iterators_per_rank) == self._world_size
        # TODO: Update DataConfig to return a List[DataIterator] directly
        # for configuring a single dataset.
        # Convert the List[Dict[str, DataIterator]] to a List[DataIterator],
        # since we only configured one dataset.
        return [iterators_per_rank[i][dataset_name] for i in range(self._world_size)]

    def _get_unsharded_dataset_iterator(
        self, dataset_info: DatasetShardMetadata
    ) -> DataIterator:
        """Returns the dataset iterator for a dataset that is excluded
        from `DataConfig.datasets_to_split`.
        Note that this method is NOT a barrier across workers and can be called
        by any subset of workers and will return immediately.
        """
        dataset_name = dataset_info.dataset_name
        world_rank = dataset_info.world_rank

        if dataset_name not in self._dataset_iterators:
            self._dataset_iterators[dataset_name] = self._create_dataset_iterators(
                dataset_info, self._datasets[dataset_name]
            )

        return self._dataset_iterators[dataset_name][world_rank]

    async def _get_sharded_dataset_iterator(
        self, dataset_info: DatasetShardMetadata
    ) -> DataIterator:
        """Returns the dataset iterator for a dataset that is included
        in `DataConfig.datasets_to_split`.
        Note that this method is a barrier across workers,
        and all workers must call this method before training.
        """
        dataset_name = dataset_info.dataset_name
        world_rank = dataset_info.world_rank

        async with self._condition:
            if dataset_name in self._dataset_iterators:
                # If the dataset iterators have already been created, return the
                # existing one.
                iterator = self._dataset_iterators[dataset_name][world_rank]
            elif world_rank == 0:
                # In this case, the dataset iterators have not been created yet.
                # The dataset only needs to be configured once globally for all workers.
                # Do it only when the rank 0 worker calls this method.
                iterators = self._create_dataset_iterators(
                    dataset_info, self._datasets[dataset_name]
                )
                self._configure_checkpoint_save_on_iterators(dataset_info, iterators)
                iterator = iterators[world_rank]

                # Cache the dataset iterators for future use.
                self._dataset_iterators[dataset_name] = iterators
                self._condition.notify_all()
            else:
                # Wait for the dataset iterators to be created by the rank 0 worker.
                await self._condition.wait_for(
                    lambda: dataset_name in self._dataset_iterators
                )
                iterator = self._dataset_iterators[dataset_name][world_rank]
        return iterator

    async def get_dataset_shard(
        self,
        dataset_info: DatasetShardMetadata,
    ) -> DataIterator:
        """Create and return the dataset shard iterator for a Ray Train worker's
        call to `ray.train.get_dataset_shard`.

        This method is a barrier that should be called by all Ray Train workers at once.
        If the dataset iterators have already been created, return the existing ones.

        Otherwise, create the dataset iterators and cache them.
        Here's an example of how this method is used with 4 workers:
        Rank 2 calls get_dataset_shard, waits on the condition variable.
        Rank 1 calls get_dataset_shard, waits on the condition variable.
        Rank 0 calls get_dataset_shard, creates the dataset iterators, caches them,
        and notifies all workers hanging on the condition variable.
        Rank 3 calls get_dataset_shard, returns the cached iterator.
        """
        dataset_name = dataset_info.dataset_name

        if dataset_name in self._datasets_to_split:
            return await self._get_sharded_dataset_iterator(dataset_info)
        else:
            return self._get_unsharded_dataset_iterator(dataset_info)
