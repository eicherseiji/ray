from typing import TYPE_CHECKING, Any, Dict, Optional

from ray.train.v2._internal.execution.train_fn_utils import get_train_fn_utils
from ray.util.annotations import PublicAPI

if TYPE_CHECKING:
    from ray.data import DataIterator


@PublicAPI(stability="stable")
def get_dataset_shard(
    dataset_name: str, state_dict: Optional[Dict[str, Any]] = None
) -> "DataIterator":
    """Returns the :class:`ray.data.DataIterator` shard for this worker.

    Call :meth:`~ray.data.DataIterator.iter_torch_batches` or
    :meth:`~ray.data.DataIterator.to_tf` on this shard to convert it to the
    appropriate framework-specific data type.

    .. testcode::

        import ray
        from ray import train
        from ray.train import ScalingConfig
        from ray.train.torch import TorchTrainer

        def train_loop_per_worker(config):
            ...
            for epoch in range(2):
                # Trainer will automatically handle sharding.
                data_shard = train.get_dataset_shard("train")
                for batch in data_shard.iter_torch_batches():
                    ...

        train_dataset = ray.data.read_csv("s3://anonymous@ray-example-data/iris.csv")
        trainer = TorchTrainer(
            train_loop_per_worker,
            scaling_config=ScalingConfig(num_workers=2),
            datasets={"train": train_dataset}
        )
        trainer.fit()

    .. testoutput::
        :hide:

        ...

    Args:
        dataset_name: If a Dictionary of Datasets was passed to ``Trainer``, then
            specifies which dataset shard to return.
        state_dict: [Experimental] Optional state dict to restore the dataset iterator.
            This requires data iterator checkpointing to be enabled via
            `ray.train.DataConfig`, and the state dict must be one that was saved with
            `DataIterator.state_dict()`.

    Returns:
        The ``DataIterator`` shard to use for this worker.
        If no dataset is passed into Trainer, then return None.
    """
    from ray.anyscale.train._internal.data_integration.interfaces import (
        DatasetShardMetadata,
    )

    train_fn_utils = get_train_fn_utils()
    return train_fn_utils.get_dataset_shard(
        DatasetShardMetadata(
            dataset_name=dataset_name,
            world_rank=train_fn_utils.get_context().get_world_rank(),
            state_dict=state_dict,
        )
    )
