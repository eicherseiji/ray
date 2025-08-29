from pathlib import Path

import pytest
import pyarrow.parquet as pq

import ray.data
from ray.anyscale.data.checkpoint.data_iterator_checkpointer import (
    RowIDBasedDataIteratorCheckpointer,
)
from ray.train import DatasetCheckpointConfig
import ray.train.collective
from ray.train.v2.api.data_parallel_trainer import DataParallelTrainer


from ray.tests.conftest import *  # noqa


def _read_checkpoint_files_for_state_dict(state_dict: dict, root_path: Path) -> list:
    checkpoint_idx = state_dict["checkpoint_idx"]
    epoch = state_dict["epoch_idx"]
    # Get all checkpoints up to and including checkpoint_idx
    checkpointed_row_ids = []
    for i in range(checkpoint_idx + 1):
        for rank_dir in root_path.glob("rank=*"):
            checkpoint_path = rank_dir / f"epoch={epoch}" / f"checkpoint={i}"
            for checkpoint_file in checkpoint_path.glob("*.parquet"):
                checkpointed_row_ids.extend(
                    pq.read_table(checkpoint_file).column("id").to_pylist()
                )
    return sorted(checkpointed_row_ids)


@pytest.mark.parametrize("reinit_iter", [True, False])
def test_multiple_checkpoints_per_epoch(ray_start_10_cpus, tmp_path, reinit_iter):
    """Test that multiple checkpoints can be created per epoch, for several epochs.

    Checkpoints are created at 1/3 and 2/3 way through each epoch.
    A checkpoint is also created at the end of each epoch.
    """
    checkpointer = RowIDBasedDataIteratorCheckpointer(
        checkpoint_config=DatasetCheckpointConfig(
            checkpoint_path=str(tmp_path), id_column="id"
        )
    )

    num_epochs = 2
    num_batches = 120
    batch_size = 10
    num_mid_epoch_checkpoints = 2
    checkpoint_at_batches = [
        i * (num_batches // (num_mid_epoch_checkpoints + 1))
        for i in range(1, num_mid_epoch_checkpoints + 1)
    ]

    # repartition(1) to ensure that batches are yielded in order.
    ds = ray.data.range(num_batches * batch_size).repartition(1)
    ds_iter = ds.iterator()
    ds_iter._enable_checkpointing(checkpointer)

    state_dict = ds_iter.state_dict()
    assert state_dict == {
        "epoch_idx": 0,
        "checkpoint_idx": -1,
    }
    # There should be no row_ids associated with a state dict initially.
    assert _read_checkpoint_files_for_state_dict(state_dict, tmp_path) == []

    batch_iter = None
    for epoch in range(num_epochs):
        checkpoint_idx = 0
        consumed_batches = 0
        if batch_iter is None or reinit_iter:
            batch_iter = ds_iter.iter_batches(batch_size=batch_size)
        for _ in batch_iter:
            consumed_batches += 1
            if consumed_batches in checkpoint_at_batches:
                state_dict = ds_iter.state_dict()
                assert state_dict == {
                    "epoch_idx": epoch,
                    "checkpoint_idx": checkpoint_idx,
                }
                assert _read_checkpoint_files_for_state_dict(
                    state_dict, tmp_path
                ) == list(range(consumed_batches * batch_size))
                checkpoint_idx += 1

        assert ds_iter.state_dict() == {
            "epoch_idx": epoch + 1,
            "checkpoint_idx": -1,
        }
        # There should be no row_ids associated with a state dict at an epoch boundary.
        assert (
            _read_checkpoint_files_for_state_dict(ds_iter.state_dict(), tmp_path) == []
        )


def test_e2e_with_ray_train(ray_start_10_cpus, tmp_path):
    """Test that the checkpointing works end-to-end with Ray Train.

    Checkpoints are created at 1/3 and 2/3 way through each epoch,
    and at the end of each epoch on 2 workers.
    """
    data_checkpoint_path = tmp_path / "data_checkpoints"
    train_checkpoint_path = tmp_path / "train_checkpoints"

    world_size = 2
    num_batches_per_worker = 120
    batch_size = 10
    num_mid_epoch_checkpoints = 2
    checkpoint_at_batches = [
        i * (num_batches_per_worker // (num_mid_epoch_checkpoints + 1)) + 1
        for i in range(1, num_mid_epoch_checkpoints + 1)
    ]
    ds = ray.data.range(num_batches_per_worker * world_size * batch_size)

    def train_fn(config):
        ds_iter = ray.train.get_dataset_shard("train")

        consumed_batches = 0
        seen_ids = []
        for batch in ds_iter.iter_batches(batch_size=batch_size):
            seen_ids.extend(batch["id"].tolist())
            consumed_batches += 1
            if consumed_batches in checkpoint_at_batches:
                state_dict = ds_iter.state_dict()
                rank_0_state_dict = ray.train.collective.broadcast_from_rank_zero(
                    state_dict
                )
                assert rank_0_state_dict == state_dict

                # Check that the global checkpointed row ids are correct.
                # NOTE: streaming_split(equal=True) splits blocks between workers,
                # so the the consumption is not in order.
                # For example, for 2 workers, the first batch could look like:
                # Worker 0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                # Worker 1: [1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009]
                checkpointed_row_ids = _read_checkpoint_files_for_state_dict(
                    state_dict, data_checkpoint_path
                )
                assert (
                    len(checkpointed_row_ids)
                    == consumed_batches * batch_size * world_size
                )

                # Check that the checkpointed row ids contains all seen batches from workers.
                assert set(seen_ids) <= set(checkpointed_row_ids)

        state_dict = ds_iter.state_dict()
        rank_0_state_dict = ray.train.collective.broadcast_from_rank_zero(state_dict)
        assert rank_0_state_dict == state_dict
        assert state_dict == {
            "epoch_idx": 1,
            "checkpoint_idx": -1,
        }

        checkpointed_row_ids = _read_checkpoint_files_for_state_dict(
            state_dict, data_checkpoint_path
        )
        assert checkpointed_row_ids == []

    trainer = DataParallelTrainer(
        train_fn,
        scaling_config=ray.train.ScalingConfig(num_workers=world_size),
        run_config=ray.train.RunConfig(storage_path=str(train_checkpoint_path)),
        datasets={"train": ds},
        dataset_config=ray.train.DataConfig(
            dataset_checkpoint_configs={
                "train": DatasetCheckpointConfig(
                    checkpoint_path=str(data_checkpoint_path), id_column="id"
                )
            }
        ),
    )
    trainer.fit()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
