import pytest

import ray.data
from ray.anyscale.data.checkpoint.data_iterator_checkpointer import (
    RowIDBasedDataIteratorCheckpointer,
)
from ray.train import DatasetCheckpointConfig

from ray.anyscale.train.tests.test_dataset_checkpointing import (
    _read_checkpoint_files_for_state_dict,
)
from ray.tests.conftest import *  # noqa


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

    ds = ray.data.range(num_batches * batch_size)
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

        seen_ids = []
        for batch in batch_iter:
            consumed_batches += 1
            seen_ids.extend(batch["id"])
            if consumed_batches in checkpoint_at_batches:
                state_dict = ds_iter.state_dict()
                assert state_dict == {
                    "epoch_idx": epoch,
                    "checkpoint_idx": checkpoint_idx,
                }
                assert set(seen_ids) == set(
                    _read_checkpoint_files_for_state_dict(state_dict, tmp_path)
                )
                checkpoint_idx += 1

        assert ds_iter.state_dict() == {
            "epoch_idx": epoch + 1,
            "checkpoint_idx": -1,
        }
        # There should be no row_ids associated with a state dict at an epoch boundary.
        assert (
            _read_checkpoint_files_for_state_dict(ds_iter.state_dict(), tmp_path) == []
        )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
