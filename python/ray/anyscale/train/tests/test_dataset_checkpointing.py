from unittest.mock import MagicMock

from pyarrow.fs import LocalFileSystem
import pytest

from ray.anyscale.train._internal.callbacks.datasets import DatasetsSetupCallback
import ray.data
import ray.train
from ray.train import DataConfig, DatasetCheckpointConfig
import ray.train.collective
from ray.train.v2.api.data_parallel_trainer import DataParallelTrainer

from ray.anyscale.data.tests.test_data_iterator_checkpointer import (
    filter_state_dict,
    read_checkpoint_files_for_state_dict,
)
from ray.train.tests.util import create_dict_checkpoint, load_dict_checkpoint
from ray.train.v2.tests.util import create_dummy_run_context


@pytest.mark.parametrize("restore_from_end_of_epoch", [True, False])
@pytest.mark.parametrize("generate_id_column", [True, False])
def test_e2e_with_ray_train(
    ray_start_4_cpus, tmp_path, restore_from_end_of_epoch, generate_id_column
):
    """Test that the checkpointing works end-to-end with Ray Train.

    Run for 2 epochs.
    Create checkpoints at 1/3, 2/3 and all the way through each epoch.
    Inject errors at 2/3 of epoch 0.
    Restore from 2/3 of epoch 0, and continue through epoch 1 to completion.
    If restore_from_end_of_epoch is True, also inject an error at the end of epoch 0
    to test that epoch 1 can start from scratch upon restoring.

    Check that the checkpointed row ids contain all seen batches from workers
    at that point.
    Check that the total number of seen rows at the end of 2 epochs is correct.
    """
    data_checkpoint_path = tmp_path / "data_checkpoints"
    train_checkpoint_path = tmp_path / "train_checkpoints"

    num_epochs = 2
    world_size = 2
    num_batches_per_worker = 30
    batch_size = 10
    num_mid_epoch_checkpoints = 2
    checkpoint_at_batches = [
        i * (num_batches_per_worker // (num_mid_epoch_checkpoints + 1))
        for i in range(1, num_mid_epoch_checkpoints + 1)
    ]
    error_at = [(0, checkpoint_at_batches[-1])]
    end_of_epoch_error_at = [0] if restore_from_end_of_epoch else []
    total_rows = num_batches_per_worker * world_size * batch_size
    ds = ray.data.range(total_rows)

    ctx = ray.data.DataContext.get_current()
    ctx.default_hash_shuffle_parallelism = 1

    id_column = "generated_id" if generate_id_column else "id"
    if generate_id_column:
        ds.write_parquet(str(tmp_path / "parquet"))
        ds = ray.data.read_parquet(str(tmp_path / "parquet"))

    def train_fn(config):
        rank = ray.train.get_context().get_world_rank()

        seen_rows = 0
        consumed_batches = 0
        start_epoch = 0
        state_dict = None

        checkpoint = ray.train.get_checkpoint()
        if checkpoint:
            checkpoint_data = load_dict_checkpoint(checkpoint)
            consumed_batches = checkpoint_data["consumed_batches"]
            start_epoch = checkpoint_data["epoch"]
            epoch_finished = checkpoint_data["epoch_finished"]
            seen_rows = checkpoint_data["seen_rows"]

            if epoch_finished:
                start_epoch = checkpoint_data["epoch"] + 1
                consumed_batches = 0

            state_dict = checkpoint_data["data_state"]

            print(
                f"[RESTORING] from checkpoint at epoch {start_epoch}, "
                f"batch {consumed_batches}, state_dict: {state_dict}"
            )

        ds_iter = ray.train.get_dataset_shard(
            "train", state_dict=state_dict if rank == 0 else None
        )

        for epoch in range(start_epoch, num_epochs):
            consumed_batches_this_epoch = consumed_batches % num_batches_per_worker

            seen_ids = []
            for batch in ds_iter.iter_batches(batch_size=batch_size):
                if generate_id_column:
                    # The generated id column should be auto-filtered out of the batch.
                    assert "generated_id" not in batch.keys()

                seen_ids.extend(batch["id"].tolist())

                consumed_batches_this_epoch += 1
                consumed_batches += 1
                seen_rows += len(batch["id"])

                if consumed_batches_this_epoch in checkpoint_at_batches:
                    state_dict = ds_iter.state_dict()
                    rank_0_state_dict = ray.train.collective.broadcast_from_rank_zero(
                        state_dict
                    )
                    assert rank_0_state_dict == state_dict

                    checkpointed_row_ids = read_checkpoint_files_for_state_dict(
                        state_dict, data_checkpoint_path, id_column
                    )
                    assert (
                        len(checkpointed_row_ids)
                        == consumed_batches_this_epoch * batch_size * world_size
                    )

                    if not generate_id_column:
                        # Check that the checkpointed row ids contains all seen batches from workers.
                        assert set(seen_ids) <= set(checkpointed_row_ids)

                    with create_dict_checkpoint(
                        {
                            "epoch": epoch,
                            "epoch_finished": False,
                            "consumed_batches": consumed_batches,
                            "seen_rows": seen_rows,
                            "data_state": state_dict,
                        }
                    ) as checkpoint:
                        ray.train.report(
                            {},
                            checkpoint=checkpoint if rank == 0 else None,
                        )
                    ray.train.collective.barrier()

                if (epoch, consumed_batches_this_epoch) in error_at:
                    raise RuntimeError(
                        f"[MID-EPOCH ERROR] at batch {consumed_batches}, state_dict: {state_dict}"
                    )

            state_dict = ds_iter.state_dict()
            rank_0_state_dict = ray.train.collective.broadcast_from_rank_zero(
                state_dict
            )
            assert rank_0_state_dict == state_dict
            assert filter_state_dict(state_dict) == {
                "epoch_idx": epoch + 1,
                "checkpoint_idx": -1,
            }
            checkpointed_row_ids = read_checkpoint_files_for_state_dict(
                state_dict, data_checkpoint_path, id_column
            )
            assert checkpointed_row_ids == []

            with create_dict_checkpoint(
                {
                    "epoch": epoch,
                    "consumed_batches": consumed_batches,
                    "seen_rows": seen_rows,
                    "epoch_finished": True,
                    "data_state": state_dict,
                }
            ) as checkpoint:
                ray.train.report(
                    {},
                    checkpoint=checkpoint if rank == 0 else None,
                )
            ray.train.collective.barrier()

            if epoch in end_of_epoch_error_at:
                raise RuntimeError(
                    f"[END-OF-EPOCH ERROR] at epoch {epoch}, state_dict: {state_dict}"
                )

        assert seen_rows == (total_rows * num_epochs) // world_size

    trainer = DataParallelTrainer(
        train_fn,
        scaling_config=ray.train.ScalingConfig(num_workers=world_size),
        datasets={"train": ds},
        run_config=ray.train.RunConfig(
            storage_path=str(train_checkpoint_path),
            failure_config=ray.train.FailureConfig(max_failures=2),
        ),
        dataset_config=ray.train.DataConfig(
            dataset_checkpoint_configs={
                "train": DatasetCheckpointConfig(
                    checkpoint_path=str(data_checkpoint_path),
                    id_column=id_column,
                    generate_id_column=generate_id_column,
                )
            }
        ),
    )
    trainer.fit()


def test_default_checkpoint_path(tmp_path):
    """Test that the default checkpoint path is updated to `storage_path/name/ray_data_checkpoints`.
    The RunConfig(filesystem) should also be taken as the default."""
    original_ds_ckpt_config = DatasetCheckpointConfig(
        id_column="id", checkpoint_path=None, override_filesystem=None
    )
    fs = LocalFileSystem()

    callback = DatasetsSetupCallback(
        train_run_context=create_dummy_run_context(
            run_config=ray.train.RunConfig(
                storage_path=str(tmp_path), name="test", storage_filesystem=fs
            ),
            datasets={"train": MagicMock()},
            dataset_config=DataConfig(
                dataset_checkpoint_configs={"train": original_ds_ckpt_config}
            ),
        )
    )
    updated_ckpt_config = callback._data_config.dataset_checkpoint_configs["train"]

    assert updated_ckpt_config.checkpoint_path == str(
        tmp_path / "test" / "ray_data_checkpoints"
    )
    assert updated_ckpt_config.override_filesystem is fs

    # Original user provided config should not be modified.
    assert original_ds_ckpt_config.checkpoint_path is None
    assert original_ds_ckpt_config.override_filesystem is None


def test_override_filesystem(tmp_path):
    """Test that the override_filesystem is taken over the RunConfig(filesystem)."""
    mock_fs = MagicMock()
    mock_fs._marker = "dummy"

    original_ds_ckpt_config = DatasetCheckpointConfig(
        id_column="id", checkpoint_path=None, override_filesystem=mock_fs
    )

    callback = DatasetsSetupCallback(
        train_run_context=create_dummy_run_context(
            run_config=ray.train.RunConfig(storage_path=str(tmp_path), name="test"),
            datasets={"train": MagicMock()},
            dataset_config=DataConfig(
                dataset_checkpoint_configs={"train": original_ds_ckpt_config}
            ),
        )
    )
    updated_ckpt_config = callback._data_config.dataset_checkpoint_configs["train"]

    # Default checkpoint path should still be set.
    assert updated_ckpt_config.checkpoint_path == str(
        tmp_path / "test" / "ray_data_checkpoints"
    )
    assert updated_ckpt_config.override_filesystem._marker == "dummy"


def test_data_config_validation():
    """Test that the data config checkpoint configuration validation works."""
    DataConfig(
        datasets_to_split="all",
        dataset_checkpoint_configs={"train": MagicMock(), "val": MagicMock()},
    )

    with pytest.raises(NotImplementedError):
        DataConfig(
            datasets_to_split=["train"],
            dataset_checkpoint_configs={"train": MagicMock(), "val": MagicMock()},
        )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
