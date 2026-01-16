from dataclasses import dataclass
from typing import Optional

import pyarrow

# Import shared classes from OSS
from ray.data.checkpoint.interfaces import (
    CheckpointBackend,
    CheckpointConfig as OSSCheckpointConfig,
    InvalidCheckpointingConfig,
    InvalidCheckpointingOperators,
)
from ray.data.datasource import PathPartitionFilter
from ray.util.annotations import PublicAPI

# Re-export for backwards compatibility
__all__ = [
    "CheckpointBackend",
    "CheckpointConfig",
    "TrainingIngestCheckpointConfig",
    "InvalidCheckpointingConfig",
    "InvalidCheckpointingOperators",
]


@PublicAPI(stability="beta")
class CheckpointConfig(OSSCheckpointConfig):
    """Configuration for checkpointing with generated ID column support.

    Extends OSS CheckpointConfig with the ability to auto-generate row IDs.

    Args:
        id_column: Name of the ID column in the input dataset.
            ID values must be unique across all rows in the dataset and must persist
            during all operators. Either `id_column` or `generated_id_column` must be
            provided.
        checkpoint_path: Path to store the checkpoint data. It can be a path to a cloud
            object storage (e.g. `s3://bucket/path`) or a file system path.
            If the latter, the path must be a network-mounted file system (e.g.
            `/mnt/cluster_storage/`) that is accessible to the entire cluster.
            If not set, defaults to `${ANYSCALE_ARTIFACT_STORAGE}/ray_data_checkpoint`.
        generated_id_column: Name of the ID column to generate a row ID for each row.
            Use this when you don't have an `id_column` in the input dataset.
            Currently, only Parquet files based data sources are supported for
            auto-generated row IDs feature.
        delete_checkpoint_on_success: If true, automatically delete checkpoint
            data when the dataset execution succeeds. Only supported for
            batch-based backend currently.
        override_filesystem: Override the :class:`pyarrow.fs.FileSystem` object used to
            read/write checkpoint data. Use this when you want to use custom credentials.
        override_backend: Override the :class:`CheckpointBackend` object used to
            access the checkpoint backend storage.
        filter_num_threads: Number of threads used to filter checkpointed rows.
        write_num_threads: Number of threads used to write checkpoint files for
            completed rows.
        checkpoint_path_partition_filter: Filter for checkpoint files to load during
            restoration when reading from `checkpoint_path`.
    """

    def __init__(
        self,
        id_column: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        *,
        generated_id_column: Optional[str] = None,
        delete_checkpoint_on_success: bool = True,
        override_filesystem: Optional["pyarrow.fs.FileSystem"] = None,
        override_backend: Optional[CheckpointBackend] = None,
        filter_num_threads: int = 3,
        write_num_threads: int = 3,
        checkpoint_path_partition_filter: Optional[PathPartitionFilter] = None,
    ):
        self.generated_id_column: Optional[str] = generated_id_column

        # Validate that we don't have both `id_column` and `generated_id_column`
        # explicitly specified
        if id_column is not None and generated_id_column is not None:
            raise InvalidCheckpointingConfig(
                "Cannot specify both `id_column` and `generated_id_column`. "
                "Use `id_column` when you have an existing ID column in your dataset, "
                "or use `generated_id_column` when you want to generate row IDs "
                "automatically."
            )

        # If no `id_column` is provided, use the generated row ID column
        elif id_column is None and generated_id_column is None:
            raise InvalidCheckpointingConfig(
                "Either `id_column` or `generated_id_column` must be provided. "
                "Use `id_column` when you have an existing ID column in your dataset, "
                "or use `generated_id_column` when you want to generate row IDs "
                "automatically."
            )
        elif id_column is None:
            # Use generated_id_column as the id_column
            id_column = generated_id_column

        # Call parent __init__ with the resolved id_column
        super().__init__(
            id_column=id_column,
            checkpoint_path=checkpoint_path,
            delete_checkpoint_on_success=delete_checkpoint_on_success,
            override_filesystem=override_filesystem,
            override_backend=override_backend,
            filter_num_threads=filter_num_threads,
            write_num_threads=write_num_threads,
            checkpoint_path_partition_filter=checkpoint_path_partition_filter,
        )


# TODO: We can pull out a common CheckpointConfig base class.
# Then, the batch inference specific logic from above can be moved
# to a BatchInferenceCheckpointConfig subclass.
# The checkpoint "restore" logic is common to both batch inference
# and training ingest, but the checkpoint "write" configuration differs.
@dataclass
class TrainingIngestCheckpointConfig:
    """Configuration for training ingest checkpointing.

    Args:
        id_column: Name of the ID column in the input dataset.
            ID values must be unique across all rows in the dataset and must persist
            during all operators.
        generate_id_column: Whether to generate the `id_column` for each row.
            Use this when you don't have a pre-existing `id_column` in the input dataset.
            Currently, only Parquet files based data sources are supported for
            auto-generated row IDs feature.
        checkpoint_path: Path to store the checkpoint data. It can be a path to a cloud
            object storage (e.g. `s3://bucket/path`) or a file system path.
            If the latter, the path must be a network-mounted file system (e.g.
            `/mnt/cluster_storage/`) that is accessible to the entire cluster.
            If not set, defaults to `{RunConfig.storage_path}/{RunConfig.name}`
            configured on the `ray.train` trainer.
        override_filesystem: Override the :class:`pyarrow.fs.FileSystem` object used to
            read/write checkpoint data. Use this when you want to use custom credentials.
            If unset, this defaults to the filesystem configured in the `ray.train.RunConfig`
            passed to the trainer.
        delete_checkpoints_after_epoch: If True, automatically delete checkpoint
            data after each epoch completion. This allows for fault tolerance from
            the latest checkpoint. If you intend to resume from a checkpoint prior
            to the latest epoch, set this to False. Defaults to True.
    """

    id_column: str
    generate_id_column: bool = False
    checkpoint_path: Optional[str] = None
    override_filesystem: Optional["pyarrow.fs.FileSystem"] = None
    delete_checkpoints_after_epoch: bool = True

    def __post_init__(self):
        if not isinstance(self.id_column, str) or len(self.id_column) == 0:
            raise InvalidCheckpointingConfig(
                "Checkpoint ID column must be a non-empty string, "
                f"but got {self.id_column}"
            )
