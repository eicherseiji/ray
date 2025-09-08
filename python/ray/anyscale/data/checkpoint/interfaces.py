from dataclasses import dataclass
import os
import warnings
from enum import Enum
from typing import Optional, Tuple

import pyarrow

from ray.data.datasource import PathPartitionFilter
from ray.util.annotations import PublicAPI


class CheckpointBackend(Enum):
    """Supported backends for storing and reading checkpoint files.

    Currently, there are two types of backends:
        * Batch-based backends: CLOUD_OBJECT_STORAGE and FILE_STORAGE.
        * Row-based backends: CLOUD_OBJECT_STORAGE_ROW and FILE_STORAGE_ROW.

    Their differences are as follows:
    1. Writing checkpoints:
       * Batch-based backends write a checkpoint file for each block.
       * Row-based backends write a checkpoint file for each individual row.
    2. Loading checkpoints and filtering input data:
       * Batch-based backends load all checkpoint data into memory prior to
         dataset execution. The checkpoint data is then passed to each
         read task to perform filtering.
       * Row-based backends do not preload any data at the execution start-up.
         Instead, during the read tasks, each row is filtered based on whether it
         already exists in the backend.

    Overall, batch-based backends are recommended due to their lower runtime
    overheads. However, they may introduce a delay in job start-up due to the
    checkpoint loading process.
    """

    # TODO(haochen): Deprecate row-based backends when we make sure the
    # checkpoint loading overhead of the batch-based backends is acceptable
    # for all workloads.

    CLOUD_OBJECT_STORAGE = "CLOUD_OBJECT_STORAGE"
    """
    Batch-based checkpoint backend that uses cloud object storage, such as
    AWS S3, Google Cloud Storage, etc.
    """

    FILE_STORAGE = "FILE_STORAGE"
    """
    Batch based checkpoint backend that uses file system storage.
    Note, when using this backend, the checkpoint path must be a network-mounted
    file system (e.g. `/mnt/cluster_storage/`).
    """

    CLOUD_OBJECT_STORAGE_ROW = "CLOUD_OBJECT_STORAGE_ROW"
    """
    Batch-based checkpoint backend that uses cloud object storage, such as
    AWS S3, Google Cloud Storage, etc.
    It's more recommended to use the batch-based version.
    """

    FILE_STORAGE_ROW = "FILE_STORAGE_ROW"
    """
    Batch based checkpoint backend that uses file system storage.
    Note, when using this backend, the checkpoint path must be a network-mounted
    file system (e.g. `/mnt/cluster_storage/`).
    It's more recommended to use the batch-based version.
    """


@PublicAPI(stability="beta")
class CheckpointConfig:
    """Configuration for row-level checkpointing.

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
            access the checkpoint backend storage. Only use this if you want to use
            the row-backend checkpoint backends. By default, batch-based backends
            are used.
        filter_num_threads: Number of threads used to filter checkpointed rows.
            Only used for row-based backends.
        write_num_threads: Number of threads used to write checkpoint files for
            completed rows.
        checkpoint_path_partition_filter: Filter for checkpoint files to load during
            restoration when reading from `checkpoint_path`.
    """

    DEFAULT_CHECKPOINT_PATH_BUCKET_ENV_VAR = "ANYSCALE_ARTIFACT_STORAGE"
    DEFAULT_CHECKPOINT_PATH_DIR = "ray_data_checkpoint"

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
        self.id_column: Optional[str] = id_column
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
        elif self.id_column is None and generated_id_column is None:
            raise InvalidCheckpointingConfig(
                "Either `id_column` or `generated_id_column` must be provided. "
                "Use `id_column` when you have an existing ID column in your dataset, "
                "or use `generated_id_column` when you want to generate row IDs "
                "automatically."
            )
        elif self.id_column is None:
            self.id_column = generated_id_column

        if not isinstance(self.id_column, str) or len(self.id_column) == 0:
            raise InvalidCheckpointingConfig(
                "Checkpoint ID column must be a non-empty string, "
                f"but got {self.id_column}"
            )

        if override_backend is not None:
            warnings.warn(
                "`override_backend` is deprecated and will be removed in August 2025.",
                FutureWarning,
                stacklevel=2,
            )

        self.checkpoint_path: str = (
            checkpoint_path or self._get_default_checkpoint_path()
        )
        inferred_backend, inferred_fs = self._infer_backend_and_fs(
            self.checkpoint_path,
            override_filesystem,
            override_backend,
        )
        self.filesystem: "pyarrow.fs.FileSystem" = inferred_fs
        self.backend: CheckpointBackend = inferred_backend
        self.delete_checkpoint_on_success: bool = delete_checkpoint_on_success
        self.filter_num_threads: int = filter_num_threads
        self.write_num_threads: int = write_num_threads
        self.checkpoint_path_partition_filter = checkpoint_path_partition_filter

    def is_row_based(self):
        """Whether the checkpoint backend is row-based."""
        return self.backend in [
            CheckpointBackend.FILE_STORAGE_ROW,
            CheckpointBackend.CLOUD_OBJECT_STORAGE_ROW,
        ]

    def is_batch_based(self):
        """Whether the checkpoint backend is batch-based."""
        return self.backend in [
            CheckpointBackend.FILE_STORAGE,
            CheckpointBackend.CLOUD_OBJECT_STORAGE,
        ]

    def _get_default_checkpoint_path(self) -> str:
        artifact_storage = os.environ.get(self.DEFAULT_CHECKPOINT_PATH_BUCKET_ENV_VAR)
        if artifact_storage is None:
            raise InvalidCheckpointingConfig(
                f"`{self.DEFAULT_CHECKPOINT_PATH_BUCKET_ENV_VAR}` env var is not set, "
                "please explictly set `CheckpointConfig.checkpoint_path`."
            )
        return f"{artifact_storage}/{self.DEFAULT_CHECKPOINT_PATH_DIR}"

    def _infer_backend_and_fs(
        self,
        checkpoint_path: str,
        override_filesystem: Optional["pyarrow.fs.FileSystem"] = None,
        override_backend: Optional[CheckpointBackend] = None,
    ) -> Tuple[CheckpointBackend, "pyarrow.fs.FileSystem"]:
        try:
            if override_filesystem is not None:
                assert isinstance(override_filesystem, pyarrow.fs.FileSystem), (
                    "override_filesystem must be an instance of "
                    f"`pyarrow.fs.FileSystem`, but got {type(override_filesystem)}"
                )
                fs = override_filesystem
            else:
                fs, _ = pyarrow.fs.FileSystem.from_uri(checkpoint_path)

            if override_backend is not None:
                assert isinstance(override_backend, CheckpointBackend), (
                    "override_backend must be an instance of `CheckpointBackend`, "
                    f"but got {type(override_backend)}"
                )
                backend = override_backend
            else:
                if isinstance(fs, pyarrow.fs.LocalFileSystem):
                    backend = CheckpointBackend.FILE_STORAGE
                else:
                    backend = CheckpointBackend.CLOUD_OBJECT_STORAGE

            return backend, fs
        except Exception as e:
            raise InvalidCheckpointingConfig(
                f"Invalid checkpoint path: {checkpoint_path}. "
            ) from e


# TODO: We can pull out a common CheckpointConfig base class.
# Then, the batch inference specific logic from above can be moved
# to a BatchInferenceCheckpointConfig subclass.
# The checkpoint "restore" logic is common to both batch inference
# and training ingest, but the checkpoint "write" configuration differs.
@dataclass
class TrainingIngestCheckpointConfig:
    """Configuration for training ingest checkpointing.

    Args:
        checkpoint_path: Path to store the checkpoint data. It can be a path to a cloud
            object storage (e.g. `s3://bucket/path`) or a file system path.
            If the latter, the path must be a network-mounted file system (e.g.
            `/mnt/cluster_storage/`) that is accessible to the entire cluster.
            If not set, defaults to `{RunConfig.storage_path}/{RunConfig.name}`
            configured on the `ray.train` trainer.
        id_column: Name of the ID column in the input dataset.
            ID values must be unique across all rows in the dataset and must persist
            during all operators.
        generate_id_column: Whether to generate the `id_column` for each row.
            Use this when you don't have a pre-existing `id_column` in the input dataset.
            Currently, only Parquet files based data sources are supported for
            auto-generated row IDs feature.
    """

    # TODO: Set default checkpoint path to `RunConfig.storage_path/name`.
    checkpoint_path: str
    id_column: str
    generate_id_column: bool = False

    def __post_init__(self):
        if not isinstance(self.id_column, str) or len(self.id_column) == 0:
            raise InvalidCheckpointingConfig(
                "Checkpoint ID column must be a non-empty string, "
                f"but got {self.id_column}"
            )


class InvalidCheckpointingConfig(Exception):
    """Exception which indicates that the checkpointing
    configuration is invalid."""

    pass


class InvalidCheckpointingOperators(Exception):
    """Exception which indicates that the DAG is not
    eligible for row-based checkpointing, due to
    one or more incompatible operators."""

    pass
