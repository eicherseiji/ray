from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

from ray._private.ray_constants import env_integer
from ray.data.checkpoint import CheckpointBackend, CheckpointConfig
from ray.data.context import (
    DEFAULT_TARGET_MAX_BLOCK_SIZE,
    DEFAULT_TARGET_MIN_BLOCK_SIZE,
    env_bool,
)

if TYPE_CHECKING:
    pass

DEFAULT_NUM_BLOCKS_PER_READ_TASK = 8

DEFAULT_DISABLE_LARGE_FILE_CHUNKING = env_bool(
    "RAY_TURBO_DISABLE_LARGE_FILE_CHUNKING", False
)

DEFAULT_PARQUET_READER_TARGET_CHUNK_SIZE = env_integer(
    "RAY_TURBO_PARQUET_CHUNKER_TARGET_CHUNK_SIZE", None
)

# Default setting for GPU-accelerated Polars joins (disabled by default for compatibility)
DEFAULT_USE_POLARS_GPU_JOIN = env_bool("RAY_TURBO_USE_POLARS_GPU_JOIN", False)

# Default GPU device ID for multi-GPU systems (None = use default device)
DEFAULT_POLARS_GPU_DEVICE_ID = env_integer("RAY_TURBO_POLARS_GPU_DEVICE_ID", None)

# Default error handling behavior for GPU failures (False = fallback to CPU)
DEFAULT_POLARS_GPU_RAISE_ON_FAIL = env_bool("RAY_TURBO_POLARS_GPU_RAISE_ON_FAIL", False)


@dataclass
class DataContextMixin:
    """A mix-in class that allows adding Anyscale proprietary
    attributes and methods to :class:`~ray.data.DataContext`."""

    # Configuration for Ray Data checkpointing.
    # If None, checkpointing is disabled.
    _checkpoint_config: Optional[CheckpointConfig] = None

    # Overrides viability of fusion for file reading ops
    _enable_read_files_fusion_override: Optional[bool] = None

    min_read_partition_size: int = DEFAULT_TARGET_MIN_BLOCK_SIZE
    # To amortize overheads associated with launching Ray tasks and using multi-
    # threading, produce multiple blocks in each read task. This doesn't change the
    # size of the blocks, but it does change the number of blocks produced by each
    # task.
    max_read_partition_size: int = (
        DEFAULT_TARGET_MAX_BLOCK_SIZE * DEFAULT_NUM_BLOCKS_PER_READ_TASK
    )

    use_polars_join: bool = True

    # Controls whether to use GPU acceleration for Polars join operations.
    # Requires NVIDIA Volta™ or higher GPU with compute capability 7.0+, CUDA 12,
    # and polars[gpu] package. If False, operations will fallback to CPU execution.
    use_polars_gpu_join: bool = DEFAULT_USE_POLARS_GPU_JOIN

    # GPU device ID to use for Polars operations in multi-GPU environments.
    # Set to None to use the default GPU device. Can be configured via
    # RAY_TURBO_POLARS_GPU_DEVICE_ID environment variable.
    polars_gpu_device_id: Optional[int] = DEFAULT_POLARS_GPU_DEVICE_ID

    # Whether to raise exceptions on GPU execution failures instead of falling back to CPU.
    # When False (default), GPU failures automatically fall back to CPU execution.
    # When True, GPU failures will raise exceptions. Can be configured via
    # RAY_TURBO_POLARS_GPU_RAISE_ON_FAIL environment variable.
    polars_gpu_raise_on_fail: bool = DEFAULT_POLARS_GPU_RAISE_ON_FAIL

    # Controls whether large file chunking is disabled.
    # When True, uses WholeFileChunker instead of more granular chunking strategies.
    disable_large_file_chunking: bool = DEFAULT_DISABLE_LARGE_FILE_CHUNKING

    # Target chunk size for ParquetFileChunker
    parquet_chunker_target_chunk_size: Optional[
        int
    ] = DEFAULT_PARQUET_READER_TARGET_CHUNK_SIZE

    @property
    def checkpoint_config(self) -> Optional[CheckpointConfig]:
        """Get the checkpoint configuration."""
        return self._checkpoint_config

    @checkpoint_config.setter
    def checkpoint_config(
        self, value: Optional[Union[CheckpointConfig, Dict[str, Any]]]
    ) -> None:
        """Set the checkpoint configuration."""
        if value is None:
            self._checkpoint_config = None
        elif isinstance(value, dict):
            if "override_backend" in value:
                if not isinstance(value["override_backend"], str):
                    raise TypeError(
                        "Expected 'override_backend' to be a string,"
                        f" but got {type(value['override_backend'])}."
                    )
                value["override_backend"] = CheckpointBackend[value["override_backend"]]
            self._checkpoint_config = CheckpointConfig(**value)
        elif isinstance(value, CheckpointConfig):
            self._checkpoint_config = value
        else:
            raise TypeError(
                "checkpoint_config must be a CheckpointConfig instance, a dict, or None."
            )
