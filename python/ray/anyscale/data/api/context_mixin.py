from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Dict, Union

from ray.anyscale.data.checkpoint.interfaces import CheckpointBackend, CheckpointConfig
from ray.data.context import (
    DEFAULT_TARGET_MAX_BLOCK_SIZE,
    DEFAULT_TARGET_MIN_BLOCK_SIZE,
    env_bool,
)
from ray._private.ray_constants import env_integer
from ray.data._internal.util import _check_import

if TYPE_CHECKING:
    import polars as pl

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


def _check_polars_gpu_import(obj: Any) -> None:
    """Check if Polars GPU dependencies are available.

    Args:
        obj: The object that has the dependency.

    Raises:
        ImportError: If polars[gpu] is not installed.
    """
    _check_import(obj, module="polars", package="polars[gpu]")

    # Additional check for GPU-specific functionality
    try:
        import polars as pl

        # Try to create a GPUEngine to verify GPU support is available
        pl.GPUEngine()
    except Exception:
        raise ImportError(
            "Polars GPU support not available. Ensure you have:\n"
            "1. NVIDIA Volta™ or higher GPU with compute capability 7.0+\n"
            "2. CUDA 12 installed (CUDA 11 support ends with RAPIDS v25.06)\n"
            "3. polars[gpu] package installed: pip install polars[gpu]\n"
            "4. Linux or Windows Subsystem for Linux 2 (WSL2)"
        )


def _check_polars_gpu_availability(
    device_id: Optional[int] = None, raise_on_fail: bool = False
) -> bool:
    """Check if Polars GPU support is available.

    Args:
        device_id: Optional GPU device ID to test
        raise_on_fail: Whether to raise exceptions instead of falling back

    Returns:
        bool: True if GPU support is available, False otherwise.
    """
    try:
        # Check if polars[gpu] is properly installed
        _check_polars_gpu_import(_check_polars_gpu_availability)

        import polars as pl

        # Create a simple test query
        test_df = pl.DataFrame({"test": [1.0, 2.0, 3.0]}).lazy()
        test_query = test_df.select(pl.col("test") * 2)

        # Configure GPU engine based on parameters
        if device_id is not None:
            gpu_engine = pl.GPUEngine(device=device_id, raise_on_fail=raise_on_fail)
        else:
            gpu_engine = pl.GPUEngine(raise_on_fail=raise_on_fail)

        # Test GPU execution
        result = test_query.collect(engine=gpu_engine)

        # Verify we got expected results
        expected_values = [2.0, 4.0, 6.0]
        actual_values = result["test"].to_list()

        if actual_values != expected_values:
            return False

        return True

    except ImportError:
        return False
    except Exception:
        return False


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

    def validate_polars_gpu_config(self) -> bool:
        """Validate Polars GPU join configuration and system requirements.

        Performs comprehensive validation of GPU join configuration including:
        - Checking if GPU joins are enabled
        - Verifying that regular Polars joins are enabled (prerequisite)
        - Testing GPU availability with configured parameters
        - Validating system requirements (GPU hardware, CUDA, packages)

        Returns:
            bool: True if configuration is valid and GPU is available, False otherwise.

        Raises:
            RuntimeError: If GPU joins are enabled but GPU support is not available
                         or system requirements are not met.
        """
        if not self.use_polars_gpu_join:
            return True

        if not self.use_polars_join:
            return False

        # Test GPU availability with the configured parameters
        if not _check_polars_gpu_availability(
            device_id=self.polars_gpu_device_id,
            raise_on_fail=self.polars_gpu_raise_on_fail,
        ):
            error_msg = (
                "GPU joins are enabled but Polars GPU support is not available. "
                "Please ensure you have:\n"
                "1. NVIDIA Volta™ or higher GPU with compute capability 7.0+\n"
                "2. CUDA 12 installed (CUDA 11 support ends with RAPIDS v25.06)\n"
                "3. polars[gpu] package installed: pip install polars[gpu]\n"
                "4. Linux or Windows Subsystem for Linux 2 (WSL2)\n"
                "5. Sufficient GPU memory available"
            )
            if self.polars_gpu_device_id is not None:
                error_msg += (
                    f"\n6. GPU device {self.polars_gpu_device_id} is accessible"
                )

            raise RuntimeError(error_msg)

        return True

    def get_polars_gpu_engine(self) -> Optional["pl.GPUEngine"]:
        """Get configured Polars GPU engine object with current settings.

        Creates a Polars GPUEngine instance configured with the current context settings
        including device ID and error handling behavior. Automatically validates that
        polars[gpu] dependencies are available.

        Returns:
            pl.GPUEngine: Configured GPU engine with current context settings,
                         or None if GPU joins are disabled or dependencies unavailable.
        """
        if not self.use_polars_gpu_join:
            return None

        try:
            _check_polars_gpu_import(self)
            import polars as pl

            kwargs = {"raise_on_fail": self.polars_gpu_raise_on_fail}
            if self.polars_gpu_device_id is not None:
                kwargs["device"] = self.polars_gpu_device_id

            return pl.GPUEngine(**kwargs)

        except ImportError:
            return None
        except Exception:
            return None
