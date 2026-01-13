from ray._private.arrow_utils import get_pyarrow_version
from ray.data._internal.arrow_ops.transform_pyarrow import (
    MIN_PYARROW_VERSION_RUN_END_ENCODED_TYPES,
)
from ray.data._internal.execution.operators.hash_shuffle import _combine
from ray.data._internal.execution.operators.join import (
    JoiningAggregation,
)
from ray.data._internal.util import _check_import


import logging
from typing import Any, List, Optional, Tuple, TYPE_CHECKING, Iterator, Dict

from ray.data.context import DataContext
from ray.data._internal.logical.operators.join_operator import JoinType
from ray.data.block import Block

if TYPE_CHECKING:
    import pyarrow as pa
    import polars as pl


_JOIN_TYPE_TO_POLARS_JOIN_TYPE_MAP = {
    JoinType.INNER: "inner",
    JoinType.LEFT_OUTER: "left",
    JoinType.RIGHT_OUTER: "right",
    JoinType.FULL_OUTER: "full",
    # Add semi/anti mappings (Polars semantics are left-semi/left-anti)
    JoinType.LEFT_SEMI: "semi",
    JoinType.LEFT_ANTI: "anti",
}

logger = logging.getLogger(__name__)


class JoiningAggregationWithPolars(JoiningAggregation):
    """Aggregation performing distributed joining of the 2 sequences,
    by utilising hash-based shuffling.

    Hash-based shuffling applied to 2 input sequences and employing the same
    partitioning scheme allows to

        - Accumulate identical keys from both sequences into the same
        (numerical) partition. In other words, all keys such that

            hash(key) % num_partitions = partition_id

        - Perform join on individual partitions independently (from other partitions)

    During the joining process, Polars native joining functionality is utilised, providing
    incredible performance while preserving the data from being deserialized.
    """

    def __init__(
        self,
        *,
        join_type: JoinType,
        left_key_col_names: Tuple[str],
        right_key_col_names: Tuple[str],
        data_context: DataContext,
        left_columns_suffix: Optional[str] = None,
        right_columns_suffix: Optional[str] = None,
    ):
        super().__init__(
            join_type=join_type,
            left_key_col_names=left_key_col_names,
            right_key_col_names=right_key_col_names,
            data_context=data_context,
            left_columns_suffix=left_columns_suffix,
            right_columns_suffix=right_columns_suffix,
        )

        join_types = list(JoinType)

        assert join_type in join_types, (
            f"Join type is not currently supported (got: {join_type}; "  # noqa: C416
            f"supported: {join_types})"  # noqa: C416
        )

        self._data_context = data_context

    def finalize(self, partition_shards_map: Dict[int, List[Block]]) -> Iterator[Block]:
        assert (
            self._data_context.use_polars_join
        ), "use_polars_join must be set to True in the DataContext"

        _check_import(self, module="pyarrow", package="pyarrow")
        _check_import(self, module="polars", package="polars")

        import polars as pl

        left_partition_shards = partition_shards_map[0]
        right_partition_shards = partition_shards_map[1]

        left_table = _combine(left_partition_shards)
        right_table = _combine(right_partition_shards)

        left_on, right_on = list(self._left_key_col_names), list(
            self._right_key_col_names
        )

        preprocess_result_l, preprocess_result_r = self._preprocess(
            left_table, right_table, left_on, right_on
        )

        left_df: pl.LazyFrame = pl.from_arrow(
            preprocess_result_l.supported_projection
        ).lazy()
        right_df: pl.LazyFrame = pl.from_arrow(
            preprocess_result_r.supported_projection
        ).lazy()

        target_join_type = self._join_type
        left_cols_suffix = self._left_columns_suffix
        right_cols_suffix = self._right_columns_suffix

        # NOTE: Polars doesn't support Right Semi/Anti joins so we have to
        #       rotate our sides to perform corresponding Left ones
        if target_join_type in (JoinType.RIGHT_SEMI, JoinType.RIGHT_ANTI):
            left_df, right_df = right_df, left_df
            left_cols_suffix, right_cols_suffix = left_cols_suffix, right_cols_suffix

            if target_join_type is JoinType.RIGHT_SEMI:
                target_join_type = JoinType.LEFT_SEMI
            elif target_join_type is JoinType.RIGHT_ANTI:
                target_join_type = JoinType.LEFT_ANTI

        polars_join_type = _JOIN_TYPE_TO_POLARS_JOIN_TYPE_MAP[target_join_type]

        # https://github.com/pola-rs/polars/issues/12418 (Polars doesn't have support for left/right suffixes)

        left_cols = set(left_df.collect_schema().names())
        right_cols = set(right_df.collect_schema().names())
        collisions = (left_cols & right_cols) - set(left_on) - set(right_on)

        if left_cols_suffix is None and right_cols_suffix is None and collisions:
            raise ValueError(
                "Left and right columns suffixes cannot be both None "
                f"(overlapping columns: {sorted(collisions)})"
            )

        if left_cols_suffix:
            left_df = left_df.rename({c: f"{c}{left_cols_suffix}" for c in collisions})

        if right_cols_suffix:
            right_df = right_df.rename(
                {c: f"{c}{right_cols_suffix}" for c in collisions}
            )

        right_suffix = right_cols_suffix or "_right"
        joined = left_df.join(
            right_df,
            how=polars_join_type,
            left_on=left_on,
            right_on=right_on,
            suffix=right_suffix,
            coalesce=True,
        )

        if self._join_type != JoinType.FULL_OUTER:
            # The goal here is to re-produce arrow's join behavior.
            # Remove duplicate join key columns (with right suffix) for non full outer joins
            # since left and right key values are identical making right keys redundant
            joined_cols = joined.collect_schema().names()
            duplicate_columns = [f"{right_key}{right_suffix}" for right_key in right_on]
            duplicate_columns = [col for col in duplicate_columns if col in joined_cols]
            if duplicate_columns:
                joined = joined.drop(duplicate_columns)

        # Determine execution engine
        gpu_engine = _get_polars_gpu_engine(self._data_context)

        batches: Optional[Iterator[pl.DataFrame]] = None

        # Execute with appropriate engine
        if gpu_engine is not None:
            try:
                import polars as pl

                with pl.Config() as cfg:
                    cfg.set_verbose(True)
                    batches = joined.collect_batches(engine=gpu_engine)
            except Exception as e:
                # Handle specific GPU execution failures
                if self._data_context.polars_gpu_raise_on_fail:
                    raise

                if isinstance(e, (NotImplementedError, ImportError)):
                    error_message = "not supported"
                else:
                    error_message = "failed"

                # Log the specific GPU failure and fall back to CPU
                logger.debug(f"GPU join {error_message}, falling back to CPU: {e}")

        if batches is None:
            batches = joined.collect_batches(engine="streaming")

        # Add back unsupported columns (join type logic is in should_index_* variables)
        for batch in batches:
            yield self._postprocess(
                batch.to_arrow(),
                preprocess_result_l.unsupported_projection,
                preprocess_result_r.unsupported_projection,
            )

    def _is_pa_join_not_supported(self, type: "pa.DataType") -> bool:
        """
        The latest pyarrow versions do not support joins where the
        tables contain the following types below (lists,
        structs, maps, unions, extension types, etc.)

        Args:
            type: The input type of column.

        Returns:
            True if the type cannot be present (non join-key) during joins.
            False if the type can be present.
        """
        import pyarrow as pa

        return pa.types.is_union(type) or (
            get_pyarrow_version() >= MIN_PYARROW_VERSION_RUN_END_ENCODED_TYPES
            and pa.types.is_run_end_encoded(type)
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
            error_msg += f"\n6. GPU device {self.polars_gpu_device_id} is accessible"

        raise RuntimeError(error_msg)

    return True


def _get_polars_gpu_engine(data_context: "DataContext") -> Optional["pl.GPUEngine"]:
    """Get configured Polars GPU engine object with current settings.

    Creates a Polars GPUEngine instance configured with the current context settings
    including device ID and error handling behavior. Automatically validates that
    polars[gpu] dependencies are available.

    Returns:
        pl.GPUEngine: Configured GPU engine with current context settings,
                     or None if GPU joins are disabled or dependencies unavailable.
    """
    if not data_context.use_polars_gpu_join:
        return None

    try:
        _check_polars_gpu_import()
        import polars as pl

        kwargs = {"raise_on_fail": data_context.polars_gpu_raise_on_fail}
        if data_context.polars_gpu_device_id is not None:
            kwargs["device"] = data_context.polars_gpu_device_id

        return pl.GPUEngine(**kwargs)

    except ImportError:
        return None
    except Exception:
        return None


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
