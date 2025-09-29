from ray._private.arrow_utils import get_pyarrow_version
from ray.data._internal.arrow_ops.transform_pyarrow import (
    MIN_PYARROW_VERSION_RUN_END_ENCODED_TYPES,
)
from ray.data._internal.execution.interfaces.physical_operator import PhysicalOperator
from ray.data._internal.execution.operators.join import (
    JoinOperator,
    JoiningShuffleAggregation,
)


import logging
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from ray.data.context import DataContext
from ray.data._internal.logical.operators.join_operator import JoinType
from ray.data.block import Block

if TYPE_CHECKING:
    import pyarrow as pa

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


class JoiningShuffleAggregationWithPolars(JoiningShuffleAggregation):
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
        aggregator_id: int,
        join_type: JoinType,
        left_key_col_names: Tuple[str],
        right_key_col_names: Tuple[str],
        target_partition_ids: List[int],
        data_context: DataContext,
        left_columns_suffix: Optional[str] = None,
        right_columns_suffix: Optional[str] = None,
    ):
        super().__init__(
            aggregator_id=aggregator_id,
            join_type=join_type,
            left_key_col_names=left_key_col_names,
            right_key_col_names=right_key_col_names,
            target_partition_ids=target_partition_ids,
            data_context=data_context,
            left_columns_suffix=left_columns_suffix,
            right_columns_suffix=right_columns_suffix,
        )
        join_types = list(JoinType)
        assert join_type in join_types, (
            f"Join type is not currently supported (got: {join_type}; "  # noqa: C416
            f"supported: {join_types})"  # noqa: C416
        )

    def finalize(self, partition_id: int) -> Block:
        assert (
            self.data_context.use_polars_join
        ), "use_polars_join must be set to True in the DataContext"
        import polars as pl

        left_on, right_on = list(self._left_key_col_names), list(
            self._right_key_col_names
        )

        preprocess_result_l, preprocess_result_r = self._preprocess(
            left_on, right_on, partition_id
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

        if left_cols_suffix or right_cols_suffix:
            if left_cols_suffix:
                left_df = left_df.rename(
                    {c: f"{c}{left_cols_suffix}" for c in collisions}
                )
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

        supported = joined.collect().to_arrow()

        # Add back unsupported columns (join type logic is in should_index_* variables)
        supported = self._postprocess(
            supported,
            preprocess_result_l.unsupported_projection,
            preprocess_result_r.unsupported_projection,
        )

        return supported

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


class JoinOperatorWithPolars(JoinOperator):
    def __init__(
        self,
        data_context: DataContext,
        left_input_op: PhysicalOperator,
        right_input_op: PhysicalOperator,
        left_key_columns: Tuple[str],
        right_key_columns: Tuple[str],
        join_type: JoinType,
        *,
        num_partitions: int,
        left_columns_suffix: Optional[str] = None,
        right_columns_suffix: Optional[str] = None,
        partition_size_hint: Optional[int] = None,
        aggregator_ray_remote_args_override: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            data_context=data_context,
            left_input_op=left_input_op,
            right_input_op=right_input_op,
            left_key_columns=left_key_columns,
            right_key_columns=right_key_columns,
            join_type=join_type,
            num_partitions=num_partitions,
            left_columns_suffix=left_columns_suffix,
            right_columns_suffix=right_columns_suffix,
            partition_size_hint=partition_size_hint,
            aggregator_ray_remote_args_override=aggregator_ray_remote_args_override,
            shuffle_aggregation_type=JoiningShuffleAggregationWithPolars,
        )
