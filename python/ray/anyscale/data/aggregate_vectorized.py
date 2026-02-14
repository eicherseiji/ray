import abc
from typing import Optional

import pyarrow.compute as pac

from ray.data._internal.arrow_ops.transform_pyarrow import (
    MIN_PYARROW_VERSION_TYPE_PROMOTION,
)
from ray.data.aggregate import (
    AbsMax,
    AccumulatorType,
    AggregateFnV2,
    AsList,
    Count,
    Max,
    Min,
    Quantile,
    Sum,
    Unique,
)
from ray.data.block import AggType, Block, BlockColumn, BlockColumnAccessor, U

MIN_PYARROW_VERSION_VECTORIZED_AGGREGATIONS = MIN_PYARROW_VERSION_TYPE_PROMOTION


# TODO move to anyscale package
class VectorizedAggregateFnV2(AggregateFnV2, abc.ABC):
    """Base class for fully vectorized aggregations"""

    def combine(self, current_accumulator: AggType, new: AggType) -> AggType:
        raise NotImplementedError(
            "this method should not be invoked for vectorized aggregations!"
        )

    @abc.abstractmethod
    def _combine_column(self, accumulator_col: BlockColumn) -> AggType:
        ...


class CountVectorized(VectorizedAggregateFnV2, Count):
    def _combine_column(self, accumulator_col: BlockColumn) -> AggType:
        return BlockColumnAccessor.for_column(accumulator_col).sum(
            ignore_nulls=self._ignore_nulls
        )


class AsListVectorized(VectorizedAggregateFnV2, AsList):
    def aggregate_block(self, block: Block) -> AccumulatorType:
        # NOTE: We simply return target column (array) as an aggregation result
        accessor = BlockColumnAccessor.for_column(block[self._target_col_name])

        if self._ignore_nulls:
            accessor = BlockColumnAccessor.for_column(accessor.dropna())

        # NOTE: We have to make sure that the column is represented in an
        #       Arrow-compatible format (either ``pyarrow.Array`` or Python's ``list``)
        #       to make sure it's not converted into a tensor downstream
        return accessor._to_arrow_compatible_container()

    def _combine_column(self, accumulator_col: BlockColumn) -> AggType:
        return BlockColumnAccessor.for_column(accumulator_col).flatten()


class SumVectorized(VectorizedAggregateFnV2, Sum):
    def _combine_column(self, accumulator_col: BlockColumn) -> AggType:
        return BlockColumnAccessor.for_column(accumulator_col).sum(
            ignore_nulls=self._ignore_nulls
        )


class MinVectorized(VectorizedAggregateFnV2, Min):
    def _combine_column(self, accumulator_col: BlockColumn) -> AggType:
        return BlockColumnAccessor.for_column(accumulator_col).min(
            ignore_nulls=self._ignore_nulls
        )


class MaxVectorized(VectorizedAggregateFnV2, Max):
    def _combine_column(self, accumulator_col: BlockColumn) -> AggType:
        return BlockColumnAccessor.for_column(accumulator_col).max(
            ignore_nulls=self._ignore_nulls
        )


class AbsMaxVectorized(VectorizedAggregateFnV2, AbsMax):
    def _combine_column(self, accumulator_col: BlockColumn) -> AggType:
        return BlockColumnAccessor.for_column(accumulator_col).max(
            ignore_nulls=self._ignore_nulls
        )


class QuantileVectorized(VectorizedAggregateFnV2, Quantile):
    def aggregate_block(self, block: Block) -> AggType:
        accessor = BlockColumnAccessor.for_column(block[self._target_col_name])
        # Return whole column as is
        #
        # NOTE: We have to make sure that the column is represented in an
        #       Arrow-compatible format (either ``pyarrow.Array`` or Python's ``list``)
        #       to make sure it's not converted into a tensor downstream
        return accessor._to_arrow_compatible_container()

    def _combine_column(self, accumulator_col: BlockColumn) -> AggType:
        # Combine all the lists in a column into a single value (ie flatten it)
        return BlockColumnAccessor.for_column(accumulator_col).flatten()

    def _finalize(self, accumulator: AggType) -> Optional[U]:
        accessor = BlockColumnAccessor.for_column(accumulator)

        return accessor.quantile(q=self._q, ignore_nulls=self._ignore_nulls)


class UniqueVectorized(VectorizedAggregateFnV2, Unique):
    def aggregate_block(self, block: Block) -> AggType:
        column = self._compute_unique(block)

        # Return column of unique values as is
        #
        # NOTE: We have to make sure that the column is represented in an
        #       Arrow-compatible format (either ``pyarrow.Array`` or Python's ``list``)
        #       to make sure it's not converted into a tensor downstream
        return BlockColumnAccessor.for_column(column)._to_arrow_compatible_container()

    def _combine_column(self, accumulator_col: BlockColumn) -> AggType:
        column_accessor = BlockColumnAccessor.for_column(accumulator_col)

        # Combine all the lists in a column into a single value (ie flatten it)
        flattened = column_accessor.flatten()
        return BlockColumnAccessor.for_column(flattened).unique()


class TopKUniqueVectorized(UniqueVectorized):
    """
    Computes unique values of the k most frequent elements globally
    using the vectorized aggregation path (Arrow >= 14).

    Preserves all values (with duplicates) per block so that global frequencies
    can be computed during ``_combine_column``. This mirrors the OSS approach in
    ``compute_unique_value_indices`` which sums counts across all partitions
    before selecting top-k, ensuring that a category appearing moderately across
    many partitions is not missed.

    This aggregator only supports the vectorized path. On Arrow < 14, the turbo
    encoder falls back to the OSS ``compute_unique_value_indices`` instead.
    """

    def __init__(
        self,
        *,
        on: str,
        k: int,
        **kwargs,
    ):
        super().__init__(on=on, **kwargs)
        self.k = k

    def aggregate_block(self, block: Block) -> AggType:
        column = block[self._target_col_name]
        column_accessor = BlockColumnAccessor.for_column(column)
        if column_accessor.is_composed_of_lists():
            column_accessor = BlockColumnAccessor.for_column(column_accessor.flatten())
            column_accessor = BlockColumnAccessor.for_column(column_accessor.dropna())
        # Return all values (not per-partition top-k) so that global frequency
        # counts can be computed correctly during _combine_column.
        return column_accessor._to_arrow_compatible_container()

    def _combine_column(self, accumulator_col: BlockColumn) -> AggType:
        column_accessor = BlockColumnAccessor.for_column(accumulator_col)
        # Flatten per-block value lists into a single global list.
        flattened = column_accessor.flatten()
        # Compute global value counts, sort descending by count, and take top-k.
        vc = pac.value_counts(flattened)
        counts = vc.field("counts")
        sorted_indices = pac.sort_indices(counts, sort_keys=[("x", "descending")])
        top_indices = sorted_indices[: self.k]
        return pac.take(vc.field("values"), top_indices)
