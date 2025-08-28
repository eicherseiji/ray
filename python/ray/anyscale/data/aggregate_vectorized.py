import abc
from typing import Optional

from ray.data._internal.arrow_ops.transform_pyarrow import (
    MIN_PYARROW_VERSION_TYPE_PROMOTION,
)
from ray.data.aggregate import (
    AggregateFnV2,
    Count,
    Sum,
    Min,
    Max,
    AbsMax,
    Quantile,
    Unique,
)
from ray.data.block import AggType, BlockColumn, BlockColumnAccessor, Block, U


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
        return accessor._as_arrow_compatible()

    def _combine_column(self, accumulator_col: BlockColumn) -> AggType:
        # Combine all the lists in a column into a single value (ie flatten it)
        return BlockColumnAccessor.for_column(accumulator_col).flatten()

    def _finalize(self, accumulator: AggType) -> Optional[U]:
        accessor = BlockColumnAccessor.for_column(accumulator)

        return accessor.quantile(q=self._q, ignore_nulls=self._ignore_nulls)


class UniqueVectorized(VectorizedAggregateFnV2, Unique):
    def __init__(self, on: str, encode_lists: bool = True, **kwargs):
        super().__init__(on=on, **kwargs)
        self.encode_lists = encode_lists

    def aggregate_block(self, block: Block) -> AggType:

        column = block[self._target_col_name]
        column_accessor = BlockColumnAccessor.for_column(column)
        if column_accessor.is_composed_of_lists():
            if self.encode_lists:
                # If lists should be encoded, flatten them and drop missing values before computing unique values.
                column_accessor = BlockColumnAccessor.for_column(
                    column_accessor.flatten()
                )
                column_accessor = BlockColumnAccessor.for_column(
                    column_accessor.dropna()
                )
            else:
                # Otherwise, hash the entire lists to prepare for uniqueness computation.
                column_accessor = BlockColumnAccessor.for_column(column_accessor.hash())

        # Return column of unique values as is
        #
        # NOTE: We have to make sure that the column is represented in an
        #       Arrow-compatible format (either ``pyarrow.Array`` or Python's ``list``)
        #       to make sure it's not converted into a tensor downstream
        return BlockColumnAccessor.for_column(
            column_accessor.unique()
        )._as_arrow_compatible()

    def _combine_column(self, accumulator_col: BlockColumn) -> AggType:
        # Combine all the lists in a column into a single value (ie flatten it)
        flattened = BlockColumnAccessor.for_column(accumulator_col).flatten()
        return BlockColumnAccessor.for_column(flattened).unique()


class TopKUniqueVectorized(UniqueVectorized):
    """
    Computes unique values of the k most frequent elements.
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
        return BlockColumnAccessor.for_column(
            column_accessor.top_k(self.k)
        )._as_arrow_compatible()
