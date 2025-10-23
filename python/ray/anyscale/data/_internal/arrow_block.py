from typing import TYPE_CHECKING, List, Union, Any, Iterator, Dict, Mapping

import numpy as np

from ray.anyscale.data._internal.block import OptimizedTableBlockMixin
from ray.data._internal.row import row_repr, row_repr_pretty, row_str

if TYPE_CHECKING:
    import pyarrow

_INTERNAL_NUM_ROWS_COUNTER_COLUMN_NAME = "__rd_internal_num_rows"


class _OptimizedArrowRow(Mapping):
    """
    Row of a tabular Dataset backed by a pyarrow RecordBatch or Table and a row index.
    """

    def __init__(
        self, batch: Union["pyarrow.Table", "pyarrow.RecordBatch"], row_idx: int
    ):
        self._batch = batch
        self._row_idx = row_idx

    def __getitem__(self, key: Union[str, List[str]]) -> Any:
        from ray.data.extensions import get_arrow_extension_tensor_types

        tensor_arrow_extension_types = get_arrow_extension_tensor_types()
        schema = self._batch.schema

        def get_item(keys: List[str]) -> Any:
            # Check for tensor extension type on first key
            if isinstance(schema.field(keys[0]).type, tensor_arrow_extension_types):
                # Build a tensor row.
                from ray.data._internal.arrow_block import ArrowBlockAccessor

                return tuple(
                    ArrowBlockAccessor._build_tensor_row(
                        self._batch, col_name=key, row_idx=self._row_idx
                    )
                    for key in keys
                )

            # Pyarrow select internally creates a new table by slicing which is expensive.
            # Instead, access the columns directly at row_idx.
            items = []
            for col_name in keys:
                col_idx = schema.get_field_index(col_name)
                if col_idx == -1:
                    # key not found
                    return None
                col = self._batch.column(col_idx)
                value = col[self._row_idx]
                items.append(value)

            if not items:
                return None

            try:
                # Try to interpret this as a pyarrow.Scalar value.
                return tuple(item.as_py() for item in items)
            except AttributeError:
                # Assume that this row is an element of an extension array, and
                # that it is bypassing pyarrow's scalar model for Arrow < 8.0.0.
                return tuple(items)

        is_single_item = isinstance(key, str)
        keys = [key] if is_single_item else key
        items = get_item(keys)

        if items is None:
            return None
        return items[0] if is_single_item else items

    def __iter__(self) -> Iterator:
        yield from self._batch.schema.names

    def __len__(self):
        return len(self._batch.schema)

    def as_pydict(self) -> Dict[str, Any]:
        return {k: self[k] for k in self}

    def __str__(self):
        return row_str(self)

    def __repr__(self):
        return row_repr(self)

    def _repr_pretty_(self, p, cycle):
        return row_repr_pretty(self, p, cycle)


class ArrowBlockMixin(OptimizedTableBlockMixin):
    """Mixin extending ``ArrowBlockAccessor`` providing optimized
    implementations for some common operations
    """

    def _get_row(self, index: int) -> _OptimizedArrowRow:
        return _OptimizedArrowRow(self._table, index)

    def _get_group_boundaries_sorted(self, keys: List[str]) -> np.ndarray:
        import pyarrow as pa
        import pyarrow.compute as pac

        if self.num_rows() == 0:
            return np.array([], dtype=np.int32)
        elif not keys:
            # If no keys are specified, whole block is considered a single group
            return np.array([0, self.num_rows()])

        # This method computes offsets for individual groups with a
        # following algorithm:
        #
        #   - Column with single int value of 1 (for every row) is appended
        ones = np.ones(self._table.num_rows, dtype=np.int32)

        extended_table = self._table.append_column(
            _INTERNAL_NUM_ROWS_COUNTER_COLUMN_NAME, pa.array(ones)
        )

        #   - Block is aggregated based on the target group-key, where
        #       newly added column is summed up (computing the size of the group)
        aggregated_extended_table = (
            extended_table.group_by(keys).aggregate(
                [(_INTERNAL_NUM_ROWS_COUNTER_COLUMN_NAME, "sum")]
            )
            # NOTE: Arrow performs hash-based aggregations and hence returned
            #       table could be out of order
            .sort_by([(k, "ascending") for k in keys])
        )

        group_size_column = aggregated_extended_table[
            f"{_INTERNAL_NUM_ROWS_COUNTER_COLUMN_NAME}_sum"
        ]

        #   - Column with respective sizes of the group is transformed into
        #       an array of offsets (by running cumulative sum on it)
        offsets_col = pac.cumulative_sum(group_size_column)

        return np.concatenate([[0], offsets_col.to_numpy()])
