import logging
from typing import Any, Dict, Iterator, List, Mapping, Union

import pandas as pd
from pandas.api.types import is_scalar

from ray.data._internal.row import row_repr, row_repr_pretty, row_str

logger = logging.getLogger(__name__)


class _OptimizedPandasRow(Mapping):
    """
    Row of a tabular Dataset backed by a Pandas DataFrame block.
    """

    def __init__(self, df: "pd.DataFrame", row_idx: int):
        self._batch = df
        self._row_idx = row_idx

    def __getitem__(self, key: Union[str, List[str]]) -> Any:
        from ray.data.extensions import TensorArrayElement

        def get_item(keys: List[str]) -> Any:
            items = []
            for col_name in keys:
                if col_name not in self._batch.columns:
                    return None
                val = self._batch[col_name].iloc[self._row_idx]
                if isinstance(val, TensorArrayElement):
                    # Getting an item in a Pandas tensor column may return
                    # a TensorArrayElement, which we have to convert to an ndarray.
                    val = val.to_numpy()
                items.append(val)

            # Try to interpret this as a numpy-type value.
            # See https://stackoverflow.com/questions/9452775/converting-numpy-dtypes-to-native-python-types.  # noqa: E501
            try:
                return tuple(v.item() if hasattr(v, "item") else v for v in items)
            except (AttributeError, ValueError) as e:
                logger.warning(
                    f"Failed to convert {items} to native Python types", exc_info=e
                )
                # Fallback to the original form.
                return tuple(items)

        is_single_item = isinstance(key, str)
        keys = [key] if is_single_item else key
        items = get_item(keys)

        if items is None:
            return None
        return items[0] if is_single_item else items

    def __iter__(self) -> Iterator:
        return iter(self._batch.columns)

    def __len__(self):
        return self._batch.shape[1]

    def as_pydict(self) -> Dict[str, Any]:
        from ray.data.extensions import TensorArrayElement

        pydict: Dict[str, Any] = {}
        for key in self:
            value = self._batch[key].iloc[self._row_idx]
            # Convert NA to None for consistency across block formats. `pd.isna`
            # returns True for both NA and NaN, but since we want to preserve NaN
            # values, we check for identity instead.
            if is_scalar(value) and value is pd.NA:
                pydict[key] = None
            elif isinstance(value, TensorArrayElement):
                pydict[key] = value.to_numpy()
            else:
                pydict[key] = value

        return pydict

    def __str__(self):
        return row_str(self)

    def __repr__(self):
        return row_repr(self)

    def _repr_pretty_(self, p, cycle):
        return row_repr_pretty(self, p, cycle)
