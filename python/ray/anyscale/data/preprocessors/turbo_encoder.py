from functools import partial
from typing import Dict, Any

import numpy
import pandas as pd
import ray.data.preprocessors as preprocessors_module

from ray.anyscale.data.aggregate_vectorized import (
    UniqueVectorized,
    TopKUniqueVectorized,
)
from ray.anyscale.data.preprocessors.turbo_preprocessor import TurboPreprocessor
from ray.data.block import BlockColumnAccessor
from ray.data.preprocessors.encoder import (
    unique_post_fn,
    _validate_df,
    _is_series_composed_of_lists,
)
from ray.data.preprocessors.utils import make_post_processor
from ray.data.preprocessors.version_support import SerializablePreprocessor

# Store original references before potential patching
_OriginalOrdinalEncoder = preprocessors_module.OrdinalEncoder
_OriginalOneHotEncoder = preprocessors_module.OneHotEncoder
_OriginalMultiHotEncoder = preprocessors_module.MultiHotEncoder
_OriginalLabelEncoder = preprocessors_module.LabelEncoder
_OriginalCategorizer = preprocessors_module.Categorizer


@SerializablePreprocessor(version=1, identifier="io.ray.preprocessors.ordinal_encoder")
class OrdinalEncoder(_OriginalOrdinalEncoder, TurboPreprocessor):
    def _fit(self, ds):
        self.stat_computation_plan.add_aggregator(
            aggregator_fn=lambda col: UniqueVectorized(
                on=col,
                encode_lists=self.encode_lists,
                alias_name=f"unique_values({col})",
            ),
            post_process_fn=unique_post_fn(),
            columns=self.columns,
        )
        return self

    def _transform_pandas(self, df: pd.DataFrame):

        _validate_df(df, *self.columns)

        def encode_list(element: list, *, name: str):
            return [self.stats_[f"unique_values({name})"].get(x) for x in element]

        def column_ordinal_encoder(s: pd.Series):
            if _is_series_composed_of_lists(s):
                if self.encode_lists:
                    return s.map(partial(encode_list, name=s.name))

                def list_as_category(element):
                    if not isinstance(element, (list, numpy.ndarray)):
                        raise ValueError("unexpected type")
                    column_accessor = BlockColumnAccessor.for_column(
                        pd.Series([element])
                    )
                    key = column_accessor.hash()
                    return self.stats_[f"unique_values({s.name})"].get(key[0])

                return s.apply(list_as_category)

            s_values = self.stats_[f"unique_values({s.name})"]
            return s.map(s_values)

        df[self.output_columns] = df[self.columns].apply(column_ordinal_encoder)
        return df


@SerializablePreprocessor(version=1, identifier="io.ray.preprocessors.one_hot_encoder")
class OneHotEncoder(_OriginalOneHotEncoder, TurboPreprocessor):
    def _fit(self, ds):
        self.stat_computation_plan.add_aggregator(
            aggregator_fn=lambda col: (
                TopKUniqueVectorized(
                    on=col,
                    k=self.max_categories[col],
                    encode_lists=False,
                    alias_name=f"unique_values({col})",
                )
                if col in self.max_categories
                else UniqueVectorized(
                    on=col, encode_lists=False, alias_name=f"unique_values({col})"
                )
            ),
            post_process_fn=unique_post_fn(),
            columns=self.columns,
        )
        return self

    def safe_get(self, v: Any, stats: Dict[str, int]):
        if not isinstance(v, (tuple, list, numpy.ndarray)):
            return super().safe_get(v, stats)
        column_accessor = BlockColumnAccessor.for_column(pd.Series([v]))
        key = column_accessor.hash()
        return stats.get(key[0], -1)


@SerializablePreprocessor(
    version=1, identifier="io.ray.preprocessors.multi_hot_encoder"
)
class MultiHotEncoder(_OriginalMultiHotEncoder, TurboPreprocessor):
    def _fit(self, ds):
        self.stat_computation_plan.add_aggregator(
            aggregator_fn=lambda col: (
                TopKUniqueVectorized(
                    on=col,
                    k=self.max_categories[col],
                    encode_lists=True,
                    alias_name=f"unique_values({col})",
                )
                if col in self.max_categories
                else UniqueVectorized(
                    on=col, encode_lists=True, alias_name=f"unique_values({col})"
                )
            ),
            post_process_fn=unique_post_fn(),
            columns=self.columns,
        )
        return self


@SerializablePreprocessor(version=1, identifier="io.ray.preprocessors.label_encoder")
class LabelEncoder(_OriginalLabelEncoder, TurboPreprocessor):
    def _fit(self, ds):
        self.stat_computation_plan.add_aggregator(
            aggregator_fn=lambda col: UniqueVectorized(
                on=col, alias_name=f"unique_values({col})"
            ),
            post_process_fn=unique_post_fn(),
            columns=[self.label_column],
        )
        return self


@SerializablePreprocessor(version=1, identifier="io.ray.preprocessors.categorizer")
class Categorizer(_OriginalCategorizer, TurboPreprocessor):
    def _fit(self, ds):
        columns_to_get = [
            column for column in self.columns if column not in self.dtypes
        ]
        self.stats_ |= self.dtypes
        if not columns_to_get:
            return self

        def callback(unique_indices: Dict[str, Dict]) -> pd.CategoricalDtype:
            return pd.CategoricalDtype(unique_indices.keys())

        self.stat_computation_plan.add_aggregator(
            aggregator_fn=lambda col: UniqueVectorized(on=col, alias_name=col),
            post_process_fn=make_post_processor(
                base_fn=unique_post_fn(drop_na_values=True),
                callbacks=[callback],
            ),
            columns=columns_to_get,
        )
        return self
