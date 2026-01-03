from typing import Dict

import pandas as pd
import ray.data.preprocessors as preprocessors_module

from ray.anyscale.data.aggregate_vectorized import (
    UniqueVectorized,
    TopKUniqueVectorized,
)
from ray.anyscale.data.preprocessors.turbo_preprocessor import TurboPreprocessor
from ray.data.preprocessors.encoder import (
    unique_post_fn,
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
                encode_lists=(
                    UniqueVectorized.ListEncodingMode.FLATTEN
                    if self.encode_lists
                    else None
                ),
                alias_name=f"unique_values({col})",
            ),
            post_process_fn=unique_post_fn(),
            columns=self.columns,
        )
        return self


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
