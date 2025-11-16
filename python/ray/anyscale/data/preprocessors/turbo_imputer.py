from typing import Dict

import numpy as np

from ray.anyscale.data.preprocessors.turbo_preprocessor import TurboPreprocessor
from ray.data.aggregate import Mean, ValueCounter
import ray.data.preprocessors as preprocessors_module
from ray.data.preprocessors.version_support import SerializablePreprocessor

# Store original reference before potential patching
_OriginalSimpleImputer = preprocessors_module.SimpleImputer


@SerializablePreprocessor(version=1, identifier="io.ray.preprocessors.simple_imputer")
class SimpleImputer(_OriginalSimpleImputer, TurboPreprocessor):
    def _fit(self, ds):
        if self.strategy == "mean":
            self.stat_computation_plan.add_aggregator(
                aggregator_fn=Mean, columns=self.columns
            )
        elif self.strategy == "most_frequent":

            def build_counters(stats: Dict) -> int:
                if not stats or not stats.get("values"):
                    return 0
                return stats["values"][np.argmax(stats["counts"])]

            self.stat_computation_plan.add_aggregator(
                aggregator_fn=ValueCounter,
                post_process_fn=build_counters,
                post_key_fn=lambda column: f"most_frequent({column})",
                columns=self.columns,
            )
        elif self.strategy == "constant":
            self._fitted = True  # No fitting needed
        else:
            raise NotImplementedError(
                f"Unsupported strategy for lazy fit: {self.strategy}"
            )
        return self
