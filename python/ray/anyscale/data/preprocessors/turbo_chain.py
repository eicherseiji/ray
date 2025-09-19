from typing import TYPE_CHECKING

from ray.anyscale.data.preprocessors.dag import _build_aggregation_dag
from ray.data.preprocessor import Preprocessor
from ray.anyscale.data.preprocessors.turbo_preprocessor import TurboPreprocessor
import ray.data.preprocessors as preprocessors_module

# Store original reference before potential patching
_OriginalChain = preprocessors_module.Chain


if TYPE_CHECKING:
    from ray.data import Dataset


class Chain(_OriginalChain, TurboPreprocessor):
    def __init__(self, *preprocessors: Preprocessor):
        _OriginalChain.__init__(self, *preprocessors)
        TurboPreprocessor.__init__(self)
        for p in preprocessors:
            p.is_chain = True
        self.is_chain = True

    def transform(self, ds: "Dataset", **kwargs) -> "Dataset":
        """
        Transforms the dataset by executing dependency-aware lazy aggregations.

        Aggregations are resolved in topological order according to column dependencies.
        Each preprocessor is applied only after all its associated aggregations are complete.

        :param ds: The input Ray Dataset to transform.
        :param kwargs: Additional keyword arguments passed to each preprocessor's transform method.
        :return: The transformed Ray Dataset.
        """
        transformed_preprocessors = set()
        aggregation_nodes = _build_aggregation_dag(self.preprocessors)
        pending_nodes = set(aggregation_nodes)

        while pending_nodes:
            ready = [n for n in pending_nodes if n.is_ready()]
            if not ready:
                raise RuntimeError("Circular dependency detected in aggregation DAG.")

            aggregates = [n.agg_fn for n in ready]
            stats = ds.aggregate(*aggregates)

            for node in ready:
                p = node.preprocessor
                stat_key = node.agg_fn.name
                post_key = (
                    node.post_key_fn(node.column)
                    if node.post_key_fn is not None
                    else stat_key
                )
                p.stats_[post_key] = node.post_process_fn(stats[stat_key])
                node.completed = True
                pending_nodes.remove(node)

            for node in ready:
                p = node.preprocessor
                if p not in transformed_preprocessors and all(
                    n.completed for n in aggregation_nodes if n.preprocessor == p
                ):
                    ds = p.transform(ds, **kwargs)
                    transformed_preprocessors.add(p)

        for preprocessor in self.preprocessors:
            if preprocessor not in transformed_preprocessors:
                ds = preprocessor.transform(ds, **kwargs)
        return ds
