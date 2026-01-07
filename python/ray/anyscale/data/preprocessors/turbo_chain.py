import logging
from typing import TYPE_CHECKING

import ray

from ray.air.util.data_batch_conversion import BatchFormat
from ray.anyscale.data.preprocessors.dag import _build_aggregation_dag
from ray.data.preprocessor import Preprocessor
from ray.anyscale.data.preprocessors.turbo_preprocessor import TurboPreprocessor
import ray.data.preprocessors as preprocessors_module

# Store original reference before potential patching
_OriginalChain = preprocessors_module.Chain


if TYPE_CHECKING:
    from ray.data import Dataset

logger = logging.getLogger(__name__)


class Chain(_OriginalChain, TurboPreprocessor):
    def __init__(self, *preprocessors: Preprocessor):
        _OriginalChain.__init__(self, *preprocessors)
        TurboPreprocessor.__init__(self)
        for p in preprocessors:
            p.is_chain = True
        self.is_chain = True

    def _transform(self, ds: "Dataset", **kwargs) -> "Dataset":
        """
        Transforms the dataset by executing dependency-aware lazy aggregations.

        Aggregations are resolved in topological order according to column dependencies.
        Each preprocessor is applied only after all its associated aggregations are complete.

        :param ds: The input Ray Dataset to transform.
        :param kwargs: Additional keyword arguments passed to each preprocessor's transform method.
        :return: The transformed Ray Dataset.
        """

        if any(
            p.stat_computation_plan.has_custom_stat_fn() for p in self.preprocessors
        ):
            return self._fallback_to_serial_execution(ds, **kwargs)
        if all(p.has_stats() for p in self.preprocessors if p._is_fittable):
            # Lazy aggregation computes stats for all preprocessors in one _transform() call.
            # If all fittable preprocessor have stats, lazy aggregation has completed.
            return super()._transform(ds, **kwargs)
        if any(p.has_stats() for p in self.preprocessors if p._is_fittable):
            logger.warning(
                f"Unexpected state: some fittable preprocessors have stats while others don't. "
                f"Fitted: {[p for p in self.preprocessors if p._is_fittable and p.has_stats()]}. "
                f"Not fitted: {[p for p in self.preprocessors if p._is_fittable and not p.has_stats()]}. "
                "Recomputing aggregations."
            )

        transformed_preprocessors = set()
        all_nodes = _build_aggregation_dag(self.preprocessors)
        pending_nodes = set(all_nodes)

        while pending_nodes:
            ready = [n for n in pending_nodes if n.is_ready()]
            if not ready:
                raise RuntimeError("Circular dependency detected in aggregation DAG.")

            # Separate placeholder nodes (non-fittable preprocessors) from real aggregation nodes
            placeholder_nodes = [n for n in ready if n.is_placeholder]
            ready_aggregation_nodes = [n for n in ready if not n.is_placeholder]

            # Remove placeholder nodes from pending (they're already marked completed in __init__)
            for node in placeholder_nodes:
                pending_nodes.remove(node)

            # Run aggregations for fittable preprocessors
            if ready_aggregation_nodes:
                aggregates = [n.agg_fn for n in ready_aggregation_nodes]

                agg_ds = ds.groupby(None).aggregate(*aggregates)
                arrow_refs = agg_ds.to_arrow_refs()
                if not arrow_refs:
                    raise ValueError("Aggregation returned no results")
                arrow_table = ray.get(arrow_refs[0]) if arrow_refs else None

                preprocessors = {n.preprocessor for n in ready_aggregation_nodes}
                logger.info(
                    f"Running {len(aggregates)} aggregations for {len(preprocessors)} preprocessors: {preprocessors}"
                )

                for node in ready_aggregation_nodes:
                    p = node.preprocessor
                    stat_key = node.agg_fn.name
                    # Aggregation returns single row - extract the first element
                    agg_result = arrow_table.column(stat_key)[0]

                    # Convert to appropriate format based on batch_format
                    if node.batch_format == BatchFormat.ARROW:
                        # Pass Arrow scalar (e.g., ListScalar) for Arrow-optimized post-processing
                        p.stats_[stat_key] = node.post_process_fn(agg_result)
                    else:
                        # Convert to Python for pandas-style post-processing
                        p.stats_[stat_key] = node.post_process_fn(agg_result.as_py())

                    node.completed = True
                    pending_nodes.remove(node)

            # Transform any preprocessor whose all nodes are complete (fittable or non-fittable)
            for node in ready:
                p = node.preprocessor
                if p not in transformed_preprocessors and all(
                    n.completed for n in all_nodes if n.preprocessor == p
                ):
                    ds = p.transform(ds, **kwargs)
                    transformed_preprocessors.add(p)

        for preprocessor in self.preprocessors:
            if preprocessor not in transformed_preprocessors:
                ds = preprocessor.transform(ds, **kwargs)
        return ds

    def _fallback_to_serial_execution(self, ds: "Dataset", **kwargs) -> "Dataset":
        """
        If any preprocessor has custom stats function which might potentially contain
        iter_batches based stats computation, fallback to serial execution.
        """
        logger.warning(
            "Falling back to serial execution because one or more preprocessors use "
            "custom stat functions (e.g., add_callable_stat). This prevents parallel "
            "deferred aggregation. Consider rewriting stats computation using "
            "AggregateV2-based aggregators (e.g., add_aggregator) to benefit from "
            "parallel execution and improved performance."
        )
        for preprocessor in self.preprocessors:
            preprocessor.is_chain = False
            ds = preprocessor._fit_execute(ds).transform(ds, **kwargs)
        return ds
