from typing import Any, Callable, Dict, Optional

import ray

from ray.data._internal.execution.operators.map_operator import MapOperator
from ray.data._internal.logical.rules.configure_map_task_memory import (
    ConfigureMapTaskMemoryRule,
)


class ConfigureMapTaskMemoryWithProfiling(ConfigureMapTaskMemoryRule):
    def __init__(
        self,
        get_cluster_resources: Callable[[], Dict[str, Any]] = ray.cluster_resources,
    ):
        # Cache the cluster resources to avoid re-fetching them on every call.
        # This should work for autoscaling clusters because object store memory
        # ratio is fixed.
        self._cluster_resources = get_cluster_resources()

    def estimate_per_task_memory_requirement(self, op: MapOperator) -> Optional[int]:
        memory = (
            op.metrics.average_max_uss_per_task or op.metrics.average_bytes_per_output
        )
        if memory is None:
            return None

        # This logic only makes sense if `memory` includes object store memory,
        # which is true right now on Anyscale and KubeRay but not on the VM
        # cluster launcher.
        # In the future, if this assumption is not true, there will be significant
        # underutilization of the cluster memory, so we can remove this logic to
        # adjust based on heap fraction.
        cluster_resources = self._cluster_resources
        total_memory = cluster_resources.get("memory", 0)
        object_store_memory = cluster_resources.get("object_store_memory", 0)
        if total_memory > 0:
            # Factor in object store memory fraction
            heap_fraction = (total_memory - object_store_memory) / total_memory
            if heap_fraction > 0:
                return int(memory / heap_fraction)
        return memory
