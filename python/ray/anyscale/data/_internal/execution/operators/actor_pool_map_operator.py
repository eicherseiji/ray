import math
from typing import Tuple

from ray.data._internal.execution.interfaces import ExecutionResources
from ray.data._internal.execution.operators.actor_pool_map_operator import (
    ActorPoolMapOperator as OSSActorPoolMapOperator,
)


class ActorPoolMapOperator(OSSActorPoolMapOperator):
    def min_max_resource_requirements(
        self,
    ) -> Tuple[ExecutionResources, ExecutionResources]:
        # The base implementation only implements the min resource requirements.
        min_resource_usage, _ = super().min_max_resource_requirements()

        if self._inputs_complete:
            max_actors = self._actor_pool.current_size()
        else:
            max_actors = self._actor_pool.max_size()
            assert max_actors is not None, max_actors

        num_cpus_per_actor = self._ray_remote_args.get("num_cpus", 0)
        num_gpus_per_actor = self._ray_remote_args.get("num_gpus", 0)
        memory_per_actor = self._ray_remote_args.get("memory", 0)
        if math.isinf(max_actors):
            max_resource_usage = ExecutionResources.inf()
        else:
            max_concurrency = self._ray_remote_args.get("max_concurrency", 1)
            max_concurrent_tasks_per_actor = min(
                self._actor_pool.max_tasks_in_flight_per_actor(), max_concurrency
            )
            max_per_actor_resource_usage = ExecutionResources(
                cpu=num_cpus_per_actor,
                gpu=num_gpus_per_actor,
                memory=memory_per_actor,
                object_store_memory=(
                    self._metrics.obj_store_mem_max_pending_output_per_task
                    * max_concurrent_tasks_per_actor
                ),
            )
            max_resource_usage = max_per_actor_resource_usage.scale(max_actors)

        return min_resource_usage, max_resource_usage
