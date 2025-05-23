import math
from typing import Tuple

import ray
from ray.data._internal.execution.interfaces import (
    ExecutionResources,
    ReportsExtraResourceUsage,
)
from ray.data._internal.execution.operators.actor_pool_map_operator import (
    ActorPoolMapOperator as OSSActorPoolMapOperator,
    _ActorPool,
)


class ActorPoolMapOperator(OSSActorPoolMapOperator, ReportsExtraResourceUsage):
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

    def extra_resource_usage(self) -> ExecutionResources:
        """Returns resources occupied by lineage reconstruction actors.

        This shouldn’t include resources used by actors that haven’t been reconstructed,
        even if they're running retried tasks.
        """
        assert hasattr(
            self, "_actor_pool"
        ), "This method assumes that the base class has an `_actor_pool` attribute."

        num_actors = self._num_lineage_reconstructed_actors(self._actor_pool)
        per_actor_resources = self._actor_pool.per_actor_resource_usage()
        return per_actor_resources.scale(num_actors)

    def _num_lineage_reconstructed_actors(self, actor_pool: _ActorPool) -> int:
        """Actors that have been recreated to lineage reconstruct an object."""
        assert isinstance(actor_pool, _ActorPool), actor_pool

        # These are all of the actors that Ray Data is aware of.
        # NOTE(@bveeramani): It's confusing how we have both logical and Ray Core actor
        # IDs. We primarily use the logical actor IDs to determine the number of
        # reconstructed actors with `get_local_ongoing_lineage_reconstruction_tasks`.
        # So, we might be able to avoid this problem by introducing a new Ray Core
        # API that directly returns the number of reconstructed actors.
        actors_in_pool = set(actor_pool.get_logical_ids())

        # These are all of the actors that are running lineage reconstruction tasks.
        # These actors may or may not have been previously removed from the actor pool.
        task_infos = (
            ray._private.internal_api.get_local_ongoing_lineage_reconstruction_tasks()
        )
        actors_running_lineage_reconstruction_tasks = {
            task_info.labels[actor_pool.get_logical_id_label_key()]
            for task_info, _ in task_infos
        }

        # The difference represents the actors that were previously released but are now
        # running lineage reconstruction tasks. This ensures we don't double-count
        # actors.
        return len(actors_running_lineage_reconstruction_tasks - actors_in_pool)
