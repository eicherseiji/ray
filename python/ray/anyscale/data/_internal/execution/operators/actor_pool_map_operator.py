from collections import defaultdict
import math
from typing import Optional, Tuple, Dict, List, Iterator

from ray.anyscale.data._internal.util.cached_ray_internals import (
    get_draining_nodes,
    get_actor_locations,
)
from ray.anyscale.data._internal.util.cached_ray_internals import (
    get_local_ongoing_lineage_reconstruction_tasks,
)
from ray.anyscale.data._internal.util.heapdict import heapdict
from ray.data._internal.execution.bundle_queue.bundle_queue import BundleQueue
from ray.data._internal.execution.interfaces.ref_bundle import RefBundle
from ray.actor import ActorHandle
from ray.data._internal.execution.interfaces import (
    ExecutionResources,
    ReportsExtraResourceUsage,
)
from ray.data._internal.execution.operators.actor_pool_map_operator import (
    ActorPoolMapOperator as OSSActorPoolMapOperator,
    _ActorTaskSelector,
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
            object_store_memory = (
                None
                if self._metrics.obj_store_mem_max_pending_output_per_task is None
                else self._metrics.obj_store_mem_max_pending_output_per_task
                * max_concurrent_tasks_per_actor
            )
            max_per_actor_resource_usage = ExecutionResources(
                cpu=num_cpus_per_actor,
                gpu=num_gpus_per_actor,
                memory=memory_per_actor,
                object_store_memory=object_store_memory,
            )
            max_resource_usage = max_per_actor_resource_usage.scale(max_actors)

        return min_resource_usage, max_resource_usage

    def extra_resource_usage(self) -> ExecutionResources:
        """Returns resources occupied by lineage reconstruction actors.

        This shouldn't include resources used by actors that haven't been reconstructed,
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
        task_infos = get_local_ongoing_lineage_reconstruction_tasks()
        actors_running_lineage_reconstruction_tasks = {
            task_info.labels[actor_pool.get_logical_id_label_key()]
            for task_info, _ in task_infos
        }

        # The difference represents the actors that were previously released but are now
        # running lineage reconstruction tasks. This ensures we don't double-count
        # actors.
        return len(actors_running_lineage_reconstruction_tasks - actors_in_pool)

    @staticmethod
    def _create_task_selector(actor_pool: "_ActorPool") -> "_ActorTaskSelector":
        return _ActorTaskSelectorImpl(actor_pool)


class _ActorTaskSelectorImpl(_ActorTaskSelector):
    def __init__(
        self,
        actor_pool: _ActorPool,
    ):
        super().__init__(actor_pool)

    def _valid_actors_in_pool(self) -> List[ActorHandle]:
        # Filter out actors that are invalid, i.e. actors with number of tasks in
        # flight >= _max_tasks_in_flight or actor_state is not ALIVE.

        draining_node_ids = get_draining_nodes()
        actor_locations = get_actor_locations(tuple(self._actor_pool.get_logical_ids()))
        draining_actors = {
            actor
            for actor in self._actor_pool.running_actors()
            if actor_locations[self._actor_pool._actor_to_logical_id[actor]]
            in draining_node_ids
        }

        return [
            actor
            for actor in self._actor_pool.running_actors()
            if self._actor_pool.running_actors()[actor].num_tasks_in_flight
            < self._actor_pool.max_tasks_in_flight_per_actor()
            and not self._actor_pool.running_actors()[actor].is_restarting
            and actor not in draining_actors
        ]

    def _build_node_to_actor_map(
        self, valid_actors: List[ActorHandle]
    ) -> Dict[str, List[ActorHandle]]:
        node_to_actor_map: Dict[str, List[ActorHandle]] = defaultdict(list)
        for actor in valid_actors:
            actor_node = self._actor_pool.running_actors()[actor].actor_location
            node_to_actor_map[actor_node].append(actor)
        return node_to_actor_map

    def _build_actor_busyness_heap(self, valid_actors: List[ActorHandle]) -> heapdict:
        actor_rank_heap = heapdict()
        for actor in valid_actors:
            actor_rank_heap[actor] = self._actor_pool.running_actors()[
                actor
            ].num_tasks_in_flight
        return actor_rank_heap

    def _find_actor_with_locality(
        self, bundle: RefBundle, node_to_actor_map: Dict[str, List[ActorHandle]]
    ) -> Optional[ActorHandle]:
        """Find the best actor to handle a bundle based on locality preferences.

        Args:
            bundle: The bundle to find an actor for
            node_to_actor_map: Mapping of node IDs to lists of actors on that node

        Returns:
            The best actor to handle the bundle, or None if no suitable actor is found
        """
        preferred_locs = bundle.get_preferred_object_locations()
        if not preferred_locs:
            return None

        # Build a list of (actor, locality_rank, busyness_rank) tuples for all actors
        # in preferred locations
        actor_ranks = []
        for node_id, total_bytes in preferred_locs.items():
            if node_id not in node_to_actor_map:
                continue
            for actor in node_to_actor_map[node_id]:
                # Negate total_bytes to maintain invariant that lower is better
                locality_rank = -total_bytes
                busyness_rank = self._actor_pool.running_actors()[
                    actor
                ].num_tasks_in_flight
                actor_ranks.append((actor, locality_rank, busyness_rank))

        if not actor_ranks:
            return None

        # Pick the actor with the highest rank (lowest value)
        # First by locality rank, then by busyness rank
        return min(actor_ranks, key=lambda x: (x[1], x[2]))[0]

    def _scheduling_will_invalidate_actor(self, actor: ActorHandle) -> bool:
        """Returns True if scheduling a task on this actor will cause it to become
        invalid.
        """
        return (
            self._actor_pool.running_actors()[actor].num_tasks_in_flight + 1
            >= self._actor_pool.max_tasks_in_flight_per_actor()
        )

    def _update_data_structures_for_actor(
        self,
        actor: ActorHandle,
        will_invalidate: bool,
        node_to_actor_map: Optional[Dict[str, List[ActorHandle]]] = None,
        actor_busyness_rank_heap: Optional[heapdict] = None,
    ) -> None:
        """Update data structures when an actor's state changes.

        Args:
            actor: The actor whose state has changed
            will_invalidate: Whether scheduling a task will make the actor invalid
            node_to_actor_map: Optional node to actor mapping to update
            actor_busyness_rank_heap: Optional actor busyness heap to update
        """
        if will_invalidate:
            # Remove actor from data structures if it will become invalid
            if node_to_actor_map is not None:
                node_list = node_to_actor_map[
                    self._actor_pool.running_actors()[actor].actor_location
                ]
                if actor in node_list:
                    node_list.remove(actor)
            if (
                actor_busyness_rank_heap is not None
                and actor in actor_busyness_rank_heap
            ):
                del actor_busyness_rank_heap[actor]
        else:
            # Update busyness in heap if actor will remain valid
            if (
                actor_busyness_rank_heap is not None
                and actor in actor_busyness_rank_heap
            ):
                actor_busyness_rank_heap[actor] = actor_busyness_rank_heap[actor] + 1

    # NOTE: This implementation has an implicit assumption that the actor pool map operator
    # will launch a task with the selected actors emitted from the operator as they are emitted.
    # E.g. for every actor selected, a task should be submitted and the pool's on_task_submitted
    # method should be called with the actor. If the operator doesn't, there might be correctness
    # issues.
    def select_actors(
        self, input_queue: BundleQueue, actor_locality_enabled: bool
    ) -> Iterator[Tuple[RefBundle, ActorHandle]]:
        if not self._actor_pool.running_actors():
            # Actor pool is empty or all actors are still pending.
            return

        # Initialize various data structures to enable more efficient task selection
        valid_actors = self._valid_actors_in_pool()

        if len(valid_actors) == 0:
            # No valid actors, return immediately.
            return

        # Initialized node to actor mapping if actor locality is enabled.
        if actor_locality_enabled:
            node_to_actor_map = self._build_node_to_actor_map(valid_actors)
        else:
            node_to_actor_map = None

        # Rank all valid actors with busyness.
        actor_busyness_rank_heap = self._build_actor_busyness_heap(valid_actors)

        while input_queue and len(actor_busyness_rank_heap) > 0:
            bundle = input_queue.peek_next()
            target_actor = None
            if actor_locality_enabled:
                target_actor = self._find_actor_with_locality(bundle, node_to_actor_map)

            if target_actor is None:
                # Either locality is not enabled or there were no local actors, find through heap strictly based on busyness
                target_actor, _ = actor_busyness_rank_heap.peekitem()

            # Update the data structures
            will_invalidate = self._scheduling_will_invalidate_actor(target_actor)
            self._update_data_structures_for_actor(
                target_actor,
                will_invalidate,
                node_to_actor_map,
                actor_busyness_rank_heap,
            )

            # We remove the bundle and yield the actor to the operator. We do not use pop()
            # in case the queue has changed the order of the bundles.
            input_queue.remove(bundle)
            yield bundle, target_actor
