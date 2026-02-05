import unittest
from unittest.mock import MagicMock, patch

import pytest

import ray

# Import with a name that doesn't start with "Test" to avoid pytest discovery
import ray.data.tests.test_actor_pool_map_operator as oss_test_module
from ray.anyscale.data._internal.execution.operators.actor_pool_map_operator import (
    ActorPoolMapOperator,
    _ActorTaskSelectorImpl,
)
from ray.data._internal.execution.bundle_queue import HashLinkedQueue
from ray.data._internal.execution.operators.actor_pool_map_operator import (
    _ActorPool,
    _ActorTaskSelector,
)
from ray.data._internal.execution.util import make_ref_bundles
from ray.data.tests.conftest import *  # noqa: F403


class AnyscaleTestActorPool(oss_test_module.TestActorPool):
    def _create_task_selector(self, pool: _ActorPool) -> _ActorTaskSelector:
        return ActorPoolMapOperator._create_task_selector(pool)

    @patch(
        "ray.anyscale.data._internal.execution.operators."
        "actor_pool_map_operator.get_draining_nodes"
    )
    @patch(
        "ray.anyscale.data._internal.execution.operators."
        "actor_pool_map_operator.get_actor_locations"
    )
    def test_selector_excludes_actors_on_draining_nodes(
        self, mock_get_actor_locations, mock_get_draining_nodes
    ):
        """Test that actors on draining nodes are excluded from scheduling.

        Verifies that:
            - can_schedule_task() returns False when all actors are on draining nodes
            - can_schedule_task() returns True when at least one actor is on non-draining node
            - select_actors() does not yield actors on draining nodes
        """

        pool = self._create_actor_pool(max_tasks_in_flight=1)

        # Add actors on different nodes
        actor1 = self._add_ready_actor(pool, node_id="node1")
        actor2 = self._add_ready_actor(pool, node_id="node2")

        assert pool.num_running_actors() == 2

        # Set up mock to return actor locations
        mock_get_actor_locations.return_value = {
            pool.get_actor_id(actor1): "node1",
            pool.get_actor_id(actor2): "node2",
        }

        selector = self._create_task_selector(pool)

        def make_queue(n):
            bundles = make_ref_bundles([[i] for i in range(n)])
            queue = HashLinkedQueue()
            for bundle in bundles:
                queue.add(bundle)
            return queue

        # Case 1: No draining nodes, both actors should be schedulable
        mock_get_draining_nodes.return_value = set()
        selector.refresh_state()
        assert selector.can_schedule_task()

        # Verify select_actors yields both actors
        queue = make_queue(2)
        results = list(
            selector.select_actors(queue, actor_locality_enabled=False, strict=True)
        )
        assert len(results) == 2
        selected_actors = {actor for _, actor in results}
        assert selected_actors == {actor1, actor2}

        # Case 2: node1 is draining, only actor2 should be schedulable
        mock_get_draining_nodes.return_value = {"node1"}
        selector.refresh_state()
        assert selector.can_schedule_task()

        # Verify select_actors only yields actor2
        queue = make_queue(1)
        results = list(
            selector.select_actors(queue, actor_locality_enabled=False, strict=True)
        )
        assert len(results) == 1
        selected_actors = {actor for _, actor in results}

        assert selected_actors == {actor2}

        # Case 3: Both nodes draining - no actors should be schedulable
        mock_get_draining_nodes.return_value = {"node1", "node2"}
        selector.refresh_state()
        assert not selector.can_schedule_task()

        # select_actors(strict=True), should raise
        queue = make_queue(1)
        with pytest.raises(AssertionError):
            list(
                selector.select_actors(queue, actor_locality_enabled=False, strict=True)
            )

        # select_actors(strict=False) returns empty iter
        queue = make_queue(1)
        results = list(
            selector.select_actors(queue, actor_locality_enabled=False, strict=False)
        )
        assert len(results) == 0

        # Case 4: node1 stops draining (actor1 becomes schedulable again)
        mock_get_draining_nodes.return_value = {"node2"}
        selector.refresh_state()
        assert selector.can_schedule_task()

        # Verify select_actors only yields actor1
        queue = make_queue(1)
        results = list(
            selector.select_actors(queue, actor_locality_enabled=False, strict=True)
        )
        assert len(results) == 1
        selected_actors = {actor for _, actor in results}

        assert selected_actors == {actor1}


class TestActorTaskSelectorImpl(unittest.TestCase):
    def setUp(self):
        self.mock_actor_pool = MagicMock()
        self.mock_actor_pool.max_tasks_in_flight_per_actor.return_value = 2
        self.mock_actor_pool.running_actors.return_value = {}
        self.selector = _ActorTaskSelectorImpl(self.mock_actor_pool)

        # TODO: Rewrite this test class with `pytest` rather than `unittest` so that we
        # can use the `ray_start_regular_shared` fixture.
        ray.init()

    def tearDown(self) -> None:
        ray.shutdown()

    def test_find_actor_with_locality_ranking(self):
        """Test that _find_actor_with_locality correctly ranks actors based on locality and busyness."""
        # Create actors on different nodes with different busyness levels
        actor1 = MagicMock()  # node1, 0 tasks
        actor2 = MagicMock()  # node1, 1 task
        actor3 = MagicMock()  # node2, 0 tasks
        actor4 = MagicMock()  # node2, 1 task

        self.mock_actor_pool.running_actors.return_value = {
            actor1: MagicMock(actor_location="node1", num_tasks_in_flight=0),
            actor2: MagicMock(actor_location="node1", num_tasks_in_flight=1),
            actor3: MagicMock(actor_location="node2", num_tasks_in_flight=0),
            actor4: MagicMock(actor_location="node2", num_tasks_in_flight=1),
        }

        # Create a bundle with preferred locations
        bundle = MagicMock()
        bundle.get_preferred_object_locations.return_value = {
            "node1": 1024,  # Higher priority (more data)
            "node2": 512,  # Lower priority (less data)
        }

        # Build node map and heap
        node_map = self.selector._build_node_to_actor_map(
            [actor1, actor2, actor3, actor4]
        )
        heap = self.selector._build_actor_busyness_heap(
            [actor1, actor2, actor3, actor4]
        )

        # Test 1: Should prefer node1 (more data) and choose least busy actor
        found_actor = self.selector._find_actor_with_locality(bundle, node_map)
        assert found_actor == actor1  # node1, 0 tasks

        # Test 2: Make actor1 busy, should choose actor2 (still on node1)
        self.mock_actor_pool.running_actors.return_value[actor1].num_tasks_in_flight = 1
        will_invalidate = self.selector._scheduling_will_invalidate_actor(actor1)
        self.selector._update_data_structures_for_actor(
            actor1,
            will_invalidate,
            node_to_actor_map=node_map,
            actor_busyness_rank_heap=heap,
        )
        found_actor = self.selector._find_actor_with_locality(bundle, node_map)
        assert found_actor == actor2  # node1, 1 task

        # Test 3: Make both node1 actors busy, should fall back to node2
        self.mock_actor_pool.running_actors.return_value[actor2].num_tasks_in_flight = 1
        will_invalidate = self.selector._scheduling_will_invalidate_actor(actor2)
        self.selector._update_data_structures_for_actor(
            actor2,
            will_invalidate,
            node_to_actor_map=node_map,
            actor_busyness_rank_heap=heap,
        )
        found_actor = self.selector._find_actor_with_locality(bundle, node_map)
        assert found_actor == actor3  # node2, 0 tasks

        # Test 4: Make all actors busy, should return None
        self.mock_actor_pool.running_actors.return_value[actor3].num_tasks_in_flight = 1
        will_invalidate = self.selector._scheduling_will_invalidate_actor(actor3)
        self.selector._update_data_structures_for_actor(
            actor3,
            will_invalidate,
            node_to_actor_map=node_map,
            actor_busyness_rank_heap=heap,
        )
        self.mock_actor_pool.running_actors.return_value[actor4].num_tasks_in_flight = 1
        will_invalidate = self.selector._scheduling_will_invalidate_actor(actor4)
        self.selector._update_data_structures_for_actor(
            actor4,
            will_invalidate,
            node_to_actor_map=node_map,
            actor_busyness_rank_heap=heap,
        )
        found_actor = self.selector._find_actor_with_locality(bundle, node_map)
        assert found_actor is None

        # Test 5: No preferred locations, should return None
        bundle.get_preferred_object_locations.return_value = {}
        found_actor = self.selector._find_actor_with_locality(bundle, node_map)
        assert found_actor is None

    @patch("ray.anyscale.data._internal.util.cached_ray_internals.get_actor_locations")
    def test_valid_actors_in_pool(self, mock_get_actor_locations):
        """Test filtering valid actors based on state and task count."""
        actor1 = MagicMock()
        actor2 = MagicMock()
        mock_get_actor_locations.return_value = {
            "actor1": "node1",
            "actor2": "node1",
        }
        self.mock_actor_pool.get_logical_ids.return_value = ["actor1", "actor2"]
        self.mock_actor_pool._actor_to_logical_id = {
            actor1: "actor1",
            actor2: "actor2",
        }
        self.mock_actor_pool.running_actors.return_value = {
            actor1: MagicMock(num_tasks_in_flight=0, is_restarting=False),
            actor2: MagicMock(num_tasks_in_flight=0, is_restarting=False),
        }

        valid_actors = self.selector._valid_actors_in_pool()
        assert len(valid_actors) == 2

        # Test with restarting actor
        self.mock_actor_pool.running_actors.return_value[actor1].is_restarting = True
        valid_actors = self.selector._valid_actors_in_pool()
        assert len(valid_actors) == 1
        assert valid_actors[0] == actor2

        # Test with max tasks in flight
        self.mock_actor_pool.running_actors.return_value[actor2].num_tasks_in_flight = 2
        valid_actors = self.selector._valid_actors_in_pool()
        assert len(valid_actors) == 0

    def test_build_node_to_actor_map(self):
        """Test building node to actor mapping."""
        actor1 = MagicMock()
        actor2 = MagicMock()
        actor3 = MagicMock()
        self.mock_actor_pool.running_actors.return_value = {
            actor1: MagicMock(actor_location="node1"),
            actor2: MagicMock(actor_location="node2"),
            actor3: MagicMock(actor_location="node1"),
        }

        node_map = self.selector._build_node_to_actor_map([actor1, actor2, actor3])
        assert len(node_map["node1"]) == 2
        assert len(node_map["node2"]) == 1
        assert actor1 in node_map["node1"]
        assert actor3 in node_map["node1"]
        assert actor2 in node_map["node2"]

    def test_build_actor_busyness_heap(self):
        """Test building actor busyness heap."""
        actor1 = MagicMock()
        actor2 = MagicMock()
        self.mock_actor_pool.running_actors.return_value = {
            actor1: MagicMock(num_tasks_in_flight=1),
            actor2: MagicMock(num_tasks_in_flight=0),
        }

        heap = self.selector._build_actor_busyness_heap([actor1, actor2])
        assert len(heap) == 2
        assert heap[actor1] == 1
        assert heap[actor2] == 0

    def test_update_data_structures_for_actor(self):
        """Test updating data structures when an actor's state changes."""
        actor1 = MagicMock()
        actor2 = MagicMock()
        self.mock_actor_pool.running_actors.return_value = {
            actor1: MagicMock(actor_location="node1", num_tasks_in_flight=0),
            actor2: MagicMock(actor_location="node2", num_tasks_in_flight=0),
        }

        # Test updating node_to_actor_map for invalid actor
        node_map = self.selector._build_node_to_actor_map([actor1, actor2])
        assert actor1 in node_map["node1"]
        self.selector._update_data_structures_for_actor(
            actor1, True, node_to_actor_map=node_map
        )
        assert actor1 not in node_map["node1"]
        assert actor2 in node_map["node2"]  # Unchanged

        # Test updating actor_busyness_rank_heap for invalid actor
        heap = self.selector._build_actor_busyness_heap([actor1, actor2])
        assert actor1 in heap
        self.selector._update_data_structures_for_actor(
            actor1, True, actor_busyness_rank_heap=heap
        )
        assert actor1 not in heap
        assert actor2 in heap  # Unchanged

        # Test updating both structures for invalid actor
        node_map = self.selector._build_node_to_actor_map([actor1, actor2])
        heap = self.selector._build_actor_busyness_heap([actor1, actor2])
        self.selector._update_data_structures_for_actor(
            actor1, True, node_to_actor_map=node_map, actor_busyness_rank_heap=heap
        )
        assert actor1 not in node_map["node1"]
        assert actor1 not in heap

        # Test updating actor_busyness_rank_heap for valid actor
        heap = self.selector._build_actor_busyness_heap([actor1, actor2])
        initial_busyness = heap[actor1]
        self.selector._update_data_structures_for_actor(
            actor1, False, actor_busyness_rank_heap=heap
        )
        assert heap[actor1] == initial_busyness + 1
        assert actor2 in heap  # Unchanged

    def test_scheduling_will_invalidate_actor(self):
        """Test the logic that determines if scheduling a task will make an actor invalid."""
        actor = MagicMock()
        self.mock_actor_pool.running_actors.return_value = {
            actor: MagicMock(num_tasks_in_flight=0)
        }

        # Set up data structures
        node_map = self.selector._build_node_to_actor_map([actor])
        heap = self.selector._build_actor_busyness_heap([actor])

        # With 0 tasks in flight, adding 1 won't invalidate (0 + 1 < 2)
        assert not self.selector._scheduling_will_invalidate_actor(actor)
        # Update data structures to reflect the new task
        self.selector._update_data_structures_for_actor(actor, False, node_map, heap)

        # With 1 task in flight, adding 1 will invalidate it (1 + 1 = 2)
        self.mock_actor_pool.running_actors.return_value[actor].num_tasks_in_flight = 1
        assert self.selector._scheduling_will_invalidate_actor(actor)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
