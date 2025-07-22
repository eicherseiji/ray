from typing import Tuple, Dict, Set
import ray
import time
from ray.anyscale.data._internal.util.cache import timed_cache
from ray.data._internal.execution.node_trackers.actor_location import (
    get_or_create_actor_location_tracker,
)


@timed_cache(ttl=60)
def get_local_ongoing_lineage_reconstruction_tasks():
    return ray._private.internal_api.get_local_ongoing_lineage_reconstruction_tasks()


@timed_cache(ttl=1)
def get_draining_nodes() -> Dict[str, int]:
    return ray._private.state.state.get_draining_nodes()


@timed_cache(ttl=1)
def get_actor_locations(logical_actor_ids: Tuple[str]):
    """Get the actor locations from logical actor ids.
    NOTE: This function is not thread-safe"""
    return ray.get(
        get_or_create_actor_location_tracker().get_actor_locations.remote(
            logical_actor_ids
        )
    )


def get_drained_nodes() -> Set[str]:
    """Returns the set of nodes that are draining and have passed its deadline."""
    now = time.time()
    return {
        node_id
        for node_id, deadline in get_draining_nodes().items()
        # deadline is in ms
        if deadline < now * 1000
    }
