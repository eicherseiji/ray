import time
from typing import Dict, Set, Tuple

import ray
from ray.anyscale.data._internal.util.cache import timed_cache
from ray.data._internal.execution.node_trackers.actor_location import (
    get_or_create_actor_location_tracker,
)

# If we submit a task immediately before the deadline,
# Ray Core might not have enough time to launch the
# task and fetch objects before the node is terminated.
# To avoid this, we stop using these inputs some time before the deadline.
DRAIN_DEADLINE_BUFFER_TIME_MS = 5000


@timed_cache(ttl=60)
def get_local_ongoing_lineage_reconstruction_tasks():
    return ray._private.internal_api.get_local_ongoing_lineage_reconstruction_tasks()


@timed_cache(ttl=1)
def get_draining_nodes() -> Dict[str, int]:
    return ray._private.state.state.get_draining_nodes()


@timed_cache(ttl=1)
def get_actor_locations(logical_actor_ids: Tuple[str]) -> Dict[str, str]:
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
        if deadline - DRAIN_DEADLINE_BUFFER_TIME_MS < now * 1000
    }
