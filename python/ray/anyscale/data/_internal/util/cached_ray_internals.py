import ray
from ray.anyscale.data._internal.util.cache import timed_cache


@timed_cache(ttl=60)
def get_local_ongoing_lineage_reconstruction_tasks():
    return ray._private.internal_api.get_local_ongoing_lineage_reconstruction_tasks()


@timed_cache(ttl=1)
def get_draining_nodes():
    return set(ray._private.state.state.get_draining_nodes())
