import ray
from ray._private.ray_constants import DEFAULT_MAX_DIRECT_CALL_OBJECT_SIZE
from typing import Dict, Optional, List, Union

from ray.anyscale.data._internal.util.cached_ray_internals import get_drained_nodes
from ray.data._internal.execution.interfaces.ref_bundle import RefBundle


def all_objects_exist_for_bundle(bundle: RefBundle) -> bool:
    object_locs: Dict[
        str, Union[Optional[int], List[str]]
    ] = ray.experimental.get_local_object_locations(bundle.block_refs)

    drained_nodes = None

    def object_does_exist_lazy(object_info) -> bool:
        nonlocal drained_nodes

        # If the object is small enough, the object isn't placed in the object store and
        # is inlined.
        object_is_inlined = (
            object_info["object_size"] is not None
            and object_info["object_size"] < DEFAULT_MAX_DIRECT_CALL_OBJECT_SIZE
        )
        if object_is_inlined:
            return True

        if drained_nodes is None:
            # Don't recompute drained nodes if all objects are inlined.
            drained_nodes = get_drained_nodes()

        # object doesn't exist on drain nodes
        return len(set(object_info["node_ids"]) - drained_nodes) > 0

    return all(object_does_exist_lazy(obj_info) for obj_info in object_locs.values())
