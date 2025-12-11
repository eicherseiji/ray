import ray
from typing import Dict, Optional, List, Union, Set

from ray.anyscale.data._internal.util.cached_ray_internals import get_drained_nodes
from ray.data._internal.execution.interfaces.ref_bundle import RefBundle

# TODO: Ray Core should provide whether an object is in plasma or not.
DEFAULT_MAX_DIRECT_CALL_OBJECT_SIZE = 100 * 1024


def object_does_exist(
    object_info: Dict[str, Union[Optional[int], List[str]]], drained_nodes: Set[str]
) -> bool:
    """Check if an object exists on non-drained nodes.

    Args:
        object_info: Dictionary containing object information including size and node_ids
        drained_nodes: Set of drained node IDs.

    Returns:
        bool: True if object exists on non-drained nodes, False otherwise
    """
    # If the object is small enough, the object isn't placed in the object store and
    # is inlined.
    object_is_inlined = (
        object_info["object_size"] is not None
        and object_info["object_size"] < DEFAULT_MAX_DIRECT_CALL_OBJECT_SIZE
    )
    if object_is_inlined:
        return True

    # object doesn't exist on drain nodes
    return len(set(object_info["node_ids"]) - drained_nodes) > 0


def all_objects_exist_for_bundle(bundle: RefBundle) -> bool:
    """Check if all objects in a bundle exist on non-drained nodes.

    Args:
        bundle: RefBundle containing block references to check

    Returns:
        bool: True if all objects exist on non-drained nodes, False otherwise
    """
    object_locs: Dict[
        str, Union[Optional[int], List[str]]
    ] = ray.experimental.get_local_object_locations(bundle.block_refs)

    drained_nodes = get_drained_nodes()

    return all(
        object_does_exist(obj_info, drained_nodes) for obj_info in object_locs.values()
    )
