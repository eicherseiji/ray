import threading
import time
from typing import TYPE_CHECKING, Dict, Optional

import ray
from ray.anyscale.data._internal.util.object_utils import all_objects_exist_for_bundle
from ray.data._internal.execution.bundle_queue import BundleQueue, FIFOBundleQueue

if TYPE_CHECKING:
    from ray.data._internal.execution.interfaces import RefBundle


DEFAULT_UPDATE_FREQUENCY_S = 30


class LocationAwareBundleQueue(BundleQueue):
    """Queue that prioritizes bundles that reside in Object Store memory.

    This class is thread-safe.
    """

    def __init__(self, update_frequency_s=DEFAULT_UPDATE_FREQUENCY_S):
        self._update_frequency_s = update_frequency_s

        self._fifo_queue = FIFOBundleQueue()
        self._bundle_nbytes: Dict["RefBundle", int] = {}
        self._last_size_refresh_ts = time.time()
        self._total_nbytes = 0
        self._lock = threading.RLock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._fifo_queue)

    def __contains__(self, bundle: "RefBundle") -> bool:
        with self._lock:
            return bundle in self._fifo_queue

    def add(self, bundle: "RefBundle") -> None:
        with self._lock:
            self._fifo_queue.add(bundle)
            # Use `"RefBundle".size_bytes()` as an initial estimate.
            self._bundle_nbytes[bundle] = bundle.size_bytes()
            self._total_nbytes += self._bundle_nbytes[bundle]

    def get_next(self) -> "RefBundle":
        with self._lock:
            if not self._fifo_queue:
                raise IndexError("You can't pop from an empty queue")

            self._try_ensure_first_bundle_exists()
            bundle = self._fifo_queue.peek_next()
            if bundle is None:
                raise IndexError("Unexpected empty queue")
            self.remove(bundle)
            return bundle

    def peek_next(self) -> Optional["RefBundle"]:
        with self._lock:
            self._try_ensure_first_bundle_exists()
            return self._fifo_queue.peek_next()

    def has_next(self) -> bool:
        bundle = self.peek_next()
        return bundle is not None and all_objects_exist_for_bundle(bundle)

    def remove(self, bundle: "RefBundle") -> None:
        with self._lock:
            if bundle not in self._bundle_nbytes:
                raise ValueError(f"Bundle {bundle} not found in the queue")

            # If there are multiple instances of the bundle in the queue, this method
            # only removes the first one.
            self._fifo_queue.remove(bundle)

            if bundle not in self._fifo_queue:
                # The underlying FIFO queue might contain multiple instances of the
                # same bundle. So, we only decrement the total size if the bundle is
                # not in the queue anymore.
                nbytes = self._bundle_nbytes[bundle]
                del self._bundle_nbytes[bundle]
                self._total_nbytes -= nbytes
                assert self._total_nbytes >= 0, (
                    "Expected the total size of objects in the queue to be non-negative, but "
                    f"got {self._total_nbytes} bytes instead."
                )

    def clear(self) -> None:
        with self._lock:
            self._fifo_queue.clear()
            self._bundle_nbytes.clear()
            self._total_nbytes = 0

    def estimate_size_bytes(self) -> int:
        with self._lock:
            now = time.time()
            # Bundle sizes can change if Ray loses objects or creates replicas. So, we
            # update the sizes every `_update_frequency_s` seconds.
            if now - self._last_size_refresh_ts >= self._update_frequency_s:
                self._refresh_bundle_sizes()
                self._total_nbytes = sum(self._bundle_nbytes.values())
                self._last_size_refresh_ts = now
            return self._total_nbytes

    def _try_ensure_first_bundle_exists(self):
        if not self._fifo_queue:
            return

        num_bundles_skipped = 0
        while num_bundles_skipped < len(self._bundle_nbytes):
            first_bundle = self._fifo_queue.peek_next()
            if first_bundle is None:
                return

            if all_objects_exist_for_bundle(first_bundle):
                break

            self._fifo_queue.get_next()
            self._fifo_queue.add(first_bundle)
            num_bundles_skipped += 1

    def _refresh_bundle_sizes(self) -> None:
        for bundle in self._bundle_nbytes:
            object_locs = ray.experimental.get_local_object_locations(bundle.block_refs)

            nbytes = 0
            for object_info in object_locs.values():
                if object_info["object_size"] is None:
                    nbytes = 0
                else:
                    # There can be copies of the object on multiple nodes. So, to
                    # calculate the total size of the object in shared object store
                    # memory, we multiply the object size by the number of nodes the
                    # object resides on.
                    nbytes += len(object_info["node_ids"]) * object_info["object_size"]

            assert nbytes >= 0, nbytes
            self._bundle_nbytes[bundle] = nbytes

    def is_empty(self) -> bool:
        with self._lock:
            return not self._fifo_queue and not self._bundle_nbytes
