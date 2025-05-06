import pytest
import threading
import time
from unittest.mock import MagicMock, patch

import ray

from ray.anyscale.data._internal.location_aware_bundle_queue import (
    LocationAwareBundleQueue,
)


def test_location_aware_bundle_queue_thread_safety():
    """Test that the LocationAwareBundleQueue is thread-safe."""
    with (
        patch.object(ray, "experimental", MagicMock()) as mock_experimental,
        patch.object(LocationAwareBundleQueue, "UPDATE_FREQUENCY_S", 0),
    ):
        mock_experimental.get_local_object_locations.return_value = {
            "": {"node_ids": ["node1"], "object_size": 100}
        }

        queue = LocationAwareBundleQueue()
        exceptions = []
        stop_event = threading.Event()

        def add_pop_worker():
            """Worker that adds and pops bundles from the queue."""
            try:
                for _ in range(1000):
                    if stop_event.is_set():
                        break

                    bundle = MagicMock(size_bytes=lambda: 100)
                    queue.add(bundle)
                    time.sleep(0.001)  # Small delay to increase thread interleaving
                    queue.pop()
            except Exception as e:
                exceptions.append(f"Add/Pop thread: {str(e)}")

        def size_estimation_worker():
            """Worker that repeatedly estimates the queue size."""
            try:
                for _ in range(2000):  # More iterations than add/pop to ensure overlap
                    if stop_event.is_set():
                        break

                    size_bytes = queue.estimate_size_bytes()
                    assert size_bytes in [0, 100]
                    time.sleep(0.0005)  # Different timing to increase interleaving
            except Exception as e:
                exceptions.append(f"Size thread: {str(e)}")

        # Create threads
        add_pop_thread = threading.Thread(target=add_pop_worker)
        size_thread = threading.Thread(target=size_estimation_worker)

        # Start threads
        add_pop_thread.start()
        size_thread.start()

        # Wait for threads to complete (with timeout to prevent hanging)
        add_pop_thread.join(timeout=5)
        size_thread.join(timeout=5)
        stop_event.set()

        assert len(exceptions) == 0, f"Exceptions occurred: {exceptions}"
        assert len(queue) == 0
        assert queue.estimate_size_bytes() == 0


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
