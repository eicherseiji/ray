import threading
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pyarrow as pa
import pytest

import ray
from ray.anyscale.data._internal.location_aware_bundle_queue import (
    LocationAwareBundleQueue,
)
from ray.data._internal.execution.interfaces import RefBundle
from ray.data.block import BlockAccessor


def _create_bundle(data: Any) -> RefBundle:
    """Create a RefBundle with a single row with the given data."""
    block = pa.Table.from_pydict({"data": [data]})
    block_ref = ray.put(block)
    metadata = BlockAccessor.for_block(block).get_metadata()
    schema = BlockAccessor.for_block(block).schema()
    return RefBundle([(block_ref, metadata)], owns_blocks=False, schema=schema)


def test_location_aware_bundle_queue_thread_safety():
    """Test that the LocationAwareBundleQueue is thread-safe."""
    with patch.object(
        ray, "experimental", MagicMock()
    ) as mock_experimental, patch.object(
        ray._private.state.state, "get_draining_nodes", MagicMock()
    ) as mock_get_draining_nodes:
        mock_experimental.get_local_object_locations.return_value = {
            "": {"node_ids": ["node1"], "object_size": 100}
        }

        mock_get_draining_nodes.return_value = {}  # node_id: deadline_ms

        queue = LocationAwareBundleQueue(update_frequency_s=0)
        exceptions = []
        stop_event = threading.Event()

        def add_pop_worker():
            """Worker that adds and pops bundles from the queue."""
            try:
                for _ in range(1000):
                    if stop_event.is_set():
                        break

                    bundle = MagicMock(size_bytes=lambda: 100, num_rows=lambda: 1)
                    queue.add(bundle)
                    time.sleep(0.001)  # Small delay to increase thread interleaving
                    queue.get_next()
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


def test_remove_duplicate_bundles():
    # Test for https://anyscale1.atlassian.net/browse/DATA-1006.
    bundle = _create_bundle(0)
    queue = LocationAwareBundleQueue(update_frequency_s=0)

    queue.add(bundle)
    queue.add(bundle)
    # At the time of writing, calling `estimate_size_bytes` is needed to reproduce the
    # bug.
    queue.estimate_size_bytes()
    queue.remove(bundle)
    queue.remove(bundle)

    assert len(queue) == 0


def test_estimate_size_bytes_with_duplicate_bundles():
    bundle = _create_bundle(0)
    queue = LocationAwareBundleQueue(update_frequency_s=0)

    queue.add(bundle)
    initial_estimate = queue.estimate_size_bytes()
    queue.add(bundle)

    # The two bundles reference the same underlying objects, so the amount of object
    # store memory used should be the same.
    assert queue.estimate_size_bytes() == initial_estimate


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
