import abc
from dataclasses import dataclass
import logging
import os
from queue import Queue
import threading
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import pyarrow as pa
import pyarrow.fs
import pyarrow.parquet as pq

from ray.data.block import Block, BlockAccessor
from ray.data.context import DataContext
from ray.data._internal.block_batching.interfaces import Batch, BatchMetadata
from ray.data._internal.util import call_with_retry


if TYPE_CHECKING:
    from ray.train.v2._internal.execution.context import DistributedContext


logger = logging.getLogger(__name__)


class DataIteratorCheckpointer(abc.ABC):
    """Abstract base class for data iterator checkpointers.

    Here's how this class should hook into an example training loop:

    load_checkpoint(...)             # (1)
    for epoch in range(num_epochs):  # (2)
        for batch in iterator:       # (3)
            train_step(model, batch)
            save_checkpoint(model)   # (4)

    (1) Load the checkpoint. Here is where the data iterator state should be loaded.
    (2) End the previous epoch and start a new one. Notify the checkpointer of these events.
    (3) Train on a new batch. The checkpointer should record the yielded batch.
    (4) Save the checkpoint. Here is where the data iterator state should be saved
        (e.g. by calling `state_dict`) along with the user's model state.

    Args:
        distributed_context: The distributed context describing the world rank/size
            if running in a distributed setting (e.g. with Ray Train).
    """

    def __init__(
        self,
        # TODO: Pass in the training ingest dataset checkpoint config class here.
        distributed_context: Optional["DistributedContext"] = None,
    ):
        self._distributed_context = distributed_context

    @abc.abstractmethod
    def state_dict(self) -> Dict[str, Any]:
        """Returns the current data iterator state."""
        ...

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Loads the data iterator state from a checkpoint.

        Args:
            state_dict: The state dictionary to load.
        """
        ...

    def record_yielded_batch(self, batch: Batch) -> None:
        """Record that the iterator yielded a batch.

        This should have as little overhead as possible, since the user iteration
        loop calls this method repeatedly. Any significant overhead here will
        impact the performance of the user iteration loop.

        NOTE: The user is responsible for ensuring that the batch is consumed
        before calling `state_dict`, since we assume that all yielded batches
        are consumed at that point. For example, the user should pass all yielded
        batches to the model before checkpointing model and data iterator state.

        Args:
            batch: The batch yielded by the iterator.
        """
        ...

    def start_epoch(self) -> None:
        """Records that a new epoch has started.

        Performs any setup needed for the new epoch.
        """
        ...

    def end_epoch(self) -> None:
        """Records that the current epoch has ended.

        Performs any cleanup needed for the current epoch.
        """
        ...

    @property
    def world_rank(self) -> int:
        """The world rank of the current worker.
        Defaults to 0 if not running in a distributed setting."""
        return self._distributed_context.world_rank if self._distributed_context else 0

    @property
    def world_size(self) -> int:
        """The world size of the current distributed setting.
        Defaults to 1 if not running in a distributed setting."""
        return self._distributed_context.world_size if self._distributed_context else 1


@dataclass
class BatchMetadataWithRowIDs(BatchMetadata):
    """Metadata for a batch with the corresponding row IDs.

    The row ids will be written to checkpoint files to record
    which rows were seen by the iterator consumer.

    Attributes:
        row_ids: The row IDs of the batch.
    """

    row_ids: Block


class RowIDBasedDataIteratorCheckpointer(DataIteratorCheckpointer):
    """Data iterator checkpointer that records yielded row IDs as checkpoint files.

    This checkpointer supports data iterators that yield batches of data, where
    the data has a column with a unique ID for each row. The checkpointer records
    the row IDs of the yielded batches and flushes them asynchronously to a parquet file.

    Upon calling `state_dict`, the checkpointer performs a blocking flush of any
    outstanding row IDs to a checkpoint file.

    Checkpoint files are stored in a hive-style partitioning format:
    {checkpoint_path}/rank={r}/epoch={x}/checkpoint={y}/chunk_{z}.parquet

    The epoch tracks the number of times iterator has reset (and called `start_epoch`).
    The checkpoint index tracks the number of committed data checkpoints so far,
    where each committed checkpoint corresponds to a call to `state_dict`.
    The chunk index tracks the number of chunks written to the current checkpoint directory.

    The rank index tracks the world rank of the current worker to avoid directory collisions.
    Setting the top-level directory to the rank allows each worker to manage a separate
    set of checkpoint files, which simplifies directory management.

    Here's an example of the checkpoint directory structure:

    /tmp/iterator_checkpoint_test/
    └── rank=0
        ├── epoch=0
        │   ├── checkpoint=0
        │   │   ├── chunk_0.parquet
        │   │   └── chunk_1.parquet
        │   ├── checkpoint=1
        │   │   └── chunk_0.parquet
        │   └── checkpoint=2
        │       └── chunk_0.parquet
        └── epoch=1
            ├── checkpoint=0
            │   └── chunk_0.parquet
            └── checkpoint=1
                └── chunk_0.parquet
    └── rank=1
        ├── epoch=0
        │   ├── checkpoint=0
        │   │   ├── chunk_0.parquet
        │   │   └── chunk_1.parquet
        │   ├── checkpoint=1
        │   │   └── chunk_0.parquet
        │   └── checkpoint=2
        │       └── chunk_0.parquet
        └── epoch=1
            ├── checkpoint=0
            │   └── chunk_0.parquet
            └── checkpoint=1
                └── chunk_0.parquet

    Calling `state_dict` triggers an on-demand snapshot of the data iterator state.
    For example: {"epoch": 1, "checkpoint_idx": 2, "epoch_running": True}.

    Upon restoring from this state dict, configure the base dataset to restore
    from a subset of the checkpoint files by using the ordering of the directory
    structure. For example, only read from directories matching this filter:
    `epoch=1/checkpoint<=2/`. Note that this dataset configuration based on a state
    dict is not in scope for this class.

    Args:
        id_column: The name of the column that contains the row IDs.
        checkpoint_path: The path to the checkpoint directory.
        distributed_context: The distributed context describing the world rank/size
            if running in a distributed setting (e.g. with Ray Train).
    """

    TARGET_CHECKPOINT_SIZE_BYTES = 128 * 1024 * 1024  # 128 MB

    def __init__(
        self,
        id_column: str,
        checkpoint_path: str,
        distributed_context: Optional["DistributedContext"] = None,
    ):
        super().__init__(distributed_context)

        # TODO: Pass in the training ingest dataset checkpoint config class instead.
        self._id_column = id_column
        self._fs, self._checkpoint_path_unwrapped = pyarrow.fs.FileSystem.from_uri(
            checkpoint_path
        )

        self._epoch_idx = 0
        self._latest_committed_checkpoint_idx = -1
        self._epoch_running = False

        # Index for tracking the current checkpoint file to write.
        self._chunk_idx = -1

        # Whether the state dict should be updated on the next `state_dict` call.
        self._should_update_state_dict = False

        # Queue for staging row IDs.
        self._row_ids_staging_queue: Queue[Optional[Block]] = Queue()
        # Background thread for flushing row IDs to a checkpoint file.
        self._flush_thread = threading.Thread(
            target=self._process_row_ids_from_queue,
            daemon=True,
            name="DataIteratorCheckpointer-flush",
        )
        # Event for signaling that a forced flush has completed.
        self._flush_completed_event = threading.Event()
        # Exception raised during a flush operation.
        self._flush_exception: Optional[Exception] = None

        self._flush_thread.start()

    def _get_current_checkpoint_directory(self) -> str:
        """Get the current checkpoint directory where files are written.

        Example: {checkpoint_path}/rank={r}/epoch={x}/checkpoint={y}
        """
        # TODO: handle multiple datasets (add the dataset name to the path)
        return os.path.join(
            self._checkpoint_path_unwrapped,
            f"rank={self.world_rank}",
            f"epoch={self._epoch_idx}",
            f"checkpoint={self._current_checkpoint_idx}",
        )

    @property
    def _current_checkpoint_idx(self) -> int:
        """The current checkpoint index.

        This is one greater than the latest committed checkpoint index
        which points to a finalized directory, compared to the current directory
        where we still write files."""
        return self._latest_committed_checkpoint_idx + 1

    def _get_current_checkpoint_path(self) -> str:
        """Get the current checkpoint file to store row IDs.

        Example: {checkpoint_path}/rank={r}/epoch={x}/checkpoint={y}/chunk_{z}.parquet
        """
        return os.path.join(
            self._get_current_checkpoint_directory(),
            f"chunk_{self._chunk_idx}.parquet",
        )

    def _flush_row_ids(self, row_ids_batches: List[pa.Array]):
        """Flush staged row IDs to a checkpoint file."""
        if not row_ids_batches:
            raise ValueError(
                "Got an empty list of row IDs batches. "
                "This method must be called with at least one batch of row IDs."
            )

        self._chunk_idx += 1
        checkpoint_path = self._get_current_checkpoint_path()

        row_ids = pa.chunked_array(row_ids_batches)
        row_ids_table = pa.table({self._id_column: row_ids})

        def _write():
            pq.write_table(row_ids_table, checkpoint_path, filesystem=self._fs)

        try:
            call_with_retry(
                _write,
                description=f"Write checkpoint file: {checkpoint_path}",
                match=DataContext.get_current().retried_io_errors,
            )
        except Exception as e:
            logger.exception(f"Failed to write checkpoint file: {checkpoint_path}")
            self._flush_exception = e

    def _process_row_ids_from_queue(self):
        """Process recorded row IDs from the staging queue.

        This method should run in a background thread and is responsible for
        flushing staged row IDs to a checkpoint file when reaching the target
        file-size or when the main thread requests a forced flush.
        """
        staged_row_ids_size_bytes = 0
        staged_row_id_batches: List[pa.Array] = []

        while True:
            row_ids = self._row_ids_staging_queue.get()
            if row_ids is None:
                # Sentinel value `None` indicates that we should force flush any
                # staged row IDs.
                if staged_row_id_batches:
                    self._flush_row_ids(staged_row_id_batches)
                staged_row_ids_size_bytes = 0
                staged_row_id_batches = []
                # Notify the main thread that the flush is complete.
                self._flush_completed_event.set()
                continue

            row_ids_accessor = BlockAccessor.for_block(row_ids)
            row_ids_array: pa.ChunkedArray = row_ids_accessor.to_arrow().column(
                self._id_column
            )
            size_bytes = row_ids_accessor.size_bytes()

            staged_row_ids_size_bytes += size_bytes
            staged_row_id_batches.extend(row_ids_array.chunks)
            if staged_row_ids_size_bytes >= self.TARGET_CHECKPOINT_SIZE_BYTES:
                self._flush_row_ids(staged_row_id_batches)
                staged_row_ids_size_bytes = 0
                staged_row_id_batches = []

    def _raise_if_flush_failed(self):
        if self._flush_exception:
            raise RuntimeError(
                "Failed to flush one or more checkpoint files. "
                "If this error is retryable, add the error message prefix to: "
                "`ray.data.DataContext.get_current().retried_io_errors`"
            ) from self._flush_exception

    def _flush_all_staged_row_ids(self):
        """Force flush all staged row IDs to a checkpoint file.

        Raises an exception if a previous flush operation failed.
        """
        self._flush_completed_event.clear()
        self._row_ids_staging_queue.put(None)
        self._flush_completed_event.wait()

        self._raise_if_flush_failed()

    def record_yielded_batch(self, batch: Batch):
        assert (
            self._epoch_running
        ), "Must call `start_epoch` before recording yielded batches."

        self._raise_if_flush_failed()

        assert isinstance(batch.metadata, BatchMetadataWithRowIDs), batch.metadata

        self._should_update_state_dict = True
        self._row_ids_staging_queue.put(batch.metadata.row_ids)

    def start_epoch(self):
        if self._epoch_running:
            # The previous epoch did not complete, so we need to end it
            # before starting a new one.
            logger.warning(
                "[DataIteratorCheckpointer] Detected that a new epoch was "
                "started before the previous one completed. "
                "This usually occurs when the dataset iterator is re-created "
                "before the previous epoch has completed. "
                "As a result, the checkpointer is moving to a new epoch. "
                "To avoid this warning, ensure the dataset is fully consumed "
                "before starting a new iterator."
            )
            self.end_epoch()

        # Create the checkpoint directory.
        checkpoint_dir = self._get_current_checkpoint_directory()
        self._setup_new_checkpoint_directory(checkpoint_dir)

        self._epoch_running = True

    def end_epoch(self):
        assert self._epoch_running, "Must call `start_epoch` before ending an epoch."

        self._epoch_running = False

        # Make sure any staged row IDs are flushed before ending the epoch.
        self._flush_all_staged_row_ids()

        self._epoch_idx += 1
        self._latest_committed_checkpoint_idx = -1
        self._should_update_state_dict = False

    def state_dict(self) -> Dict[str, Any]:
        # TODO: make this a barrier across train workers

        if self._should_update_state_dict:
            # Force flush and wait for completion
            self._flush_all_staged_row_ids()

            self._latest_committed_checkpoint_idx += 1
            self._chunk_idx = -1

            # Create the next checkpoint directory.
            new_checkpoint_dir = self._get_current_checkpoint_directory()
            self._setup_new_checkpoint_directory(new_checkpoint_dir)

            self._should_update_state_dict = False

        return {
            "epoch_idx": self._epoch_idx,
            "checkpoint_idx": self._latest_committed_checkpoint_idx,
        }

    def _setup_new_checkpoint_directory(self, new_checkpoint_dir: str):
        """Setup a new checkpoint directory.

        Deletes a partially written checkpoint directory from a previous run.
        """
        from ray.train.v2._internal.execution.storage import (
            _exists_at_fs_path,
            _create_directory,
            _delete_fs_path,
        )

        if _exists_at_fs_path(self._fs, new_checkpoint_dir):
            _delete_fs_path(self._fs, new_checkpoint_dir)

        _create_directory(self._fs, new_checkpoint_dir)

    def load_state_dict(self, state_dict: Dict[str, Any]):
        if self._epoch_running:
            raise RuntimeError(
                "Cannot load state dict while iterating through the dataset mid-epoch."
            )

        self._epoch_idx = state_dict["epoch_idx"]
        self._latest_committed_checkpoint_idx = state_dict["checkpoint_idx"]
