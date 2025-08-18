import dataclasses
from contextlib import contextmanager
from typing import Optional, Dict, Any, Iterator

from ray.anyscale.data.checkpoint.data_iterator_checkpointer import (
    BatchMetadataWithRowIDs,
    DataIteratorCheckpointer,
    RowIDBasedDataIteratorCheckpointer,
)
from ray.data._internal.block_batching.iter_batches import BatchIterator
from ray.data._internal.execution.interfaces import RefBundle
from ray.data._internal.block_batching.interfaces import Batch
from ray.data.block import Block, BlockAccessor
from ray.util.annotations import PublicAPI


class DataIteratorMixin:
    @PublicAPI(stability="alpha")
    def state_dict(self) -> Dict[str, Any]:
        """Returns the state of the iterator.

        This snapshot is useful upon restoration for resuming the dataset
        and iterator to the same state.

        Returns:
            A dictionary containing the state of the iterator.

        Raises:
            ValueError: If checkpointing is not enabled on this iterator.
        """
        checkpointer = self._get_checkpointer()
        if not checkpointer:
            raise ValueError("Checkpointing is not enabled on this iterator.")

        return checkpointer.state_dict()

    def _enable_checkpointing(self, checkpointer: DataIteratorCheckpointer) -> None:
        self._checkpointer = checkpointer

    def _get_checkpointer(self) -> Optional[DataIteratorCheckpointer]:
        return getattr(self, "_checkpointer", None)

    def _create_batch_iterator(
        self, ref_bundles_iter: Iterator[RefBundle], **kwargs
    ) -> BatchIterator:
        return CheckpointingBatchIterator(
            ref_bundles_iter, checkpointer=self._get_checkpointer(), **kwargs
        )


class CheckpointingBatchIterator(BatchIterator):
    def __init__(
        self,
        ref_bundles_iter: Iterator[RefBundle],
        *,
        checkpointer: Optional[RowIDBasedDataIteratorCheckpointer] = None,
        **kwargs,
    ):
        super().__init__(ref_bundles_iter, **kwargs)
        self._checkpointer = checkpointer

    def _blocks_to_batches(self, blocks: Iterator[Block]) -> Iterator[Batch]:
        for batch in super()._blocks_to_batches(blocks):
            if self._checkpointer:
                row_ids = BlockAccessor.for_block(batch.data).select(
                    columns=[self._checkpointer._id_column]
                )
                batch = dataclasses.replace(
                    batch,
                    metadata=BatchMetadataWithRowIDs(
                        batch_idx=batch.metadata.batch_idx, row_ids=row_ids
                    ),
                )
            yield batch

    def before_epoch_start(self):
        super().before_epoch_start()

        if self._checkpointer:
            self._checkpointer.start_epoch()

    def after_epoch_end(self):
        super().after_epoch_end()

        if self._checkpointer:
            self._checkpointer.end_epoch()

    @contextmanager
    def yield_batch_context(self, batch: Batch):
        if self._checkpointer:
            self._checkpointer.record_yielded_batch(batch)

        with super().yield_batch_context(batch):
            yield
