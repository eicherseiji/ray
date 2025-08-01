import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict

from ray.anyscale.data.checkpoint.checkpoint_filter import RowBasedCheckpointFilter
from ray.anyscale.data.checkpoint.checkpoint_writer import CheckpointWriter
from ray.anyscale.data.checkpoint.interfaces import (
    CheckpointConfig,
)
from ray.anyscale.data.checkpoint.util import normalize_id
from ray.data._internal.delegating_block_builder import DelegatingBlockBuilder
from ray.data.block import Block, BlockAccessor

logger = logging.getLogger(__name__)


class RowBasedFileStorageCheckpointFilter(RowBasedCheckpointFilter):
    """CheckpointFilter implementation for FILE_STORAGE backend, reading
    one checkpoint file per input row.

    For a more efficient implementation, see `FileStorageCheckpointFilter`."""

    def __init__(self, config: CheckpointConfig):
        super().__init__(config)

        self._checkpoint_dir_does_not_exist = False
        if not os.path.exists(self.checkpoint_path_unwrapped):
            self._checkpoint_dir_does_not_exist = True
            logger.warning(
                f"Checkpoint directory {config.checkpoint_path} does not exist. "
                f"No rows will be skipped from checkpoint filtering. "
                f"Ensure that the correct checkpoint directory was configured."
            )

    def filter_rows_for_block(self, block: Block) -> Block:
        if self._checkpoint_dir_does_not_exist:
            # If the checkpoint directory does not exist,
            # simply return the original block.
            return block

        # Generate list of checkpoint file paths to check.
        block_accessor = BlockAccessor.for_block(block)
        files = []
        for row in block_accessor.iter_rows(False):
            _id = row[self.id_column]
            normalized_id = normalize_id(_id)
            files.append(f"{normalized_id}.jsonl")

        # Check if each checkpoint file exists, and re-build
        # the block with only rows that do not have a checkpoint file.
        mask_file_exists = self.check_files_exist_concurrent(files)
        builder = DelegatingBlockBuilder()
        for idx, row in enumerate(block_accessor.iter_rows(False)):
            ckpt_file_key = files[idx]
            if not mask_file_exists[ckpt_file_key]:
                builder.add(row)
        filtered_block = builder.build()
        return filtered_block

    def check_files_exist_concurrent(self, files) -> Dict[str, bool]:
        """Use a `ThreadPoolExecutor` to concurrently check if each file exists.

        Returns a dictionary mapping each file path to a boolean indicating
        whether the file exists or not."""

        def _exists(fpath: str):
            return os.path.exists(
                os.path.join(self.checkpoint_path_unwrapped, fpath),
            )

        with ThreadPoolExecutor(max_workers=self.filter_num_threads) as executor:
            file_exists = list(executor.map(_exists, files))
        return dict(zip(files, file_exists))


class RowBasedFileStorageCheckpointWriter(CheckpointWriter):
    """CheckpointWriter implementation for FILE_STORAGE backend, writing
    one checkpoint file per input row.

    For a more efficient implementation, see `FileStorageCheckpointWriter`."""

    def __init__(self, config: CheckpointConfig):
        super().__init__(config)
        # If the checkpoint output directory does not exist, create it.
        if not os.path.exists(self.checkpoint_path_unwrapped):
            os.makedirs(self.checkpoint_path_unwrapped)

    def write_block_checkpoint(self, block: BlockAccessor):
        with ThreadPoolExecutor(max_workers=self.write_num_threads) as executor:
            futures = []
            future_ids = []
            for row in block.iter_rows(public_row_format=False):
                futures.append(executor.submit(self.write_row_checkpoint, row))
                future_ids.append(row[self.id_col])

            completed_ids = []
            for result in as_completed(futures):
                completed_ids.append(result.result())

            # Verify that all checkpoints were written successfully.
            assert set(future_ids) == set(completed_ids), (
                f"Checkpoint writes failed for rows with IDs: "
                f"`{set(future_ids) - set(completed_ids)}`."
            )

    def write_row_checkpoint(self, row: Dict[str, Any]):
        """Write a checkpoint for a single row to the checkpoint
        output directory given by `self.checkpoint_path_unwrapped`.

        The name of the checkpoint file is `f"{normalize_id(row[self.id_col])}.jsonl"`."""

        row_id = row[self.id_col]
        normalized_id = normalize_id(row_id)
        file_key = os.path.join(
            self.checkpoint_path_unwrapped, f"{normalized_id}.jsonl"
        )

        file = Path(file_key)
        file.parent.mkdir(parents=True, exist_ok=True)
        # TODO: add some checkpoint metadata, like timestamp, etc. in file content
        file.write_text("")
        return row_id
