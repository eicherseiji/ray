import pyarrow

import ray
from ray.anyscale.data.checkpoint.checkpoint_filter import BatchBasedCheckpointFilter
from ray.anyscale.data.checkpoint.interfaces import (
    CheckpointConfig as AnyscaleCheckpointConfig,
)
from ray.data._internal.execution.streaming_executor import StreamingExecutor
from ray.data.block import Block
from ray.data.checkpoint.load_checkpoint_callback import (
    LoadCheckpointCallback as OSSLoadCheckpointCallback,
)
from ray.data.datasource import PartitionStyle, PathPartitionFilter
from ray.types import ObjectRef


class LoadCheckpointCallback(OSSLoadCheckpointCallback):
    """Anyscale LoadCheckpointCallback with generated_id_column support."""

    def _load_checkpoint_data(self) -> ObjectRef[Block]:
        """Override to skip loading checkpoint if restoration is disabled."""
        assert isinstance(self._config, AnyscaleCheckpointConfig)

        if not self._config._should_restore:
            return ray.put(pyarrow.table({}))
        return super()._load_checkpoint_data()

    def after_execution_succeeds(self, executor: StreamingExecutor):
        """Disable checkpoint restoration for subsequent epochs to avoid
        loading the same mid-epoch state multiple times.
        """
        super().after_execution_succeeds(executor)
        checkpoint_config = executor._data_context.checkpoint_config

        if checkpoint_config:
            # Disable checkpoint restoration for subsequent epochs.
            # Example: [ restored at 50% for epoch N ] [ start at the beginning for epoch N+1 ]
            checkpoint_config._should_restore = False
            checkpoint_config.checkpoint_path_partition_filter = PathPartitionFilter.of(
                filter_fn=lambda _: False,
                style=PartitionStyle.HIVE,
            )

    def _create_checkpoint_filter(
        self, config: AnyscaleCheckpointConfig
    ) -> BatchBasedCheckpointFilter:
        """Override to use Anyscale BatchBasedCheckpointFilter.

        The Anyscale version supports generated_id_column.
        """
        return BatchBasedCheckpointFilter(config)
