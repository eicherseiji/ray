from ray.anyscale.data.checkpoint.checkpoint_filter import BatchBasedCheckpointFilter
from ray.anyscale.data.checkpoint.interfaces import CheckpointConfig
from ray.data._internal.execution.streaming_executor import StreamingExecutor
from ray.data.checkpoint.load_checkpoint_callback import (
    LoadCheckpointCallback as OSSLoadCheckpointCallback,
)
from ray.data.datasource import PartitionStyle, PathPartitionFilter


class LoadCheckpointCallback(OSSLoadCheckpointCallback):
    """Anyscale LoadCheckpointCallback with generated_id_column support."""

    def after_execution_succeeds(self, executor: StreamingExecutor):
        """
        Clear the set of files to restore from after the first epoch to avoid
        loading the mid-epoch state on every subsequent epoch.
        """
        super().after_execution_succeeds(executor)
        checkpoint_config = getattr(executor._data_context, "checkpoint_config", None)

        if checkpoint_config:
            checkpoint_config.checkpoint_path_partition_filter = PathPartitionFilter.of(
                filter_fn=lambda _: False,
                style=PartitionStyle.HIVE,
            )

    def _create_checkpoint_filter(
        self, config: CheckpointConfig
    ) -> BatchBasedCheckpointFilter:
        """Override to use Anyscale BatchBasedCheckpointFilter.

        The Anyscale version supports generated_id_column.
        """
        return BatchBasedCheckpointFilter(config)
