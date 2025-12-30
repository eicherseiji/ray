from ray.anyscale.data.checkpoint.checkpoint_filter import BatchBasedCheckpointFilter
from ray.anyscale.data.checkpoint.interfaces import CheckpointConfig
from ray.data.checkpoint.load_checkpoint_callback import (
    LoadCheckpointCallback as OSSLoadCheckpointCallback,
)


class LoadCheckpointCallback(OSSLoadCheckpointCallback):
    """Anyscale LoadCheckpointCallback with generated_id_column support."""

    def _create_checkpoint_filter(
        self, config: CheckpointConfig
    ) -> BatchBasedCheckpointFilter:
        """Override to use Anyscale BatchBasedCheckpointFilter.

        The Anyscale version supports generated_id_column.
        """
        return BatchBasedCheckpointFilter(config)
