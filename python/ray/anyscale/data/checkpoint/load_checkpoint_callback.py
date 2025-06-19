import logging
import time
from typing import Optional

import ray
from ray.anyscale.data.checkpoint import CheckpointConfig
from ray.anyscale.data.checkpoint.interfaces import (
    BatchBasedCheckpointFilter,
)
from ray.data._internal.execution.execution_callback import (
    ExecutionCallback,
    remove_execution_callback,
)
from ray.data._internal.execution.streaming_executor import StreamingExecutor
from ray.data.block import Block, BlockAccessor
from ray.types import ObjectRef
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

logger = logging.getLogger(__name__)


@ray.remote(num_cpus=0)
def load_checkpoint(ckpt_filter: BatchBasedCheckpointFilter) -> Block:
    start_t = time.time()
    checkpoint = ckpt_filter.load_checkpoint()
    num_rows = BlockAccessor.for_block(checkpoint).num_rows()
    logger.info(
        "Checkpoint loaded in %.2f seconds with %d rows.",
        time.time() - start_t,
        num_rows,
    )
    return checkpoint


class LoadCheckpointCallback(ExecutionCallback):
    """ExecutionCallback that handles checkpoints."""

    def __init__(self, config: CheckpointConfig):
        assert config is not None
        assert config.is_batch_based()
        self._config = config

        self._ckpt_filter: Optional[
            BatchBasedCheckpointFilter
        ] = BatchBasedCheckpointFilter.create(config)
        self._checkpoint_ref: Optional[ObjectRef[Block]] = None

    def before_execution_starts(self, executor: StreamingExecutor):
        assert self._config is executor._data_context.checkpoint_config

        # Load checkpoint data before execution starts.
        scheduling_strategy = NodeAffinitySchedulingStrategy(
            ray.get_runtime_context().get_node_id(),
            soft=False,
        )
        self._checkpoint_ref = load_checkpoint.options(
            scheduling_strategy=scheduling_strategy,
        ).remote(self._ckpt_filter)

    def after_execution_succeeds(self, executor: StreamingExecutor):
        assert self._config is executor._data_context.checkpoint_config

        # Remove the callback from the DataContext.
        remove_execution_callback(self, executor._data_context)
        # Delete checkpoint data.
        try:
            if self._config.delete_checkpoint_on_success:
                self._ckpt_filter.delete_checkpoint()
        except Exception:
            logger.warning("Failed to delete checkpoint data.", exc_info=True)

    def after_execution_fails(self, executor: StreamingExecutor, error: Exception):
        assert self._config is executor._data_context.checkpoint_config

        # Remove the callback from the DataContext.
        remove_execution_callback(self, executor._data_context)

    def get_checkpoint_ref(self) -> ObjectRef[Block]:
        assert self._checkpoint_ref is not None
        return self._checkpoint_ref
