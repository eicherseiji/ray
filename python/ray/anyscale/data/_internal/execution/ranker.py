"""RayTurbo-specific ranker implementation."""

from typing import TYPE_CHECKING, Tuple

from ray.data._internal.execution.interfaces import PhysicalOperator
from ray.data._internal.execution.resource_manager import ResourceManager
from ray.data._internal.execution.ranker import Ranker

if TYPE_CHECKING:
    from ray.data._internal.execution.streaming_executor_state import OpState, Topology


class LocationAwareRanker(Ranker[Tuple[int, int, int]]):
    """A ranker implementation that takes into consideration bundles
    that are available."""

    def rank_operator(
        self,
        op: PhysicalOperator,
        topology: "Topology",
        resource_manager: ResourceManager,
    ) -> Tuple[int, int, int]:
        """Computes rank for op. *Lower means better rank*

            1. Whether operator's could be throttled (int)
            2. Whether the operator has available input bundles (int)
            3. Operators' object store utilization

        Args:
            op: Operator to rank
            topology: Current execution topology
            resource_manager: Resource manager for usage information

        Returns:
            Rank (tuple) of operator
        """
        state: OpState = topology[op]

        # Calculate ranks. Lower is better.
        throttling_rank = 0 if op.throttling_disabled() else 1
        has_available_bundle = any(
            input_queue._queue.has_next() for input_queue in state.input_queues
        )
        available_bundles_rank = 0 if has_available_bundle else 1

        return (
            throttling_rank,
            available_bundles_rank,
            resource_manager.get_op_usage(op).object_store_memory,
        )
