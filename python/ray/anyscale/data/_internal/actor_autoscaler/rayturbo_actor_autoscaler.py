import abc
import math
import time
from collections import defaultdict
from logging import getLogger
from typing import TYPE_CHECKING, Dict, Optional, OrderedDict, Union

from ray.anyscale.data._internal.util.average_calculator import (
    TimeWindowAverageCalculator,
)
from ray.data._internal.actor_autoscaler import (
    DefaultActorAutoscaler,
    ActorPoolScalingRequest,
    AutoscalingActorPool,
    _get_max_scale_up,
)
from ray.data.context import AutoscalingConfig

if TYPE_CHECKING:
    from ray.data._internal.execution.interfaces import PhysicalOperator
    from ray.data._internal.execution.resource_manager import ResourceManager
    from ray.data._internal.execution.streaming_executor_state import OpState, Topology

logger = getLogger(__name__)


class ActorPoolResizingPolicy(abc.ABC):
    """Interface for determining how to resize an actor pool.

    When the actor pool needs to scaling up, `num_to_scale_up` will be called
    to determine how many new actors to add.
    When the actor pool needs to scaling down, `num_to_scale_down` will be called
    to determine how many actors to remove.
    """

    @abc.abstractmethod
    def num_to_scale_up(self, actor_pool: AutoscalingActorPool) -> int:
        """Determine how many new actors to add when the actor pool needs to
        scale up.
        """
        ...

    @abc.abstractmethod
    def num_to_scale_down(self, actor_pool: AutoscalingActorPool) -> int:
        """Determine how many actors to remove when the actor pool needs to
        scale down.
        """
        ...


def _normalize_scaling_up_factor(
    scaling_up_factor: Union[float, Dict[float, float]]
) -> OrderedDict[float, float]:
    """Normalize the input scaling up factor configuration.

    Converts the scaling factor from either a single value or a dictionary mapping
    to an OrderedDict sorted by key in ascending order.
    """
    if isinstance(scaling_up_factor, (float, int)):
        assert scaling_up_factor > 1
        return OrderedDict([(1, scaling_up_factor)])
    else:
        assert isinstance(scaling_up_factor, dict), scaling_up_factor
        for k, v in scaling_up_factor.items():
            assert (
                0 <= k <= 1
            ), f"The key of scaling_up_factor should be in [0, 1], but got {k}"
            assert (
                v > 1
            ), f"The value of scaling_up_factor should be greater than 1, but got {v}"
        # Sort by the key in ascending order
        scaling_up_factor = OrderedDict(sorted(scaling_up_factor.items()))
        return scaling_up_factor


def _get_scaling_up_factor(
    capacity_ratio: float, scaling_up_factor: OrderedDict[float, float]
) -> float:
    """Get the scaling up factor based on the current capacity ratiof.

    NOTE: the input `scaling_up_factor` must be sorted by key in ascending order.
    """
    for k, v in scaling_up_factor.items():
        # Find the first key that is larger than the current utilization.
        if k >= capacity_ratio:
            return v
    return 1.0


class DefaultActorPoolResizingPolicy(ActorPoolResizingPolicy):
    """
    This policy scales up the actor with a factor based on the current capacity
    ratio (current pool / max pool size).
    And always scales down the actor pool by 1 each time.
    """

    # Default scaling up factor for actor pool autoscaling.
    # See docstring of `actor_pool_scaling_up_factor` for the format.
    DEFAULT_SCALING_UP_FACTOR: Dict[float, float] = {0.02: 10, 0.10: 5, 1: 2}

    def __init__(
        self,
        scaling_up_factor: Union[float, Dict[float, float]] = DEFAULT_SCALING_UP_FACTOR,
    ):
        """Initialize the actor pool scaling policy.

        Args:
            scaling_up_factor: Factor by which to scale up actor pools.
                Can be a float value or a dictionary.
                When it's a float value, it means using a constant scaling up factor.
                When it's a dictionary, it means using a scaling up factor based on
                the current capacity ratio (current pool size / max pool size). Each
                key/value pair in the dict means when the capacity ratio is below the
                key, scale up the actor pool size by the value.
                E.g., {0.15: 5, 1: 2} means: when the pool size is below 15% of the
                max size, scale up by 5x; when it's between 15% and 100%, scale
                up by 2x.
        """
        self._actor_pool_scaling_up_factor: OrderedDict[
            float, float
        ] = _normalize_scaling_up_factor(scaling_up_factor)

    def num_to_scale_up(self, actor_pool: AutoscalingActorPool) -> int:
        current_size = actor_pool.current_size()
        factor = _get_scaling_up_factor(
            current_size / actor_pool.max_size(),
            self._actor_pool_scaling_up_factor,
        )
        return (
            min(
                math.ceil(current_size * factor),
                actor_pool.max_size(),
            )
            - current_size
        )

    def num_to_scale_down(self, actor_pool: AutoscalingActorPool) -> int:
        return 1


class RayTurboActorAutoscaler(DefaultActorAutoscaler):
    """Anyscale's proprietary Ray Data actor autoscaler implementation.

    It works in the following way:

      * For each actor pool, check the average actor pool utilization in a time window
        (`actor_pool_util_avg_window_s`) and other factors (see
        `_actor_pool_should_scale_up/down`) to decide whether to scale up or down
        the actor pool.
      * The actor pool size will be increased by `_actor_pool_scaling_up_factor`
        each time when scaling up. And will be decreased by 1
        each time when scaling down.
    """

    # Default interval in seconds to check actor pool utilization.
    DEFAULT_ACTOR_POOL_UTIL_CHECK_INTERVAL_S: float = 0.5
    # Default time window in seconds to calculate the average of
    # actor pool utilization.
    DEFAULT_ACTOR_POOL_UTIL_AVG_WINDOW_S: int = 10

    def __init__(
        self,
        topology: "Topology",
        resource_manager: "ResourceManager",
        *,
        config: AutoscalingConfig,
        actor_pool_util_avg_window_s: float = DEFAULT_ACTOR_POOL_UTIL_AVG_WINDOW_S,
        actor_pool_util_check_interval_s: float = DEFAULT_ACTOR_POOL_UTIL_CHECK_INTERVAL_S,
        actor_pool_resizing_policy: Optional[ActorPoolResizingPolicy] = None,
    ):
        super().__init__(topology, resource_manager, config=config)
        self._actor_pool_scaling_up_threshold = (
            config.actor_pool_util_upscaling_threshold
        )
        self._actor_pool_scaling_down_threshold = (
            config.actor_pool_util_downscaling_threshold
        )
        assert actor_pool_util_avg_window_s > 0
        self._actor_pool_util_calculators = defaultdict(
            lambda: TimeWindowAverageCalculator(window_s=actor_pool_util_avg_window_s)
        )
        assert actor_pool_util_check_interval_s >= 0
        self._actor_pool_util_check_interval_s = actor_pool_util_check_interval_s
        # Last time when the actor pool utilization was checked.
        self._last_actor_pool_util_check_time = 0

        self._actor_pool_resizing_policy = (
            actor_pool_resizing_policy or DefaultActorPoolResizingPolicy()
        )

    def try_trigger_scaling(self):
        """Override to add check interval rate limiting."""
        now = time.time()
        if (
            now - self._last_actor_pool_util_check_time
            < self._actor_pool_util_check_interval_s
        ):
            return

        self._last_actor_pool_util_check_time = now

        super().try_trigger_scaling()

    def _calculate_actor_pool_util(self, actor_pool: AutoscalingActorPool):
        """Calculate the utilization of the given actor pool."""
        if actor_pool.num_running_actors() == 0:
            return None

        util = actor_pool.get_pool_util()
        self._actor_pool_util_calculators[actor_pool].report(util)
        return self._actor_pool_util_calculators[actor_pool].get_average()

    def _derive_target_scaling_config(
        self,
        actor_pool: AutoscalingActorPool,
        op: "PhysicalOperator",
        op_state: "OpState",
    ) -> ActorPoolScalingRequest:
        if op.completed() or (
            op._inputs_complete and op_state.total_enqueued_input_blocks() == 0
        ):
            # Currently we copy-paste the logic from the parent class. Because we have our own calculation of the actor pool utilization.
            num_to_scale_down = self._actor_pool_resizing_policy.num_to_scale_down(
                actor_pool
            )
            return ActorPoolScalingRequest.downscale(
                delta=-num_to_scale_down, force=True, reason="consumed all inputs"
            )

        if actor_pool.current_size() < actor_pool.min_size():
            return ActorPoolScalingRequest.upscale(
                delta=actor_pool.min_size() - actor_pool.current_size(),
                reason="pool below min size",
            )
        elif actor_pool.current_size() > actor_pool.max_size():
            num_to_scale_down = min(
                self._actor_pool_resizing_policy.num_to_scale_down(actor_pool),
                actor_pool.current_size() - actor_pool.max_size(),
            )
            return ActorPoolScalingRequest.downscale(
                delta=-num_to_scale_down,
                reason="pool exceeding max size",
            )

        util = self._calculate_actor_pool_util(actor_pool)
        if util is None:
            return ActorPoolScalingRequest.no_op(reason="no running actors")

        if util >= self._actor_pool_scaling_up_threshold:
            # Do not scale up, if the op has enough free slots for the existing inputs.
            # TODO: this should be normalized with op.metrics.average_num_inputs_per_task.
            if (
                op_state.total_enqueued_input_blocks()
                <= actor_pool.num_free_task_slots()
            ):
                return ActorPoolScalingRequest.no_op(
                    reason="enough free task slots to consume the existing inputs"
                )

            if actor_pool.num_pending_actors() > 0:
                return ActorPoolScalingRequest.no_op(reason="pending actors")
            elif actor_pool.current_size() >= actor_pool.max_size():
                return ActorPoolScalingRequest.no_op(reason="reached max size")
            if not op_state._scheduling_status.under_resource_limits:
                return ActorPoolScalingRequest.no_op(
                    reason="operator exceeding resource quota"
                )

            num_to_scale_up = self._actor_pool_resizing_policy.num_to_scale_up(
                actor_pool
            )

            budget = self._resource_manager.get_budget(op)
            max_scale = _get_max_scale_up(actor_pool, budget)
            if max_scale is not None:
                num_to_scale_up = min(num_to_scale_up, max_scale)

            if num_to_scale_up == 0:
                return ActorPoolScalingRequest.no_op(reason="exceeded resource limits")

            return ActorPoolScalingRequest.upscale(
                delta=num_to_scale_up,
                reason=(
                    f"utilization of {util} >= "
                    f"{self._actor_pool_scaling_up_threshold} "
                    f"(scaling by {num_to_scale_up})"
                ),
            )

        elif util <= self._actor_pool_scaling_down_threshold:
            if actor_pool.current_size() <= actor_pool.min_size():
                return ActorPoolScalingRequest.no_op(reason="reached min size")

            num_to_scale_down = self._actor_pool_resizing_policy.num_to_scale_down(
                actor_pool
            )
            max_can_remove = actor_pool.current_size() - actor_pool.min_size()
            num_to_scale_down = min(num_to_scale_down, max_can_remove)

            return ActorPoolScalingRequest.downscale(
                delta=-num_to_scale_down,
                reason=(
                    f"utilization of {util} <= "
                    f"{self._actor_pool_scaling_down_threshold}"
                ),
            )
        else:
            return ActorPoolScalingRequest.no_op(
                reason=(
                    f"utilization of {util} w/in limits "
                    f"[{self._actor_pool_scaling_down_threshold}, "
                    f"{self._actor_pool_scaling_up_threshold}]"
                )
            )
