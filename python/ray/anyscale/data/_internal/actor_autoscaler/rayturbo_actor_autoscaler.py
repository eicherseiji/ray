import math
from logging import getLogger
from typing import TYPE_CHECKING, Dict, Optional, OrderedDict, Union

from ray.data._internal.actor_autoscaler import (
    AutoscalingActorPool,
    DefaultActorAutoscaler,
)
from ray.data._internal.actor_autoscaler.actor_pool_resizing_policy import (
    ActorPoolResizingPolicy,
)
from ray.data.context import AutoscalingConfig

if TYPE_CHECKING:
    from ray.data._internal.execution.resource_manager import ResourceManager
    from ray.data._internal.execution.streaming_executor_state import Topology

logger = getLogger(__name__)


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
    """Get the scaling up factor based on the current capacity ratio.

    NOTE: the input `scaling_up_factor` must be sorted by key in ascending order.
    """
    for k, v in scaling_up_factor.items():
        # Find the first key that is larger than the current utilization.
        if k >= capacity_ratio:
            return v
    return 1.0


class RayTurboResizingPolicy(ActorPoolResizingPolicy):
    """
    This policy scales up the actor with a factor based on the current capacity
    ratio (current pool / max pool size).
    And always scales down the actor pool by 1 each time.
    """

    # Default scaling up factor for actor pool autoscaling.
    # See docstring of `scaling_up_factor` parameter for the format.
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
        self._scaling_up_factor: OrderedDict[
            float, float
        ] = _normalize_scaling_up_factor(scaling_up_factor)

    def compute_upscale_delta(
        self, actor_pool: AutoscalingActorPool, util: float
    ) -> int:
        current_size = actor_pool.current_size()
        factor = _get_scaling_up_factor(
            current_size / actor_pool.max_size(),
            self._scaling_up_factor,
        )
        return (
            min(
                math.ceil(current_size * factor),
                actor_pool.max_size(),
            )
            - current_size
        )

    def compute_downscale_delta(self, actor_pool: AutoscalingActorPool) -> int:
        return 1


class RayTurboActorAutoscaler(DefaultActorAutoscaler):
    def __init__(
        self,
        topology: "Topology",
        resource_manager: "ResourceManager",
        *,
        config: AutoscalingConfig,
        actor_pool_resizing_policy: Optional[ActorPoolResizingPolicy] = None,
    ):
        super().__init__(
            topology,
            resource_manager,
            config=config,
            actor_pool_resizing_policy=(
                actor_pool_resizing_policy or RayTurboResizingPolicy()
            ),
        )
