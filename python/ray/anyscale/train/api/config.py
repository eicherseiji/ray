from dataclasses import dataclass
from typing import Tuple, Union

from ray.train.v2.api.config import ScalingConfig as RayScalingConfig


@dataclass
class ScalingConfig(RayScalingConfig):
    num_workers: Union[int, Tuple[int, int]]

    # TODO: Rewrite docstring, and possibly move this into an ElasticScalingConfig
    # While the worker group is healthy, consider resizing the worker group
    # every `elastic_resize_monitor_interval_s` seconds.
    elastic_resize_monitor_interval_s: float = 60.0

    @property
    def elasticity_enabled(self):
        return isinstance(self.num_workers, tuple)

    @property
    def min_workers(self):
        return (
            self.num_workers
            if isinstance(self.num_workers, int)
            else self.num_workers[0]
        )

    @property
    def max_workers(self):
        return (
            self.num_workers
            if isinstance(self.num_workers, int)
            else self.num_workers[1]
        )

    def __post_init__(self):
        super().__post_init__()

        is_fixed = isinstance(self.num_workers, int)
        is_elastic = (
            isinstance(self.num_workers, tuple)
            and len(self.num_workers) == 2
            and all(isinstance(x, int) for x in self.num_workers)
        )
        if not (is_fixed or is_elastic):
            raise ValueError(
                "ScalingConfig(num_workers) must be an int or a tuple of two ints."
            )

        if self.elastic_resize_monitor_interval_s < 0:
            raise ValueError(
                "ScalingConfig(elastic_resize_monitor_interval_s) must be non-negative."
            )
        if self.min_workers > self.max_workers:
            raise ValueError(
                f"Invalid ScalingConfig(num_workers={self.num_workers}): "
                f"min_workers={self.min_workers} must be <= max_workers={self.max_workers}."
            )
        if (
            is_elastic
            and isinstance(self.label_selector, list)
            and len(self.label_selector) != self.max_workers
        ):
            raise ValueError(
                f"`label_selector` is a list of length {len(self.label_selector)}, "
                f"but it must be of length `max_workers={self.max_workers}` instead."
            )
