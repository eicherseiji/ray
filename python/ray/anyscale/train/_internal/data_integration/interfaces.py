from dataclasses import dataclass
from ray.train.v2._internal.callbacks.datasets import (
    DatasetShardMetadata as RayDatasetShardMetadata,
)


@dataclass
class DatasetShardMetadata(RayDatasetShardMetadata):
    world_rank: int
