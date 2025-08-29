from dataclasses import dataclass
from typing import Any, Dict, Optional

from ray.train.v2._internal.callbacks.datasets import (
    DatasetShardMetadata as RayDatasetShardMetadata,
)


@dataclass
class DatasetShardMetadata(RayDatasetShardMetadata):
    world_rank: int
    state_dict: Optional[Dict[str, Any]] = None
