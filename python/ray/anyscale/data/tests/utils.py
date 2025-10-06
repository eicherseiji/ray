from typing import Any, Dict, Iterable, List

from ray.data._internal.delegating_block_builder import DelegatingBlockBuilder
from ray.data.block import BlockAccessor, DataBatch


def batches_to_rows(batches: Iterable[DataBatch]) -> List[Dict[str, Any]]:
    builder = DelegatingBlockBuilder()
    for batch in batches:
        builder.add_batch(batch)
    block = builder.build()
    block_accessor = BlockAccessor.for_block(block)
    return list(block_accessor.iter_rows(public_row_format=True))
