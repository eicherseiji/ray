from abc import ABC, abstractmethod
from typing import List, Optional, Callable, Any, Iterable

from ray.data._internal.execution.interfaces import TaskContext
from ray.data._internal.execution.operators.map_transformer import (
    MapTransformFn,
    MapTransformer,
    BatchMapTransformFn,
    BlockMapTransformFn,
    RowMapTransformFn,
    Row,
)
from ray.data._internal.output_buffer import OutputBlockSizeOption
from ray.data.block import DataBatch, Block


class OptimizedMapTransformer(MapTransformer):
    """Optimized ``MapTransformer`` enabling fusion of ``MapTransformFn``s"""

    def __init__(
        self,
        transform_fns: List[MapTransformFn],
        *,
        init_fn: Optional[Callable[[], None]] = None,
        output_block_size_option_override: Optional[OutputBlockSizeOption] = None,
    ):
        super().__init__(
            transform_fns,
            init_fn=init_fn,
            output_block_size_option_override=output_block_size_option_override,
        )

    @classmethod
    def _combine_transformations(
        cls, ones: List[MapTransformFn], others: List[MapTransformFn]
    ) -> list[Any]:
        return _fuse_transform_fns(ones + others)


class OptimizedMapTransformFn(MapTransformFn, ABC):
    """Optimized version of ``MapTransformFn`` requiring fusion semantic to be
    implemented"""

    @abstractmethod
    def can_fuse(self, next: "OptimizedMapTransformFn") -> bool:
        pass

    @abstractmethod
    def fuse(self, next: "OptimizedMapTransformFn") -> "OptimizedMapTransformFn":
        pass


class OptimizedBatchMapTransformFn(BatchMapTransformFn, OptimizedMapTransformFn):
    def can_fuse(self, next: "MapTransformFn") -> bool:
        if not isinstance(next, OptimizedBatchMapTransformFn):
            # Cannot fuse with downstream map transform fns
            return False

        return (
            self._batch_format == next._batch_format
            and self._batch_size == next._batch_size
        )

    def fuse(self, next: "MapTransformFn") -> "MapTransformFn":
        assert self.can_fuse(
            next
        ), f"Trying to fuse unfusable transformers ({self}, {next})"

        next_batch_transform_fn: OptimizedBatchMapTransformFn = next

        def _fused_batch_fn(
            batches: Iterable[DataBatch], ctx: TaskContext
        ) -> Iterable[DataBatch]:
            return next_batch_transform_fn._batch_fn(self._batch_fn(batches, ctx), ctx)

        return OptimizedBatchMapTransformFn(
            batch_fn=_fused_batch_fn,
            batch_size=next._batch_size,
            batch_format=self._batch_format,
            zero_copy_batch=(
                # Fused batch transformation is zero-copy only if both of the
                # fused ones are
                self._zero_copy_batch
                and next_batch_transform_fn._zero_copy_batch
            ),
            is_udf=self._is_udf or next_batch_transform_fn._is_udf,
            output_block_size_option=(
                # NOTE: Latest output block-size option overrides prior one
                next._output_block_size_option
                if next._output_block_size_option is not None
                else self._output_block_size_option
            ),
        )


class OptimizedBlockMapTransformFn(BlockMapTransformFn, OptimizedMapTransformFn):
    def can_fuse(self, next: "MapTransformFn") -> bool:
        return (
            isinstance(next, BlockMapTransformFn)
            and
            # NOTE: Only transformations can only be fused in case block-shaping
            #       configuration could be merged
            self._disable_block_shaping == next._disable_block_shaping
        )

    def fuse(self, next: "MapTransformFn") -> "MapTransformFn":
        assert self.can_fuse(
            next
        ), f"Trying to fuse unfusable transformers ({self}, {next})"

        next_block_transform: OptimizedBlockMapTransformFn = next

        def _fused_transform(
            blocks: Iterable[Block], ctx: TaskContext
        ) -> Iterable[Block]:
            return next_block_transform._block_fn(self._block_fn(blocks, ctx), ctx)

        return OptimizedBlockMapTransformFn(
            block_fn=_fused_transform,
            is_udf=self._is_udf or next_block_transform._is_udf,
            # NOTE: Latter transformation overrides the block-shaping
            disable_block_shaping=next_block_transform._disable_block_shaping,
            output_block_size_option=(
                # NOTE: Latest output block-size option overrides prior one
                next._output_block_size_option
                if next._output_block_size_option is not None
                else self._output_block_size_option
            ),
        )


class OptimizedRowMapTransformFn(RowMapTransformFn, OptimizedMapTransformFn):
    def can_fuse(self, next: "MapTransformFn") -> bool:
        return isinstance(next, RowMapTransformFn)

    def fuse(self, next: "MapTransformFn") -> "MapTransformFn":
        assert self.can_fuse(
            next
        ), f"Trying to fuse unfusable transformers ({self}, {next})"

        next_row_transform: OptimizedRowMapTransformFn = next

        def _fused_row_fn(inputs: Iterable[Row], ctx: TaskContext) -> Iterable[Row]:
            return next_row_transform._row_fn(self._row_fn(inputs, ctx), ctx)

        return OptimizedRowMapTransformFn(
            row_fn=_fused_row_fn,
            is_udf=self._is_udf or next_row_transform._is_udf,
            output_block_size_option=(
                # NOTE: Latest output block-size option overrides prior one
                next._output_block_size_option
                if next._output_block_size_option is not None
                else self._output_block_size_option
            ),
        )


def _fuse_transform_fns(
    transform_fns: List[OptimizedMapTransformFn],
) -> List[OptimizedMapTransformFn]:
    if len(transform_fns) <= 1:
        return transform_fns

    fused_stack = [transform_fns[0]]

    for next_transform_fn in transform_fns[1:]:
        prev_transform_fn = fused_stack[-1]

        if prev_transform_fn.can_fuse(next_transform_fn):
            # Replace the last element with the fused version
            fused_stack[-1] = prev_transform_fn.fuse(next_transform_fn)
        elif isinstance(prev_transform_fn, OptimizedBatchMapTransformFn) and isinstance(
            next_transform_fn, OptimizedRowMapTransformFn
        ):
            # Skip block shaping for batch transform when following with row transform
            prev_transform_fn._output_block_size_option = OutputBlockSizeOption(
                disable_block_shaping=True
            )
            fused_stack.append(next_transform_fn)
        else:
            fused_stack.append(next_transform_fn)

    return fused_stack
