import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import pyarrow as pa

from ray.anyscale.data._internal.file_indexer import filter_file_manifest
from ray.anyscale.data._internal.logical.operators.list_files_operator import (
    FileManifest,
    ListFiles,
)
from ray.anyscale.data._internal.readers import FileReader
from ray.anyscale.data._internal.readers.supports_metadata import SupportsSchema
from ray.data._internal.compute import TaskPoolStrategy
from ray.data._internal.logical.interfaces import LogicalOperator, SourceOperator
from ray.data._internal.logical.operators.map_operator import AbstractMap
from ray.data._internal.planner.plan_expression.expression_evaluator import (
    ExpressionEvaluator,
)
from ray.data.block import BlockAccessor, Schema

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import pyarrow.dataset as pd

    from ray.data._internal.execution.interfaces.ref_bundle import RefBundle


class ReadFiles(SourceOperator, AbstractMap):
    def __init__(
        self,
        input_dependency: LogicalOperator,
        *,
        reader: FileReader,
        filesystem,
        filter_expr: Optional["pd.Expression"] = None,
        columns: Optional[List[str]] = None,
        columns_rename: Optional[Dict[str, str]] = None,
        ray_remote_args: Optional[Dict[str, Any]] = None,
        concurrency: Optional[int] = None,
    ):
        super().__init__(
            name="ReadFiles",
            input_op=input_dependency,
            ray_remote_args=ray_remote_args,
            compute=TaskPoolStrategy(concurrency),
        )

        self.reader = reader
        self.filesystem = filesystem
        # TODO assert that projected columns include filtered ones as this
        #      isn't working correctly
        # See https://github.com/apache/arrow/issues/47493
        self.filter_expr = filter_expr
        if columns is not None:
            if not isinstance(columns, list):
                raise TypeError("`columns` must be a list of strings.")
            if not columns:
                raise ValueError("`columns` cannot be an empty list.")
            if not all(isinstance(col, str) for col in columns):
                raise TypeError("All elements in `columns` must be strings.")
        if columns is not None and columns_rename is not None:
            assert set(columns_rename.keys()).issubset(columns), (
                f"All column rename keys must be a subset of the columns list. "
                f"Invalid keys: {set(columns_rename.keys()) - set(columns)}"
            )
        self.columns = columns
        self.columns_rename = columns_rename

    def pushdown_filter(self, filter_expr_strs: List[str]) -> None:
        filter_expr = self._create_filter_expr(filter_expr_strs)
        if self.filter_expr is not None:
            self.filter_expr &= filter_expr
        else:
            self.filter_expr = filter_expr

    def _create_filter_expr(self, filter_expr_strs: List[str]) -> "pd.Expression":
        # This is to handle a case where user specifies
        # read->rename(a->x)->filter("x>10")
        # When filter is pushed down to read, underlying schema wont know about column 'x' and fails.
        # So we need to reconstruct the filter expression with the original column names
        # Note: It is okay if there is a rename after filter pushdown as it doesnt break underlying read
        if not filter_expr_strs:
            return None
        field_changes = {}
        if self.columns_rename:
            for old_col, new_col in self.columns_rename.items():
                field_changes[new_col] = old_col
        filter_expr: "pd.Expression" = ExpressionEvaluator.get_filters(
            filter_expr_strs[0], field_changes=field_changes
        )
        for filter_expr_str in filter_expr_strs[1:]:
            filter_expr &= ExpressionEvaluator.get_filters(
                filter_expr_str, field_changes=field_changes
            )
        return filter_expr

    def infer_schema(self) -> Optional["Schema"]:
        # This method is used by the execution plan to efficiently return metadata
        # without triggering execution.
        if isinstance(self.input_dependency, ListFiles) and isinstance(
            self.reader, SupportsSchema
        ):
            paths = pa.array(self.input_dependency.paths)
            gen = self.input_dependency.file_indexer.list_files(
                paths, filesystem=self.filesystem
            )
            first_file_manifest = next(gen)
            if first_file_manifest and len(first_file_manifest) > 0:
                first_file_manifest = filter_file_manifest(
                    first_file_manifest,
                    self.input_dependency.file_extensions,
                    self.input_dependency.partition_filter,
                )
                first_file_manifest = FileManifest(
                    BlockAccessor.for_block(first_file_manifest.as_block()).slice(0, 1)
                )
                return self.reader.read_schema(
                    first_file_manifest,
                    filesystem=self.filesystem,
                    columns=self.columns,
                )
        return super().infer_schema()

    def output_data(self) -> Optional[List["RefBundle"]]:
        """The output data of this operator if already known, or ``None``."""
        return None
