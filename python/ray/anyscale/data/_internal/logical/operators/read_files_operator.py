import copy
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import pyarrow as pa

from ray.anyscale.data._internal.logical.operators.list_files_operator import (
    FileManifest,
    ListFiles,
)
from ray.anyscale.data._internal.readers import FileReader
from ray.anyscale.data._internal.readers.supports_metadata import SupportsSchema
from ray.data._internal.compute import TaskPoolStrategy
from ray.data._internal.datasource.parquet_datasource import (
    _combine_projection,
    _combine_rename_map,
)
from ray.data._internal.logical.interfaces import (
    LogicalOperator,
    SourceOperator,
    LogicalOperatorSupportsProjectionPushdown,
)
from ray.data._internal.logical.operators.map_operator import AbstractMap
from ray.data.block import BlockAccessor, Schema

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import pyarrow.dataset as pd
    from ray.data.expressions import Expr
    from ray.data._internal.execution.interfaces.ref_bundle import RefBundle


def _rename_columns_in_expr(expr: "Expr", column_mapping: Dict[str, str]) -> "Expr":
    """Rename columns in a native expression based on a column mapping."""
    from ray.data.expressions import (
        AliasExpr,
        BinaryExpr,
        ColumnExpr,
        LiteralExpr,
        UnaryExpr,
        UDFExpr,
    )

    if isinstance(expr, ColumnExpr):
        original_name = column_mapping.get(expr.name, expr.name)
        if original_name != expr.name:
            return ColumnExpr(original_name)
        return expr
    elif isinstance(expr, LiteralExpr):
        return expr
    elif isinstance(expr, BinaryExpr):
        return BinaryExpr(
            expr.op,
            _rename_columns_in_expr(expr.left, column_mapping),
            _rename_columns_in_expr(expr.right, column_mapping),
        )
    elif isinstance(expr, UnaryExpr):
        return UnaryExpr(
            expr.op,
            _rename_columns_in_expr(expr.operand, column_mapping),
        )
    elif isinstance(expr, UDFExpr):
        return UDFExpr(
            expr.fn,
            [_rename_columns_in_expr(arg, column_mapping) for arg in expr.args],
            {
                k: _rename_columns_in_expr(v, column_mapping)
                for k, v in expr.kwargs.items()
            },
            expr.data_type,
        )
    elif isinstance(expr, AliasExpr):
        return AliasExpr(
            _rename_columns_in_expr(expr.expr, column_mapping),
            expr.alias,
        )
    else:
        return expr


class ReadFiles(LogicalOperatorSupportsProjectionPushdown, SourceOperator, AbstractMap):
    def __init__(
        self,
        input_dependency: LogicalOperator,
        *,
        reader: FileReader,
        filesystem,
        predicate_expr: Optional[Union["Expr", "pd.Expression"]] = None,
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
        self.predicate_expr = predicate_expr

        if columns is not None:
            if not isinstance(columns, list):
                raise TypeError("`columns` must be a list of strings.")
            if not all(isinstance(col, str) for col in columns):
                raise TypeError("All elements in `columns` must be strings.")
        if columns is not None and columns_rename is not None:
            assert set(columns_rename.keys()).issubset(columns), (
                f"All column rename keys must be a subset of the columns list. "
                f"Invalid keys: {set(columns_rename.keys()) - set(columns)}"
            )
        self.columns = columns
        self.columns_rename = columns_rename

    def supports_projection_pushdown(self) -> bool:
        return True

    def get_current_projection(self) -> Optional[List[str]]:
        return self.columns

    def apply_projection(
        self,
        columns: Optional[List[str]],
        column_rename_map: Optional[Dict[str, str]],
    ) -> LogicalOperator:
        clone = copy.copy(self)

        clone.columns = _combine_projection(self.columns, columns)
        clone.columns_rename = _combine_rename_map(
            self.columns_rename, column_rename_map
        )

        return clone

    def pushdown_predicate(
        self, predicate_expr: Union["Expr", "pd.Expression"]
    ) -> None:
        """Push down predicate expressions (either Ray Data or PyArrow)."""
        from ray.data.expressions import Expr
        import pyarrow.compute as pc

        # Handle Ray Data expressions with column renaming
        if isinstance(predicate_expr, Expr):
            if self.columns_rename:
                column_mapping = {
                    new_col: old_col for old_col, new_col in self.columns_rename.items()
                }
                predicate_expr = _rename_columns_in_expr(predicate_expr, column_mapping)

        # For simplicity, just combine expressions if they're the same type
        if self.predicate_expr is not None:
            if isinstance(self.predicate_expr, Expr) and isinstance(
                predicate_expr, Expr
            ):
                # Both are Ray Data expressions - combine with AND
                self.predicate_expr = self.predicate_expr & predicate_expr
            elif not isinstance(self.predicate_expr, Expr) and not isinstance(
                predicate_expr, Expr
            ):
                # Both are PyArrow expressions - combine with AND
                try:
                    self.predicate_expr = self.predicate_expr & predicate_expr
                except Exception:
                    # If combination fails, replace with new predicate
                    self.predicate_expr = predicate_expr
            else:
                # Mixed types - convert to PyArrow and combine
                try:
                    # Convert both to PyArrow expressions
                    if isinstance(self.predicate_expr, Expr):
                        existing_pa_expr = self.predicate_expr.to_pyarrow()
                    else:
                        existing_pa_expr = self.predicate_expr

                    if isinstance(predicate_expr, Expr):
                        new_pa_expr = predicate_expr.to_pyarrow()
                    else:
                        new_pa_expr = predicate_expr

                    # Combine with AND using PyArrow
                    self.predicate_expr = pc.and_kleene(existing_pa_expr, new_pa_expr)
                except (ValueError, TypeError, AttributeError) as e:
                    # If conversion or combination fails, log warning and keep new predicate
                    logger.warning(
                        f"Failed to combine predicates of mixed types: {e}. "
                        "Using only the new predicate expression."
                    )
                    self.predicate_expr = predicate_expr
        else:
            self.predicate_expr = predicate_expr

    def infer_schema(self) -> Optional["Schema"]:
        # This method is used by the execution plan to efficiently return metadata
        # without triggering execution.
        if isinstance(self.input_dependency, ListFiles) and isinstance(
            self.reader, SupportsSchema
        ):
            paths = pa.array(self.input_dependency.paths)
            gen = self.input_dependency.file_indexer.list_files(
                paths,
                filesystem=self.filesystem,
                file_extensions=self.input_dependency.file_extensions,
                partition_filter=self.input_dependency.partition_filter,
            )
            first_file_manifest = next(gen)
            if first_file_manifest and len(first_file_manifest) > 0:
                first_file_manifest = FileManifest(
                    BlockAccessor.for_block(first_file_manifest.as_block()).slice(0, 1)
                )
                return self.reader.read_schema(
                    first_file_manifest,
                    filesystem=self.filesystem,
                    columns=self.columns,
                )
        return super().infer_schema()

    def can_modify_num_rows(self) -> bool:
        return not self.reader.produces_one_row_per_file()

    def output_data(self) -> Optional[List["RefBundle"]]:
        """The output data of this operator if already known, or ``None``."""
        return None
