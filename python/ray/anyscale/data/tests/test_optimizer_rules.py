from typing import List, Callable, Optional

import pandas as pd
import pyarrow.compute as pc
import pytest
import numpy as np
from PIL import Image
import soundfile as sf

import ray
from ray.anyscale.data._internal.logical.operators.read_files_operator import ReadFiles
from ray.data import Dataset, DataContext
from ray.data._internal.execution.operators.map_transformer import (
    MapTransformFn,
)
from ray.anyscale.data._internal.execution.operators.map_transformer import (
    OptimizedBatchMapTransformFn,
    OptimizedBlockMapTransformFn,
    OptimizedRowMapTransformFn,
)
from ray.data.expressions import col
from ray.data._internal.logical.operators.map_operator import Project
from ray.data._internal.logical.optimizers import LogicalOptimizer, get_execution_plan
from ray.data.tests.conftest import *  # noqa
from ray.data.tests.test_execution_optimizer_limit_pushdown import (
    _check_valid_plan_and_result,
)
from ray.data.tests.util import column_udf
from ray.tests.conftest import *  # noqa


@pytest.fixture
def parquet_ds(ray_start_regular_shared):
    """Fixture to load the Parquet dataset for testing."""
    ds = ray.data.read_parquet("example://iris.parquet")
    assert ds.count() == 150
    return ds


@pytest.fixture
def csv_ds(ray_start_regular_shared):
    """Fixture to load the CSV dataset for testing."""
    ds = ray.data.read_csv("example://iris.csv")
    assert ds.count() == 150
    return ds


def test_apply_local_limit(ray_start_regular_shared):
    def f1(x):
        return x

    ds = ray.data.range(100, parallelism=2).map(f1).limit(1)
    _check_valid_plan_and_result(
        ds,
        "Read[ReadRange] -> Limit[limit=1] -> MapRows[Map(f1)]",
        [{"id": 0}],
        ["ReadRange", "limit=1"],
    )
    assert ds._block_num_rows() == [1]

    # Test larger parallelism still only yields one block.
    ds = ray.data.range(10000, parallelism=50).map(f1).limit(50)
    _check_valid_plan_and_result(
        ds,
        "Read[ReadRange] -> Limit[limit=50] -> MapRows[Map(f1)]",
        [{"id": i} for i in range(50)],
        ["ReadRange", "limit=50"],
    )
    assert ds._block_num_rows() == [50]


def test_filter_with_udfs(parquet_ds):
    """Test filtering with UDFs where predicate pushdown does not occur."""
    filtered_udf_ds = parquet_ds.filter(lambda r: r["sepal.length"] > 5.0)
    filtered_udf_data = filtered_udf_ds.take_all()
    assert filtered_udf_ds.count() == 118
    assert all(record["sepal.length"] > 5.0 for record in filtered_udf_data)
    _check_valid_plan_and_result(
        filtered_udf_ds,
        "ListFiles[ListFiles] -> ReadFiles[ReadFiles] -> Filter[Filter(<lambda>)]",
        filtered_udf_data,
    )


def test_filter_with_expressions(parquet_ds):
    """Test filtering with expressions where predicate pushdown occurs."""
    filtered_udf_data = parquet_ds.filter(lambda r: r["sepal.length"] > 5.0).take_all()
    filtered_expr_ds = parquet_ds.filter(expr="sepal.length > 5.0")
    _check_valid_plan_and_result(
        filtered_expr_ds,
        "ListFiles[ListFiles] -> ReadFiles[ReadFiles]",
        filtered_udf_data,
    )


@pytest.mark.parametrize(
    "source_expr,filter_expr,check",
    [
        # Test with PyArrow compute expressions
        (
            pc.greater(pc.field("sepal.length"), pc.scalar(5.0)),
            "sepal.width > 3.0",
            lambda r: r["sepal.length"] > 5.0 and r["sepal.width"] > 3.0,
        ),
        # Test with PyArrow DNF form
        (
            [("sepal.length", "<", 4.0)],
            "sepal.width < 2.0",
            lambda r: r["sepal.length"] < 4.0 and r["sepal.width"] < 2.0,
        ),
        (
            [[("variety", "=", "Setosa"), ("sepal.length", ">", 5.0)]],
            "petal.length > 1.0",
            lambda r: (r["variety"] == "Setosa" and r["sepal.length"] > 5.0)
            and r["petal.length"] > 1.0,
        ),
    ],
)
def test_filter_pushdown_source_and_op(
    ray_start_regular_shared, source_expr, filter_expr, check
):
    """Test filtering when expressions are provided both in source and operator.

    Tests both PyArrow compute expressions and DNF form filters for source.
    """
    ds = ray.data.read_parquet("example://iris.parquet", filter=source_expr).filter(
        expr=filter_expr
    )
    result = ds.take_all()
    assert all(check(k) for k in result)
    _check_valid_plan_and_result(
        ds,
        "ListFiles[ListFiles] -> ReadFiles[ReadFiles]",
        result,
    )


def test_chained_filter_with_expressions(parquet_ds):
    """Test chained filtering with expressions where combined pushdown occurs."""
    filtered_expr_chained_ds = (
        parquet_ds.filter(expr="sepal.length > 1.0")
        .filter(expr="sepal.length > 2.0")
        .filter(expr="sepal.length > 3.0")
        .filter(expr="sepal.length > 3.0")
        .filter(expr="sepal.length > 5.0")
    )
    filtered_udf_data = parquet_ds.filter(lambda r: r["sepal.length"] > 5.0).take_all()
    _check_valid_plan_and_result(
        filtered_expr_chained_ds,
        "ListFiles[ListFiles] -> ReadFiles[ReadFiles]",
        filtered_udf_data,
    )


@pytest.mark.parametrize(
    "filter_fn,expected_plan",
    [
        (
            lambda ds: ds.filter(lambda r: r["sepal.length"] > 5.0),
            "ListFiles[ListFiles] -> ReadFiles[ReadFiles] -> Filter[Filter(<lambda>)]",
        ),
        (
            lambda ds: ds.filter(expr="sepal.length > 5.0"),
            "ListFiles[ListFiles] -> ReadFiles[ReadFiles]",
        ),
    ],
)
def test_filter_pushdown_csv(csv_ds, filter_fn, expected_plan):
    """Test filtering on CSV files with and without predicate pushdown."""
    filtered_ds = filter_fn(csv_ds)
    filtered_data = filtered_ds.take_all()
    assert filtered_ds.count() == 118
    assert all(record["sepal.length"] > 5.0 for record in filtered_data)
    _check_valid_plan_and_result(
        filtered_ds,
        expected_plan,
        filtered_data,
    )


def test_filter_mixed(csv_ds):
    """Test that mixed function and expressions work."""
    csv_ds = csv_ds.filter(lambda r: r["sepal.length"] < 5.0)
    csv_ds = csv_ds.filter(expr="sepal.length > 3.0")
    csv_ds = csv_ds.filter(expr="sepal.length > 4.0")
    csv_ds = csv_ds.map(lambda x: x)
    csv_ds = csv_ds.filter(expr="sepal.length > 2.0")
    csv_ds = csv_ds.filter(expr="sepal.length > 1.0")
    filtered_expr_data = csv_ds.take_all()
    assert csv_ds.count() == 22
    assert all(record["sepal.length"] < 5.0 for record in filtered_expr_data)
    assert all(record["sepal.length"] > 4.0 for record in filtered_expr_data)
    _check_valid_plan_and_result(
        csv_ds,
        "ListFiles[ListFiles] -> ReadFiles[ReadFiles] -> "
        "Filter[Filter(<lambda>)] -> "
        "Filter[Filter(<expression>)] -> MapRows[Map(<lambda>)] -> "
        "Filter[Filter(<expression>)]",
        filtered_expr_data,
    )


@pytest.mark.parametrize(
    "ds_creator",
    [
        lambda: ray.data.read_parquet("example://iris.parquet"),
        lambda: ray.data.read_csv("example://iris.csv"),
    ],
)
def test_filter_mixed_expression_first(ds_creator):
    """Test that mixed functional and expressions work."""
    ds = ds_creator()
    ds = ds.filter(expr="sepal.length > 3.0")
    ds = ds.filter(expr="sepal.length > 4.0")
    ds = ds.filter(lambda r: r["sepal.length"] < 5.0)
    filtered_expr_data = ds.take_all()
    assert ds.count() == 22
    assert all(record["sepal.length"] < 5.0 for record in filtered_expr_data)
    assert all(record["sepal.length"] > 4.0 for record in filtered_expr_data)
    _check_valid_plan_and_result(
        ds,
        "ListFiles[ListFiles] -> ReadFiles[ReadFiles] -> Filter[Filter(<lambda>)]",
        filtered_expr_data,
    )


def test_filter_mixed_expression_not_readfiles(ray_start_regular_shared):
    """Test that mixed functional and expressions work."""
    ds = ray.data.range(100).filter(expr="id > 1.0")
    ds = ds.filter(expr="id > 2.0")
    ds = ds.filter(lambda r: r["id"] < 5.0)
    filtered_expr_data = ds.take_all()
    assert ds.count() == 2
    assert all(record["id"] < 5.0 for record in filtered_expr_data)
    assert all(record["id"] > 2.0 for record in filtered_expr_data)
    _check_valid_plan_and_result(
        ds,
        "Read[ReadRange] -> Filter[Filter(<expression>)] -> "
        "Filter[Filter(<lambda>)]",
        filtered_expr_data,
    )


def test_read_range_union_with_filter_pushdown(ray_start_regular_shared):
    ds1 = ray.data.range(100, parallelism=2)
    ds2 = ray.data.range(100, parallelism=2)
    ds = ds1.union(ds2).filter(expr="id >= 50")
    result = ds.take_all()
    assert ds.count() == 100
    _check_valid_plan_and_result(
        ds,
        "Read[ReadRange] -> Filter[Filter(<expression>)], "
        "Read[ReadRange] -> Filter[Filter(<expression>)] -> Union[Union]",
        result,
    )


def test_multiple_union_with_filter_pushdown(ray_start_regular_shared):
    ds1 = ray.data.read_parquet("example://iris.parquet")
    ds2 = ray.data.read_parquet("example://iris.parquet")
    ds3 = ray.data.read_parquet("example://iris.parquet")
    ds = ds1.union(ds2).union(ds3).filter(expr="sepal.length > 5.0")
    result = ds.take_all()
    assert ds.count() == 354
    assert all(record["sepal.length"] > 5.0 for record in result)
    _check_valid_plan_and_result(
        ds,
        "ListFiles[ListFiles] -> ReadFiles[ReadFiles], "
        "ListFiles[ListFiles] -> ReadFiles[ReadFiles] -> Union[Union], "
        "ListFiles[ListFiles] -> ReadFiles[ReadFiles] -> Union[Union]",
        result,
    )


def test_multiple_filter_with_union_pushdown_parquet(ray_start_regular_shared):
    ds1 = ray.data.read_parquet("example://iris.parquet")
    ds1 = ds1.filter(expr="sepal.width > 2.0")
    ds2 = ray.data.read_parquet("example://iris.parquet")
    ds2 = ds2.filter(expr="sepal.width > 2.0")
    ds = ds1.union(ds2).filter(expr="sepal.length < 5.0")
    result = ds.take_all()
    assert all(record["sepal.width"] > 2.0 for record in result)
    assert all(record["sepal.length"] < 5.0 for record in result)

    assert ds.count() == 44
    _check_valid_plan_and_result(
        ds,
        "ListFiles[ListFiles] "
        "-> ReadFiles[ReadFiles], "
        "ListFiles[ListFiles] "
        "-> ReadFiles[ReadFiles] -> Union[Union]",
        result,
    )


def test_projection_pushdown(ray_start_regular_shared):
    """Tests that Projection Pushdown works for Parquet."""
    path = "example://iris.parquet"
    ds = ray.data.read_parquet(path)
    cols = ["sepal.length", "petal.width"]
    ds = ds.select_columns(cols)
    # check plan
    logical_plan = ds._plan._logical_plan
    op = logical_plan.dag
    assert isinstance(op, Project), op.name

    # Optimize it
    optimized_logical_plan = LogicalOptimizer().optimize(logical_plan)
    new_op = optimized_logical_plan.dag

    assert isinstance(new_op, ReadFiles), new_op.name
    assert not any(isinstance(op, Project) for op in new_op.post_order_iter())

    readfiles = new_op
    assert readfiles.columns == cols

    target = ray.data.read_parquet(path).to_pandas()[cols]
    df = ds.to_pandas()
    pd.testing.assert_frame_equal(
        df.sort_values(cols).reset_index(drop=True),
        target.sort_values(cols).reset_index(drop=True),
        check_like=True,
    )


def test_projection_pushdown_on_csv(ray_start_regular_shared):
    """Tests that Proj Pushdown works for Native File-Reader codepath"""
    path = "example://iris.csv"
    ds = ray.data.read_csv(path)
    cols = ["sepal.length", "petal.width"]
    ds = ds.select_columns(cols)

    # Optimize it
    optimized_logical_plan = LogicalOptimizer().optimize(ds._plan._logical_plan)
    new_op = optimized_logical_plan.dag

    assert isinstance(new_op, ReadFiles), new_op.name
    assert not any(isinstance(op, Project) for op in new_op.post_order_iter())

    readfiles = new_op
    assert readfiles.columns == cols

    target = ray.data.read_csv(path).to_pandas()[cols]
    df = ds.to_pandas()
    pd.testing.assert_frame_equal(
        df.sort_values(cols).reset_index(drop=True),
        target.sort_values(cols).reset_index(drop=True),
        check_like=True,
    )


def test_projection_pushdown_avoided(ray_start_regular_shared):
    """Tests that Proj Pushdown is avoided when UDFs are provided."""
    path = "example://iris.parquet"
    ds = ray.data.read_parquet(path)
    ds = ds.map_batches(lambda d: d)
    cols = ["sepal.length", "petal.width"]
    ds = ds.select_columns(cols)

    logical_plan = ds._plan._logical_plan
    op = logical_plan.dag
    assert isinstance(op, Project), op.name

    optimized_logical_plan = LogicalOptimizer().optimize(logical_plan)
    new_op = optimized_logical_plan.dag
    assert isinstance(new_op, Project), new_op.name

    target = ray.data.read_parquet(path).to_pandas()[cols]
    df = ds.to_pandas()
    pd.testing.assert_frame_equal(
        df.sort_values(cols).reset_index(drop=True),
        target.sort_values(cols).reset_index(drop=True),
        check_like=True,
    )


def test_projection_pushdown_no_intersection(ray_start_regular_shared):
    """Check that sequential selects with no intersection are not merged."""
    path = "example://iris.parquet"
    ds = ray.data.read_parquet(path)
    ds = ds.select_columns(["sepal.length", "petal.width"])
    ds = ds.select_columns(["sepal.width"])

    logical_plan = ds._plan._logical_plan
    op = logical_plan.dag
    assert isinstance(op, Project), op.name

    expected_error_msg = "Column(s) ['sepal.width'] not found. Available columns: ['petal.width', 'sepal.length']"

    with pytest.raises(KeyError) as excinfo:
        LogicalOptimizer().optimize(logical_plan)

    error_msg = str(excinfo.value)
    assert expected_error_msg in error_msg


def test_projection_select_rename_merge(ray_start_regular_shared):
    """Test that select on renamed column is handled."""
    path = "example://iris.parquet"
    ds = ray.data.read_parquet(path)
    ds = ds.map_batches(lambda d: d)
    ds = ds.select_columns(["sepal.length", "petal.width"])
    ds = ds.rename_columns({"sepal.length": "length", "petal.width": "width"})
    ds = ds.select_columns(["length"])

    logical_plan = ds._plan._logical_plan
    op = logical_plan.dag
    assert isinstance(op, Project), op.name

    optimized_logical_plan = LogicalOptimizer().optimize(logical_plan)
    assert isinstance(optimized_logical_plan.dag, Project)

    select_op = optimized_logical_plan.dag

    assert select_op.exprs == [col("sepal.length").alias("length")]


def test_projection_pushdown_read_rename_columns(ray_start_regular_shared):
    """Test that select columns on ReadFiles is handled."""
    path = "example://iris.parquet"
    ds = ray.data.read_parquet(
        path, columns=["sepal.length", "sepal.width"]
    ).rename_columns({"sepal.length": "length"})

    ds2 = ray.data.read_parquet(path, columns=["sepal.length", "sepal.width"])
    assert ds.count() == ds2.count()
    assert sorted(ds.schema().names) == sorted(["length", "sepal.width"])


def test_projection_pushdown_rename_nonexistent_column(ray_start_regular_shared):
    """
    Test that renaming a column that doesn't exist in projecting_op raises
    an error.
    """
    path = "example://iris.parquet"
    ds = ray.data.read_parquet(path)

    # First projection has no renames, just selects the columns
    ds = ds.select_columns(["sepal.length", "petal.width"])

    # Second projection tries to rename a non-existing column 'col3'
    ds = ds.rename_columns({"col3": "new_col3"})

    logical_plan = ds._plan._logical_plan
    op = logical_plan.dag
    assert isinstance(op, Project), op.name

    # Pattern to match in the error message
    error_msg_pattern = "Column(s) ['col3'] not found. Available columns: ['petal.width', 'sepal.length']"

    with pytest.raises(KeyError) as excinfo:
        LogicalOptimizer().optimize(logical_plan)

    # Use re.search to check for a part of the error message with a pattern
    assert error_msg_pattern in str(excinfo.value)


def test_projection_pushdown_merge_rename(ray_start_regular_shared):
    """
    Test that valid select and renaming merges correctly.
    """
    path = "example://iris.parquet"
    ds = ray.data.read_parquet(path)
    ds = ds.map_batches(lambda d: d)
    ds = ds.select_columns(["sepal.length", "petal.width"])

    # First projection renames 'sepal.length' to 'length'
    ds = ds.rename_columns({"sepal.length": "length"})

    # Second projection renames 'petal.width' to 'width'
    ds = ds.rename_columns({"petal.width": "width"})

    logical_plan = ds._plan._logical_plan
    op = logical_plan.dag
    assert isinstance(op, Project), op.name

    optimized_logical_plan = LogicalOptimizer().optimize(logical_plan)
    assert isinstance(optimized_logical_plan.dag, Project)

    select_op = optimized_logical_plan.dag

    # Check that both "sepal.length" and "petal.width" are present in the columns,
    # regardless of their order.
    assert select_op.exprs == [
        col("sepal.length").alias("length"),
        col("petal.width").alias("width"),
    ]


def test_projection_pushdown_merge_rename_chaining(ray_start_regular_shared):
    """
    Test that valid renaming merges correctly, including renaming chains.
    """
    path = "example://iris.parquet"
    ds = ray.data.read_parquet(path).limit(1)
    ds = ds.map_batches(lambda d: d)

    # First projection renames 'sepal.length' to 'length'
    ds = ds.rename_columns({"sepal.length": "length"})

    # Second projection renames 'length' to 'short_length'
    ds = ds.rename_columns({"length": "short_length"})

    # Third projection renames 'petal.width' to 'width'
    ds = ds.rename_columns({"petal.width": "width"})

    logical_plan = ds._plan._logical_plan
    op = logical_plan.dag
    assert isinstance(op, Project), op.name

    optimized_logical_plan = LogicalOptimizer().optimize(logical_plan)
    assert isinstance(optimized_logical_plan.dag, Project)

    assert ds.take_all() == [
        {
            "petal.length": 1.4,
            "sepal.width": 3.5,
            "short_length": 5.1,
            "variety": "Setosa",
            "width": 0.2,
        }
    ]


def test_projection_pushdown_merge(ray_start_regular_shared):
    """Check that sequential selects with intersection are merged."""
    path = "example://iris.parquet"
    ds = ray.data.read_parquet(path)
    ds = ds.map_batches(lambda d: d)
    cols = ["sepal.length", "petal.width"]
    ds = ds.select_columns(cols)
    ds = ds.select_columns(["petal.width"])

    logical_plan = ds._plan._logical_plan
    op = logical_plan.dag
    assert isinstance(op, Project), op.name
    assert op.exprs == [col("petal.width")]

    optimized_logical_plan = LogicalOptimizer().optimize(logical_plan)
    assert isinstance(optimized_logical_plan.dag, Project)

    select_op = optimized_logical_plan.dag
    assert select_op.exprs == [col("petal.width")]


def test_pushdown_divergent_branches(ray_start_regular_shared):
    """Check that sequential selects with intersection are merged."""
    path = "example://iris.parquet"
    ds = ray.data.read_parquet(path)
    ds2 = ds.select_columns(["petal.width"])

    # Execute ds2 with projection pushdown
    ds2.take(1)

    # Execute ds without projection pushdown
    result = ds.take(1)[0]
    result_keys = list(result.keys())
    assert all(
        key in result_keys
        for key in [
            "sepal.length",
            "sepal.width",
            "petal.length",
            "petal.width",
        ]
    )


def match_transform_fns(
    expected_fns: List[MapTransformFn], got_fns: List[MapTransformFn]
) -> bool:
    assert len(expected_fns) == len(
        got_fns
    ), f"Expected {len(expected_fns)} functions, but got {len(got_fns)}."
    for expected_fn, got_fn in zip(expected_fns, got_fns):
        if not isinstance(got_fn, expected_fn):
            return False
    return True


def match_transform_fns_with_batch_size(
    expected_fns: List[MapTransformFn],
    got_fns: List[MapTransformFn],
    expected_batch_sizes: List[Optional[int]],
) -> bool:
    """Match transform functions and validate their batch sizes."""
    assert len(expected_fns) == len(
        got_fns
    ), f"Expected {len(expected_fns)} functions, but got {len(got_fns)}."
    assert len(expected_fns) == len(
        expected_batch_sizes
    ), f"Expected {len(expected_fns)} batch sizes, but got {len(expected_batch_sizes)}."

    for i, (expected_fn, got_fn, expected_batch_size) in enumerate(
        zip(expected_fns, got_fns, expected_batch_sizes)
    ):
        if not isinstance(got_fn, expected_fn):
            assert False, (
                f"Function {i}: Expected {expected_fn.__name__}, "
                f"but got {got_fn.__class__.__name__}"
            )

        # Validate batch_size for OptimizedBatchMapTransformFn
        if isinstance(got_fn, OptimizedBatchMapTransformFn):
            actual_batch_size = got_fn._batch_size
            assert actual_batch_size == expected_batch_size, (
                f"Function {i}: Expected batch_size {expected_batch_size}, "
                f"but got {actual_batch_size} for {got_fn.__class__.__name__}"
            )

    return True


def match_ds_result(ds: Dataset, expected_output: List[int]) -> bool:
    output = [item["id"] for item in ds.take_all()]
    assert output == expected_output, f"{output} == {expected_output}"
    return True


def test_pushdown_rename_filter(ray_start_regular_shared):
    """rename("sepal.length" -> a).filter(a)."""
    path = "example://iris.parquet"
    ds = (
        ray.data.read_parquet(path)
        .rename_columns({"sepal.length": "a"})
        .filter(expr="a > 2.0")
    )
    ds.take_all()
    assert (
        ds._plan._logical_plan.dag.dag_str
        == "ListFiles[ListFiles] -> ReadFiles[ReadFiles]"
    )
    ds1 = ray.data.read_parquet(path).filter(expr="sepal.length > 2.0")
    assert ds.count() == ds1.count()
    df = ds.to_pandas().rename(columns={"a": "sepal.length"})
    df1 = ds1.to_pandas()
    pd.testing.assert_frame_equal(df, df1)

    logical_plan = ds._plan._logical_plan
    optimized_logical_plan = LogicalOptimizer().optimize(logical_plan)
    read_op: ReadFiles = optimized_logical_plan.dag
    # Check that predicate expression was pushed down and can be converted to expected PyArrow expression
    assert read_op.predicate_expr is not None
    converted_expr = read_op.predicate_expr.to_pyarrow()
    expected_expr = pc.greater(pc.field("sepal.length"), pc.scalar(2.0))
    assert converted_expr.equals(expected_expr)


def test_maprows_repartition_fusion(ray_start_regular_shared):
    """StreamingRepartition after MapRows should be removed and fused into MapRows."""

    def f1(r):
        return {"id": r["id"]}

    ds = ray.data.range(50, override_num_blocks=5).map(f1)
    ds = ds.repartition(target_num_rows_per_block=5)

    # Execute first, then assert on optimized logical plan.
    result = ds.take_all()
    # Expect the StreamingRepartition to be removed from the logical plan.
    # Only MapRows should remain on top of Read.
    assert ds._plan._logical_plan.dag.dag_str == "Read[ReadRange] -> MapRows[Map(f1)]"

    # Validate correctness and that no block exceeds the target.
    assert result == [{"id": i} for i in range(50)]
    block_rows = ds._block_num_rows()
    assert sum(block_rows) == 50


def test_flatmap_repartition_fusion(ray_start_regular_shared):
    """StreamingRepartition after FlatMap should be removed and fused into FlatMap."""

    def duplicate_row(row):
        return [{"id": row["id"]}]

    ds = ray.data.range(50, override_num_blocks=5).flat_map(duplicate_row)
    ds = ds.repartition(target_num_rows_per_block=5)

    # Execute first, then assert on optimized logical plan.
    result = ds.take_all()
    # Expect the StreamingRepartition to be removed from the logical plan.
    assert (
        ds._plan._logical_plan.dag.dag_str
        == "Read[ReadRange] -> FlatMap[FlatMap(duplicate_row)]"
    )

    # Validate correctness and that no block exceeds the target.
    expected = [{"id": i} for i in range(50)]
    assert result == expected
    block_rows = ds._block_num_rows()
    assert sum(block_rows) == 50


def test_pushdown_rename_filter_rename(ray_start_regular_shared):
    """rename("sepal.length" -> a).filter(a).rename(a -> b)."""
    path = "example://iris.parquet"
    ds = (
        ray.data.read_parquet(path)
        .rename_columns({"sepal.length": "a"})
        .filter(expr="a > 2.0")
        .rename_columns({"a": "b"})
    )
    ds.take_all()
    assert (
        ds._plan._logical_plan.dag.dag_str
        == "ListFiles[ListFiles] -> ReadFiles[ReadFiles]"
    )
    ds1 = ray.data.read_parquet(path).filter(expr="sepal.length > 2.0")
    assert ds.count() == ds1.count()
    df = ds.to_pandas().rename(columns={"b": "sepal.length"})
    df1 = ds1.to_pandas()
    pd.testing.assert_frame_equal(df, df1)

    logical_plan = ds._plan._logical_plan
    optimized_logical_plan = LogicalOptimizer().optimize(logical_plan)
    read_op: ReadFiles = optimized_logical_plan.dag
    # Check that predicate expression was pushed down and can be converted to expected PyArrow expression
    assert read_op.predicate_expr is not None
    converted_expr = read_op.predicate_expr.to_pyarrow()
    expected_expr = pc.greater(pc.field("sepal.length"), pc.scalar(2.0))
    assert converted_expr.equals(expected_expr)


def test_pushdown_rename_filter_rename_filter(ray_start_regular_shared):
    """rename("sepal.length" -> a).filter(a).rename(a -> b).filter(b)."""
    path = "example://iris.parquet"
    ds = (
        ray.data.read_parquet(path)
        .rename_columns({"sepal.length": "a"})
        .filter(expr="a > 2.0")
        .rename_columns({"a": "b"})
        .filter(expr="b < 5.0")
    )

    ds.take_all()
    assert (
        ds._plan._logical_plan.dag.dag_str
        == "ListFiles[ListFiles] -> ReadFiles[ReadFiles]"
    )
    ds1 = ray.data.read_parquet(path).filter(
        expr="sepal.length > 2.0 and sepal.length < 5.0"
    )
    assert ds.count() == ds1.count()
    df = ds.to_pandas().rename(columns={"b": "sepal.length"})
    df1 = ds1.to_pandas()
    pd.testing.assert_frame_equal(df, df1)

    logical_plan = ds._plan._logical_plan
    optimized_logical_plan = LogicalOptimizer().optimize(logical_plan)
    read_op: ReadFiles = optimized_logical_plan.dag
    expected_filter_expr = pc.greater(
        pc.field("sepal.length"), pc.scalar(2.0)
    ) & pc.less(pc.field("sepal.length"), pc.scalar(5.0))
    # Check that predicate expression was pushed down and can be converted to expected PyArrow expression
    assert read_op.predicate_expr is not None
    converted_expr = read_op.predicate_expr.to_pyarrow()
    assert converted_expr.equals(expected_filter_expr)


def test_pushdown_rename_filter_rename_filter_rename(ray_start_regular_shared):
    """rename("sepal.length" -> a).filter(a).rename(a -> b).filter(b).rename("sepal.width" -> a).
    Here column a is referred multiple times in rename
    """
    path = "example://iris.parquet"
    ds = (
        ray.data.read_parquet(path)
        .rename_columns({"sepal.length": "a"})
        .filter(expr="a > 2.0")
        .rename_columns({"a": "b"})
        .filter(expr="b < 5.0")
        .rename_columns({"sepal.width": "a"})
    )
    ds.take_all()
    assert (
        ds._plan._logical_plan.dag.dag_str
        == "ListFiles[ListFiles] -> ReadFiles[ReadFiles]"
    )
    ds1 = ray.data.read_parquet(path).filter(
        expr="sepal.length > 2.0 and sepal.length < 5.0"
    )
    assert ds.count() == ds1.count()
    df = ds.to_pandas().rename(columns={"b": "sepal.length", "a": "sepal.width"})
    df1 = ds1.to_pandas()
    pd.testing.assert_frame_equal(df, df1)
    expected_filter_expr = pc.greater(
        pc.field("sepal.length"), pc.scalar(2.0)
    ) & pc.less(pc.field("sepal.length"), pc.scalar(5.0))
    logical_plan = ds._plan._logical_plan
    optimized_logical_plan = LogicalOptimizer().optimize(logical_plan)
    read_op: ReadFiles = optimized_logical_plan.dag
    # Check that predicate expression was pushed down and can be converted to expected PyArrow expression
    assert read_op.predicate_expr is not None
    converted_expr = read_op.predicate_expr.to_pyarrow()
    assert converted_expr.equals(expected_filter_expr)


@pytest.fixture
def fusion_test_cases():
    """Fixture providing test cases for map_batches fusion scenarios."""
    return [
        {
            "name": "basic_fusion",
            "dataset": (
                ray.data.range(5)
                .map_batches(column_udf("id", lambda x: x + 1))  # None batch_size
                .map_batches(column_udf("id", lambda x: x + 2))  # None batch_size
                .map_batches(column_udf("id", lambda x: x + 3))  # None batch_size
                .map_batches(column_udf("id", lambda x: x + 4))  # None batch_size
                .map_batches(column_udf("id", lambda x: x + 5))  # None batch_size
            ),
            "expected_fns": [
                OptimizedBlockMapTransformFn,  # Read operation
                OptimizedBatchMapTransformFn,  # All 5 batch transforms fused into one
            ],
            "expected_batch_sizes": [
                None,  # OptimizedBlockMapTransformFn doesn't have batch_size
                None,  # All None batch_sizes fused together
            ],
            "expected_result": [15, 16, 17, 18, 19],
        },
        {
            "name": "fusion_with_none_batch_size",
            "dataset": (
                ray.data.range(5)
                .map_batches(column_udf("id", lambda x: x + 1))  # None batch_size
                .map_batches(column_udf("id", lambda x: x + 2))  # None batch_size
                .map_batches(column_udf("id", lambda x: x + 3), batch_size=2)
                .map_batches(column_udf("id", lambda x: x + 4), batch_size=3)
            ),
            "expected_fns": [
                OptimizedBatchMapTransformFn,  # None batch_size
                OptimizedBatchMapTransformFn,  # batch_size=2
                OptimizedBatchMapTransformFn,  # batch_size=3
            ],
            "expected_batch_sizes": [
                None,  # None batch_size
                2,  # batch_size=2
                3,  # batch_size=3
            ],
            "expected_result": [10, 11, 12, 13, 14],
        },
    ]


def test_map_batches_transformer_fusion(ray_start_regular_shared, fusion_test_cases):
    """Test various fusion cases for map_batches transformations."""
    for test_case in fusion_test_cases:
        plan = get_execution_plan(test_case["dataset"]._plan._logical_plan)
        fns = plan.dag.get_map_transformer().get_transform_fns()

        # Validate function types and batch sizes
        assert match_transform_fns_with_batch_size(
            test_case["expected_fns"], fns, test_case["expected_batch_sizes"]
        ), f"Failed for test case: {test_case['name']}"
        assert match_ds_result(
            test_case["dataset"], test_case["expected_result"]
        ), f"Failed for test case: {test_case['name']}"


def test_repartition_build_output(ray_start_regular_shared):
    """Test repartition with target_num_rows_per_block"""
    target_num_rows_per_block = 4
    ds = ray.data.range(20).repartition(
        target_num_rows_per_block=target_num_rows_per_block
    )
    plan = get_execution_plan(ds._plan._logical_plan)
    fns = plan.dag.get_map_transformer().get_transform_fns()
    expected_fns = [
        OptimizedBlockMapTransformFn,
    ]
    assert match_transform_fns(expected_fns, fns)
    repartition_fn = fns[-1]
    assert repartition_fn.target_num_rows_per_block == target_num_rows_per_block
    assert match_ds_result(ds, list(range(20)))


def test_repartition_fusion_build_output(ray_start_regular_shared):
    """Test repartition fusion with target_num_rows_per_block"""
    target_num_rows_per_block = 20
    ds = (
        ray.data.range(20)
        .map(column_udf("id", lambda x: x))
        .map_batches(column_udf("id", lambda x: x), batch_size=1)
        .repartition(target_num_rows_per_block=4)
        .repartition(target_num_rows_per_block=5)
        .repartition(target_num_rows_per_block=10)
        .repartition(target_num_rows_per_block=target_num_rows_per_block)
    )
    plan = get_execution_plan(ds._plan._logical_plan)

    assert (
        "InputDataBuffer[Input] -> "
        "TaskPoolMapOperator[ReadRange] -> "
        "TaskPoolMapOperator[Map(<lambda>)->MapBatches(<lambda>)] -> "
        "TaskPoolMapOperator[StreamingRepartition]" == plan.dag.dag_str
    )

    fns = plan.dag.get_map_transformer().get_transform_fns()
    expected_fns = [
        OptimizedBlockMapTransformFn,
    ]
    assert match_transform_fns(expected_fns, fns)
    repartition_fn = fns[-1]
    assert repartition_fn.target_num_rows_per_block == target_num_rows_per_block
    assert match_ds_result(ds, list(range(20)))


@pytest.fixture
def non_fusion_test_cases():
    """Fixture providing test cases for map_batches non-fusion scenarios."""
    return [
        {
            "name": "non_fusion_with_other_none_batch_size",
            "dataset": (
                ray.data.range(5)
                .map_batches(column_udf("id", lambda x: x + 1), batch_size=2)
                .map_batches(column_udf("id", lambda x: x + 2), batch_size=3)
                .map_batches(column_udf("id", lambda x: x + 3))  # None batch_size
            ),
            "expected_fns": [
                OptimizedBatchMapTransformFn,  # batch_size=2
                OptimizedBatchMapTransformFn,  # batch_size=3
                OptimizedBatchMapTransformFn,  # None
            ],
            "expected_batch_sizes": [
                2,  # batch_size=2
                3,  # batch_size=3
                None,  # None batch_size
            ],
            "expected_result": [6, 7, 8, 9, 10],
        },
        {
            "name": "non_fusion_with_incompatible_batch_sizes",
            "dataset": (
                ray.data.range(5)
                .map_batches(column_udf("id", lambda x: x + 1), batch_size=3)
                .map_batches(column_udf("id", lambda x: x + 2), batch_size=2)
                .map_batches(column_udf("id", lambda x: x + 3), batch_size=5)
            ),
            "expected_fns": [
                OptimizedBatchMapTransformFn,  # batch_size=3
                OptimizedBatchMapTransformFn,  # batch_size=2
                OptimizedBatchMapTransformFn,  # batch_size=5
            ],
            "expected_batch_sizes": [
                3,  # batch_size=3
                2,  # batch_size=2
                5,  # batch_size=5
            ],
            "expected_result": [6, 7, 8, 9, 10],
        },
    ]


def test_map_batches_transformer_non_fusion(
    ray_start_regular_shared, non_fusion_test_cases
):
    """Test various non-fusion cases for map_batches transformations."""
    for test_case in non_fusion_test_cases:
        plan = get_execution_plan(test_case["dataset"]._plan._logical_plan)
        fns = plan.dag.get_map_transformer().get_transform_fns()

        # Validate function types and batch sizes
        assert match_transform_fns_with_batch_size(
            test_case["expected_fns"], fns, test_case["expected_batch_sizes"]
        ), f"Failed for test case: {test_case['name']}"
        assert match_ds_result(
            test_case["dataset"], test_case["expected_result"]
        ), f"Failed for test case: {test_case['name']}"


def test_map_rows_transformer_fusion(ray_start_regular_shared):
    """Test fusion of multiple map row transformations."""

    ds = (
        ray.data.range(5)
        .map(column_udf("id", lambda x: x + 1))
        .map(column_udf("id", lambda x: x + 2))
        .map(column_udf("id", lambda x: x + 3))
        .map(column_udf("id", lambda x: x + 4))
        .map(column_udf("id", lambda x: x + 5))
    )

    plan = get_execution_plan(ds._plan._logical_plan)
    fns = plan.dag.get_map_transformer().get_transform_fns()
    # With new architecture, all OptimizedRowMapTransformFn are fused into one
    expected_fns = [
        OptimizedBlockMapTransformFn,  # ReadRange
        OptimizedRowMapTransformFn,  # All 5 row transforms fused into one
    ]
    assert match_transform_fns(expected_fns, fns)
    assert match_ds_result(ds, [15, 16, 17, 18, 19])


@pytest.fixture(scope="module")
def data_context_override(request):
    overrides = getattr(request, "param", {})

    ctx = DataContext.get_current()
    copy = ctx.copy()

    for k, v in overrides.items():
        assert hasattr(ctx, k), f"Key '{k}' not found in DataContext"

        setattr(ctx, k, v)

    yield ctx

    DataContext._set_current(copy)


@pytest.mark.parametrize(
    "data_context_override",
    [
        {"_enable_read_files_fusion_override": True},
        {"_enable_read_files_fusion_override": False},
        {"_enable_read_files_fusion_override": None},
    ],
    indirect=True,
)
def test_read_files_fusion(ray_start_regular_shared, data_context_override):
    """Test that ReadFiles gets fused with the following map,
    but not with the previous ListFiles."""

    if data_context_override._enable_read_files_fusion_override:
        expected_plan_str = (
            "TaskPoolMapOperator[ListFiles] -> "
            "TaskPoolMapOperator[ReadFiles->Map(<lambda>)]"
        )
    else:
        expected_plan_str = (
            "TaskPoolMapOperator[ListFiles] -> "
            "TaskPoolMapOperator[ReadFiles] -> "
            "TaskPoolMapOperator[Map(<lambda>)]"
        )

    ds = ray.data.read_parquet("example://iris.parquet").map(lambda x: x)
    dag_str = get_execution_plan(ds._logical_plan).dag.dag_str
    assert expected_plan_str in dag_str


def test_map_batches_to_map_rows_transforms(ray_start_regular_shared):
    """Test map_batches() and map() transforms with skipped block shaping optimization."""
    ds = ray.data.range(5).map_batches(lambda x: x, batch_size=2).map(lambda x: x)

    plan = get_execution_plan(ds._plan._logical_plan)
    fns = plan.dag.get_map_transformer().get_transform_fns()
    # With new architecture: OptimizedBatchMapTransformFn handles blocks->batches internally,
    # OptimizedRowMapTransformFn handles blocks->rows internally
    expected_fns = [
        OptimizedBatchMapTransformFn,
        OptimizedRowMapTransformFn,
    ]
    assert match_transform_fns(expected_fns, fns)

    # Verify that the batch transform fn has skipped block shaping optimization
    batch_transform_fn = fns[0]
    assert batch_transform_fn._output_block_size_option.disable_block_shaping is True

    # Verify that the row transform fn does NOT skip block shaping
    row_transform_fn = fns[1]
    assert row_transform_fn._output_block_size_option.disable_block_shaping is False

    assert match_ds_result(ds, list(range(5)))


@pytest.mark.parametrize(
    "first_op,second_op,expected_columns,unexpected_columns",
    [
        # with_column -> select_columns
        (
            lambda ds: ds.with_column("length_plus_one", col("sepal.length") + 1),
            lambda ds: ds.select_columns(["sepal.length", "length_plus_one"]),
            ["sepal.length", "length_plus_one"],
            ["sepal.width", "petal.length", "petal.width", "variety"],
        ),
        # with_column -> rename_columns
        (
            lambda ds: ds.with_column("length_plus_one", col("sepal.length") + 1),
            lambda ds: ds.rename_columns({"sepal.length": "renamed_length"}),
            [
                "renamed_length",
                "length_plus_one",
                "sepal.width",
                "petal.length",
                "petal.width",
                "variety",
            ],
            ["sepal.length"],
        ),
        # TODO: Re-add these parameters after deprecating cols, cols_rename in Project.
        # # select_columns -> with_column
        # (
        #     lambda ds: ds.select_columns(["sepal.length", "sepal.width"]),
        #     lambda ds: ds.with_column("length_plus_one", col("sepal.length") + 1),
        #     ["sepal.length", "sepal.width", "length_plus_one"],
        #     ["petal.length", "petal.width", "variety"],
        # ),
        # # rename_columns -> with_column
        # (
        #     lambda ds: ds.rename_columns({"sepal.length": "renamed_length"}),
        #     lambda ds: ds.with_column("length_doubled", col("renamed_length") * 2),
        #     [
        #         "renamed_length",
        #         "length_doubled",
        #         "sepal.width",
        #         "petal.length",
        #         "petal.width",
        #         "variety",
        #     ],
        #     ["sepal.length"],
        # ),
    ],
)
def test_projection_pushdown_exprs_and_cols_combinations(
    ray_start_regular_shared, first_op, second_op, expected_columns, unexpected_columns
):
    """Test ProjectionPushdown correctly handles combinations of expressions and column operations.

    This test ensures that when Project operations with expressions (from with_column)
    are combined with column operations (from select_columns/rename_columns), both
    types of operations are preserved and executed correctly.
    """
    path = "example://iris.parquet"
    ds = ray.data.read_parquet(path)

    # Apply the two operations
    ds = first_op(ds)
    ds = second_op(ds)

    # Execute to trigger optimization
    result = ds.take(1)[0]
    result_keys = list(result.keys())

    # Verify expected columns are present
    for col_name in expected_columns:
        assert (
            col_name in result_keys
        ), f"Expected '{col_name}' in result keys: {result_keys}"

    # Verify unexpected columns are not present
    for col_name in unexpected_columns:
        assert (
            col_name not in result_keys
        ), f"Unexpected '{col_name}' in result keys: {result_keys}"


def test_projection_pushdown_multiple_exprs_with_select(ray_start_regular_shared):
    """Test that multiple expressions combined with column selection work correctly."""

    path = "example://iris.parquet"
    ds = ray.data.read_parquet(path)

    # Add multiple columns with expressions
    ds = ds.with_column("length_plus_one", col("sepal.length") + 1)
    ds = ds.with_column("width_times_two", col("sepal.width") * 2)

    # Select specific columns including both new ones
    ds = ds.select_columns(["sepal.length", "length_plus_one", "width_times_two"])

    result = ds.take(1)[0]
    result_keys = list(result.keys())

    # Should have exactly the selected columns
    expected = ["sepal.length", "length_plus_one", "width_times_two"]
    unexpected = ["sepal.width", "petal.length", "petal.width", "variety"]

    for col_name in expected:
        assert (
            col_name in result_keys
        ), f"Expected '{col_name}' in result keys: {result_keys}"

    for col_name in unexpected:
        assert (
            col_name not in result_keys
        ), f"Unexpected '{col_name}' in result keys: {result_keys}"


def test_limit_pushdown_dont_push_through_readfiles(ray_start_regular_shared):
    """Test that limit is NOT incorrectly pushed through ReadFiles operator
       for Datasources that produce more than 1 row per file.

    This test reproduces the specific bug scenario where:
    - Original correct pipeline: ListFiles -> ReadFiles -> Limit
    - Incorrect "optimized" pipeline: ListFiles -> Limit -> ReadFiles

    """

    ds = ray.data.read_parquet("example://iris.parquet")
    ds = ds.limit(1)
    # Verify correctness: should get exactly 1 row, not the entire first file
    ds_rows = ds.take_all()

    # Verify the pipeline structure is correct
    plan_str = ds._plan._logical_plan.dag.dag_str

    # The correct pipeline should be: ListFiles -> ReadFiles -> Limit
    # NOT: ListFiles -> Limit -> ReadFiles
    expected_correct_order = (
        "ListFiles[ListFiles] -> ReadFiles[ReadFiles] -> Limit[limit=1]"
    )

    assert plan_str == expected_correct_order, (
        f"Limit pushdown incorrectly optimized the pipeline!\n"
        f"Expected: {expected_correct_order}\n"
        f"Got:      {plan_str}\n"
        f"This would cause reading only 1 file manifest instead of limiting actual data rows."
    )

    assert len(ds_rows) == 1, (
        f"Expected exactly 1 row after limit(1), but got {len(ds_rows)} rows. "
        f"This suggests limit was incorrectly pushed through ReadFiles."
    )


@pytest.fixture
def image_files_fixture(tmp_path) -> str:
    """Create test image files fixture."""
    # Create 3 test image files
    for i in range(3):
        img = Image.new("RGB", (100, 100), color=(i * 50, i * 50, i * 50))
        img_path = tmp_path / f"test_{i}.jpg"
        img.save(img_path)

    return str(tmp_path)


@pytest.fixture
def audio_files_fixture(tmp_path) -> str:
    """Create test audio files fixture."""
    # Create 3 test audio files
    for i in range(3):
        # Generate a simple sine wave
        sample_rate = 44100
        duration = 1.0  # 1 second
        frequency = 440 + i * 100  # Different frequencies
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio_data = np.sin(2 * np.pi * frequency * t)

        audio_path = tmp_path / f"test_{i}.wav"
        sf.write(audio_path, audio_data, sample_rate)

    return str(tmp_path)


@pytest.fixture
def binary_files_fixture(tmp_path) -> str:
    """Create test binary files fixture."""
    # Create 3 test binary files
    for i in range(3):
        # Create binary data with different content for each file
        binary_data = (
            b"Binary file content " + str(i).encode() + b" " + b"x" * (100 + i * 10)
        )
        binary_path = tmp_path / f"test_{i}.bin"
        with open(binary_path, "wb") as f:
            f.write(binary_data)

    return str(tmp_path)


def _test_single_row_file_limit_pushdown(
    file_type: str,
    read_func: Callable[[str], Dataset],
    temp_dir: str,
    expected_rows: int = 2,
) -> None:
    """Helper function to test limit pushdown for single-row file types."""
    ds = read_func(temp_dir)
    ds = ds.limit(expected_rows)

    # Verify correctness: should get exactly expected_rows
    rows = ds.take_all()

    # Verify the pipeline structure allows pushdown
    plan_str = ds._plan._logical_plan.dag.dag_str
    expected_optimized_order = (
        "ListFiles[ListFiles] -> Limit[limit=2] -> ReadFiles[ReadFiles]"
    )

    assert plan_str == expected_optimized_order, (
        f"Limit pushdown should be allowed for {file_type} files!\n"
        f"Expected: {expected_optimized_order}\n"
        f"Got:      {plan_str}\n"
        f"{file_type} files produce 1 row per file, so pushdown is safe."
    )

    assert len(rows) == expected_rows, (
        f"Expected exactly {expected_rows} rows after limit({expected_rows}) on {file_type}, "
        f"but got {len(rows)} rows."
    )


@pytest.mark.parametrize(
    "file_type,read_func,fixture_name",
    [
        ("image", ray.data.read_images, "image_files_fixture"),
        ("audio", ray.data.read_audio, "audio_files_fixture"),
        ("binary", ray.data.read_binary_files, "binary_files_fixture"),
    ],
)
def test_limit_pushdown_allows_single_row_files(
    ray_start_regular_shared, request, file_type, read_func, fixture_name
):
    """Test that limit IS correctly pushed through ReadFiles for Datasources that produce 1 row per file.

    Image, audio, and binary Datasources produce 1 row per file, so limit pushdown is safe and beneficial.
    """
    fixture = request.getfixturevalue(fixture_name)
    _test_single_row_file_limit_pushdown(file_type, read_func, fixture)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
