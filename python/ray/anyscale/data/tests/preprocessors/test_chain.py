import pandas as pd
import pytest
from unittest.mock import patch

import ray
from ray.data.preprocessors import (
    Chain,
    LabelEncoder,
    OrdinalEncoder,
    SimpleImputer,
    StandardScaler,
)
from ray.anyscale.data.preprocessors.dag import _build_aggregation_dag


def test_transform_multiple_times():
    """Test that calling transform() multiple times doesn't redundantly run aggregations.

    This test verifies that after the first transform() call, subsequent calls
    skip lazy aggregation and reuse the already-computed stats.
    """
    col_a = [-1, -1, 1, 1]
    col_b = [1, 1, 1, None]
    col_c = ["sunday", "monday", "tuesday", "tuesday"]
    in_df = pd.DataFrame.from_dict({"A": col_a, "B": col_b, "C": col_c})
    ds = ray.data.from_pandas(in_df)

    imputer = SimpleImputer(["B"])
    scaler = StandardScaler(["A", "B"])
    encoder = LabelEncoder("C")
    chain = Chain(scaler, imputer, encoder)

    # Fit the chain
    chain.fit(ds)

    # With Turbo Chain, stats are computed lazily during first transform, not during fit
    assert (
        not imputer.has_stats()
    ), "Turbo Chain should not have stats immediately after fit"
    assert (
        not scaler.has_stats()
    ), "Turbo Chain should not have stats immediately after fit"
    assert (
        not encoder.has_stats()
    ), "Turbo Chain should not have stats immediately after fit"

    # Transform once - this is where lazy aggregation happens
    transformed1 = chain.transform(ds)
    out_df1 = transformed1.to_pandas()

    # After first transform, all preprocessors should have stats
    assert imputer.has_stats(), "SimpleImputer should have stats after first transform"
    assert scaler.has_stats(), "StandardScaler should have stats after first transform"
    assert encoder.has_stats(), "LabelEncoder should have stats after first transform"

    # Transform again - this should skip lazy aggregation
    # Mock ds.aggregate to verify it's NOT called (proving aggregations are skipped)
    with patch.object(ds, "aggregate") as mock_aggregate:
        transformed2 = chain.transform(ds)
        out_df2 = transformed2.to_pandas()

        # Assert aggregate was NOT called - lazy aggregation was skipped
        mock_aggregate.assert_not_called()

    # Transform a third time with same assertion
    with patch.object(ds, "aggregate") as mock_aggregate:
        transformed3 = chain.transform(ds)
        out_df3 = transformed3.to_pandas()

        # Assert aggregate was NOT called again
        mock_aggregate.assert_not_called()

    # Verify all results are identical
    assert out_df1.equals(
        out_df2
    ), "First and second transform results should be identical"
    assert out_df2.equals(
        out_df3
    ), "Second and third transform results should be identical"

    # Verify the expected output
    processed_col_a = [-1.0, -1.0, 1.0, 1.0]
    processed_col_b = [0.0, 0.0, 0.0, 0.0]
    processed_col_c = [1, 0, 2, 2]
    expected_df = pd.DataFrame.from_dict(
        {"A": processed_col_a, "B": processed_col_b, "C": processed_col_c}
    )

    assert out_df1.equals(expected_df), "Transform output should match expected values"


def test_has_stats_method():
    """Test that the has_stats() method works correctly."""
    col_a = [1, 2, 3, 4]
    in_df = pd.DataFrame.from_dict({"A": col_a})
    ds = ray.data.from_pandas(in_df)

    scaler = StandardScaler(["A"])

    # Before fit, should not have stats
    assert not scaler.has_stats(), "Should not have stats before fit"

    # After fit, should have stats
    scaler.fit(ds)
    assert scaler.has_stats(), "Should have stats after fit"


def test_dag_independent_preprocessors():
    """Test DAG construction with independent preprocessors (no column dependencies).

    When preprocessors operate on different columns with no overlap,
    the DAG should have no dependencies between their aggregation nodes.
    """
    col_a = [1, 2, 3, 4]
    col_b = [5, 6, 7, 8]
    col_c = ["x", "y", "z", "w"]
    in_df = pd.DataFrame.from_dict({"A": col_a, "B": col_b, "C": col_c})
    ds = ray.data.from_pandas(in_df)

    # Three independent preprocessors operating on different columns
    scaler_a = StandardScaler(["A"], output_columns=["A_scaled"])
    scaler_b = StandardScaler(["B"], output_columns=["B_scaled"])
    encoder = LabelEncoder("C")

    # Test with Chain - all should work correctly
    chain = Chain(scaler_a, scaler_b, encoder)
    chain.fit(ds)

    # After fit, preprocessors have stat_computation_plan populated
    # Build DAG to verify structure
    nodes = _build_aggregation_dag([scaler_a, scaler_b, encoder])

    # Each preprocessor should have aggregation nodes
    assert len(nodes) > 0, "Should have aggregation nodes after fit"

    result = chain.transform(ds)
    out_df = result.to_pandas()

    # Verify all columns are present
    assert "A_scaled" in out_df.columns
    assert "B_scaled" in out_df.columns
    assert "C" in out_df.columns


def test_dag_with_column_dependencies():
    """Test DAG construction when one preprocessor reads columns written by another.

    This tests the core dependency resolution: if Preprocessor B reads a column
    that Preprocessor A writes, then B's aggregations must wait for A's.
    """
    col_a = [1, 2, 3, 4]
    col_b = [5, None, 7, None]
    in_df = pd.DataFrame.from_dict({"A": col_a, "B": col_b})
    ds = ray.data.from_pandas(in_df)

    # SimpleImputer fills missing values in B
    imputer = SimpleImputer(["B"], output_columns=["B"])

    # StandardScaler operates on B after imputation
    # This creates a dependency: scaler reads "B" which imputer writes
    scaler = StandardScaler(["B"], output_columns=["B_scaled"])

    # Test with Chain - should work correctly with proper ordering
    chain = Chain(imputer, scaler)
    chain.fit(ds)

    # After fit, build DAG to examine structure
    nodes = _build_aggregation_dag([imputer, scaler])

    # Find nodes for each preprocessor
    imputer_nodes = [n for n in nodes if n.preprocessor == imputer]
    scaler_nodes = [n for n in nodes if n.preprocessor == scaler]

    assert len(imputer_nodes) > 0, "Imputer should have aggregation nodes"
    assert len(scaler_nodes) > 0, "Scaler should have aggregation nodes"

    # Scaler nodes should depend on imputer nodes because scaler reads "B" which imputer writes
    for scaler_node in scaler_nodes:
        # Check if any imputer node is in dependencies
        has_imputer_dep = any(
            dep.preprocessor == imputer for dep in scaler_node.dependencies
        )
        # Since scaler reads column "B" which imputer writes to, there should be a dependency
        assert (
            has_imputer_dep
        ), "Scaler should depend on imputer (scaler reads 'B', imputer writes 'B')"

    result = chain.transform(ds)
    out_df = result.to_pandas()

    # Verify imputed and scaled columns exist
    assert "B" in out_df.columns
    assert "B_scaled" in out_df.columns
    # Verify no NaN values in output
    assert not out_df["B"].isna().any()


def test_dag_complex_dependencies():
    """Test DAG with complex multi-level dependencies.

    Creates a chain where:
    1. Imputer fills missing values in column A
    2. Scaler normalizes the imputed A
    3. Encoder encodes a separate column B
    4. All outputs are used
    """
    col_a = [1, None, 3, None, 5]
    col_b = ["cat", "dog", "cat", "bird", "dog"]
    in_df = pd.DataFrame.from_dict({"A": col_a, "B": col_b})
    ds = ray.data.from_pandas(in_df)

    # Complex chain with dependencies
    imputer = SimpleImputer(["A"], output_columns=["A"])
    scaler = StandardScaler(["A"], output_columns=["A_scaled"])
    encoder = OrdinalEncoder(["B"], output_columns=["B_encoded"])

    chain = Chain(imputer, scaler, encoder)
    chain.fit(ds)

    # Verify stats are computed lazily
    assert not imputer.has_stats(), "Should not have stats before transform"
    assert not scaler.has_stats(), "Should not have stats before transform"
    assert not encoder.has_stats(), "Should not have stats before transform"

    result = chain.transform(ds)
    out_df = result.to_pandas()

    # After transform, all should have stats
    assert imputer.has_stats(), "Imputer should have stats after transform"
    assert scaler.has_stats(), "Scaler should have stats after transform"
    assert encoder.has_stats(), "Encoder should have stats after transform"

    # Verify outputs
    assert "A" in out_df.columns
    assert "A_scaled" in out_df.columns
    assert "B_encoded" in out_df.columns
    assert not out_df["A"].isna().any(), "No NaN values should remain"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-sv", __file__]))
