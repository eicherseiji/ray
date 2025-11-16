import pandas as pd
import pytest
from unittest.mock import patch
import numpy as np

import ray
from ray.data.preprocessors import (
    Categorizer,
    Chain,
    Concatenator,
    CountVectorizer,
    FeatureHasher,
    HashingVectorizer,
    LabelEncoder,
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
    RobustScaler,
    SimpleImputer,
    StandardScaler,
    Tokenizer,
    TorchVisionPreprocessor,
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


@pytest.mark.parametrize(
    "preprocessor_name,preprocessor_factory,input_data,expected_columns",
    [
        # Scalers
        (
            "StandardScaler",
            lambda: StandardScaler(["A"], output_columns=["A_scaled"]),
            {"A": [1.0, 2.0, 3.0, 4.0]},
            ["A_scaled"],
        ),
        (
            "MinMaxScaler",
            lambda: MinMaxScaler(["A"], output_columns=["A_scaled"]),
            {"A": [1.0, 2.0, 3.0, 4.0]},
            ["A_scaled"],
        ),
        (
            "MaxAbsScaler",
            lambda: MaxAbsScaler(["A"], output_columns=["A_scaled"]),
            {"A": [1.0, 2.0, 3.0, 4.0]},
            ["A_scaled"],
        ),
        (
            "RobustScaler",
            lambda: RobustScaler(["A"], output_columns=["A_scaled"]),
            {"A": [1.0, 2.0, 3.0, 4.0]},
            ["A_scaled"],
        ),
        # Encoders
        (
            "LabelEncoder",
            lambda: LabelEncoder("B", output_column="B_encoded"),
            {"B": ["cat", "dog", "cat", "bird"]},
            ["B_encoded"],
        ),
        (
            "OrdinalEncoder",
            lambda: OrdinalEncoder(["B"], output_columns=["B_encoded"]),
            {"B": ["cat", "dog", "cat", "bird"]},
            ["B_encoded"],
        ),
        (
            "OneHotEncoder",
            lambda: OneHotEncoder(["B"], output_columns=["B_encoded"]),
            {"B": ["cat", "dog", "cat", "bird"]},
            ["B_encoded"],
        ),
        (
            "Categorizer",
            lambda: Categorizer(["B"], output_columns=["B_cat"]),
            {"B": ["cat", "dog", "cat", "bird"]},
            ["B_cat"],
        ),
        # Imputer
        (
            "SimpleImputer",
            lambda: SimpleImputer(["A"], output_columns=["A_imputed"]),
            {"A": [1.0, None, 3.0, None]},
            ["A_imputed"],
        ),
        # Transformers
        (
            "Normalizer",
            lambda: Normalizer(["A", "C"], output_columns=["A_norm", "C_norm"]),
            {"A": [1.0, 2.0, 3.0, 4.0], "C": [5.0, 6.0, 7.0, 8.0]},
            ["A_norm", "C_norm"],
        ),
        (
            "PowerTransformer",
            lambda: PowerTransformer(["A"], power=0.5, output_columns=["A_power"]),
            {"A": [1.0, 2.0, 3.0, 4.0]},
            ["A_power"],
        ),
        # Concatenator - uses output_column_name instead of output_columns
        (
            "Concatenator",
            lambda: Concatenator(["A", "C"], output_column_name="AC_concat"),
            {"A": [1.0, 2.0, 3.0, 4.0], "C": [5.0, 6.0, 7.0, 8.0]},
            ["AC_concat"],
        ),
        # Text preprocessors
        (
            "Tokenizer",
            lambda: Tokenizer(["text"], output_columns=["text_tokens"]),
            {"text": ["hello world", "foo bar", "test case", "example data"]},
            ["text_tokens"],
        ),
        (
            "HashingVectorizer",
            lambda: HashingVectorizer(
                ["text"], num_features=8, output_columns=["text_vec"]
            ),
            {"text": ["hello world", "foo bar", "test case", "example data"]},
            ["text_vec"],
        ),
        (
            "CountVectorizer",
            lambda: CountVectorizer(["text"], output_columns=["text_counts"]),
            {"text": ["hello world", "foo bar", "test case", "example data"]},
            ["text_counts"],
        ),
        # FeatureHasher - uses output_column instead of output_columns
        (
            "FeatureHasher",
            lambda: FeatureHasher(
                ["token1", "token2"], num_features=8, output_column="hashed"
            ),
            {"token1": [1, 2, 3, 4], "token2": [5, 6, 7, 8]},
            ["hashed"],
        ),
        # TorchVisionPreprocessor - uses _columns and _output_columns
        (
            "TorchVisionPreprocessor",
            lambda: TorchVisionPreprocessor(
                ["image"], transform=lambda x: x, output_columns=["image_transformed"]
            ),
            {
                "image": [
                    np.array([[[1, 2, 3]]]),
                    np.array([[[4, 5, 6]]]),
                    np.array([[[7, 8, 9]]]),
                    np.array([[[10, 11, 12]]]),
                ]
            },
            ["image_transformed"],
        ),
    ],
)
def test_preprocessor_in_chain(
    preprocessor_name, preprocessor_factory, input_data, expected_columns
):
    """Test that each preprocessor works correctly in a turbo Chain.

    This test verifies that all preprocessors have correct get_input_columns() and
    get_output_columns() implementations, which are required for turbo_chain's
    dependency tracking.
    """
    # Create dataset from input data
    df = pd.DataFrame(input_data)
    ds = ray.data.from_pandas(df)

    # Create preprocessor
    preprocessor = preprocessor_factory()

    # Verify methods return lists
    input_cols = preprocessor.get_input_columns()
    output_cols = preprocessor.get_output_columns()
    assert isinstance(
        input_cols, list
    ), f"{preprocessor_name}.get_input_columns() must return a list"
    assert isinstance(
        output_cols, list
    ), f"{preprocessor_name}.get_output_columns() must return a list"

    # Test in a Chain
    chain = Chain(preprocessor)
    chain.fit(ds)

    # Transform
    result = chain.transform(ds)
    out_df = result.to_pandas()

    # Verify expected columns exist in output
    for col in expected_columns:
        assert (
            col in out_df.columns
        ), f"Expected column '{col}' not found in output for {preprocessor_name}"

    # Verify preprocessor has stats if it's fittable
    if preprocessor._is_fittable:
        assert (
            preprocessor.has_stats()
        ), f"{preprocessor_name} should have stats after transform"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-sv", __file__]))
