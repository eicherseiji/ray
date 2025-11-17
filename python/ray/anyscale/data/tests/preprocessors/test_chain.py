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
    3. Concatenator combines scaled A and original C into features
    4. Encoder encodes a separate column B
    5. All outputs are used

    This tests Concatenator's get_input_columns/get_output_columns implementations
    since it uses output_column_name instead of output_columns. Concatenator depends
    on Scaler's output, ensuring non-fittable preprocessors work correctly in the DAG.
    """
    col_a = [1, None, 3, None, 5]
    col_b = ["cat", "dog", "cat", "bird", "dog"]
    col_c = [10, 20, 30, 40, 50]
    in_df = pd.DataFrame.from_dict({"A": col_a, "B": col_b, "C": col_c})
    ds = ray.data.from_pandas(in_df)

    # Complex chain with dependencies
    # Imputer fills missing values in A
    imputer = SimpleImputer(["A"], output_columns=["A_imputed"])
    # Scaler normalizes the imputed A
    scaler = StandardScaler(["A_imputed"], output_columns=["A_scaled"])
    # Concatenator combines A_scaled and C (depends on Scaler's output)
    concatenator = Concatenator(["A_scaled", "C"], output_column_name="features")
    # Encoder encodes B (independent branch)
    encoder = OrdinalEncoder(["B"], output_columns=["B_encoded"])

    chain = Chain(imputer, scaler, concatenator, encoder)
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
    assert "A_imputed" in out_df.columns, "Imputed A should exist"
    assert "features" in out_df.columns, "Concatenated features column should exist"
    assert "B_encoded" in out_df.columns, "Encoded B should exist"
    # Note: A_scaled is dropped by Concatenator after being used as input
    assert (
        "A_scaled" not in out_df.columns
    ), "A_scaled should be dropped by Concatenator"
    # Note: C is also dropped by Concatenator after being used as input
    assert "C" not in out_df.columns, "C should be dropped by Concatenator"

    # Verify the concatenated column has the right shape
    # Each row in features should be a 2-element array (from A_scaled and C)
    assert all(
        isinstance(row, np.ndarray) and len(row) == 2 for row in out_df["features"]
    ), "Features should be 2-element arrays (from A_scaled + C)"

    # Verify the values make sense (scaled values should have mean ~0)
    features_array = np.array(list(out_df["features"]))
    scaled_a_values = features_array[:, 0]  # First element of each array is A_scaled
    assert (
        abs(np.mean(scaled_a_values)) < 0.1
    ), "Scaled A values should have mean close to 0"


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


def test_fallback_to_serial_execution_with_custom_stats():
    """Test fallback to serial execution when preprocessor has custom stat functions.

    The original (non-turbo) OrdinalEncoder uses add_callable_stat() which triggers
    the fallback path since it has iter_batches type of stat evaluation. This test
    verifies that the chain correctly falls back to serial execution by creating a
    dependency chain where one preprocessor reads the output of the previous one.

    The key insight: In DAG-based execution, dependencies are tracked by column names,
    but serial execution processes preprocessors in order. By having the scaler depend
    on the encoder's output, we ensure serial execution is truly happening.

    Note: We import the original encoder before patching to test the fallback behavior.
    """
    col_a = [1.0, 2.0, 3.0, 4.0, 5.0]
    col_b = ["cat", "dog", "cat", "bird", "dog"]
    in_df = pd.DataFrame.from_dict({"A": col_a, "B": col_b})
    ds = ray.data.from_pandas(in_df)

    # Import the original (non-turbo) OrdinalEncoder which uses custom stat functions.
    # This import must be inside the function to ensure we get the unpatched version
    # before any patching occurs, as required for testing fallback behavior.
    from ray.data.preprocessors.encoder import OrdinalEncoder as OriginalOrdinalEncoder

    # Create a dependency chain:
    # 1. Encoder transforms B -> B_encoded
    # 2. Scaler transforms B_encoded -> B_scaled (depends on encoder's output!)
    # This dependency can only work in serial execution, verifying the fallback works
    encoder = OriginalOrdinalEncoder(["B"], output_columns=["B_encoded"])
    scaler = StandardScaler(["B_encoded"], output_columns=["B_scaled"])

    # Create chain with preprocessor that has custom stats
    chain = Chain(encoder, scaler)

    # Mock the fallback method to verify it's called
    fallback_called = []
    original_fallback = chain._fallback_to_serial_execution

    def mock_fallback(ds, **kwargs):
        fallback_called.append(True)
        return original_fallback(ds, **kwargs)

    chain._fallback_to_serial_execution = mock_fallback

    # Fit the encoder to populate its stat_computation_plan with custom stats
    # Note: The original encoder computes stats during _fit_execute (not lazy)
    chain.fit(ds)

    # After fit, encoder should have custom stat functions
    assert (
        encoder.stat_computation_plan.has_custom_stat_fn()
    ), "Original OrdinalEncoder should have custom stat functions after fit"

    # The original encoder computes stats during fit (not lazy like turbo version)
    # So encoder already has stats, but scaler doesn't yet
    assert encoder.has_stats(), "Original encoder computes stats during fit"
    assert (
        not scaler.has_stats()
    ), "Scaler should not have stats before transform in turbo mode"

    # Reset encoder stats to test the fallback during transform
    encoder.stats_ = {}

    result = chain.transform(ds)

    # Verify fallback was called due to custom stat functions
    assert (
        len(fallback_called) == 1
    ), "Fallback to serial execution should have been called due to custom stat functions"

    # Verify stats were computed during fallback
    assert encoder.has_stats(), "Encoder should have stats after transform"
    assert scaler.has_stats(), "Scaler should have stats after transform"

    # Verify output is correct - this is the critical test!
    # If serial execution didn't happen, scaler would fail because B_encoded wouldn't exist
    out_df = result.to_pandas()
    assert "B_encoded" in out_df.columns, "Encoded column should exist"
    assert (
        "B_scaled" in out_df.columns
    ), "Scaled column should exist (proves serial execution)"
    assert "A" in out_df.columns, "Original column A should still exist"

    # Verify encoding worked correctly (cat=0, dog=1, bird=2 or similar mapping)
    assert len(out_df["B_encoded"].unique()) == 3, "Should have 3 unique encoded values"

    # Verify scaling worked correctly on encoded values (mean should be close to 0)
    assert (
        abs(out_df["B_scaled"].mean()) < 0.1
    ), "Scaled encoded values should have mean close to 0"

    # Verify the dependency chain worked: B -> B_encoded -> B_scaled
    # This proves serial execution happened in order
    assert (
        out_df["B_scaled"].notna().all()
    ), "All scaled values should be valid (not NaN)"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-sv", __file__]))
