from unittest.mock import MagicMock, patch

import pytest

import ray
from ray._private.arrow_utils import get_pyarrow_version
from ray.anyscale.data._internal.location_aware_bundle_queue import (
    LocationAwareBundleQueue,
)
from ray.anyscale.data.aggregate_vectorized import (
    MIN_PYARROW_VERSION_VECTORIZED_AGGREGATIONS,
)
from ray.anyscale.data.api.context_mixin import DataContextMixin
from ray.anyscale.data.api.dataset_mixin import DatasetMixin
from ray.anyscale.data.apply_anyscale_patches import (
    _patch_aggregations,
    _patch_class_with_dataclass_mixin,
    _patch_class_with_mixin,
    _register_anyscale_lineage_tracking_callback,
)
from ray.data._internal.execution.execution_callback import (
    EXECUTION_CALLBACKS_CONFIG_KEY,
)
from ray.data._internal.execution.bundle_queue import (
    FIFOBundleQueue,
    create_bundle_queue,
)
from ray.tests.conftest import *  # noqa


def test__patch_class_with_mixin(ray_start_regular_shared):
    _patch_class_with_mixin(ray.data.Dataset, DatasetMixin)

    # Check that Dataset has custom rayturbo methods and attributes.
    assert hasattr(ray.data.Dataset, "write_snowflake")
    assert hasattr(ray.data.Dataset, "streaming_aggregate")


def test__patch_class_with_dataclass_mixin(ray_start_regular_shared):
    _patch_class_with_dataclass_mixin(ray.data.DataContext, DataContextMixin)

    # Check that DataContext has custom rayturbo methods and attributes.
    assert hasattr(ray.data.DataContext, "checkpoint_config")

    # Check that GPU join configuration is available
    ctx = ray.data.DataContext.get_current()
    assert hasattr(ctx, "use_polars_gpu_join")
    assert hasattr(ctx, "polars_gpu_device_id")
    assert hasattr(ctx, "polars_gpu_raise_on_fail")
    assert hasattr(ctx, "validate_polars_gpu_config")
    assert hasattr(ctx, "get_polars_gpu_engine")

    # Test default values
    assert ctx.use_polars_gpu_join is False
    assert ctx.polars_gpu_device_id is None
    assert ctx.polars_gpu_raise_on_fail is False


def test_patch_aggregations(ray_start_regular_shared):
    _patch_aggregations()

    from ray.anyscale.data import aggregate_vectorized
    from ray.data import aggregate

    should_be_vectorized = (
        get_pyarrow_version() >= MIN_PYARROW_VERSION_VECTORIZED_AGGREGATIONS
    )

    assert should_be_vectorized == (
        aggregate.Count is aggregate_vectorized.CountVectorized
    )
    assert should_be_vectorized == (aggregate.Sum is aggregate_vectorized.SumVectorized)
    assert should_be_vectorized == (aggregate.Min is aggregate_vectorized.MinVectorized)
    assert should_be_vectorized == (aggregate.Max is aggregate_vectorized.MaxVectorized)
    assert should_be_vectorized == (
        aggregate.AbsMax is aggregate_vectorized.AbsMaxVectorized
    )
    assert should_be_vectorized == (
        aggregate.Quantile is aggregate_vectorized.QuantileVectorized
    )
    assert should_be_vectorized == (
        aggregate.Unique is aggregate_vectorized.UniqueVectorized
    )


@pytest.mark.parametrize(
    "env_value, expected_bundle_queue_type",
    [
        ("1", LocationAwareBundleQueue),
        ("0", FIFOBundleQueue),
        (None, LocationAwareBundleQueue),
    ],
)
def test_create_bundle_queue_returns_correct_type(
    env_value, expected_bundle_queue_type, monkeypatch
):
    if env_value is not None:
        monkeypatch.setenv("RAY_DATA_ENABLE_LOCATION_AWARE_BUNDLE_QUEUES", env_value)

    assert isinstance(create_bundle_queue(), expected_bundle_queue_type)


@pytest.mark.parametrize(
    "feature_enabled, expect_callback_registered",
    [
        (False, False),
        (True, True),
    ],
)
def test_register_anyscale_lineage_tracking_callback(
    feature_enabled, expect_callback_registered, monkeypatch
):
    import ray.anyscale.data.apply_anyscale_patches as patches_module
    from ray.data.context import DataContext

    # Get a fresh context and clear any existing callbacks
    context = DataContext.get_current()
    context.remove_config(EXECUTION_CALLBACKS_CONFIG_KEY)

    # Mock the feature flag
    monkeypatch.setattr(
        patches_module, "ANYSCALE_LINEAGE_TRACKING_ENABLED", feature_enabled
    )

    # Create a mock callback class (to avoid importing openlineage)
    mock_callback_instance = MagicMock()
    mock_callback_class = MagicMock(return_value=mock_callback_instance)

    # Mock the callback import to avoid openlineage dependency
    with patch.dict(
        "sys.modules",
        {
            "ray.anyscale.lineage.ray_lineage.data.main": MagicMock(
                RayDataOpenLineageExecutionCallback=mock_callback_class
            )
        },
    ):
        # Call the function
        _register_anyscale_lineage_tracking_callback()

    # Verify the result
    callbacks = context.get_config(EXECUTION_CALLBACKS_CONFIG_KEY, [])

    if expect_callback_registered:
        assert len(callbacks) == 1
        assert callbacks[0] is mock_callback_instance
        mock_callback_class.assert_called_once()
    else:
        assert len(callbacks) == 0
        mock_callback_class.assert_not_called()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
