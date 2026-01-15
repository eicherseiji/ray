import pytest

from ray.anyscale.train.api.config import ScalingConfig


def test_scaling_config_validation():
    elastic_scaling_config = ScalingConfig(
        num_workers=(1, 2),
        label_selector=[
            {"subcluster": "my_subcluster"},
            {"subcluster": "other_subcluster"},
        ],
    )
    assert elastic_scaling_config.min_workers == 1
    assert elastic_scaling_config.max_workers == 2
    assert elastic_scaling_config.elasticity_enabled

    with pytest.raises(ValueError, match="must be an int or a tuple of two ints."):
        ScalingConfig(num_workers=(1, 2, 3))

    with pytest.raises(
        ValueError,
        match=r"ScalingConfig\(elastic_resize_monitor_interval_s\) must be non-negative.",
    ):
        ScalingConfig(num_workers=(1, 2), elastic_resize_monitor_interval_s=-1)

    with pytest.raises(ValueError, match="min_workers=2 must be <= max_workers=1"):
        ScalingConfig(num_workers=(2, 1))

    with pytest.raises(
        ValueError,
        match="`label_selector` is a list of length 1, but it must be of length `max_workers=2` instead.",
    ):
        ScalingConfig(
            num_workers=(1, 2), label_selector=[{"subcluster": "my_subcluster"}]
        )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", "-x", __file__]))
