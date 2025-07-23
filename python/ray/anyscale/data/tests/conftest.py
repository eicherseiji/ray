import copy

import pytest

import ray
from ray.anyscale.data._internal.util.cache import _disable_timed_cache_for_tests


@pytest.fixture
def restore_data_context(request):
    """Restore any DataContext changes after the test runs"""
    original = copy.deepcopy(ray.data.context.DataContext.get_current())
    # TODO: `copy.deepcopy` doesn't work with fields in `DataContextMixin`, so we
    # need to do a manual copy.
    original_issue_detectors_config = copy.deepcopy(original.issue_detectors_config)
    yield
    original.issue_detectors_config = original_issue_detectors_config
    ray.data.context.DataContext._set_current(original)


@pytest.fixture
def disable_timed_cache_fixture():
    with _disable_timed_cache_for_tests():
        yield
