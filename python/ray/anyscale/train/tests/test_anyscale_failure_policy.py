import pytest

from ray.anyscale.train._internal.execution.failure_handling.anyscale_failure_policy import (  # noqa: E501
    _contains_preemption_error,
)
from ray.exceptions import RayActorError
from ray.train import FailureConfig
from ray.train.v2._internal.exceptions import WorkerHealthCheckFailedError
from ray.train.v2._internal.execution.failure_handling import (
    FailureDecision,
    create_failure_policy,
)
from ray.train.v2.api.exceptions import WorkerGroupError


def _worker_group_error_from_errors(errors):
    return WorkerGroupError(
        "Worker group failed",
        dict(enumerate(errors)),
    )


def test_worker_group_status_has_preemption():
    class PreemptionRayActorError(RayActorError):
        def preempted(self) -> bool:
            return True

    worker_health_check_failure_error = WorkerHealthCheckFailedError(
        message="Worker health check failed due to node preemption.",
        failure=PreemptionRayActorError(),
    )
    worker_group_error = _worker_group_error_from_errors(
        [None, worker_health_check_failure_error, None, RuntimeError(), None]
    )
    assert _contains_preemption_error(worker_group_error.worker_failures)

    worker_group_error = _worker_group_error_from_errors(
        [None, RuntimeError(), None, RuntimeError(), None]
    )
    assert not _contains_preemption_error(worker_group_error.worker_failures)

    worker_group_error = _worker_group_error_from_errors([None, None, None, None])
    assert not _contains_preemption_error(worker_group_error.worker_failures)


def test_failure_on_preemption_errors():
    """Check that the failure counts preemption errors correctly."""

    policy = create_failure_policy(FailureConfig(max_failures=0))

    class PreemptionRayActorError(RayActorError):
        def preempted(self) -> bool:
            return True

    worker_health_check_failure_error = WorkerHealthCheckFailedError(
        message="Worker health check failed due to node preemption.",
        failure=PreemptionRayActorError(),
    )

    worker_group_error = _worker_group_error_from_errors(
        [None, worker_health_check_failure_error, None, RuntimeError(), None]
    )

    assert policy.make_decision(worker_group_error) == FailureDecision.RETRY

    worker_group_error = _worker_group_error_from_errors(
        [None, RuntimeError(), None, RuntimeError(), None]
    )

    assert policy.make_decision(worker_group_error) == FailureDecision.RAISE


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", "-x", __file__]))
