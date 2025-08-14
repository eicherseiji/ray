import re
import pytest

from ray import serve
from ray._common.test_utils import SignalActor
from ray.anyscale.serve._private.replica_result import gRPCReplicaResult
from ray.serve._private.common import OBJ_REF_NOT_SUPPORTED_ERROR
from ray.serve._private.replica_result import ActorReplicaResult, ReplicaResult
from ray.serve.tests.conftest import *  # noqa
from ray.serve.tests.conftest import _shared_serve_instance  # noqa


@pytest.mark.parametrize(
    "by_reference,expected_result",
    [(True, ActorReplicaResult), (False, gRPCReplicaResult)],
)
def test_init_by_reference(
    serve_instance, by_reference: bool, expected_result: ReplicaResult
):
    @serve.deployment
    def f():
        return "hi"

    h = serve.run(f.bind())

    resp = h.options(_by_reference=by_reference).remote()
    assert resp.result() == "hi"
    assert isinstance(resp._replica_result, expected_result)


@pytest.mark.timeout(60)
async def test_by_reference_false_raises_error(serve_instance):
    signal = SignalActor.remote()

    @serve.deployment
    async def f():
        await signal.wait.remote()
        return "hi"

    h = serve.run(f.bind())
    with pytest.raises(
        RuntimeError, match=re.escape(OBJ_REF_NOT_SUPPORTED_ERROR.args[0])
    ):
        response = h.options(_by_reference=False).remote()
        await response._to_object_ref()

    await signal.send.remote()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", "-s", __file__]))
