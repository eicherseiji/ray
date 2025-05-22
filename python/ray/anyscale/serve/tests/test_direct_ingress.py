import grpc
import pytest
import requests
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple
from uuid import UUID

from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import PlainTextResponse

import ray
from ray import serve
from ray.actor import ActorHandle
from ray._private.test_utils import SignalActor, Semaphore, wait_for_condition
from ray.serve.generated import serve_pb2, serve_pb2_grpc
from ray.anyscale.serve._private.constants import (
    ANYSCALE_RAY_SERVE_ENABLE_DIRECT_INGRESS,
    RAY_SERVE_DIRECT_INGRESS_MIN_HTTP_PORT,
    RAY_SERVE_DIRECT_INGRESS_MIN_GRPC_PORT,
    RAY_SERVE_DIRECT_INGRESS_MAX_HTTP_PORT,
    RAY_SERVE_DIRECT_INGRESS_PORT_RETRY_COUNT,
)
from ray.serve.schema import ApplicationStatus, RequestProtocol
from ray.serve._private.client import ServeControllerClient
from ray.serve.schema import ServeInstanceDetails
from ray.dashboard.modules.serve.sdk import ServeSubmissionClient
from ray.serve.schema import DeploymentStatus


@pytest.fixture
def _skip_if_ff_not_enabled():
    if not ANYSCALE_RAY_SERVE_ENABLE_DIRECT_INGRESS:
        pytest.skip(
            reason="ANYSCALE_RAY_SERVE_ENABLE_DIRECT_INGRESS not set.",
        )


@serve.deployment(name="default-deployment")
class Hybrid:
    def __init__(
        self,
        *,
        message: str = "",
        raise_error: bool = False,
        wait_signal: Optional[ActorHandle] = None,
        fail_hc_signal: Optional[ActorHandle] = None,
        shutdown_signal: Optional[ActorHandle] = None,
        initialize_signal: Optional[ActorHandle] = None,
    ):
        self._message = message
        self._raise_error = raise_error
        self._wait_signal = wait_signal
        self._fail_hc_signal = fail_hc_signal
        self._shutdown_signal = shutdown_signal

        if initialize_signal is not None:
            ray.get(initialize_signal.wait.remote())

    def check_health(self):
        # Fail health check once the signal is sent, else pass.
        if self._fail_hc_signal is not None:
            obj_ref = self._fail_hc_signal.wait.remote()
            ready, _ = ray.wait([obj_ref], timeout=0.1)
            if len(ready) == 1:
                raise RuntimeError("Failing health check!")

    def __del__(self):
        if self._shutdown_signal is not None:
            ray.get(self._shutdown_signal.wait.remote())

    async def __call__(self, request: Request):
        if self._raise_error:
            raise RuntimeError("oops!")

        if self._wait_signal:
            await self._wait_signal.wait.remote()

        return self._message

    async def Method1(
        self, request: serve_pb2.UserDefinedMessage
    ) -> serve_pb2.UserDefinedResponse:
        if self._raise_error:
            raise RuntimeError("oops!")

        if self._wait_signal:
            await self._wait_signal.wait.remote()

        return serve_pb2.UserDefinedResponse(greeting=self._message)


def test_proxy_is_started_on_head_only_mode(_skip_if_ff_not_enabled, serve_instance):
    assert len(serve.status().proxies) == 1


def get_http_ports(
    serve_instance: ServeControllerClient, route_prefix=None, first_only=True
):
    serve_details = ServeInstanceDetails(
        **ServeSubmissionClient("http://localhost:8265").get_serve_details()
    )
    target_groups = serve_details.target_groups
    if first_only:
        http_target_group = next(
            (
                tg
                for tg in target_groups
                if tg.protocol == RequestProtocol.HTTP
                and (route_prefix is None or tg.route_prefix == route_prefix)
            )
        )
        http_targets = http_target_group.targets
        http_ports = [target.port for target in http_targets]
        return http_ports
    else:
        http_ports = []
        for target_group in target_groups:
            if target_group.protocol == RequestProtocol.HTTP and (
                route_prefix is None or target_group.route_prefix == route_prefix
            ):
                http_ports.extend([target.port for target in target_group.targets])
        return http_ports


def get_grpc_ports(
    serve_instance: ServeControllerClient, route_prefix=None, first_only=True
):
    serve_details = ServeInstanceDetails(
        **ServeSubmissionClient("http://localhost:8265").get_serve_details()
    )
    target_groups = serve_details.target_groups
    if first_only:
        grpc_target_group = next(
            (
                tg
                for tg in target_groups
                if tg.protocol == RequestProtocol.GRPC
                and (route_prefix is None or tg.route_prefix == route_prefix)
            )
        )
        grpc_targets = grpc_target_group.targets
        grpc_ports = [target.port for target in grpc_targets]
        return grpc_ports
    else:
        grpc_ports = []
        for target_group in target_groups:
            if target_group.protocol == RequestProtocol.GRPC and (
                route_prefix is None or target_group.route_prefix == route_prefix
            ):
                grpc_ports.extend([target.port for target in target_group.targets])
        return grpc_ports


def test_basic(_skip_if_ff_not_enabled, serve_instance):
    serve.run(Hybrid.bind(message="Hello world!"))

    http_port = get_http_ports(serve_instance)[0]
    grpc_port = get_grpc_ports(serve_instance)[0]

    # Basic HTTP request.
    r = requests.get(f"http://localhost:{http_port}/")
    r.raise_for_status()
    assert r.text == "Hello world!"

    # Basic gRPC request.
    channel = grpc.insecure_channel(f"localhost:{grpc_port}")
    stub = serve_pb2_grpc.UserDefinedServiceStub(channel)
    assert stub.Method1(serve_pb2.UserDefinedMessage()).greeting == "Hello world!"


def test_internal_server_error(_skip_if_ff_not_enabled, serve_instance):
    pytest.skip("TODO(abrar): fix this test")
    serve.run(Hybrid.bind(raise_error=True))

    http_port = get_http_ports(serve_instance)[0]
    grpc_port = get_grpc_ports(serve_instance)[0]

    # Basic HTTP request.
    r = requests.get(f"http://localhost:{http_port}/")
    assert r.status_code == 500
    assert r.text == "Internal Server Error"

    # Basic gRPC request.
    channel = grpc.insecure_channel(f"localhost:{grpc_port}")
    stub = serve_pb2_grpc.UserDefinedServiceStub(channel)
    try:
        with pytest.raises(grpc.RpcError) as exception_info:
            stub.Method1(serve_pb2.UserDefinedMessage())

        rpc_error = exception_info.value
        assert rpc_error.code() == grpc.StatusCode.UNKNOWN
    finally:
        # Force close the gRPC channel to ensure ports are released
        channel.close()


def test_fastapi_app(_skip_if_ff_not_enabled, serve_instance):

    fastapi_app = FastAPI()

    @serve.deployment
    @serve.ingress(fastapi_app)
    class FastAPIDeployment:
        @fastapi_app.get("/")
        def root(self) -> PlainTextResponse:
            return PlainTextResponse("Hello from root!")

        @fastapi_app.post("/{wildcard}")
        def post(self, wildcard: str) -> PlainTextResponse:
            return PlainTextResponse(
                f"Hello from {wildcard}!",
                status_code=201,
            )

    serve.run(FastAPIDeployment.bind())
    http_port = get_http_ports(serve_instance)[0]

    # Test GET /.
    r = requests.get(f"http://localhost:{http_port}/")
    r.raise_for_status()
    assert r.text == "Hello from root!"

    # Test POST /{wildcard}.
    r = requests.post(f"http://localhost:{http_port}/foobar")
    assert r.status_code == 201
    assert r.text == "Hello from foobar!"


@pytest.mark.parametrize("use_fastapi", [False, True])
def test_http_request_id(_skip_if_ff_not_enabled, serve_instance, use_fastapi: bool):

    if use_fastapi:
        fastapi_app = FastAPI()

        @serve.deployment
        @serve.ingress(fastapi_app)
        class EchoRequestID:
            @fastapi_app.get("/")
            async def root(self, request: Request) -> PlainTextResponse:
                return PlainTextResponse(request.headers.get("x-request-id", ""))

    else:

        @serve.deployment
        class EchoRequestID:
            async def __call__(self, request: Request) -> str:
                return PlainTextResponse(request.headers.get("x-request-id", ""))

    serve.run(EchoRequestID.bind())
    http_port = get_http_ports(serve_instance)[0]

    # Case 1: no x-request-id passed, should get populated and returned as a header.
    r = requests.get(f"http://localhost:{http_port}/")
    r.raise_for_status()
    assert r.text != "" and r.text == r.headers["x-request-id"]
    # This call would raise if the request ID isn't a valid UUID.
    UUID(r.text, version=4)

    # Case 2: x-request-id passed, result and header should match it.
    r = requests.get(
        f"http://localhost:{http_port}/", headers={"x-request-id": "TEST-HEADER"}
    )
    r.raise_for_status()
    assert r.text == "TEST-HEADER" and r.text == r.headers["x-request-id"]


def test_grpc_request_id(_skip_if_ff_not_enabled, serve_instance):
    pytest.skip("TODO: duplicate HTTP tests for gRPC")


def test_multiplexed_model_id(_skip_if_ff_not_enabled, serve_instance):
    pytest.skip("TODO: test that sends a MM ID and checks that it's set correctly")


def test_health_check(_skip_if_ff_not_enabled, serve_instance):

    wait_signal = SignalActor.remote()
    fail_hc_signal = SignalActor.remote()
    shutdown_signal = SignalActor.remote()
    initialize_signal = SignalActor.remote()

    # Use private `_run` API so we can test the behavior before replicas initialize.
    serve._run(
        # Set a high health check period so we have time to check behavior before the
        # controller restarts the replica.
        Hybrid.options(health_check_period_s=1).bind(
            wait_signal=wait_signal,
            fail_hc_signal=fail_hc_signal,
            shutdown_signal=shutdown_signal,
            initialize_signal=initialize_signal,
        ),
        _blocking=False,
    )
    # Here I am assuming that min port will always be available. But that may be true
    # since that port maybe occupied by some other parallel test. But we have no way of
    # knowing which port will be used ahead of replica initialization. May need to revisit
    # this in the future.
    http_port = RAY_SERVE_DIRECT_INGRESS_MIN_HTTP_PORT
    grpc_port = RAY_SERVE_DIRECT_INGRESS_MIN_GRPC_PORT

    def _do_grpc_hc() -> Tuple[grpc.StatusCode, str]:
        channel = grpc.insecure_channel(f"localhost:{grpc_port}")
        stub = serve_pb2_grpc.RayServeAPIServiceStub(channel)
        try:
            response, call = stub.Healthz.with_call(serve_pb2.HealthzRequest())
            return call.code(), response.message
        except grpc.RpcError as e:
            return e.code(), ""

    # Wait for replica constructor to start. The direct ingress server should not be
    # listening on the port at all yet.
    wait_for_condition(lambda: ray.get(initialize_signal.cur_num_waiters.remote()) == 1)
    for _ in range(10):
        with pytest.raises(requests.ConnectionError):
            requests.get(f"http://localhost:{http_port}/-/healthz")

        code, _ = _do_grpc_hc()
        assert code == grpc.StatusCode.UNAVAILABLE

    def _verify_health_check(
        *,
        passing: bool,
        message: str,
    ) -> bool:
        # Check HTTP health check.
        expected_status = 200 if passing else 503
        r = requests.get(f"http://localhost:{http_port}/-/healthz")
        assert r.status_code == expected_status
        assert r.text == message

        # Check gRPC health check.
        expected_code = grpc.StatusCode.OK if passing else grpc.StatusCode.UNAVAILABLE
        code, response_message = _do_grpc_hc()
        assert code == expected_code
        # NOTE(edoakes): we can't access the response message if the gRPC call fails
        # due to StatusCode.UNAVAILABLE.
        if passing:
            assert response_message == message

        return True

    # Signal the constructor to finish and verify that health checks start to pass.
    ray.get(initialize_signal.send.remote())
    wait_for_condition(
        lambda: _verify_health_check(passing=True, message="OK"),
    )

    # Signal the health check method to fail and verify that health checks fail.
    ray.get(fail_hc_signal.send.remote())
    wait_for_condition(
        lambda: _verify_health_check(passing=False, message="UNHEALTHY"),
    )

    # Signal the health check method to pass and verify that health checks pass.
    ray.get(fail_hc_signal.send.remote(clear=True))
    wait_for_condition(
        lambda: _verify_health_check(passing=True, message="OK"),
    )

    # Initiate graceful shutdown and verify that health checks fail.
    serve.delete("default", _blocking=False)
    wait_for_condition(
        lambda: ray.get(shutdown_signal.cur_num_waiters.remote()) == 1,
    )
    for _ in range(10):
        assert _verify_health_check(passing=False, message="DRAINING")

    ray.get(shutdown_signal.send.remote())
    wait_for_condition(
        lambda: len(serve.status().applications) == 0,
    )


def test_max_ongoing_requests(_skip_if_ff_not_enabled, serve_instance):
    wait_signal = SignalActor.remote()

    serve.run(
        Hybrid.options(max_ongoing_requests=5).bind(
            message="done waiting!", wait_signal=wait_signal
        )
    )
    http_port = get_http_ports(serve_instance)[0]

    def _do_http_request() -> bool:
        r = requests.get(f"http://localhost:{http_port}/")
        if r.status_code == 200:
            return True
        elif r.status_code == 503:
            return False
        else:
            raise RuntimeError(f"Unexpected status code: {r.status_code}")

    grpc_port = get_grpc_ports(serve_instance)[0]

    def _do_grpc_request() -> bool:
        channel = grpc.insecure_channel(f"localhost:{grpc_port}")
        stub = serve_pb2_grpc.UserDefinedServiceStub(channel)

        try:
            stub.Method1(serve_pb2.UserDefinedMessage())
            return True
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.RESOURCE_EXHAUSTED:
                return False

            raise RuntimeError(f"Unexpected status code: {e.code()}")

    for _do_request in [_do_grpc_request, _do_http_request]:
        with ThreadPoolExecutor() as tpe:
            # Submit `max_ongoing_requests` blocking requests.
            futures = [tpe.submit(_do_request) for _ in range(5)]
            wait_for_condition(
                lambda: ray.get(wait_signal.cur_num_waiters.remote()) == 5
            )
            assert all(not f.done() for f in futures)

            # Send another request beyond `max_ongoing_requests`, should error.
            assert _do_request() is False

            # Unblock the requests, check they finish successfully.
            ray.get(wait_signal.send.remote())
            assert all(f.result() is True for f in futures)

        # Now a new request showld succeed.
        assert _do_request() is True

        ray.get(wait_signal.send.remote(clear=True))


def test_port_retry_logic(_skip_if_ff_not_enabled, serve_instance):
    """Test that replicas retry port allocation when ports are in use."""
    import socket

    # Create a function to occupy a port
    def occupy_port(port):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("localhost", port))
        sock.listen(1)
        return sock

    # Start occupying the min HTTP and gRPC ports
    http_sock = occupy_port(RAY_SERVE_DIRECT_INGRESS_MIN_HTTP_PORT)
    grpc_sock = occupy_port(RAY_SERVE_DIRECT_INGRESS_MIN_GRPC_PORT)

    try:
        # Deploy a service - it should retry port allocation and eventually fall back
        # to shared ingress since we're occupying the ports
        serve.run(Hybrid.bind(message="Hello world!"))

        serve_details = ServeInstanceDetails(
            **ServeSubmissionClient("http://localhost:8265").get_serve_details()
        )
        target_groups = serve_details.target_groups

        # Check HTTP target group
        http_target_group = next(
            (tg for tg in target_groups if tg.protocol == RequestProtocol.HTTP)
        )
        assert (
            http_target_group.targets[0].port != RAY_SERVE_DIRECT_INGRESS_MIN_HTTP_PORT
        )

        # Check gRPC target group
        grpc_target_group = next(
            (tg for tg in target_groups if tg.protocol == RequestProtocol.GRPC)
        )
        assert (
            grpc_target_group.targets[0].port != RAY_SERVE_DIRECT_INGRESS_MIN_GRPC_PORT
        )

        # Verify the service still works through shared ingress
        r = requests.get(f"http://localhost:{http_target_group.targets[0].port}/")
        r.raise_for_status()
        assert r.text == "Hello world!"

    finally:
        # Clean up the sockets
        http_sock.close()
        grpc_sock.close()


def test_replica_gives_up_after_max_port_retries_for_http(
    _skip_if_ff_not_enabled, serve_instance
):
    """Test that replicas give up after max port retries."""
    import socket

    occupied_ports = []
    # TODO(sheikh): Control env variables
    for port in range(
        RAY_SERVE_DIRECT_INGRESS_MIN_HTTP_PORT,
        RAY_SERVE_DIRECT_INGRESS_MIN_HTTP_PORT
        + RAY_SERVE_DIRECT_INGRESS_PORT_RETRY_COUNT,
    ):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("localhost", port))
        sock.listen(1)
        occupied_ports.append(sock)

    serve._run(Hybrid.bind(message="Hello world!"), _blocking=False)

    # wait to deployment to be DEPLOY_FAILED
    def _func():
        serve_details = ServeInstanceDetails(
            **ServeSubmissionClient("http://localhost:8265").get_serve_details()
        )
        status = (
            serve_details.applications["default"]
            .deployments["default-deployment"]
            .status
        )
        assert status == DeploymentStatus.DEPLOY_FAILED
        return True

    wait_for_condition(_func, timeout=20)

    serve.delete("default", _blocking=True)


def test_replica_gives_up_after_max_port_retries_for_grpc(
    _skip_if_ff_not_enabled, serve_instance
):
    """Test that replicas give up after max port retries."""
    import socket

    occupied_ports = []
    for port in range(
        RAY_SERVE_DIRECT_INGRESS_MIN_GRPC_PORT,
        RAY_SERVE_DIRECT_INGRESS_MIN_GRPC_PORT
        + RAY_SERVE_DIRECT_INGRESS_PORT_RETRY_COUNT,
    ):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("localhost", port))
            sock.listen(1)
        except socket.error:
            # Port may already be in use, continue to next port
            pass
        occupied_ports.append(sock)

    serve._run(Hybrid.bind(message="Hello world!"), _blocking=False)

    # wait to deployment to be DEPLOY_FAILED
    def _func():
        serve_details = ServeInstanceDetails(
            **ServeSubmissionClient("http://localhost:8265").get_serve_details()
        )
        status = (
            serve_details.applications["default"]
            .deployments["default-deployment"]
            .status
        )
        assert status == DeploymentStatus.DEPLOY_FAILED
        return True

    wait_for_condition(_func, timeout=20)

    serve.delete("default", _blocking=True)


def test_no_port_available(_skip_if_ff_not_enabled, serve_instance):
    """Test that replicas give up after max port retries."""
    import socket

    occupied_ports = []
    for port in range(
        RAY_SERVE_DIRECT_INGRESS_MIN_HTTP_PORT, RAY_SERVE_DIRECT_INGRESS_MAX_HTTP_PORT
    ):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("localhost", port))
        sock.listen(1)
        occupied_ports.append(sock)

    """Test that multiple replicas on the same node occupy unique ports."""
    serve._run(
        Hybrid.options(name="default-deployment").bind(message="Hello world!"),
        _blocking=False,
    )

    # check that the deployment failed
    def _func():
        serve_details = ServeInstanceDetails(
            **ServeSubmissionClient("http://localhost:8265").get_serve_details()
        )
        assert (
            serve_details.applications["default"]
            .deployments["default-deployment"]
            .status
            == DeploymentStatus.DEPLOY_FAILED
        )
        assert (
            serve_details.applications["default"].status
            == ApplicationStatus.DEPLOY_FAILED
        )
        return True

    wait_for_condition(_func, timeout=20)


def test_replica_releases_ports_on_shutdown(_skip_if_ff_not_enabled, serve_instance):
    """Test that replicas release ports on shutdown."""
    serve.run(Hybrid.options(num_replicas=4).bind(message="Hello world!"))

    http_ports = get_http_ports(serve_instance)
    grpc_ports = get_grpc_ports(serve_instance)
    assert set(http_ports) == {30000, 30001, 30002, 30003}
    assert set(grpc_ports) == {40000, 40001, 40002, 40003}

    assert len(http_ports) == 4
    assert len(grpc_ports) == 4

    def _is_port_in_use(port):
        import socket

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        return sock.connect_ex(("0.0.0.0", port)) == 0

    # Check that the ports are occupied
    for http_port in http_ports:
        assert _is_port_in_use(http_port)
    for grpc_port in grpc_ports:
        assert _is_port_in_use(grpc_port)

    # Shutdown the replica
    serve.delete("default", _blocking=True)

    # Check that the ports are released
    for http_port in http_ports:
        assert not _is_port_in_use(http_port)
    for grpc_port in grpc_ports:
        assert not _is_port_in_use(grpc_port)


def test_get_serve_instance_details(_skip_if_ff_not_enabled, serve_instance):
    """Test that get_serve_instance_details returns the correct information."""
    serve.run(Hybrid.options(num_replicas=4).bind(message="Hello world!"))

    serve_details = ServeInstanceDetails(
        **ServeSubmissionClient("http://localhost:8265").get_serve_details()
    )

    assert len(serve_details.target_groups) == 2
    assert len(serve_details.target_groups[0].targets) == 4
    assert len(serve_details.target_groups[1].targets) == 4


def test_only_ingress_deployment_replicas_are_used_for_target_groups(
    _skip_if_ff_not_enabled, serve_instance
):
    @serve.deployment(num_replicas=2)
    class DownstreamDeployment:
        def __init__(self):
            pass

        def __call__(self):
            return "downstream-deployment"

    @serve.deployment(num_replicas=3)
    class IngressDeployment:
        def __init__(self, downstream_deployment: DownstreamDeployment):
            self.downstream_deployment = downstream_deployment

        async def __call__(self):
            res = await self.downstream_deployment.remote()
            return f"ingress-deployment-{res}"

        async def Method1(
            self, request: serve_pb2.UserDefinedMessage
        ) -> serve_pb2.UserDefinedResponse:
            res = await self.downstream_deployment.remote()
            return serve_pb2.UserDefinedResponse(greeting=f"ingress-deployment-{res}")

    serve.run(
        IngressDeployment.options(name="ingress-deployment").bind(
            DownstreamDeployment.options(name="downstream-deployment").bind()
        )
    )

    serve_details = ServeInstanceDetails(
        **ServeSubmissionClient("http://localhost:8265").get_serve_details()
    )

    assert len(serve_details.target_groups) == 2
    assert len(serve_details.target_groups[0].targets) == 3
    assert len(serve_details.target_groups[1].targets) == 3

    # test that the target groups are unique and contain the correct ports for ingress deployment
    http_ports = get_http_ports(serve_instance)
    grpc_ports = get_grpc_ports(serve_instance)
    assert len(set(http_ports) & {30000, 30001, 30002, 30003, 30004}) == 3
    assert len(set(grpc_ports) & {40000, 40001, 40002, 40003, 40004}) == 3

    for http_port in http_ports:
        req = requests.get(f"http://localhost:{http_port}/")
        assert req.status_code == 200
        assert req.text == "ingress-deployment-downstream-deployment"

    for grpc_port in grpc_ports:
        channel = grpc.insecure_channel(f"localhost:{grpc_port}")
        stub = serve_pb2_grpc.UserDefinedServiceStub(channel)
        assert (
            stub.Method1(serve_pb2.UserDefinedMessage()).greeting
            == "ingress-deployment-downstream-deployment"
        )


def test_crashed_replica_port_is_released_and_reused(
    _skip_if_ff_not_enabled, serve_instance
):
    """Test that crashed replica port is released and reused."""
    serve.run(Hybrid.options(num_replicas=4).bind(message="Hello world!"))

    http_ports = get_http_ports(serve_instance)
    grpc_ports = get_grpc_ports(serve_instance)
    assert set(http_ports) == {30000, 30001, 30002, 30003}
    assert set(grpc_ports) == {40000, 40001, 40002, 40003}

    # delete the application
    serve.delete("default", _blocking=True)

    # run the deployment again
    serve.run(Hybrid.options(num_replicas=4).bind(message="Hello world!"))

    new_http_ports = get_http_ports(serve_instance)
    new_grpc_ports = get_grpc_ports(serve_instance)

    assert set(http_ports) == set(new_http_ports)
    assert set(grpc_ports) == set(new_grpc_ports)

    # get pid of the replicas
    serve_details = ServeInstanceDetails(
        **ServeSubmissionClient("http://localhost:8265").get_serve_details()
    )
    replicas = (
        serve_details.applications["default"].deployments["default-deployment"].replicas
    )
    pids = [replica.pid for replica in replicas]

    # kill the replicas
    import signal
    import os

    # force kill the replicas
    os.kill(pids[0], signal.SIGKILL)
    # keyboard interrupt the replicas
    os.kill(pids[1], signal.SIGINT)
    # TODO(sheikh): Find a way to gracefully stop the replicas

    def _func():
        # get pid of the replicas
        serve_details = ServeInstanceDetails(
            **ServeSubmissionClient("http://localhost:8265").get_serve_details()
        )
        replicas = (
            serve_details.applications["default"]
            .deployments["default-deployment"]
            .replicas
        )
        new_pids = [replica.pid for replica in replicas]
        assert new_pids != pids and len(new_pids) == 4
        return True

    wait_for_condition(_func, timeout=20)

    # wait for deployment to be running
    def _func2():
        serve_details = ServeInstanceDetails(
            **ServeSubmissionClient("http://localhost:8265").get_serve_details()
        )
        assert (
            serve_details.applications["default"]
            .deployments["default-deployment"]
            .status
            == DeploymentStatus.HEALTHY
        )
        return True

    wait_for_condition(lambda: _func2(), timeout=30)

    # check that the ports are released
    after_crash_http_ports = get_http_ports(serve_instance)
    after_crash_grpc_ports = get_grpc_ports(serve_instance)

    assert len(after_crash_http_ports) == 4
    assert len(after_crash_grpc_ports) == 4

    # show that smart port selection is working even with crashed ports
    assert set(after_crash_http_ports) == set(http_ports)
    assert set(after_crash_grpc_ports) == set(grpc_ports)


def test_multiple_applications_on_same_node(_skip_if_ff_not_enabled, serve_instance):
    """Test that multiple applications, such that each app has a ingress deployment"""

    @serve.deployment(num_replicas=2)
    def deployment_1():
        return "deployment-1"

    @serve.deployment(num_replicas=2)
    def deployment_2():
        return "deployment-2"

    serve.run(
        deployment_1.options(name="deployment-1").bind(),
        name="app-1",
        route_prefix="/app-1",
    )
    serve.run(
        deployment_2.options(name="deployment-2").bind(),
        name="app-2",
        route_prefix="/app-2",
    )

    http_ports_1 = get_http_ports(serve_instance, "/app-1")
    http_ports_2 = get_http_ports(serve_instance, "/app-2")
    grpc_ports_1 = get_grpc_ports(serve_instance, "/app-1")
    grpc_ports_2 = get_grpc_ports(serve_instance, "/app-2")

    assert set(http_ports_1) == {30000, 30001}
    assert set(http_ports_2) == {30002, 30003}
    assert set(grpc_ports_1) == {40000, 40001}
    assert set(grpc_ports_2) == {40002, 40003}

    # make a request to the ingress deployment
    req = requests.get(f"http://localhost:{http_ports_1[0]}/app-1")
    assert req.status_code == 200
    assert req.text == "deployment-1"

    # make a request to the other ingress deployment
    req = requests.get(f"http://localhost:{http_ports_2[0]}/app-2")
    assert req.status_code == 200
    assert req.text == "deployment-2"


def test_app_with_composite_deployments(_skip_if_ff_not_enabled, serve_instance):
    """Test that an app with composite deployments can be deployed. verify
    that ports are occupied by all deployments in the app but only the ingress
    deployment is used for the target groups"""

    @serve.deployment(num_replicas=3)
    class ChildDeployment:
        def __call__(self):
            return "child-deployment"

    @serve.deployment(num_replicas=2)
    class IngressDeployment:
        def __init__(self, child_deployment: ChildDeployment):
            self.child_deployment = child_deployment

        async def __call__(self):
            return await self.child_deployment.remote()

        async def Method1(
            self, request: serve_pb2.UserDefinedMessage
        ) -> serve_pb2.UserDefinedResponse:
            res = await self.child_deployment.remote()
            return serve_pb2.UserDefinedResponse(greeting=res)

    serve.run(
        IngressDeployment.options(name="ingress-deployment").bind(
            ChildDeployment.options(name="child-deployment").bind()
        ),
        name="app-1",
        route_prefix="/app-1",
    )

    # test that the target groups are unique and contain the correct ports for ingress deployment
    http_ports = get_http_ports(serve_instance)
    grpc_ports = get_grpc_ports(serve_instance)
    # difficult to say which ports are used for the target groups
    assert len(set(http_ports) & {30000, 30001, 30002, 30003, 30004}) == 2
    assert len(set(grpc_ports) & {40000, 40001, 40002, 40003, 40004}) == 2

    # make a request to the ingress deployment
    req = requests.get(f"http://localhost:{http_ports[0]}/app-1")
    assert req.status_code == 200
    assert req.text == "child-deployment"

    # grpc request
    channel = grpc.insecure_channel(f"localhost:{grpc_ports[0]}")
    stub = serve_pb2_grpc.UserDefinedServiceStub(channel)
    assert stub.Method1(serve_pb2.UserDefinedMessage()).greeting == "child-deployment"


def test_only_running_apps_are_used_for_target_groups(
    _skip_if_ff_not_enabled, serve_instance
):
    """Test that only running apps are used for target groups"""

    signal_actor = SignalActor.remote()

    @serve.deployment(num_replicas=2)
    def deployment_1():
        return "deployment-1"

    @serve.deployment(num_replicas=2)
    class Deployment2:
        async def __init__(self, signal_actor: SignalActor):
            self.signal_actor = signal_actor
            await self.signal_actor.wait.remote()

        async def __call__(self):
            return "deployment-2"

    serve.run(
        deployment_1.options(name="deployment-1").bind(),
        name="app-1",
        route_prefix="/app-1",
    )
    serve._run(
        Deployment2.options(name="deployment-2").bind(signal_actor=signal_actor),
        name="app-2",
        route_prefix="/app-2",
        _blocking=False,
    )
    wait_for_condition(
        lambda: ray.get(signal_actor.cur_num_waiters.remote()) == 2, timeout=10
    )
    serve_details = ServeInstanceDetails(
        **ServeSubmissionClient("http://localhost:8265").get_serve_details()
    )
    assert (
        serve_details.applications["app-2"].deployments["deployment-2"].status
        == DeploymentStatus.UPDATING
    )
    assert serve_details.applications["app-2"].status == ApplicationStatus.DEPLOYING
    assert serve_details.applications["app-1"].status == ApplicationStatus.RUNNING

    http_ports = get_http_ports(serve_instance, first_only=False)
    grpc_ports = get_grpc_ports(serve_instance, first_only=False)
    assert set(http_ports) == {30000, 30001, 8000}
    assert set(grpc_ports) == {40000, 40001, 9000}

    ray.get(signal_actor.send.remote())

    def _func():
        serve_details = ServeInstanceDetails(
            **ServeSubmissionClient("http://localhost:8265").get_serve_details()
        )
        assert serve_details.applications["app-2"].status == ApplicationStatus.RUNNING
        return True

    wait_for_condition(_func, timeout=10)

    http_ports = get_http_ports(serve_instance, "/app-1", first_only=False)
    grpc_ports = get_grpc_ports(serve_instance, "/app-1", first_only=False)
    assert set(http_ports) == {30000, 30001}
    assert set(grpc_ports) == {40000, 40001}

    http_ports = get_http_ports(serve_instance, "/app-2", first_only=False)
    grpc_ports = get_grpc_ports(serve_instance, "/app-2", first_only=False)
    assert set(http_ports) == {30002, 30003}
    assert set(grpc_ports) == {40002, 40003}


def test_some_replicas_not_running(_skip_if_ff_not_enabled, serve_instance):
    signal_actor = Semaphore.remote(2)

    @serve.deployment(num_replicas=4)
    class Deployment1:
        async def __init__(self):
            await signal_actor.acquire.remote()

        def __call__(self):
            return "deployment-1"

    serve._run(
        Deployment1.options(name="deployment-1").bind(),
        name="app-1",
        route_prefix="/app-1",
        _blocking=False,
    )

    def _func():
        http_ports = get_http_ports(serve_instance, "/app-1", first_only=False)
        grpc_ports = get_grpc_ports(serve_instance, "/app-1", first_only=False)
        assert set(http_ports) == {30000, 30001}
        assert set(grpc_ports) == {40000, 40001}
        return True

    wait_for_condition(_func, timeout=10)

    # check status of the deployment
    serve_details = ServeInstanceDetails(
        **ServeSubmissionClient("http://localhost:8265").get_serve_details()
    )
    assert (
        serve_details.applications["app-1"].deployments["deployment-1"].status
        == DeploymentStatus.UPDATING
    )
    assert serve_details.applications["app-1"].status == ApplicationStatus.DEPLOYING


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", "-s", __file__]))
