import asyncio
import sys
import threading
import time
from unittest import mock
import os
import pytest
import pytest_asyncio
import tempfile
from ray._common.test_utils import async_wait_for_condition, wait_for_condition
import requests
import uvicorn

from ray.cluster_utils import Cluster
from fastapi import FastAPI, Response, Request
import ray
from ray import serve
from ray._private.test_utils import find_free_port
from ray.actor import ActorHandle
from ray.anyscale.serve._private.haproxy import (
    BackendConfig,
    HAProxyApi,
    HAProxyConfig,
    ServerConfig,
)
from ray.serve._private.constants import (
    DEFAULT_UVICORN_KEEP_ALIVE_TIMEOUT_S,
    SERVE_NAMESPACE,
)
from ray.serve.context import _get_global_client
from ray.serve.schema import (
    ProxyStatus,
    ServeDeploySchema,
    ServeInstanceDetails,
)
from ray.serve.tests.conftest import *  # noqa
from ray.tests.conftest import call_ray_stop_only  # noqa: F401
from ray.util.state import list_actors
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Skip all tests in this module if the HAProxy feature flag is not enabled
# pytestmark = pytest.mark.skipif(
#     not ANYSCALE_RAY_SERVE_ENABLE_HA_PROXY,
#     reason="ANYSCALE_RAY_SERVE_ENABLE_HA_PROXY not set.",
# )


def check_haproxy_ready(stats_port: int, timeout: int = 2) -> bool:
    """Check if HAProxy is ready by verifying the stats endpoint is accessible."""
    try:
        response = requests.get(f"http://127.0.0.1:{stats_port}/stats", timeout=timeout)
        return response.status_code == 200
    except Exception:
        return False


def create_test_backend_server(port: int):
    """Create a test backend server with slow and fast endpoints using uvicorn."""
    app = FastAPI()

    @app.get("/health")
    async def health_endpoint():
        return {"status": "OK"}

    @app.get("/slow")
    async def slow_endpoint():
        await asyncio.sleep(3)  # 3-second delay
        return "Slow response completed"

    @app.get("/fast")
    async def fast_endpoint(req: Request, res: Response):
        res.headers["x-haproxy-reload-id"] = req.headers.get("x-haproxy-reload-id", "")

        return "Fast response"

    # Configure uvicorn server with 60s keep-alive timeout
    config = uvicorn.Config(
        app=app,
        host="127.0.0.1",
        port=port,
        log_level="error",  # Reduce log noise
        access_log=False,
        timeout_keep_alive=60,  # 60 seconds keep-alive timeout
    )
    server = uvicorn.Server(config)

    # Run server in a separate thread
    def run_server():
        asyncio.run(server.serve())

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()

    # Wait for the server to start
    def wait_for_server():
        r = requests.get(f"http://127.0.0.1:{port}/health")
        assert r.status_code == 200
        return True

    wait_for_condition(wait_for_server)
    return server, thread


def process_exists(pid: int) -> bool:
    """Check if a process with the given PID exists."""
    try:
        # Send signal 0 to check if process exists without actually sending a signal
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def make_test_request(
    url: str,
    track_results: list = None,
    signal_started: threading.Event = None,
    timeout: int = 10,
):
    """Unified function to make test requests with optional result tracking."""
    try:
        if signal_started:
            signal_started.set()  # Signal that request has started

        start_time = time.time()
        response = requests.get(url, timeout=timeout)
        end_time = time.time()

        if track_results is not None:
            track_results.append(
                {
                    "status": response.status_code,
                    "duration": end_time - start_time,
                    "content": response.content,
                }
            )
    except Exception as ex:
        if track_results is not None:
            track_results.append({"error": str(ex)})


@pytest.fixture
def shutdown_ray():
    if ray.is_initialized():
        ray.shutdown()
    yield
    if ray.is_initialized():
        ray.shutdown()


def test_deploy_with_no_applications(ray_shutdown):
    """Deploy an empty list of applications, serve should just be started."""
    ray.init(num_cpus=8)
    serve.start(http_options=dict(port=8003))
    client = _get_global_client()
    config = ServeDeploySchema.parse_obj({"applications": []})
    client.deploy_apps(config)

    def serve_running():
        ServeInstanceDetails.parse_obj(
            ray.get(client._controller.get_serve_instance_details.remote())
        )
        actors = list_actors(
            filters=[
                ("ray_namespace", "=", SERVE_NAMESPACE),
                ("state", "=", "ALIVE"),
            ]
        )
        actor_names = [actor["class_name"] for actor in actors]
        return "ServeController" in actor_names and "HAProxyManager" in actor_names

    wait_for_condition(serve_running)
    client.shutdown()


def test_single_app_shutdown_actors(ray_shutdown):
    """Tests serve.shutdown() works correctly in single-app case

    Ensures that after deploying a (nameless) app using serve.run(), serve.shutdown()
    deletes all actors (controller, haproxy, all replicas) in the "serve" namespace.
    """
    address = ray.init(num_cpus=8)["address"]
    serve.start(http_options=dict(port=8003))

    @serve.deployment
    def f():
        pass

    serve.run(f.bind(), name="app")

    actor_names = {
        "ServeController",
        "HAProxyManager",
        "ServeReplica:app:f",
    }

    def check_alive():
        actors = list_actors(
            address=address,
            filters=[("ray_namespace", "=", SERVE_NAMESPACE), ("state", "=", "ALIVE")],
        )
        return {actor["class_name"] for actor in actors} == actor_names

    def check_dead():
        actors = list_actors(
            address=address,
            filters=[("ray_namespace", "=", SERVE_NAMESPACE), ("state", "=", "ALIVE")],
        )
        return len(actors) == 0

    wait_for_condition(check_alive)
    serve.shutdown()
    wait_for_condition(check_dead)


@pytest.mark.asyncio
async def test_single_app_shutdown_actors_async(ray_shutdown):
    """Tests serve.shutdown_async() works correctly in single-app case

    Ensures that after deploying a (nameless) app using serve.run(), serve.shutdown_async()
    deletes all actors (controller, haproxy, all replicas) in the "serve" namespace.
    """
    address = ray.init(num_cpus=8)["address"]
    serve.start(http_options=dict(port=8003))

    @serve.deployment
    def f():
        pass

    serve.run(f.bind(), name="app")

    actor_names = {
        "ServeController",
        "HAProxyManager",
        "ServeReplica:app:f",
    }

    def check_alive():
        actors = list_actors(
            address=address,
            filters=[("ray_namespace", "=", SERVE_NAMESPACE), ("state", "=", "ALIVE")],
        )
        return {actor["class_name"] for actor in actors} == actor_names

    def check_dead():
        actors = list_actors(
            address=address,
            filters=[("ray_namespace", "=", SERVE_NAMESPACE), ("state", "=", "ALIVE")],
        )
        return len(actors) == 0

    wait_for_condition(check_alive)
    await serve.shutdown_async()
    wait_for_condition(check_dead)


# TODO(alexyang): Delete these tests and run test_proxy.py instead once HAProxy is fully supported.
class TestTimeoutKeepAliveConfig:
    """Test setting keep_alive_timeout_s in config and env."""

    def get_proxy_actor(self) -> ActorHandle:
        [proxy_actor] = list_actors(filters=[("class_name", "=", "HAProxyManager")])
        return ray.get_actor(proxy_actor.name, namespace=SERVE_NAMESPACE)

    def test_default_keep_alive_timeout_s(self, ray_shutdown):
        """Test when no keep_alive_timeout_s is set.

        When the keep_alive_timeout_s is not set, the uvicorn keep alive is 5.
        """
        serve.start()
        proxy_actor = self.get_proxy_actor()
        assert (
            ray.get(proxy_actor._get_http_options.remote()).keep_alive_timeout_s
            == DEFAULT_UVICORN_KEEP_ALIVE_TIMEOUT_S
        )

    def test_set_keep_alive_timeout_in_http_configs(self, ray_shutdown):
        """Test when keep_alive_timeout_s is in http configs.

        When the keep_alive_timeout_s is set in http configs, the uvicorn keep alive
        is set correctly.
        """
        keep_alive_timeout_s = 222
        serve.start(http_options={"keep_alive_timeout_s": keep_alive_timeout_s})
        proxy_actor = self.get_proxy_actor()
        assert (
            ray.get(proxy_actor._get_http_options.remote()).keep_alive_timeout_s
            == keep_alive_timeout_s
        )

    @pytest.mark.parametrize(
        "ray_instance",
        [
            {"RAY_SERVE_HTTP_KEEP_ALIVE_TIMEOUT_S": "333"},
        ],
        indirect=True,
    )
    def test_set_keep_alive_timeout_in_env(self, ray_instance, ray_shutdown):
        """Test when keep_alive_timeout_s is in env.

        When the keep_alive_timeout_s is set in env, the uvicorn keep alive
        is set correctly.
        """
        serve.start()
        proxy_actor = self.get_proxy_actor()
        assert (
            ray.get(proxy_actor._get_http_options.remote()).keep_alive_timeout_s == 333
        )

    @pytest.mark.parametrize(
        "ray_instance",
        [
            {"RAY_SERVE_HTTP_KEEP_ALIVE_TIMEOUT_S": "333"},
        ],
        indirect=True,
    )
    def test_set_timeout_keep_alive_in_both_config_and_env(
        self, ray_instance, ray_shutdown
    ):
        """Test when keep_alive_timeout_s is in both http configs and env.

        When the keep_alive_timeout_s is set in env, the uvicorn keep alive
        is set to the one in env.
        """
        keep_alive_timeout_s = 222
        serve.start(http_options={"keep_alive_timeout_s": keep_alive_timeout_s})
        proxy_actor = self.get_proxy_actor()
        assert (
            ray.get(proxy_actor._get_http_options.remote()).keep_alive_timeout_s == 333
        )


def test_drain_and_undrain_haproxy_manager(
    monkeypatch, shutdown_ray, call_ray_stop_only  # noqa: F811
):
    """Test the state transtion of the haproxy manager between
    HEALTHY, DRAINING and DRAINED
    """
    monkeypatch.setenv("RAY_SERVE_PROXY_MIN_DRAINING_PERIOD_S", "10")

    cluster = Cluster()
    head_node = cluster.add_node(num_cpus=0)
    cluster.add_node(num_cpus=1)
    cluster.add_node(num_cpus=1)
    cluster.wait_for_nodes()
    ray.init(address=head_node.address)
    serve.start(http_options={"location": "EveryNode"})

    @serve.deployment
    class HelloModel:
        def __call__(self):
            return "hello"

    serve.run(HelloModel.options(num_replicas=2).bind())

    # 3 proxies, 1 controller, 2 replicas.
    wait_for_condition(lambda: len(list_actors()) == 6)
    assert len(ray.nodes()) == 3

    client = _get_global_client()
    serve_details = ServeInstanceDetails(
        **ray.get(client._controller.get_serve_instance_details.remote())
    )
    proxy_actor_ids = {proxy.actor_id for _, proxy in serve_details.proxies.items()}

    assert len(proxy_actor_ids) == 3

    serve.run(HelloModel.options(num_replicas=1).bind())
    # 1 proxy should be draining

    def check_proxy_status(proxy_status_to_count):
        serve_details = ServeInstanceDetails(
            **ray.get(client._controller.get_serve_instance_details.remote())
        )
        proxy_status_list = [proxy.status for _, proxy in serve_details.proxies.items()]
        print("all proxies!!!", [proxy for _, proxy in serve_details.proxies.items()])
        current_status = {
            status: proxy_status_list.count(status) for status in proxy_status_list
        }
        return current_status == proxy_status_to_count, current_status

    wait_for_condition(
        condition_predictor=check_proxy_status,
        proxy_status_to_count={ProxyStatus.HEALTHY: 2, ProxyStatus.DRAINING: 1},
    )

    serve.run(HelloModel.options(num_replicas=2).bind())
    # The draining proxy should become healthy.
    wait_for_condition(
        condition_predictor=check_proxy_status,
        proxy_status_to_count={ProxyStatus.HEALTHY: 3},
    )
    serve_details = ServeInstanceDetails(
        **ray.get(client._controller.get_serve_instance_details.remote())
    )

    assert {
        proxy.actor_id for _, proxy in serve_details.proxies.items()
    } == proxy_actor_ids

    serve.run(HelloModel.options(num_replicas=1).bind())
    # 1 proxy should be draining and eventually be drained.
    wait_for_condition(
        condition_predictor=check_proxy_status,
        timeout=40,
        proxy_status_to_count={ProxyStatus.HEALTHY: 2},
    )

    # Clean up serve.
    serve.shutdown()


def test_haproxy_failure(ray_shutdown):
    """Test HAProxyManager is successfully restarted after being killed."""
    ray.init(num_cpus=1)
    serve.start()

    @serve.deployment(name="proxy_failure")
    def function(_):
        return "hello1"

    serve.run(function.bind())

    def check_proxy_alive():
        actors = list_actors(
            filters=[("ray_namespace", "=", SERVE_NAMESPACE), ("state", "=", "ALIVE")],
        )
        return "HAProxyManager" in {actor["class_name"] for actor in actors}

    wait_for_condition(check_proxy_alive)

    [proxy_actor] = list_actors(
        filters=[("class_name", "=", "HAProxyManager"), ("state", "=", "ALIVE")]
    )
    proxy_actor_id = proxy_actor.actor_id

    proxy_actor = ray.get_actor(proxy_actor.name, namespace=SERVE_NAMESPACE)
    ray.kill(proxy_actor, no_restart=False)

    def check_new_proxy():
        proxies = list_actors(
            filters=[("class_name", "=", "HAProxyManager"), ("state", "=", "ALIVE")]
        )
        return len(proxies) == 1 and proxies[0].actor_id != proxy_actor_id

    wait_for_condition(check_new_proxy, timeout=45)
    serve.shutdown()


def test_haproxy_loop_get_target_groups(shutdown_ray):
    """Test that haproxy get_target_groups retrieves the correct target groups."""
    ray.init(num_cpus=4)
    serve.start()

    @serve.deployment
    def function(_):
        return "hello1"

    # Deploy the application
    serve.run(
        function.options(num_replicas=1).bind(), name="test_app", route_prefix="/test"
    )

    def check_proxy_alive():
        actors = list_actors(
            filters=[("ray_namespace", "=", SERVE_NAMESPACE), ("state", "=", "ALIVE")],
        )
        return "HAProxyManager" in {actor["class_name"] for actor in actors}

    wait_for_condition(check_proxy_alive)

    [proxy_actor] = list_actors(
        filters=[("class_name", "=", "HAProxyManager"), ("state", "=", "ALIVE")]
    )
    proxy_actor = ray.get_actor(proxy_actor.name, namespace=SERVE_NAMESPACE)

    def has_n_targets(route_prefix: str, n: int):
        target_groups = ray.get(proxy_actor.get_target_groups.remote())
        for tg in target_groups:
            if tg.route_prefix == route_prefix and len(tg.targets) == n:
                return True
        return False

    wait_for_condition(has_n_targets, route_prefix="/test", n=1)

    serve.run(
        function.options(num_replicas=2).bind(), name="test_app", route_prefix="/test2"
    )
    wait_for_condition(has_n_targets, route_prefix="/test2", n=2)

    serve.shutdown()


@pytest_asyncio.fixture
async def haproxy_api_cleanup():
    registered_apis = []

    def register(api: Optional[HAProxyApi]) -> None:
        if api is not None:
            registered_apis.append(api)

    yield register

    for api in registered_apis:
        proc = getattr(api, "proc", None)
        if proc and proc.returncode is None:
            try:
                await api.stop()
            except Exception as exc:  # pragma: no cover - best effort cleanup
                logger.warning(f"Failed to stop HAProxy API cleanly: {exc}")
                try:
                    proc.kill()
                    await proc.wait()
                except Exception as kill_exc:
                    logger.error(
                        f"Failed to kill HAProxy process {proc.pid}: {kill_exc}"
                    )
        elif proc and proc.returncode is not None:
            continue


@pytest.mark.asyncio
async def test_generate_config_file_internal(haproxy_api_cleanup):
    """Test that initialize writes the correct config_stub file content using the actual template."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config_file_path = os.path.join(temp_dir, "haproxy.cfg")

        config_stub = HAProxyConfig(
            socket_path="/test/admin.sock",
            maxconn=1000,
            nbthread=2,
            timeout_connect_s=5,
            timeout_client_s=30,
            timeout_server_s=30,
            timeout_http_request_s=10,
            timeout_http_keep_alive_s=55,
            timeout_queue_s=1,
            stats_port=8080,
            stats_uri="/mystats",
            frontend_port=8000,
            frontend_host="0.0.0.0",
            health_check_fall=3,
            health_check_rise=2,
            health_check_inter="2s",
            health_check_path="/health",
        )
        backend_config_stub = {
            "api_backend": BackendConfig(
                name="api_backend",
                path_prefix="/api",
                timeout_http_keep_alive_s=60,
                timeout_tunnel_s=60,
                health_check_path="/api/health",
                health_check_fall=2,
                health_check_rise=3,
                health_check_inter="5s",
                servers=[
                    ServerConfig(name="api_server1", host="127.0.0.1", port=8001),
                    ServerConfig(name="api_server2", host="127.0.0.1", port=8002),
                ],
            ),
            "web_backend": BackendConfig(
                name="web_backend",
                path_prefix="/web",
                timeout_connect_s=3,
                timeout_server_s=25,
                timeout_http_keep_alive_s=45,
                timeout_tunnel_s=45,
                servers=[
                    ServerConfig(name="web_server1", host="127.0.0.1", port=8003),
                ]
                # No health check overrides - should use global defaults
            ),
        }

        with mock.patch(
            "ray.anyscale.serve._private.haproxy.HAPROXY_CONFIG_FILE_LOC",
            config_file_path,
        ):

            api = HAProxyApi(
                cfg=config_stub,
                backend_configs=backend_config_stub,
                config_file_path=config_file_path,
            )

            try:
                await api._generate_config_file_internal()

                # Read and verify the generated file
                with open(config_file_path, "r") as f:
                    actual_content = f.read()

                # Expected configuration stub (matching the actual template output)
                expected_config = """
global
    # Log to the standard system log socket with debug level.
    log /dev/log local0 debug
    stats socket /test/admin.sock mode 666 level admin expose-fd listeners
    stats timeout 30s
    maxconn 1000
    nbthread 2
defaults
    mode http
    option log-health-checks
    timeout connect 5s
    timeout client 30s
    timeout server 30s
    timeout http-request 10s
    timeout http-keep-alive 55s
    timeout queue 1s
    log global
    option httplog
frontend http_frontend
    bind 0.0.0.0:8000
    # Health check endpoint
    acl healthcheck path -i /-/healthz
    http-request return status 200 content-type text/plain string "OK" if healthcheck
    # Static routing based on path prefixes
    acl is_api_backend path_beg /api
    use_backend backend_api_backend if is_api_backend
    acl is_web_backend path_beg /web
    use_backend backend_web_backend if is_web_backend
    default_backend default_backend
backend default_backend
    http-request deny deny_status 404
backend backend_api_backend
    log global
    balance leastconn
    # Enable HTTP connection reuse for better performance
    http-reuse always
    # Set backend-specific timeouts, overriding defaults if specified
    # Set timeouts to support keep-alive connections
    timeout http-keep-alive 60s
    timeout tunnel 60s
    # Health check configuration - use backend-specific or global defaults
    # HTTP health check with custom path
    option httpchk GET /api/health
    http-check expect status 200
    default-server fall 2 rise 3 inter 5s check
    # Servers in this backend
    server api_server1 127.0.0.1:8001 check
    server api_server2 127.0.0.1:8002 check
backend backend_web_backend
    log global
    balance leastconn
    # Enable HTTP connection reuse for better performance
    http-reuse always
    # Set backend-specific timeouts, overriding defaults if specified
    timeout connect 3s
    timeout server 25s
    # Set timeouts to support keep-alive connections
    timeout http-keep-alive 45s
    timeout tunnel 45s
    # Health check configuration - use backend-specific or global defaults
    # HTTP health check with custom path
    option httpchk GET /health
    http-check expect status 200
    default-server fall 3 rise 2 inter 2s check
    # Servers in this backend
    server web_server1 127.0.0.1:8003 check
listen stats
  bind *:8080
  stats enable
  stats uri /mystats
  stats refresh 1s
"""

                # Compare the entire configuration
                assert actual_content.strip() == expected_config.strip()
            finally:
                # Clean up any temporary files created by initialize()
                temp_files = ["haproxy.cfg", "routes.map"]
                for temp_file in temp_files:
                    try:
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                    except (FileNotFoundError, OSError):
                        pass  # File already removed or doesn't exist


@pytest.mark.asyncio
async def test_graceful_reload(haproxy_api_cleanup):
    """Test that graceful reload preserves long-running connections."""

    with tempfile.TemporaryDirectory() as temp_dir:
        # Setup ports
        haproxy_port = find_free_port()
        backend_port = find_free_port()

        # Create and start a backend server
        backend_server, backend_thread = create_test_backend_server(backend_port)

        # Configure HAProxy

        config = HAProxyConfig(
            frontend_port=haproxy_port,
            frontend_host="127.0.0.1",
            stats_port=find_free_port(),
            timeout_http_keep_alive_s=58,
            inject_process_id_header=True,  # Enable for testing graceful reload
            reload_id=f"initial-{int(time.time() * 1000)}",  # Set initial reload ID
            socket_path=os.path.join(temp_dir, "admin.sock"),
        )

        backend_config = BackendConfig(
            name="test_backend",
            path_prefix="/",
            servers=[ServerConfig(name="backend", host="127.0.0.1", port=backend_port)],
            timeout_http_keep_alive_s=58,
        )

        config_file_path = os.path.join(temp_dir, "haproxy.cfg")

        api = HAProxyApi(
            cfg=config,
            backend_configs={"test_backend": backend_config},
            config_file_path=config_file_path,
        )

        haproxy_api_cleanup(api)

        try:
            await api.start()

            # Wait for HAProxy to be ready (check stat endpoint)
            def check_stats_ready():
                try:
                    response = requests.get(
                        f"http://127.0.0.1:{config.stats_port}/stats", timeout=2
                    )
                    return response.status_code == 200
                except Exception:
                    return False

            wait_for_condition(check_stats_ready, timeout=10, retry_interval_ms=100)

            # Track slow request results
            slow_results = []
            request_started = threading.Event()

            slow_thread = threading.Thread(
                target=make_test_request,
                args=[f"http://127.0.0.1:{haproxy_port}/slow"],
                kwargs={
                    "track_results": slow_results,
                    "signal_started": request_started,
                },
            )

            slow_thread.start()
            wait_for_condition(
                lambda: request_started.is_set(), timeout=5, retry_interval_ms=10
            )

            assert api.proc is not None
            original_pid = api.proc.pid

            await api._graceful_reload()

            assert api.proc is not None
            new_pid = api.proc.pid

            def check_for_new_reload_id():
                fast_response = requests.get(
                    f"http://127.0.0.1:{haproxy_port}/fast", timeout=5
                )

                # Reload ID should always match what exists in the config.
                return (
                    fast_response.headers.get("x-haproxy-reload-id")
                    == api.cfg.reload_id
                    and fast_response.status_code == 200
                )

            wait_for_condition(
                check_for_new_reload_id, timeout=1, retry_interval_ms=100
            )

            slow_thread.join(timeout=10)

            assert (
                original_pid != new_pid
            ), "Process should have been reloaded with new PID"

            wait_for_condition(
                lambda: not process_exists(original_pid),
                timeout=15,
                retry_interval_ms=100,
            )

            assert len(slow_results) == 1, "Slow request should have completed"

            result = slow_results[0]
            assert "error" not in result, f"Slow request failed: {result.get('error')}"
            assert result["status"] == 200, "Slow request should have succeeded"
            assert result["duration"] >= 3.0, "Slow request should have taken full time"
            assert (
                b"Slow response completed" in result["content"]
            ), "Slow request should have completed"

        finally:
            # Backend server cleanup
            try:
                backend_server.should_exit = True
                backend_thread.join(timeout=5)  # Wait for thread to finish
            except Exception as e:
                print(f"Error occurred while shutting down server stub. Error: {e}")


@pytest.mark.asyncio
async def test_start(haproxy_api_cleanup):
    """Test HAProxy start functionality."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config_file_path = os.path.join(temp_dir, "haproxy.cfg")
        socket_path = os.path.join(temp_dir, "admin.sock")

        # Create HAProxy config
        config = HAProxyConfig(
            frontend_port=find_free_port(),
            frontend_host="127.0.0.1",
            stats_port=find_free_port(),
            pass_health_checks=True,
            socket_path=socket_path,
            timeout_http_keep_alive_s=58,
        )

        api = HAProxyApi(cfg=config, config_file_path=config_file_path)

        haproxy_api_cleanup(api)

        await api.start()
        await asyncio.sleep(0.5)

        assert api.proc is not None, "HAProxy process should exist"
        assert api._is_running(), "HAProxy should be running"

        # Verify config file contains expected content
        with open(config_file_path, "r") as f:
            config_content = f.read()
            assert "frontend http_frontend" in config_content
            assert f"bind 127.0.0.1:{config.frontend_port}" in config_content
            assert "acl healthcheck path -i /-/healthz" in config_content

        assert (
            "http-request return status 200" in config_content
        ), "Health checks should be enabled in config"

        # Verify config file contains expected content
        with open(config_file_path, "r") as f:
            config_content = f.read()
            assert "frontend http_frontend" in config_content
            assert f"bind 127.0.0.1:{config.frontend_port}" in config_content
            assert "acl healthcheck path -i /-/healthz" in config_content
            assert (
                "http-request return status 200" in config_content
            )  # Health checks enabled

        health_response = requests.get(
            f"http://127.0.0.1:{config.frontend_port}/-/healthz", timeout=5
        )
        assert health_response.status_code == 200, "Health check should return 200"
        assert b"OK" in health_response.content, "Health check should return 'OK'"

        await api.stop()
        assert api.proc is None
        assert not api._is_running()


@pytest.mark.asyncio
async def test_stop(haproxy_api_cleanup):
    """Test HAProxy stop functionality."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config_file_path = os.path.join(temp_dir, "haproxy.cfg")

        config = HAProxyConfig(
            frontend_port=find_free_port(),
            frontend_host="127.0.0.1",
            stats_port=find_free_port(),
            socket_path=os.path.join(temp_dir, "admin.sock"),
        )

        api = HAProxyApi(cfg=config, config_file_path=config_file_path)

        haproxy_api_cleanup(api)

        # Start HAProxy
        await api.start()

        haproxy_api_cleanup(api)

        await api.stop()

        # Verify it's stopped
        assert not api._is_running(), "HAProxy should be stopped after shutdown"


@pytest.mark.asyncio
async def test_get_stats_integration(haproxy_api_cleanup):
    with tempfile.TemporaryDirectory() as temp_dir:
        config_file_path = os.path.join(temp_dir, "haproxy.cfg")
        socket_path = os.path.join(temp_dir, "admin.sock")

        # Create test backend servers
        backend_port1 = find_free_port()
        backend_port2 = find_free_port()
        backend_server1, backend_thread1 = create_test_backend_server(backend_port1)
        backend_server2, backend_thread2 = create_test_backend_server(backend_port2)

        # Configure HAProxy with multiple backends
        config = HAProxyConfig(
            socket_path=socket_path,
            frontend_port=find_free_port(),
            stats_port=find_free_port(),
            timeout_http_keep_alive_s=58,
        )

        backend_configs = {
            "test_backend1": BackendConfig(
                name="test_backend1",
                path_prefix="/api",
                servers=[
                    ServerConfig(name="server1", host="127.0.0.1", port=backend_port1)
                ],
                timeout_http_keep_alive_s=58,
            ),
            "test_backend2": BackendConfig(
                name="test_backend2",
                path_prefix="/web",
                servers=[
                    ServerConfig(name="server2", host="127.0.0.1", port=backend_port2)
                ],
                timeout_http_keep_alive_s=58,
            ),
        }

        api = HAProxyApi(
            cfg=config,
            backend_configs=backend_configs,
            config_file_path=config_file_path,
        )

        haproxy_api_cleanup(api)

        try:
            # Start HAProxy
            await api.start()

            # Wait for HAProxy to be ready
            wait_for_condition(
                lambda: check_haproxy_ready(config.stats_port),
                timeout=10,
                retry_interval_ms=500,
            )

            # Make some API calls to generate sessions and traffic
            request_threads = []

            for i in range(3):
                thread = threading.Thread(
                    target=make_test_request,
                    args=[f"http://127.0.0.1:{config.frontend_port}/api/slow"],
                )
                thread.start()
                request_threads.append(thread)

            for i in range(3):
                thread = threading.Thread(
                    target=make_test_request,
                    args=[f"http://127.0.0.1:{config.frontend_port}/web/slow"],
                )
                thread.start()
                request_threads.append(thread)

            # Wait for HAProxy socket to be available for stats commands
            def socket_available():
                return os.path.exists(socket_path)

            wait_for_condition(socket_available, timeout=10, retry_interval_ms=500)

            # Get actual stats
            async def two_servers_up():
                stats = await api.get_haproxy_stats()
                return stats.active_servers == 2

            await async_wait_for_condition(
                two_servers_up, timeout=10, retry_interval_ms=200
            )

            all_stats = await api.get_all_stats()
            haproxy_stats = await api.get_haproxy_stats()

            # Assert against the expected stub with exact values
            assert (
                len(all_stats) == 2
            ), f"Should have exactly 2 backends, got {len(all_stats)}"
            assert (
                haproxy_stats.total_backends == 2
            ), f"Should have exactly 2 backends, got {haproxy_stats.total_backends}"
            assert (
                haproxy_stats.total_servers == 2
            ), f"Should have exactly 2 servers, got {haproxy_stats.total_servers}"
            assert (
                haproxy_stats.active_servers == 2
            ), f"Should have exactly 2 active servers, got {haproxy_stats.active_servers}"

            # Wait for request threads to complete
            for thread in request_threads:
                thread.join(timeout=1)
        finally:
            # Stop HAProxy
            await api.stop()

            # Cleanup backend servers
            try:
                backend_server1.should_exit = True
                backend_server2.should_exit = True
                backend_thread1.join(timeout=5)  # Wait for the thread to finish
                backend_thread2.join(timeout=5)  # Wait for the thread to finish
            except Exception as e:
                print(f"Error cleaning up backend servers: {e}")


@pytest.mark.asyncio
async def test_update_and_reload(haproxy_api_cleanup):
    with tempfile.TemporaryDirectory() as temp_dir:
        config_file_path = os.path.join(temp_dir, "haproxy.cfg")
        socket_path = os.path.join(temp_dir, "admin.sock")

        backend = BackendConfig(
            name="backend",
            path_prefix="/",
            servers=[
                ServerConfig(name="server", host="127.0.0.1", port=find_free_port())
            ],
        )

        config = HAProxyConfig(
            frontend_port=find_free_port(),
            frontend_host="127.0.0.1",
            stats_port=find_free_port(),
            socket_path=socket_path,
            inject_process_id_header=True,
        )

        api = HAProxyApi(
            cfg=config,
            backend_configs={backend.name: backend},
            config_file_path=config_file_path,
        )

        await api.start()
        haproxy_api_cleanup(api)

        original_proc = api.proc
        original_pid = original_proc.pid

        # Add another backend
        backend2 = BackendConfig(
            name="backend_2",
            path_prefix="/",
            servers=[
                ServerConfig(name="server", host="127.0.0.1", port=find_free_port())
            ],
        )

        await api.update_and_reload({backend.name: backend, backend2.name: backend2})

        assert api.proc is not None
        assert api.proc.pid != original_pid

        # The original process should eventually exit once the reload completes.
        await asyncio.sleep(0.5)
        assert original_proc.returncode is not None


@pytest.mark.asyncio
async def test_toggle_health_checks(haproxy_api_cleanup):
    """Test that disable()/enable() toggle HAProxy health checks end-to-end."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config_file_path = os.path.join(temp_dir, "haproxy.cfg")
        socket_path = os.path.join(temp_dir, "admin.sock")

        backend = BackendConfig(
            name="backend",
            path_prefix="/",
            servers=[
                ServerConfig(name="server", host="127.0.0.1", port=find_free_port())
            ],
        )

        config = HAProxyConfig(
            frontend_port=find_free_port(),
            frontend_host="127.0.0.1",
            stats_port=find_free_port(),
            socket_path=socket_path,
            inject_process_id_header=True,
        )

        api = HAProxyApi(
            cfg=config,
            backend_configs={backend.name: backend},
            config_file_path=config_file_path,
        )

        await api.start()
        haproxy_api_cleanup(api)

        # Verify HAProxy is running
        assert api._is_running(), "HAProxy should be running"

        # Test health check endpoint works initially
        health_response = requests.get(
            f"http://127.0.0.1:{config.frontend_port}{config.health_check_endpoint}",
            timeout=5,
        )
        assert (
            health_response.status_code == 200
        ), "Health check should return 200 initially"
        assert (
            b"OK" in health_response.content
        ), "Health check should return 'OK' initially"

        # Verify a config file contains health check enabled
        with open(api.config_file_path, "r") as f:
            config_content = f.read()
            assert (
                "http-request return status 200" in config_content
            ), "Health checks should be enabled in config"

        # Disable health checks
        await api.disable()

        # Verify HAProxy is still running after calling disable()
        assert api._is_running(), "HAProxy should still be running after disable"

        # Config should now deny the health endpoint
        with open(api.config_file_path, "r") as f:
            config_content = f.read()
            assert (
                "http-request return status 503" in config_content
            ), "Health checks should be disabled in config"

        def health_check_condition(status_code: int):
            # Test health check endpoint now fails
            health_response = requests.get(
                f"http://127.0.0.1:{config.frontend_port}{config.health_check_endpoint}",
                timeout=5,
            )

            return health_response.status_code == status_code

        wait_for_condition(health_check_condition, timeout=1, status_code=503)

        # Re-enable health checks
        await api.enable()

        # Config should contain the 200 response again
        with open(api.config_file_path, "r") as f:
            config_content = f.read()
            assert (
                "http-request return status 200" in config_content
            ), "Health checks should be re-enabled in config"

        wait_for_condition(health_check_condition, timeout=1, status_code=200)


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", "-s", __file__]))
