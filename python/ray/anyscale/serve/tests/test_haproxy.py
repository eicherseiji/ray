import asyncio
import httpx
import logging
import pytest
import sys

import ray
from ray import serve
from ray._common.test_utils import (
    SignalActor,
    wait_for_condition,
)
from ray.actor import ActorHandle
from ray.anyscale.serve._private.constants import (
    ANYSCALE_RAY_SERVE_ENABLE_HA_PROXY,
)
from ray.anyscale.serve._private.haproxy import HAProxyManager
from ray.cluster_utils import Cluster
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

logger = logging.getLogger(__name__)

# Skip all tests in this module if the HAProxy feature flag is not enabled
pytestmark = pytest.mark.skipif(
    not ANYSCALE_RAY_SERVE_ENABLE_HA_PROXY,
    reason="ANYSCALE_RAY_SERVE_ENABLE_HA_PROXY not set.",
)


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


@pytest.mark.asyncio
async def test_drain_and_undrain_haproxy_manager(
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

    signal_actor = SignalActor.remote()

    @serve.deployment
    class HelloModel:
        async def __call__(self):
            await signal_actor.wait.remote()
            return "hello"

    serve.run(HelloModel.options(num_replicas=2).bind())

    # 3 proxies, 1 controller, 2 replicas, 1 signal actor
    wait_for_condition(lambda: len(list_actors()) == 7)
    assert len(ray.nodes()) == 3

    client = _get_global_client()
    serve_details = ServeInstanceDetails(
        **ray.get(client._controller.get_serve_instance_details.remote())
    )
    proxy_actor_ids = {proxy.actor_id for _, proxy in serve_details.proxies.items()}

    assert len(proxy_actor_ids) == 3

    httpx.get("http://localhost:8000/")

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

    # should stay in draining status until the signal is sent
    await asyncio.sleep(1)
    assert check_proxy_status(
        proxy_status_to_count={ProxyStatus.HEALTHY: 2, ProxyStatus.DRAINING: 1}
    )

    serve.run(HelloModel.options(num_replicas=2).bind())
    # The proxy should return to healthy status
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
    await signal_actor.send.remote()
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


def test_haproxy_get_target_groups(shutdown_ray):
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


@pytest.mark.asyncio
async def test_haproxy_update_target_groups(ray_shutdown):
    """Test that the haproxy correctly updates the target groups."""
    ray.init(num_cpus=4)
    serve.start(http_options={"host": "0.0.0.0"})

    @serve.deployment
    def function(_):
        return "hello1"

    serve.run(
        function.options(num_replicas=1).bind(), name="app1", route_prefix="/test"
    )
    wait_for_condition(lambda: httpx.get("http://localhost:8000/test").text == "hello1")
    wait_for_condition(
        lambda: httpx.get("http://localhost:8000/test2").status_code == 404
    )

    serve.run(
        function.options(num_replicas=1).bind(), name="app2", route_prefix="/test2"
    )
    wait_for_condition(lambda: httpx.get("http://localhost:8000/test").text == "hello1")
    wait_for_condition(
        lambda: httpx.get("http://localhost:8000/test2").text == "hello1"
    )

    serve.delete("app1")
    wait_for_condition(
        lambda: httpx.get("http://localhost:8000/test").status_code == 404
    )
    wait_for_condition(
        lambda: httpx.get("http://localhost:8000/test2").text == "hello1"
    )

    serve.run(
        function.options(num_replicas=1).bind(), name="app1", route_prefix="/test"
    )
    wait_for_condition(lambda: httpx.get("http://localhost:8000/test").text == "hello1")
    wait_for_condition(
        lambda: httpx.get("http://localhost:8000/test2").text == "hello1"
    )

    serve.shutdown()


@pytest.mark.asyncio
async def test_haproxy_update_draining_health_checks(ray_shutdown):
    """Test that the haproxy update_draining method updates the HAProxy health checks."""
    ray.init(num_cpus=4)
    serve.start()

    signal_actor = SignalActor.remote()

    @serve.deployment
    async def function(_):
        await signal_actor.wait.remote()
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
    proxy_actor = ray.get_actor(proxy_actor.name, namespace=SERVE_NAMESPACE)

    wait_for_condition(
        lambda: httpx.get("http://localhost:8000/-/healthz").status_code == 200
    )

    await proxy_actor.update_draining.remote(draining=True)
    wait_for_condition(
        lambda: httpx.get("http://localhost:8000/-/healthz").status_code == 503
    )

    await proxy_actor.update_draining.remote(draining=False)
    wait_for_condition(
        lambda: httpx.get("http://localhost:8000/-/healthz").status_code == 200
    )
    assert not await proxy_actor._is_draining.remote()

    serve.shutdown()


def test_haproxy_safe_name():
    """Test that the safe name is generated correctly."""
    assert HAProxyManager.get_safe_name("HTTP:test") == "HTTP:test"
    assert HAProxyManager.get_safe_name("HTTP:test/foo") == "HTTP:test.foo"
    assert HAProxyManager.get_safe_name("replica#abc") == "replica-abc"


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", "-s", __file__]))
