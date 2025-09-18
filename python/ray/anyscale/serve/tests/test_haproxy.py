import pytest
import sys

import ray
from ray import serve
from ray._common.test_utils import wait_for_condition
from ray.actor import ActorHandle
from ray.anyscale.serve._private.constants import ANYSCALE_RAY_SERVE_ENABLE_HA_PROXY
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


@pytest.fixture
def _skip_if_ff_not_enabled():
    if not ANYSCALE_RAY_SERVE_ENABLE_HA_PROXY:
        pytest.skip(
            reason="ANYSCALE_RAY_SERVE_ENABLE_HA_PROXY not set.",
        )


@pytest.fixture
def shutdown_ray():
    if ray.is_initialized():
        ray.shutdown()
    yield
    if ray.is_initialized():
        ray.shutdown()


def test_deploy_with_no_applications(_skip_if_ff_not_enabled, ray_shutdown):
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


def test_single_app_shutdown_actors(_skip_if_ff_not_enabled, ray_shutdown):
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
async def test_single_app_shutdown_actors_async(_skip_if_ff_not_enabled, ray_shutdown):
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

    def test_default_keep_alive_timeout_s(self, _skip_if_ff_not_enabled, ray_shutdown):
        """Test when no keep_alive_timeout_s is set.

        When the keep_alive_timeout_s is not set, the uvicorn keep alive is 5.
        """
        serve.start()
        proxy_actor = self.get_proxy_actor()
        assert (
            ray.get(proxy_actor._get_http_options.remote()).keep_alive_timeout_s
            == DEFAULT_UVICORN_KEEP_ALIVE_TIMEOUT_S
        )

    def test_set_keep_alive_timeout_in_http_configs(
        self, _skip_if_ff_not_enabled, ray_shutdown
    ):
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
    def test_set_keep_alive_timeout_in_env(
        self, _skip_if_ff_not_enabled, ray_instance, ray_shutdown
    ):
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
        self, _skip_if_ff_not_enabled, ray_instance, ray_shutdown
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
    monkeypatch, _skip_if_ff_not_enabled, shutdown_ray, call_ray_stop_only  # noqa: F811
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


def test_haproxy_failure(_skip_if_ff_not_enabled, ray_shutdown):
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

    wait_for_condition(check_new_proxy, timeout=30)
    serve.shutdown()


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", "-s", __file__]))
