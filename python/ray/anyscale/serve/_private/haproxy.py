import asyncio
import json
import logging
import os
import time

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from jinja2 import Environment
from typing import Any, Dict, List, Optional, Set

from ray.anyscale.serve._private.haproxy_templates import HAPROXY_CONFIG_TEMPLATE
from ray.anyscale.serve._private.constants import (
    ANYSCALE_RAY_SERVE_HAPROXY_MAXCONN,
    ANYSCALE_RAY_SERVE_HAPROXY_NBTHREAD,
    ANYSCALE_RAY_SERVE_HAPROXY_SOCKET_PATH,
)
import ray
from ray._common.utils import get_or_create_event_loop
from ray.serve._private.common import (
    NodeId,
    ReplicaID,
    RequestMetadata,
)
from ray.serve._private.constants import (
    PROXY_MIN_DRAINING_PERIOD_S,
    SERVE_CONTROLLER_NAME,
    SERVE_LOGGER_NAME,
    SERVE_NAMESPACE,
)
from ray.serve._private.logging_utils import get_component_logger_file_path
from ray.serve._private.long_poll import LongPollClient, LongPollNamespace
from ray.serve._private.proxy import ProxyActorInterface
from ray.serve.config import HTTPOptions, gRPCOptions
from ray.serve.schema import (
    LoggingConfig,
    TargetGroup,
)

logger = logging.getLogger(SERVE_LOGGER_NAME)


HAPROXY_MAP_ENTRY_FILE = "/etc/haproxy/routes.map"
HAPROXY_CONFIG_FILE_LOC = "/etc/haproxy/haproxy.cfg"


@dataclass
class ServerConfig:
    """Configuration for a single server."""

    name: str
    host: str
    port: int


@dataclass
class HAProxyConfig:
    """Configuration for HAProxy."""

    socket_path: str = ANYSCALE_RAY_SERVE_HAPROXY_SOCKET_PATH
    maxconn: int = ANYSCALE_RAY_SERVE_HAPROXY_MAXCONN
    nbthread: int = ANYSCALE_RAY_SERVE_HAPROXY_NBTHREAD
    stats_port: int = 8404
    stats_uri: str = "/stats"
    # All timeout values are in seconds
    timeout_queue_s: Optional[int] = None
    timeout_connect_s: Optional[int] = None
    timeout_client_s: Optional[int] = None
    timeout_server_s: Optional[int] = None
    timeout_http_request_s: Optional[int] = None
    timeout_http_keep_alive_s: Optional[int] = None
    custom_global: Dict[str, str] = field(default_factory=dict)
    custom_defaults: Dict[str, str] = field(default_factory=dict)

    # Configurable frontend parameters
    frontend_port: int = 80
    frontend_host: str = "*"

    # Global health check parameters (used as defaults for backends)
    # Number of consecutive failed health checks that must occur before a service instance is marked as unhealthy
    health_check_fall: Optional[int] = 2

    # Number of consecutive successful health checks required to mark an unhealthy service instance as healthy again
    health_check_rise: Optional[int] = 2

    # Interval, or the amount of time, between each health check attempt
    health_check_inter: Optional[str] = "1s"
    health_check_path: Optional[str] = None  # For HTTP health checks


@dataclass
class BackendConfig:
    """Configuration for a single application backend."""

    # Name of the target group.
    name: str

    # Path prefix for the target group. This will be used to route requests to the target group.
    path_prefix: str

    # Maximum time HAProxy will wait for a successful TCP connection to be established with the backend server.
    timeout_connect_s: Optional[int] = None

    # Maximum time that the backend server can be inactive while sending data back to HAProxy.
    # This is also active during the initial response phase.
    timeout_server_s: Optional[int] = None

    # Maximum time that the client can be inactive while sending data to HAProxy.
    # This is active during the initial request phase.
    timeout_client_s: Optional[int] = None
    timeout_http_request_s: Optional[int] = None

    # Maximum time HAProxy will wait for a request in the queue.
    timeout_queue_s: Optional[int] = None

    # Maximum time HAProxy will keep the connection alive.
    # This has to be same or greater than the client side keep-alive timeout.
    timeout_http_keep_alive_s: Optional[int] = None

    # Control the inactivity timeout for established WebSocket connections.
    # Without this setting, a WebSocket connection could be prematurely terminated by other,
    # more general timeout settings like timeout client or timeout server,
    # which are intended for the initial phases of a connection.
    timeout_tunnel_s: Optional[int] = None

    # Number of consecutive failed health checks that must occur before a service instance is marked as unhealthy
    health_check_fall: Optional[int] = None

    # Number of consecutive successful health checks required to mark an unhealthy service instance as healthy again
    health_check_rise: Optional[int] = None

    # Interval, or the amount of time, between each health check attempt
    health_check_inter: Optional[str] = None

    # Endpoint path that the health check mechanism will send a request to. It's typically an HTTP path.
    health_check_path: Optional[str] = None

    # List of servers in this backend
    servers: List[ServerConfig] = field(default_factory=list)


@dataclass
class ServerStats:
    """Server statistics from HAProxy."""

    backend: str
    server: str
    status: str
    current_sessions: int = 0
    queued: int = 0

    @property
    def is_up(self) -> bool:
        return self.status == "UP"

    @property
    def is_draining(self) -> bool:
        return self.status in ["DRAIN", "NOLB"]


class ProxyApi(ABC):
    """Generic interface for load balancer management operations."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initializes proxy configuration files."""
        pass

    @abstractmethod
    async def add_servers(self, backend: str, servers: List[ServerConfig]) -> None:
        """Add new servers to the backend."""
        pass

    @abstractmethod
    async def remove_servers(self, backend: str, servers: List[ServerConfig]) -> None:
        """Remove servers from a backend."""
        pass

    @abstractmethod
    async def get_all_stats(self) -> Dict[str, Dict[str, ServerStats]]:
        """Get statistics for all servers in all backends."""
        pass

    @abstractmethod
    async def get_server_stats(self, backend: str, server: str) -> ServerStats:
        """Get statistics for a specific server."""
        pass

    @abstractmethod
    async def start_proxy(self) -> None:
        """Start the proxy."""
        pass

    @abstractmethod
    async def stop_proxy(self) -> None:
        """Stop the proxy."""
        pass

    @abstractmethod
    async def add_backend(self, backend_config: BackendConfig) -> None:
        """Add a new target group or a backend configuration and reload the service."""
        pass

    @abstractmethod
    async def remove_backend(self, backend_name: str) -> None:
        """Remove a target group or a backend configuration and reload the service."""
        pass


class HAProxyApi(ProxyApi):
    """ProxyApi implementation for HAProxy."""

    def __init__(
        self,
        socket_path: str = ANYSCALE_RAY_SERVE_HAPROXY_SOCKET_PATH,
        cfg: HAProxyConfig = None,
        backend_configs: Dict[str, BackendConfig] = None,
        http_options: Optional[HTTPOptions] = None,
    ):
        self.cfg = HAProxyConfig() if cfg is None else cfg
        self.backend_configs = backend_configs or {}
        self.socket_path = socket_path
        self.http_options = http_options

    # TODO: (ok-scale) Add implementation for socket connection for add/remove servers
    #   instead of using subprocess. In event of rapid up-scaling/down-scaling,
    #   subprocess, specially in case of large stdout's can lag.
    @staticmethod
    async def _run_subprocess(
        cmd: List[str], input_data: Optional[bytes] = None, timeout: float = 10.0
    ) -> str:
        """
        Run a subprocess command and return its output.

        Args:
            cmd: List of command arguments.
            input_data: Optional input data to pass to the subprocess.
            timeout: Timeout in seconds.

        Returns:
            Output string of the subprocess.
        """
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE if input_data else None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=input_data), timeout=timeout
            )

            if proc.returncode != 0:
                error_msg = stderr.decode("utf-8").strip()
                raise Exception(f"Command failed: {error_msg}")

            return stdout.decode("utf-8").strip()
        except asyncio.TimeoutError:
            raise Exception(f"Command timed out after {timeout}s")
        except Exception as e:
            logger.error(f"Subprocess execution failed: {e}")
            raise

    async def initialize(self) -> None:
        try:
            env = Environment()
            template = env.from_string(HAPROXY_CONFIG_TEMPLATE)

            if self.http_options:
                self.cfg.frontend_port = self.http_options.port

                # Convert host format: HTTPOptions uses "0.0.0.0" while HAProxy uses "*"
                if self.http_options.host == "0.0.0.0":
                    self.cfg.frontend_host = "*"
                elif self.http_options.host:
                    self.cfg.frontend_host = self.http_options.host

            # target groups = backend configs in haproxy lingo
            config_content = template.render(
                {"config": self.cfg, "backends": list(self.backend_configs.values())}
            )

            # Ensure the config ends with a newline
            if not config_content.endswith("\n"):
                config_content += "\n"

            with open(HAPROXY_CONFIG_FILE_LOC, "w") as f:
                f.write(config_content)

            logger.debug(f"Generated HAProxy configuration: {HAPROXY_CONFIG_FILE_LOC}")

            # Ensure routes map file exists
            if not os.path.exists(HAPROXY_MAP_ENTRY_FILE):
                await self._run_subprocess(["sudo", "touch", HAPROXY_MAP_ENTRY_FILE])
                logger.debug(f"Created empty routes map file: {HAPROXY_MAP_ENTRY_FILE}")
        except Exception as e:
            logger.error(f"Failed to create HAProxy configuration files: {e}")
            raise

    async def add_servers(self, backend: str, servers: List[ServerConfig]) -> None:
        raise NotImplementedError()

    async def remove_servers(self, backend: str, servers: List[ServerConfig]) -> None:
        raise NotImplementedError()

    async def get_all_stats(self) -> Dict[str, Dict[str, ServerStats]]:
        raise NotImplementedError()

    async def get_server_stats(self, backend: str, server: str) -> ServerStats:
        raise NotImplementedError()

    async def start_proxy(self) -> None:
        raise NotImplementedError()

    async def stop_proxy(self) -> None:
        raise NotImplementedError()

    async def add_backend(self, backend_config: BackendConfig) -> None:
        raise NotImplementedError()

    async def remove_backend(self, backend_name: str) -> None:
        raise NotImplementedError()


@ray.remote(num_cpus=0)
class HAProxyManager(ProxyActorInterface):
    def __init__(
        self,
        http_options: HTTPOptions,
        grpc_options: gRPCOptions,
        *,
        node_id: NodeId,
        node_ip_address: str,
        logging_config: LoggingConfig,
        long_poll_client: Optional[LongPollClient] = None,
    ):  # noqa: F821
        super().__init__(
            node_id=node_id,
            node_ip_address=node_ip_address,
            logging_config=logging_config,
        )

        self._grpc_options = grpc_options
        self._http_options = http_options

        self._controller_handle = ray.get_actor(
            SERVE_CONTROLLER_NAME, namespace=SERVE_NAMESPACE
        )
        # The time when the node starts to drain.
        # The node is not draining if it's None.
        self._draining_start_time: Optional[float] = None

        event_loop = get_or_create_event_loop()

        # TODO: create async task to start haproxy

        self._target_groups: List[TargetGroup] = []

        self.long_poll_client = LongPollClient(
            self._controller_handle,
            {
                LongPollNamespace.TARGET_GROUPS: self.update_target_groups,
            },
            call_in_event_loop=event_loop,
        )

    async def ready(self) -> str:
        # TODO: wait for haproxy task to finish and health check to pass

        # Return proxy metadata used by the controller.
        # NOTE(zcin): We need to convert the metadata to a json string because
        # of cross-language scenarios. Java can't deserialize a Python tuple.
        return json.dumps(
            [
                ray.get_runtime_context().get_worker_id(),
                get_component_logger_file_path(),
            ]
        )

    def _is_draining(self) -> bool:
        """Whether is haproxy is in the draining status or not."""
        return self._draining_start_time is not None

    async def _fail_health_check(self) -> None:
        """Fail the health check."""
        # TODO: tell haproxy to fail the health check
        pass

    async def _pass_health_check(self) -> None:
        """Pass the health check."""
        # TODO: tell haproxy to pass the health check
        pass

    async def _has_ongoing_requests(self) -> bool:
        """Check whether the haproxy has ongoing requests or not."""
        # TODO: check whether the haproxy has ongoing requests
        return False

    async def update_draining(
        self, draining: bool, _after: Optional[Any] = None
    ) -> None:
        """Update the draining status of the proxy.

        This is called by the proxy state manager
        to drain or un-drain the haproxy.
        """

        if draining and (not self._is_draining()):
            logger.info(
                f"Start to drain the HAProxy on node {self._node_id}.",
                extra={"log_to_stderr": False},
            )
            await self._fail_health_check()
            self._draining_start_time = time.time()
        if (not draining) and self._is_draining():
            logger.info(
                f"Stop draining the HAProxy on node {self._node_id}.",
                extra={"log_to_stderr": False},
            )
            await self._pass_health_check()
            self._draining_start_time = None

    async def is_drained(self, _after: Optional[Any] = None) -> bool:
        """Check whether the haproxy is drained or not.

        An haproxy is drained if it has no ongoing requests
        AND it has been draining for more than
        `PROXY_MIN_DRAINING_PERIOD_S` seconds.
        """
        if not self._is_draining():
            return False

        return (not self._has_ongoing_requests()) and (
            (time.time() - self._draining_start_time) > PROXY_MIN_DRAINING_PERIOD_S
        )

    async def check_health(self) -> None:
        logger.debug("Received health check.", extra={"log_to_stderr": False})
        # TODO: implement haproxy health check

    def pong(self) -> str:
        pass

    async def receive_asgi_messages(self, request_metadata: RequestMetadata) -> bytes:
        raise NotImplementedError("Receive is handled by the ingress replicas.")

    def _get_http_options(self) -> HTTPOptions:
        return self._http_options

    def _get_logging_config(self) -> Optional[str]:
        """Get the logging configuration (for testing purposes)."""
        log_file_path = None
        for handler in logger.handlers:
            if isinstance(handler, logging.handlers.MemoryHandler):
                log_file_path = handler.target.baseFilename
        return log_file_path

    def update_target_groups(self, target_groups: List[TargetGroup]) -> None:
        self._target_groups = target_groups

    def get_target_groups(self) -> List[TargetGroup]:
        """Get current target groups."""
        return self._target_groups

    def _dump_ingress_replicas_for_testing(self, route: str) -> Set[ReplicaID]:
        return set()
