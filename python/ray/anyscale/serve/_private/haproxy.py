import asyncio
import csv
import io
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
HAPROXY_CONFIG_FILE_LOC = "/etc/haproxy/haproxy.cfg"


@dataclass
class ServerStats:
    """Server statistics from HAProxy."""

    backend: str  # Which backend this server belongs to
    server: str  # Server name within the backend
    status: str  # Current status: "UP", "DOWN", "DRAIN", etc.
    current_sessions: int  # Active sessions (HAProxy 'scur')
    queued: int  # Queued requests (HAProxy 'qcur')

    @property
    def is_up(self) -> bool:
        return self.status == "UP"

    @property
    def is_draining(self) -> bool:
        return self.status in ["DRAIN", "NOLB"]

    @property
    def can_drain_safely(self) -> bool:
        """
        Return True if the server can be drained safely based on the current load.
        Safe to drain when:
        - No current active sessions (0)
        - No queued requests waiting
        This ensures no active user sessions are disrupted during draining.
        """
        return self.current_sessions == 0 and self.queued == 0


@dataclass
class ServerConfig:
    """Configuration for a single server."""

    name: str  # Server identifier for HAProxy config
    host: str  # IP/hostname to connect to
    port: int  # Port to connect to


@dataclass
class HAProxyStats:
    """Complete HAProxy statistics with both individual server data and aggregate metrics."""

    # Individual server statistics by backend and server name
    backend_to_servers: Dict[str, Dict[str, ServerStats]] = field(default_factory=dict)

    # Computed aggregate metrics (calculated from server data)
    @property
    def total_backends(self) -> int:
        """Total number of backends."""
        return len(self.backend_to_servers)

    @property
    def total_servers(self) -> int:
        """Total number of servers across all backends."""
        return sum(
            len(backend_servers) for backend_servers in self.backend_to_servers.values()
        )

    @property
    def active_servers(self) -> int:
        """Number of servers currently UP."""
        return sum(
            1
            for backend_servers in self.backend_to_servers.values()
            for server in backend_servers.values()
            if server.is_up
        )

    @property
    def draining_servers(self) -> int:
        """Number of servers currently draining."""
        return sum(
            1
            for backend_servers in self.backend_to_servers.values()
            for server in backend_servers.values()
            if server.is_draining
        )

    @property
    def total_active_sessions(self) -> int:
        """Sum of all active sessions across all servers."""
        return sum(
            server.current_sessions
            for backend_servers in self.backend_to_servers.values()
            for server in backend_servers.values()
        )

    @property
    def total_queued_requests(self) -> int:
        """Sum of all queued requests across all servers."""
        return sum(
            server.queued
            for backend_servers in self.backend_to_servers.values()
            for server in backend_servers.values()
        )

    @property
    def is_system_idle(self) -> bool:
        """Return True if the entire system has no active load."""
        return self.total_active_sessions == 0 and self.total_queued_requests == 0

    @property
    def draining_progress_pct(self) -> float:
        """Return percentage of servers currently draining (0.0 to 100.0)."""
        if self.total_servers == 0:
            return 0.0
        return (self.draining_servers / self.total_servers) * 100.0


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
    # Testing/debugging options
    inject_process_id_header: bool = False
    reload_id: Optional[str] = None  # Unique ID for each reload

    pass_health_checks: bool = True
    health_check_endpoint: str = "/-/healthz"
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
    # This has to be the same or greater than the client side keep-alive timeout.
    timeout_http_keep_alive_s: Optional[int] = None

    # Control the inactivity timeout for established WebSocket connections.
    # Without this setting, a WebSocket connection could be prematurely terminated by other,
    # more general timeout settings like timeout client or timeout server,
    # which are intended for the initial phases of a connection.
    timeout_tunnel_s: Optional[int] = None

    # The number of consecutive failed health checks that must occur before a service instance is marked as unhealthy
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
    async def start(self) -> None:
        """Initializes proxy configuration files."""
        pass

    @abstractmethod
    async def get_all_stats(self) -> Dict[str, Dict[str, ServerStats]]:
        """Get statistics for all servers in all backends."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the proxy."""
        pass

    @abstractmethod
    async def disable(self) -> None:
        """Disables the proxy from receiving any HTTP requests"""
        pass

    @abstractmethod
    async def enable(self) -> None:
        """Enables the proxy from receiving any HTTP requests"""
        pass

    @abstractmethod
    async def update_and_reload(
        self, backend_configs: Dict[str, BackendConfig]
    ) -> None:
        """Gracefully reload the service."""
        pass


class HAProxyApi(ProxyApi):
    """ProxyApi implementation for HAProxy."""

    def __init__(
        self,
        cfg: HAProxyConfig = None,
        backend_configs: Dict[str, BackendConfig] = None,
        http_options: Optional[HTTPOptions] = None,
        config_file_path: str = HAPROXY_CONFIG_FILE_LOC,
    ):
        self.cfg = HAProxyConfig() if cfg is None else cfg
        self.backend_configs = backend_configs or {}
        self.http_options = http_options
        self.config_file_path = config_file_path
        # Lock to prevent concurrent config modifications
        self._config_lock = asyncio.Lock()
        self.proc = None

    def _is_running(self) -> bool:
        """Check if the HAProxy process is still running."""
        return self.proc is not None and self.proc.returncode is None

    async def _start_and_wait_for_haproxy(
        self, *extra_args: str, timeout_s: int = 5
    ) -> asyncio.subprocess.Process:
        proc = await asyncio.create_subprocess_exec(
            "haproxy",
            "-db",
            "-f",
            self.config_file_path,
            *extra_args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        start_time = time.time()

        # TODO: update this to use health checks
        while time.time() - start_time < timeout_s:
            if proc.returncode is not None:
                stdout = await proc.stdout.read() if proc.stdout else b""
                stderr = await proc.stderr.read() if proc.stderr else b""
                output = (
                    stderr.decode("utf-8", errors="ignore").strip()
                    or stdout.decode("utf-8", errors="ignore").strip()
                )
                raise RuntimeError(
                    f"HAProxy exited during startup: {output or proc.returncode}"
                )

            try:
                if await self.has_stats():
                    return proc
            except Exception:
                pass

        raise RuntimeError(f"HAProxy startup timed out after {timeout_s} seconds")

    async def has_stats(self) -> bool:
        """Check if the HAProxy process is running and has stats."""
        try:
            stats_output = await self._send_socket_command("show stat")
            return len(stats_output) > 0
        except RuntimeError as e:
            logger.error(f"HAProxy exited during startup: {e}")
            return False

    async def _graceful_reload(self) -> None:
        """Perform a graceful reload of HAProxy using the `-sf` flag."""
        try:
            if not self._is_running():
                logger.info("HAProxy not running, cannot perform graceful reload")
                return

            old_proc = self.proc
            if old_proc is None or old_proc.returncode is not None:
                logger.warning(
                    "Existing HAProxy process already exited; skipping reload"
                )
                return

            if self.cfg.inject_process_id_header:
                self.cfg.reload_id = f"reload-{int(time.time() * 1000)}"
                await self._generate_config_file_internal()

            self.proc = await self._start_and_wait_for_haproxy("-sf", str(old_proc.pid))

            logger.info("Successfully performed graceful HAProxy reload")
        except Exception as e:
            logger.error(f"HAProxy graceful reload failed: {e}")
            raise

    async def _generate_config_file_internal(self) -> None:
        """Internal config generation without locking (for use within locked sections)."""
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

            with open(self.config_file_path, "w") as f:
                f.write(config_content)

            logger.debug(f"Generated HAProxy configuration: {self.config_file_path}")
        except Exception as e:
            logger.error(f"Failed to create HAProxy configuration files: {e}")
            raise

    async def start(self) -> None:
        """
        Generate HAProxy configuration files and start the HAProxy server process.

        This method creates the necessary configuration files and launches the HAProxy
        process in foreground mode, ensuring that the proxy is running with the latest
        configuration and that the parent retains control of the subprocess handle.
        """
        try:
            async with self._config_lock:
                # Set initial reload ID if header injection is enabled and ID is not set
                if self.cfg.inject_process_id_header and self.cfg.reload_id is None:
                    self.cfg.reload_id = f"initial-{int(time.time() * 1000)}"

                await self._generate_config_file_internal()
            logger.debug("Generated HAProxy config file")

            self.proc = await self._start_and_wait_for_haproxy()
            logger.debug("HAProxy started successfully")
        except Exception as e:
            logger.error(f"Failed to initialize and start HAProxy configuration: {e}")
            raise

    async def get_all_stats(self) -> Dict[str, Dict[str, ServerStats]]:
        """Get statistics for all servers in all backends (implements abstract method)."""
        try:
            stats_output = await self._send_socket_command("show stat")
            return self._parse_haproxy_stats(stats_output)
        except Exception as e:
            logger.error(f"Failed to get HAProxy stats: {e}")
            return {}

    async def get_haproxy_stats(self) -> HAProxyStats:
        """Get complete HAProxy statistics including both individual and aggregate data."""
        server_stats = await self.get_all_stats()
        return HAProxyStats(backend_to_servers=server_stats)

    # TODO: use socket library instead of subprocess
    async def _send_socket_command(self, command: str) -> str:
        """Send a command to the HAProxy stats socket via subprocess."""
        try:
            # Check if a socket file exists
            if not os.path.exists(self.cfg.socket_path):
                logger.warning(
                    f"HAProxy socket file does not exist: {self.cfg.socket_path}"
                )
                return ""

            proc = await asyncio.create_subprocess_exec(
                "socat",
                "-",
                f"UNIX-CONNECT:{self.cfg.socket_path}",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(f"{command}\n".encode("utf-8")), timeout=5.0
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                raise RuntimeError(
                    f"Timeout while sending command '{command}' to HAProxy socket"
                )

            if proc.returncode != 0:
                err = stderr.decode("utf-8", errors="ignore").strip()
                raise RuntimeError(
                    f"Command '{command}' failed with code {proc.returncode}: {err}"
                )

            result = stdout.decode("utf-8", errors="ignore")
            logger.debug(f"Socket command '{command}' returned {len(result)} chars")
            return result
        except Exception as e:
            raise RuntimeError(f"Failed to send socket command '{command}': {e}")

    @staticmethod
    def _parse_haproxy_stats(stats_output: str) -> Dict[str, Dict[str, ServerStats]]:
        """Parse HAProxy stats CSV output into structured data."""
        if not stats_output.strip():
            return {}

        # HAProxy stats start with '#' comment - replace with nothing for CSV parsing
        csv_data = stats_output.replace("# ", "", 1)

        def safe_int(v):
            return int(v) if v and v.strip() else 0

        backend_stats = {}

        for row in csv.DictReader(io.StringIO(csv_data)):
            # Skip non-server entries
            if row.get("svname") in ["FRONTEND", "BACKEND"]:
                continue

            # Direct dictionary-to-dataclass mapping
            server = ServerStats(
                backend=row["pxname"],
                server=row["svname"],
                status=row["status"],
                current_sessions=safe_int(row["scur"]),
                queued=safe_int(row["qcur"]),
            )

            backend_stats.setdefault(server.backend, {})[server.server] = server

        return backend_stats

    async def stop(self) -> None:
        proc = self.proc
        if proc is None:
            logger.info("HAProxy process not running, skipping shutdown")
            return

        try:
            if proc.returncode is None:
                proc.kill()
                await proc.wait()
                self.proc = None

            logger.info("Stopped HAProxy process")
        except RuntimeError as e:
            logger.error(f"Error during HAProxy shutdown: {e}")

    async def update_and_reload(
        self, backend_configs: Dict[str, BackendConfig]
    ) -> None:
        try:
            self.backend_configs = backend_configs
            await self._graceful_reload()
        except Exception as e:
            raise RuntimeError(f"Failed to update and reload HAProxy: {e}")

    async def disable(self) -> None:
        """Force health checks to fail by denying all requests at the frontend."""
        try:
            # Disable health checks (set to fail)
            self.cfg.pass_health_checks = False

            # Regenerate the config file with the deny rule
            await self._generate_config_file_internal()

            # Perform a graceful reload to apply changes
            await self._graceful_reload()
            logger.info("Successfully disabled health checks")
        except Exception as e:
            logger.error(f"Failed to disable health checks: {e}")
            raise

    async def enable(self) -> None:
        """Force health checks to fail by denying all requests at the frontend."""
        try:
            self.cfg.pass_health_checks = True

            await self._generate_config_file_internal()
            # Perform a graceful reload to apply changes
            await self._graceful_reload()
            logger.info("Successfully enabled health checks")
        except Exception as e:
            logger.error(f"Failed to disable health checks: {e}")
            raise


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
