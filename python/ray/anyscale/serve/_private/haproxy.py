import json
import logging
import time
from typing import Optional, Any, Set

import ray
from ray.serve._private.common import (
    NodeId,
    ReplicaID,
    RequestMetadata,
)
from ray.serve._private.constants import (
    PROXY_MIN_DRAINING_PERIOD_S,
    SERVE_LOGGER_NAME,
)
from ray.serve._private.logging_utils import get_component_logger_file_path
from ray.serve._private.long_poll import LongPollClient
from ray.serve._private.proxy import ProxyActorInterface
from ray.serve.config import HTTPOptions, gRPCOptions
from ray.serve.schema import LoggingConfig

logger = logging.getLogger(SERVE_LOGGER_NAME)


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

        # The time when the node starts to drain.
        # The node is not draining if it's None.
        self._draining_start_time: Optional[float] = None

        # TODO: create async task to start haproxy

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

    def _dump_ingress_replicas_for_testing(self, route: str) -> Set[ReplicaID]:
        return set()
