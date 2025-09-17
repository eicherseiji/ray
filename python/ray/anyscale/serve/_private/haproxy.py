import json
import logging
from typing import Optional, Any, Set

import ray
from ray.serve._private.common import (
    NodeId,
    ReplicaID,
    RequestMetadata,
)
from ray.serve._private.constants import SERVE_LOGGER_NAME
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

    async def ready(self) -> str:
        # Return proxy metadata used by the controller.
        # NOTE(zcin): We need to convert the metadata to a json string because
        # of cross-language scenarios. Java can't deserialize a Python tuple.
        return json.dumps(
            [
                ray.get_runtime_context().get_worker_id(),
                get_component_logger_file_path(),
            ]
        )

    async def update_draining(
        self, draining: bool, _after: Optional[Any] = None
    ) -> None:
        pass

    async def is_drained(self, _after: Optional[Any] = None) -> bool:
        return True

    async def check_health(self) -> None:
        logger.debug("Received health check.", extra={"log_to_stderr": False})

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
