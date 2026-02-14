import logging
from typing import Optional

from ray.anyscale.serve._private.constants import (
    ANYSCALE_FREEZE_GC_ON_STARTUP,
)
from ray.serve._private.constants import (
    RAY_SERVE_ENABLE_DIRECT_INGRESS,
    RAY_SERVE_LOG_TO_STDERR,
    RAY_SERVE_REQUEST_PATH_LOG_BUFFER_SIZE,
    RAY_SERVE_RUN_ROUTER_IN_SEPARATE_LOOP,
    RAY_SERVE_RUN_USER_CODE_IN_SEPARATE_THREAD,
    RAY_SERVE_THROUGHPUT_OPTIMIZED,
    RAY_SERVE_USE_GRPC_BY_DEFAULT,
    SERVE_LOGGER_NAME,
)
from ray.serve._private.controller import ServeController
from ray.serve.config import HTTPOptions, gRPCOptions
from ray.serve.schema import LoggingConfig

logger = logging.getLogger(SERVE_LOGGER_NAME)


class AnyscaleServeController(ServeController):
    """Anyscale-specific ServeController that handles direct ingress functionality.
    This controller extends the base ServeController to support direct ingress,
    where each replica listens directly on its own ports rather than going through
    a proxy. This is useful for Kubernetes deployments where we want each replica
    to be directly accessible via the ingress controller.
    """

    async def __init__(
        self,
        *,
        http_options: HTTPOptions,
        global_logging_config: LoggingConfig,
        grpc_options: Optional[gRPCOptions] = None,
    ):
        # Set the feature flags for throughput optimized Ray Serve.
        if RAY_SERVE_THROUGHPUT_OPTIMIZED:
            self._log_throughput_opt_message()

        await super().__init__(
            http_options=http_options,
            global_logging_config=global_logging_config,
            grpc_options=grpc_options,
        )

    def _log_throughput_opt_message(self) -> None:
        msg = "Throughput optimized Ray Serve enabled with the following configurations:\n"
        if RAY_SERVE_ENABLE_DIRECT_INGRESS:
            msg += "  • Direct ingress enabled\n"
        if RAY_SERVE_USE_GRPC_BY_DEFAULT:
            msg += "  • gRPC communication enabled\n"
        if not RAY_SERVE_RUN_USER_CODE_IN_SEPARATE_THREAD:
            msg += "  • User code running in main thread (not separate)\n"
        if not RAY_SERVE_RUN_ROUTER_IN_SEPARATE_LOOP:
            msg += "  • Router running in main thread (not separate)\n"
        if not RAY_SERVE_LOG_TO_STDERR:
            msg += "  • Log to stderr disabled\n"
        if ANYSCALE_FREEZE_GC_ON_STARTUP:
            msg += "  • Garbage collector is frozen on startup\n"
        msg += f"  • Request path log buffer size: {RAY_SERVE_REQUEST_PATH_LOG_BUFFER_SIZE}\n"
        logger.info(msg)
