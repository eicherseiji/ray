import asyncio
from ray.serve._private.thirdparty.get_asgi_route_name import get_asgi_route_name
import inspect
import logging
import pickle
import time
import errno
from collections import defaultdict, deque
from contextlib import contextmanager
from functools import wraps
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    List,
    Dict,
    Generator,
    Optional,
    Tuple,
)

import grpc
from ray.serve._private.proxy_request_response import ResponseStatus
from ray.serve.config import HTTPOptions, gRPCOptions
from starlette.types import Receive, Scope, Send

import ray
from ray.anyscale.serve._private.http_util import ASGIDIReceiveProxy
from ray.anyscale.serve._private.constants import (
    ANYSCALE_RAY_SERVE_REPLICA_GRPC_MAX_MESSAGE_LENGTH,
    ANYSCALE_RAY_SERVE_ENABLE_DIRECT_INGRESS,
    RAY_SERVE_DIRECT_INGRESS_PORT_RETRY_COUNT,
    ANYSCALE_RAY_SERVE_DIRECT_INGRESS_MIN_DRAINING_PERIOD_S,
)
from ray.anyscale.serve._private.tracing_utils import (
    TraceContextManager,
    extract_propagated_context,
    is_span_recording,
    is_tracing_enabled,
    set_http_span_attributes,
    set_rpc_span_attributes,
    set_span_attributes,
    set_span_exception,
    setup_tracing,
)
from ray.serve.generated.serve_pb2 import HealthzResponse, ListApplicationsResponse
from ray.serve._private.http_util import (
    convert_object_to_asgi_messages,
    start_asgi_http_server,
    configure_http_options_with_defaults,
    configure_http_middlewares,
)
from ray.serve.context import _get_in_flight_requests
from ray.anyscale.serve.utils import asyncio_grpc_exception_handler
from ray.serve._private.common import (
    RequestProtocol,
    ReplicaQueueLengthInfo,
    RequestMetadata,
    ServeComponentType,
)
from ray.serve._private.constants import (
    GRPC_CONTEXT_ARG_NAME,
    HEALTHY_MESSAGE,
    REQUEST_LATENCY_BUCKETS_MS,
    SERVE_LOGGER_NAME,
    SERVE_CONTROLLER_NAME,
    SERVE_HTTP_REQUEST_ID_HEADER,
    SERVE_HTTP_REQUEST_TIMEOUT_S_HEADER,
    SERVE_HTTP_REQUEST_DISCONNECT_DISABLED_HEADER,
    SERVE_NAMESPACE,
)
from ray.anyscale.serve._private.replica_response_generator import (
    ReplicaResponseGenerator,
)
from ray.serve._private.replica import (
    ReplicaBase,
    ReplicaMetricsManager,
    StatusCodeCallback,
)
from ray.serve._private.utils import generate_request_id, is_grpc_enabled
from ray.serve.generated import serve_proprietary_pb2, serve_proprietary_pb2_grpc
from ray.serve.grpc_util import (
    RayServegRPCContext,
)
from ray.serve._private.grpc_util import (
    get_grpc_response_status,
    set_grpc_code_and_details,
    start_grpc_server,
)
from ray.util import metrics
from ray.anyscale.serve._private.serialization import RPCSerializer


logger = logging.getLogger(SERVE_LOGGER_NAME)


async def send_http_response(message, status_code, send):
    for msg in convert_object_to_asgi_messages(
        message,
        status_code=status_code,
    ):
        await send(msg)


def _wrap_grpc_call(f):
    """Decorator that processes grpc methods."""

    def serialize(result, metadata):
        if metadata.is_streaming and metadata.is_http_request:
            return result
        else:
            # Use cached serializer to avoid per-request instantiation overhead
            serializer = RPCSerializer.get_cached_serializer(
                metadata.request_serialization,
                metadata.response_serialization,
            )
            return serializer.dumps_response(result)

    @wraps(f)
    async def wrapper(
        self,
        request: serve_proprietary_pb2.ASGIRequest,
        context: grpc.aio.ServicerContext,
    ):
        request_metadata = pickle.loads(request.pickled_request_metadata)

        # Get cached serializer with options from metadata
        serializer = RPCSerializer.get_cached_serializer(
            request_metadata.request_serialization,
            request_metadata.response_serialization,
        )

        request_args = serializer.loads_request(request.request_args)
        request_kwargs = serializer.loads_request(request.request_kwargs)

        if request_metadata.is_http_request or request_metadata.is_grpc_request:
            request_args = (pickle.loads(request_args[0]),)

        try:
            result = await f(
                self, context, request_metadata, *request_args, **request_kwargs
            )
            return serve_proprietary_pb2.ASGIResponse(
                serialized_message=serialize(result, request_metadata)
            )
        except (Exception, asyncio.CancelledError) as e:
            return serve_proprietary_pb2.ASGIResponse(
                serialized_message=serializer.dumps_response(e),
                is_error=True,
            )

    @wraps(f)
    async def gen_wrapper(
        self,
        request: serve_proprietary_pb2.ASGIRequest,
        context: grpc.aio.ServicerContext,
    ):
        request_metadata = pickle.loads(request.pickled_request_metadata)

        # Get cached serializer with options from metadata
        serializer = RPCSerializer.get_cached_serializer(
            request_metadata.request_serialization,
            request_metadata.response_serialization,
        )

        request_args = serializer.loads_request(request.request_args)
        request_kwargs = serializer.loads_request(request.request_kwargs)

        if request_metadata.is_http_request or request_metadata.is_grpc_request:
            request_args = (pickle.loads(request_args[0]),)

        try:
            async for result in f(
                self, context, request_metadata, *request_args, **request_kwargs
            ):
                yield serve_proprietary_pb2.ASGIResponse(
                    serialized_message=serialize(result, request_metadata)
                )
        except (Exception, asyncio.CancelledError) as e:
            yield serve_proprietary_pb2.ASGIResponse(
                serialized_message=serializer.dumps_response(e),
                is_error=True,
            )

    if inspect.isasyncgenfunction(f):
        return gen_wrapper
    else:
        return wrapper


class AnyscaleReplicaMetricsManager(ReplicaMetricsManager):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self._is_direct_ingress:
            # TODO(alexyang): De-duplicate these metrics from the those collected by
            # the proxy. https://anyscale1.atlassian.net/browse/SERVE-871
            self.ingress_http_request_counter = self._init_ingress_request_counter(
                "HTTP"
            )

            self.ingress_http_request_error_counter = (
                self._init_ingress_request_error_counter("HTTP")
            )

            self.deployment_http_request_error_counter = (
                self._init_deployment_request_error_counter("HTTP")
            )

            # log REQUEST_LATENCY_BUCKET_MS
            logger.debug(f"REQUEST_LATENCY_BUCKET_MS: {REQUEST_LATENCY_BUCKETS_MS}")
            self.ingress_http_processing_latency_tracker = (
                self._init_ingress_processing_latency_tracker("HTTP")
            )

            node_id = ray.get_runtime_context().get_node_id()
            node_ip_address = ray.util.get_node_ip_address()
            self.ingress_num_ongoing_http_requests_gauge = (
                self._init_ingress_num_ongoing_requests_gauge(
                    "HTTP", node_id, node_ip_address
                )
            )
            self._ingress_ongoing_http_requests = 0

            if self._cached_metrics_enabled:
                # Cache metrics in the following format: protocol -> tags -> value
                self._cached_ingress_request_counter = defaultdict(
                    lambda: defaultdict(int)
                )
                self._cached_ingress_request_error_counter = defaultdict(
                    lambda: defaultdict(int)
                )
                self._cached_deployment_request_error_counter = defaultdict(
                    lambda: defaultdict(int)
                )
                self._cached_ingress_processing_latencies = defaultdict(
                    lambda: defaultdict(deque)
                )

    @property
    def _is_direct_ingress(self) -> bool:
        return self._ingress and ANYSCALE_RAY_SERVE_ENABLE_DIRECT_INGRESS

    def _init_ingress_request_counter(self, protocol: str):
        return metrics.Counter(
            f"serve_num_{protocol.lower()}_requests",
            description=f"The number of {protocol} requests processed.",
            tag_keys=("route", "method", "application", "status_code"),
        )

    def _init_ingress_request_error_counter(self, protocol: str):
        return metrics.Counter(
            f"serve_num_{protocol.lower()}_error_requests",
            description=(f"The number of errored {protocol} responses."),
            tag_keys=(
                "route",
                "error_code",
                "method",
                "application",
            ),
        )

    def _init_deployment_request_error_counter(self, protocol: str):
        return metrics.Counter(
            f"serve_num_deployment_{protocol.lower()}_error_requests",
            description=(
                f"The number of errored {protocol} responses returned by each deployment."
            ),
            tag_keys=(
                "deployment",
                "error_code",
                "method",
                "route",
                "application",
            ),
        )

    def _init_ingress_processing_latency_tracker(self, protocol: str):
        return metrics.Histogram(
            f"serve_{protocol.lower()}_request_latency_ms",
            description=(
                f"The end-to-end latency of {protocol} requests "
                f"(measured from the Serve ingress)."
            ),
            boundaries=REQUEST_LATENCY_BUCKETS_MS,
            tag_keys=(
                "method",
                "route",
                "application",
                "status_code",
            ),
        )

    def _init_ingress_num_ongoing_requests_gauge(
        self, protocol: str, node_id: str, node_ip_address: str
    ):
        return metrics.Gauge(
            name=f"serve_num_ongoing_{protocol.lower()}_requests",
            description=f"The number of ongoing requests in this {protocol} ingress.",
            tag_keys=("node_id", "node_ip_address"),
        ).set_default_tags(
            {
                "node_id": node_id,
                "node_ip_address": node_ip_address,
            }
        )

    def should_collect_metrics(self) -> bool:
        """
        For direct ingress deployments, metrics must be collected from replicas regardless
        of whether autoscaling metrics are being collected via handles. This is necessary
        because direct ingress traffic bypasses deployment handles and goes directly to
        the replicas.
        """
        return (
            self._is_direct_ingress and self._autoscaling_config
        ) or super().should_collect_metrics()

    def record_ingress_request_metrics(
        self,
        *,
        protocol: RequestProtocol,
        method: str,
        route: str,
        app_name: str,
        deployment_name: str,
        latency_ms: float,
        was_error: bool,
        status_code: str,
    ):
        """Record per-request metrics."""
        if not self._is_direct_ingress:
            return

        if protocol == RequestProtocol.HTTP:
            latency_tracker = self.ingress_http_processing_latency_tracker
            request_error_counter = self.ingress_http_request_error_counter
            deployment_error_counter = self.deployment_http_request_error_counter
            request_counter = self.ingress_http_request_counter
        else:
            # TODO(alexyang): Add metrics for gRPC.
            # https://anyscale1.atlassian.net/browse/SERVE-872
            return

        request_tags = {
            "route": route,
            "method": method,
            "application": app_name,
            "status_code": status_code,
        }
        latency_tags = request_tags
        request_error_tags = {
            "route": route,
            "method": method,
            "application": app_name,
            "error_code": status_code,
        }
        deployment_error_tags = {
            "route": route,
            "method": method,
            "application": app_name,
            "error_code": status_code,
            "deployment": deployment_name,
        }

        if self._cached_metrics_enabled:
            self._cached_ingress_request_counter[protocol][
                frozenset(request_tags.items())
            ] += 1
            self._cached_ingress_processing_latencies[protocol][
                frozenset(latency_tags.items())
            ].append(latency_ms)
            if was_error:
                self._cached_ingress_request_error_counter[protocol][
                    frozenset(request_error_tags.items())
                ] += 1
                self._cached_deployment_request_error_counter[protocol][
                    frozenset(deployment_error_tags.items())
                ] += 1
        else:
            request_counter.inc(tags=request_tags)
            latency_tracker.observe(latency_ms, tags=latency_tags)
            if was_error:
                request_error_counter.inc(tags=request_error_tags)
                deployment_error_counter.inc(tags=deployment_error_tags)

    def inc_num_ongoing_requests(self, request_metadata: RequestMetadata) -> int:
        self._num_ongoing_requests += 1

        if self._is_direct_ingress and request_metadata.is_direct_ingress:
            self._ingress_ongoing_http_requests += 1

        if not self._cached_metrics_enabled:
            self._num_ongoing_requests_gauge.set(self._num_ongoing_requests)

            if self._is_direct_ingress and request_metadata.is_direct_ingress:
                if request_metadata.is_http_request:
                    self.ingress_num_ongoing_http_requests_gauge.set(
                        self._ingress_ongoing_http_requests
                    )

    def dec_num_ongoing_requests(self, request_metadata: RequestMetadata) -> int:
        self._num_ongoing_requests -= 1

        if self._is_direct_ingress and request_metadata.is_direct_ingress:
            self._ingress_ongoing_http_requests -= 1

        if not self._cached_metrics_enabled:
            self._num_ongoing_requests_gauge.set(self._num_ongoing_requests)

            if self._is_direct_ingress and request_metadata.is_direct_ingress:
                if request_metadata.is_http_request:
                    self.ingress_num_ongoing_http_requests_gauge.set(
                        self._ingress_ongoing_http_requests
                    )

    def _report_cached_metrics(self):
        super()._report_cached_metrics()

        if not self._is_direct_ingress:
            return

        for protocol in [RequestProtocol.HTTP]:
            if protocol == RequestProtocol.HTTP:
                ingress_request_counter = self.ingress_http_request_counter
                ingress_request_error_counter = self.ingress_http_request_error_counter
                deployment_request_error_counter = (
                    self.deployment_http_request_error_counter
                )
                ingress_processing_latencies = (
                    self.ingress_http_processing_latency_tracker
                )
                self.ingress_num_ongoing_http_requests_gauge.set(
                    self._ingress_ongoing_http_requests
                )
            else:
                # TODO(alexyang): Add metrics for gRPC.
                continue

            for request_tags, count in self._cached_ingress_request_counter[
                protocol
            ].items():
                ingress_request_counter.inc(count, tags=dict(request_tags))

            for request_tags, count in self._cached_ingress_request_error_counter[
                protocol
            ].items():
                ingress_request_error_counter.inc(count, tags=dict(request_tags))

            for request_tags, count in self._cached_deployment_request_error_counter[
                protocol
            ].items():
                deployment_request_error_counter.inc(count, tags=dict(request_tags))

            for latency_tags, latencies in self._cached_ingress_processing_latencies[
                protocol
            ].items():
                for latency_ms in latencies:
                    ingress_processing_latencies.observe(
                        latency_ms, tags=dict(latency_tags)
                    )

        self._cached_ingress_request_counter.clear()
        self._cached_ingress_request_error_counter.clear()
        self._cached_deployment_request_error_counter.clear()
        self._cached_ingress_processing_latencies.clear()


class AnyscaleReplica(ReplicaBase):
    def __init__(self, **kwargs):
        self._server = grpc.aio.server(
            options=[
                (
                    "grpc.max_receive_message_length",
                    ANYSCALE_RAY_SERVE_REPLICA_GRPC_MAX_MESSAGE_LENGTH,
                )
            ]
        )

        self._direct_ingress_http_server_task: Optional[asyncio.Task] = None
        self._direct_ingress_grpc_server_task: Optional[asyncio.Task] = None

        super().__init__(**kwargs)

        # Silence spammy false positive errors from gRPC Python
        self._event_loop.set_exception_handler(asyncio_grpc_exception_handler)

        # Set up tracing
        try:
            is_tracing_setup_successful = setup_tracing(
                component_type=ServeComponentType.REPLICA,
                component_name=self._component_name,
                component_id=self._component_id,
            )
            if is_tracing_setup_successful:
                logger.info("Successfully set up tracing for replica")
        except Exception as e:
            logger.warning(
                f"Failed to set up tracing: {e}. "
                "The replica will continue running, but traces will not be exported."
            )

        self._controller_handle = ray.get_actor(
            SERVE_CONTROLLER_NAME, namespace=SERVE_NAMESPACE
        )

        # get node ID
        self._node_id = ray.get_runtime_context().get_node_id()
        self._http_options: Optional[HTTPOptions] = None
        self._grpc_options: Optional[gRPCOptions] = None

        self._num_queued_requests = 0

    @property
    def max_queued_requests(self) -> int:
        return self._deployment_config.max_queued_requests

    async def _maybe_start_direct_ingress_servers(self):
        if not ANYSCALE_RAY_SERVE_ENABLE_DIRECT_INGRESS:
            return

        if not self._ingress:
            return

        async def allocate_and_start_server(start_server_fn, protocol):
            """Attempt to allocate a port and start the server with retries."""
            is_port_in_use = False
            for _ in range(RAY_SERVE_DIRECT_INGRESS_PORT_RETRY_COUNT):
                port = await self._controller_handle.allocate_replica_port.remote(
                    self._node_id, self._replica_id.unique_id, protocol
                )
                logger.info(f"Allocated port {port} for {protocol}")

                try:
                    server_task = await start_server_fn(port)
                    logger.info(
                        f"Successfully started {protocol} server on port {port}"
                    )
                    return port, server_task
                except RuntimeError as e:
                    logger.warning(
                        f"Failed to start {protocol} server on port {port}: {e}. Retrying..."
                    )

                    # `start_asgi_http_server` raises a RuntimeError with the original OSError as the cause.
                    if isinstance(e.__cause__, OSError) and e.__cause__.errno in (
                        errno.EADDRINUSE,
                        errno.EADDRNOTAVAIL,
                    ):
                        is_port_in_use = True
                    else:
                        is_port_in_use = False

                    # setting block_port to True because we are concluding that the port is
                    # in use by another service on the same node. Blocking port here is a small
                    # optimization to avoid trying to start the server on a the same port
                    # multiple times by other replicas.
                    await self._controller_handle.release_replica_port.remote(
                        self._node_id,
                        self._replica_id.unique_id,
                        port,
                        protocol,
                        block_port=True,
                    )

            err_msg = f"Failed to allocate and start {protocol} server after retries"
            if is_port_in_use:
                err_msg = f"""
                Failed to start {protocol} server: port already in use. Suggestion: Ensure that the Ray Serve direct ingress port ranges do not overlap with the Ray worker port range (min_worker_port to max_worker_port).
                """

            raise RuntimeError(err_msg)

        # Fetch configs
        self._http_options, self._grpc_options = ray.get(
            [
                self._controller_handle.get_http_config.remote(),
                self._controller_handle.get_grpc_config.remote(),
            ]
        )

        grpc_enabled = is_grpc_enabled(self._grpc_options)

        # Allocate and start HTTP server
        async def start_http_server(port):
            options = configure_http_middlewares(
                configure_http_options_with_defaults(
                    HTTPOptions(**{**self._http_options.dict(), "port": port})
                )
            )

            return await start_asgi_http_server(
                self._direct_ingress_asgi,
                options,
                event_loop=self._event_loop,
                enable_so_reuseport=False,
            )

        (
            self._http_port,
            self._direct_ingress_http_server_task,
        ) = await allocate_and_start_server(
            start_server_fn=start_http_server,
            protocol=RequestProtocol.HTTP,
        )

        # Allocate and start gRPC server if enabled
        if grpc_enabled:

            async def start_grpc_server_fn(port):
                options = gRPCOptions(**{**self._grpc_options.dict(), "port": port})
                return await start_grpc_server(
                    self._direct_ingress_service_handler_factory,
                    options,
                    event_loop=self._event_loop,
                    enable_so_reuseport=False,
                )

            (
                self._grpc_port,
                self._direct_ingress_grpc_server_task,
            ) = await allocate_and_start_server(
                start_server_fn=start_grpc_server_fn,
                protocol=RequestProtocol.GRPC,
            )

        logger.info(
            f"Started HTTP server on port {self._http_port}"
            + (f" and gRPC server on port {self._grpc_port}" if grpc_enabled else "")
        )

    async def _on_initialized(self):
        serve_proprietary_pb2_grpc.add_ASGIServiceServicer_to_server(self, self._server)
        self._port = self._server.add_insecure_port("[::]:0")
        await self._server.start()

        await self._maybe_start_direct_ingress_servers()

        self._set_internal_replica_context(
            servable_object=self._user_callable_wrapper.user_callable
        )

        # Save the initialization latency if the replica is initializing
        # for the first time.
        if self._initialization_latency is None:
            self._initialization_latency = time.time() - self._initialization_start_time

    def _on_request_cancelled(
        self, metadata: RequestMetadata, e: asyncio.CancelledError
    ):
        """Recursively cancel child requests.

        This includes all requests that are pending assignment, and gRPC
        requests that have already been assigned.
        """
        # Cancel child requests pending assignment
        requests_pending_assignment = (
            ray.serve.context._get_requests_pending_assignment(
                metadata.internal_request_id
            )
        )
        for task in requests_pending_assignment.values():
            task.cancel()

        # Cancel child requests that have already been assigned.
        # This is for gRPC requests and direct ingress requests.
        in_flight_requests = _get_in_flight_requests(metadata.internal_request_id)
        for replica_result in in_flight_requests.values():
            replica_result.cancel()

    def _on_request_failed(self, request_metadata: RequestMetadata, e: Exception):
        if ray.util.pdb._is_ray_debugger_post_mortem_enabled():
            ray.util.pdb._post_mortem()

    def _can_accept_request(self, request_metadata: RequestMetadata):
        if request_metadata.is_direct_ingress:
            limit = self.max_queued_requests
            if limit != -1 and self._num_queued_requests >= limit:
                return False

            return True
        else:
            return super()._can_accept_request(request_metadata)

    @contextmanager
    def _tracing_context(self, request_metadata: RequestMetadata):
        # TODO (abrar): for http requests on ASGI, request_metadata.call_method is __call__
        #   this is not nice, figure out a better way to map this to method name.
        if not is_tracing_enabled():
            yield
            return

        call_method = request_metadata.call_method
        trace_context = extract_propagated_context(request_metadata.tracing_context)
        trace_manager = TraceContextManager(
            trace_name=f"replica_handle_request {self._deployment_id.name} {call_method}",
            trace_context=trace_context,
        )
        with trace_manager:
            if is_span_recording():
                trace_attributes = {
                    "request_id": request_metadata.request_id,
                    "replica_id": self._replica_id.unique_id,
                    "deployment": self._deployment_id.name,
                    "app": self._deployment_id.app_name,
                    "call_method": request_metadata.call_method,
                    "route": request_metadata.route,
                    "multiplexed_model_id": request_metadata.multiplexed_model_id,
                    "is_streaming": request_metadata.is_streaming,
                }
                set_span_attributes(trace_attributes)
            yield

    @contextmanager
    def _wrap_request(
        self, request_metadata: RequestMetadata
    ) -> Generator[StatusCodeCallback, None, None]:
        """Context manager that wraps user method calls.

        1) Sets the request context var with appropriate metadata.
        2) Records the access log message (if not disabled).
        3) Records per-request metrics via the metrics manager.
        """
        with self._tracing_context(request_metadata):
            ray.serve.context._serve_request_context.set(
                ray.serve.context._RequestContext(
                    route=request_metadata.route,
                    request_id=request_metadata.request_id,
                    _internal_request_id=request_metadata.internal_request_id,
                    app_name=self._deployment_id.app_name,
                    multiplexed_model_id=request_metadata.multiplexed_model_id,
                    grpc_context=request_metadata.grpc_context,
                    cancel_on_parent_request_cancel=self._ingress
                    and ANYSCALE_RAY_SERVE_ENABLE_DIRECT_INGRESS,
                )
            )
            with self._handle_errors_and_metrics(
                request_metadata
            ) as status_code_callback:
                yield status_code_callback

    def _record_errors_and_metrics(
        self,
        user_exception: Optional[BaseException],
        status_code: Optional[str],
        latency_ms: float,
        request_metadata: RequestMetadata,
    ):
        super()._record_errors_and_metrics(
            user_exception, status_code, latency_ms, request_metadata
        )
        if request_metadata.is_direct_ingress and status_code is not None:
            self._metrics_manager.record_ingress_request_metrics(
                protocol=RequestProtocol.HTTP,
                method=request_metadata._http_method,
                route=self._route_prefix,
                app_name=self._deployment_id.app_name,
                deployment_name=self._deployment_id.name,
                latency_ms=latency_ms,
                was_error=status_code.startswith(("4", "5")),
                status_code=status_code,
            )

        if is_span_recording():
            http_route = request_metadata.route
            call_method = request_metadata.call_method
            if request_metadata.is_http_request:
                set_http_span_attributes(
                    method=request_metadata._http_method,
                    status_code=status_code,
                    route=http_route,
                )
            else:
                # in this case we are either in grpc or undefined. I think
                # undefined is the case where we call the user method as ray
                # tasks. Treating it as grpc for now from POV of tracing.
                set_rpc_span_attributes(
                    system=request_metadata._request_protocol,
                    method=call_method,
                    status_code=status_code,
                    service=self._deployment_id.name,
                )
            if user_exception is not None:
                set_span_exception(user_exception, escaped=False)

    @_wrap_grpc_call
    async def HandleRequest(
        self,
        context: grpc.aio.ServicerContext,
        request_metadata: RequestMetadata,
        *request_args,
        **request_kwargs,
    ):
        result = await self.handle_request(
            request_metadata, *request_args, **request_kwargs
        )
        if request_metadata.is_grpc_request:
            result = (request_metadata.grpc_context, result.SerializeToString())

        return result

    @_wrap_grpc_call
    async def HandleRequestStreaming(
        self,
        context: grpc.aio.ServicerContext,
        request_metadata: RequestMetadata,
        *request_args,
        **request_kwargs,
    ):
        async for result in self.handle_request_streaming(
            request_metadata, *request_args, **request_kwargs
        ):
            if request_metadata.is_grpc_request:
                result = (request_metadata.grpc_context, result.SerializeToString())

            yield result

    @_wrap_grpc_call
    async def HandleRequestWithRejection(
        self,
        context: grpc.aio.ServicerContext,
        request_metadata: RequestMetadata,
        *request_args,
        **request_kwargs,
    ):
        """gRPC entrypoint for all unary requests with strict max_ongoing_requests enforcement

        This generator yields a system message indicating if the request was accepted,
        then the actual response.

        If an exception occurred while processing the request, whether it's a user
        exception or an error intentionally raised by Serve, it will be returned as
        a gRPC response instead of raised directly.
        """
        result_gen = self.handle_request_with_rejection(
            request_metadata, *request_args, **request_kwargs
        )
        queue_len_info: ReplicaQueueLengthInfo = await result_gen.__anext__()
        await context.send_initial_metadata(
            [
                ("accepted", str(int(queue_len_info.accepted))),
                ("num_ongoing_requests", str(queue_len_info.num_ongoing_requests)),
            ]
        )
        if not queue_len_info.accepted:
            # NOTE(edoakes): in gRPC, it's not guaranteed that the initial metadata sent
            # by the server will be delivered for a stream with no messages. Therefore,
            # we send a dummy message here to ensure it is populated in every case.
            return b""

        result = await result_gen.__anext__()
        # Consume the result generator to ensure all request operations are completed.
        async for _ in result_gen:
            pass

        if request_metadata.is_grpc_request:
            result = (request_metadata.grpc_context, result.SerializeToString())

        return result

    @_wrap_grpc_call
    async def HandleRequestWithRejectionStreaming(
        self,
        context: grpc.aio.ServicerContext,
        request_metadata: RequestMetadata,
        *request_args,
        **request_kwargs,
    ) -> AsyncGenerator[Any, None]:
        """gRPC entrypoint for all streaming requests with strict max_ongoing_requests enforcement

        This generator yields a system message indicating if the request was accepted,
        then the actual response(s).

        If an exception occurred while processing the request, whether it's a user
        exception or an error intentionally raised by Serve, it will be returned as
        a gRPC response instead of raised directly.
        """
        result_gen = self.handle_request_with_rejection(
            request_metadata, *request_args, **request_kwargs
        )
        queue_len_info: ReplicaQueueLengthInfo = await result_gen.__anext__()
        await context.send_initial_metadata(
            [
                ("accepted", str(int(queue_len_info.accepted))),
                ("num_ongoing_requests", str(queue_len_info.num_ongoing_requests)),
            ]
        )
        if not queue_len_info.accepted:
            # NOTE(edoakes): in gRPC, it's not guaranteed that the initial metadata sent
            # by the server will be delivered for a stream with no messages. Therefore,
            # we send a dummy message here to ensure it is populated in every case.
            yield b""
            return

        async for result in result_gen:
            if request_metadata.is_grpc_request:
                result = (request_metadata.grpc_context, result.SerializeToString())

            yield result

    async def _dataplane_health_check(self) -> Tuple[bool, str]:
        healthy, message = True, HEALTHY_MESSAGE
        if self._shutting_down:
            healthy = False
            message = "DRAINING"
        elif not self._healthy:
            healthy = False
            message = "UNHEALTHY"

        return healthy, message

    def get_grpc_tracing_context(self, context: grpc._cython.cygrpc._ServicerContext):
        """Populate tracing context for gRPC requests.

        This method extracts the "traceparent" metadata from the request headers and
        sets the tracing context from it.
        """
        if not is_tracing_enabled():
            return

        tracing_ctx = {}
        for key, value in context.invocation_metadata():
            if key in ("traceparent", "tracestate"):
                tracing_ctx = tracing_ctx or {}
                tracing_ctx[key] = value

        return tracing_ctx

    async def _direct_ingress_unary_unary(
        self,
        service_method: str,
        request_proto: Any,
        context: grpc._cython.cygrpc._ServicerContext,
    ) -> bytes:
        if service_method == "/ray.serve.RayServeAPIService/Healthz":
            healthy, message = await self._dataplane_health_check()
            context.set_code(
                grpc.StatusCode.OK if healthy else grpc.StatusCode.UNAVAILABLE
            )
            context.set_details(message)
            return HealthzResponse(message=message).SerializeToString()

        if service_method == "/ray.serve.RayServeAPIService/ListApplications":
            # NOTE(edoakes): ListApplications may currently be used by Anyscale for
            # health checking. We should clean this up in the future.
            healthy, message = await self._dataplane_health_check()
            context.set_code(
                grpc.StatusCode.OK if healthy else grpc.StatusCode.UNAVAILABLE
            )
            context.set_details(message)
            # ListApplications returns only the app name the replica is serving.
            application_names = [self._deployment_id.app_name]
            return ListApplicationsResponse(
                application_names=application_names
            ).SerializeToString()

        request_id = generate_request_id()
        c = RayServegRPCContext(context)
        c.set_trailing_metadata([("request_id", request_id)])
        request_metadata = RequestMetadata(
            # TODO: pick up the request ID from gRPC initial metadata.
            request_id=request_id,
            internal_request_id=generate_request_id(),
            call_method=service_method.split("/")[-1],
            _request_protocol=RequestProtocol.GRPC,
            grpc_context=c,
            app_name=self._deployment_id.app_name,
            # TODO(edoakes): populate this.
            multiplexed_model_id="",
            route=self._deployment_id.app_name,
            tracing_context=self.get_grpc_tracing_context(context),
            is_streaming=False,
            is_direct_ingress=True,
        )

        if not self._can_accept_request(request_metadata):
            status = ResponseStatus(
                code=grpc.StatusCode.RESOURCE_EXHAUSTED,
                message="Request dropped due to backpressure",
            )
            set_grpc_code_and_details(context, status)
            return

        method_info = self._user_callable_wrapper.get_user_method_info(
            request_metadata.call_method
        )
        request_args = (request_proto,)
        request_kwargs = (
            {GRPC_CONTEXT_ARG_NAME: request_metadata.grpc_context}
            if method_info.takes_grpc_context_kwarg
            else {}
        )

        async def call_unary():
            yield await self._user_callable_wrapper.call_user_method(
                request_metadata, request_args, request_kwargs
            )

        with self._wrap_request(request_metadata):
            self._num_queued_requests += 1
            async with self._start_request(request_metadata):
                self._num_queued_requests -= 1

                # Use the generic disconnect detecting wrapper
                result_gen = call_unary()
                replica_response_generator = ReplicaResponseGenerator(
                    result_gen,
                    timeout_s=self._grpc_options.request_timeout_s,
                )
                try:
                    result = await replica_response_generator.__anext__()
                    c._set_on_grpc_context(context)
                    status = ResponseStatus(code=grpc.StatusCode.OK)

                    # NOTE(edoakes): we need to fully consume the generator otherwise the
                    # finalizers that run after the `yield` statement won't run. There might
                    # be a cleaner way to structure this.
                    try:
                        await replica_response_generator.__anext__()
                    except StopAsyncIteration:
                        pass
                except BaseException as e:
                    status = get_grpc_response_status(
                        e,
                        self._grpc_options.request_timeout_s,
                        request_metadata.request_id,
                    )
                    set_rpc_span_attributes(
                        system=RequestProtocol.GRPC,
                        method=request_metadata.call_method,
                        status_code=status.code.name,
                        service=self._deployment_id.name,
                    )
                    set_span_exception(e, escaped=True)
                    return
                finally:
                    set_grpc_code_and_details(context, status)

                return result.SerializeToString()

    async def _direct_ingress_unary_stream(
        self,
        service_method: str,
        request: Any,
        context: grpc._cython.cygrpc._ServicerContext,
    ):
        raise NotImplementedError("unary_stream not implemented.")

    def _direct_ingress_service_handler_factory(
        self, service_method: str, stream: bool
    ) -> Callable:
        if stream:

            async def handler(*args, **kwargs):
                return await self._direct_ingress_unary_stream(
                    service_method, *args, **kwargs
                )

        else:

            async def handler(*args, **kwargs):
                return await self._direct_ingress_unary_unary(
                    service_method, *args, **kwargs
                )

        return handler

    def get_asgi_tracing_context(self, headers: List[Tuple[bytes, bytes]]):
        """Extract tracing context from ASGI request headers.

        This method extracts both "traceparent" and "tracestate" headers from the
        request headers to maintain proper trace context propagation.
        """
        if not is_tracing_enabled():
            return None

        tracing_ctx = None
        for key, value in headers:
            key_str = key.decode()
            if key_str in ("traceparent", "tracestate"):
                tracing_ctx = tracing_ctx or {}
                tracing_ctx[key_str] = value.decode()

        return tracing_ctx

    def _determine_http_route(self, scope: Scope) -> str:
        # Default to route prefix for consistency with non-DI mode
        route = self._route_prefix
        if self._user_callable_asgi_app is not None:
            try:
                matched_route = get_asgi_route_name(self._user_callable_asgi_app, scope)
                if matched_route is not None:
                    route = matched_route
            except Exception:
                # If route matching fails, keep the route prefix
                pass

        return route

    def _parse_request_timeout(self, headers: Dict[str, str]) -> Optional[float]:
        """Gets the desired request timeout from the headers.
        If the header is missing or invalid, returns the default request timeout
        from HttpOptions. If the header is non-positive, timeout is disabled.
        """
        header_name = SERVE_HTTP_REQUEST_TIMEOUT_S_HEADER.encode("utf-8")
        if header_name not in headers:
            return self._http_options.request_timeout_s

        value = headers.get(header_name).decode("utf-8")
        try:
            timeout = float(value)
            if timeout > 0:
                return timeout
            return None
        except ValueError:
            return self._http_options.request_timeout_s

    async def _direct_ingress_asgi(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ):
        # NOTE(edoakes): it's important to only start the replica server after the
        # constructor runs because we are using SO_REUSEPORT. We don't want a new
        # replica to start handling connections until it's ready to serve traffic.
        #
        # This can be loosened to listen on the port but fail health checks once we no
        # longer rely on SO_REUSEPORT.
        assert (
            self._user_callable_initialized
        ), "Replica server should only be started *after* the replica is initialized."

        if self._route_prefix and self._route_prefix != "/":
            scope["root_path"] = self._route_prefix

        start_time = time.time()
        method = scope.get("method", "WS").upper()
        route = scope.get("path", "")

        # Handle health check or routes request.
        if route in ["/-/healthz", "/-/routes"]:
            healthy, message = await self._dataplane_health_check()
            status_code = 200 if healthy else 503
            if route == "/-/routes" and healthy:
                # routes endpoint returns only the route prefix andapp name the replica is serving.
                message = {
                    self._route_prefix: self._deployment_id.app_name,
                }
            for msg in convert_object_to_asgi_messages(
                message,
                status_code=status_code,
            ):
                await send(msg)

            latency_ms = (time.time() - start_time) * 1000.0
            self._metrics_manager.record_ingress_request_metrics(
                protocol=RequestProtocol.HTTP,
                method=method,
                route=route,
                app_name=self._deployment_id.app_name,
                deployment_name=self._deployment_id.name,
                latency_ms=latency_ms,
                was_error=not healthy,
                status_code=str(status_code),
            )
            return

        # If the HTTP path does not match the deployment route prefix,
        # it is invalid and we should not serve it.
        if not route.startswith(self._route_prefix):
            for msg in convert_object_to_asgi_messages(
                f"Path '{route}' not found. "
                "Ping http://.../-/routes for available routes.",
                status_code=404,
            ):
                await send(msg)
            return

        headers = dict(scope["headers"])
        request_id = (
            headers.get(SERVE_HTTP_REQUEST_ID_HEADER.encode("utf-8")).decode("utf-8")
            or generate_request_id()
        )
        request_disconnect_disabled = (
            headers.get(
                SERVE_HTTP_REQUEST_DISCONNECT_DISABLED_HEADER.encode("utf-8"), b"?0"
            ).decode("utf-8")
        ) == "?1"
        request_timeout_s = self._parse_request_timeout(headers)

        request_metadata = RequestMetadata(
            request_id=request_id,
            internal_request_id=generate_request_id(),
            call_method="__call__",
            route=self._determine_http_route(scope),
            app_name=self._deployment_id.app_name,
            # TODO(edoakes): populate the multiplexed model ID.
            multiplexed_model_id="",
            is_streaming=True,
            _request_protocol=RequestProtocol.HTTP,
            tracing_context=self.get_asgi_tracing_context(scope["headers"]),
            _http_method=scope.get("method", "WS"),
            is_direct_ingress=True,
        )

        if not self._can_accept_request(request_metadata):
            for msg in convert_object_to_asgi_messages(
                "Request dropped due to backpressure",
                status_code=503,
            ):
                await send(msg)
            return

        # Optimization: we can avoid creating an async receive task if the client
        # has disabled handling disconnects for this request.
        if request_disconnect_disabled:
            receive_proxy = receive
            receive_task = None
        else:
            receive_proxy = ASGIDIReceiveProxy(
                scope, receive, self._user_callable_wrapper.event_loop
            )
            receive_task = receive_proxy.fetch_until_disconnect_task()

        response_started = False
        response_finished = False
        first_message_peeked = False

        with self._wrap_request(request_metadata) as status_code_callback:
            self._num_queued_requests += 1

            async def send_user_message(msg: Dict):
                nonlocal response_started
                nonlocal response_finished
                nonlocal first_message_peeked

                if not first_message_peeked:
                    first_message_peeked = True
                    if msg["type"] == "http.response.start":
                        status_code_callback(str(msg["status"]))

                await send(msg)
                response_started = True
                if msg.get("more_body") is False:
                    response_finished = True

            async def call_asgi():
                async with self._start_request(request_metadata):
                    self._num_queued_requests -= 1

                    if (
                        not self._user_callable_wrapper._run_user_code_in_separate_thread
                    ):
                        user_method_info = (
                            self._user_callable_wrapper.get_user_method_info(
                                request_metadata.call_method
                            )
                        )
                        # `_call_http_entrypoint` will have already called
                        # `send_user_message`, so the ASGI messages will have
                        # already been sent back to the client.
                        await self._user_callable_wrapper._call_http_entrypoint(
                            user_method_info, scope, receive_proxy, send_user_message
                        )
                    else:
                        async for asgi_messages in self._user_callable_wrapper.call_http_entrypoint(
                            request_metadata, status_code_callback, scope, receive_proxy
                        ):
                            for message in asgi_messages:
                                await send_user_message(message)

            # Optimization: if Serve doesn't need to handle disconnects and
            # timeouts for this request, we can avoid event loop overhead by
            # directly awaiting the user code.
            if receive_task is None and request_timeout_s is None:
                return await call_asgi()

            # Otherwise, we'd always need the call_asgi() task.
            request_task = asyncio.create_task(call_asgi())
            tasks = [request_task]
            if receive_task is not None:
                tasks.append(receive_task)

            done, _ = await asyncio.wait(
                tasks,
                timeout=request_timeout_s,
                return_when=asyncio.FIRST_COMPLETED,
            )

            # NOTE(zcin): it's possible that the request task has finished sending
            # all ASGI messages, but the task is suspended and before it can fully
            # complete, the client has sent a disconnect message after the request
            # is completed. That is why we check for `response_finished` here.
            if request_task in done or response_finished:
                if receive_task is not None:
                    receive_task.cancel()
                await request_task
            elif receive_task in done:
                request_task.cancel()
                status_code_callback("499")
                if not response_started:
                    msg = (
                        f"Client for request {request_id} disconnected, "
                        "cancelling request."
                    )
                    await send_http_response(msg, 499, send)
                raise asyncio.CancelledError
            else:
                request_task.cancel()
                status_code_callback("408")
                if not response_started:
                    msg = (
                        f"Request {request_id} timed out after "
                        f"{self._http_options.request_timeout_s}s."
                    )
                    await send_http_response(msg, 408, send)
                raise asyncio.CancelledError

    async def perform_graceful_shutdown(self):
        if not ANYSCALE_RAY_SERVE_ENABLE_DIRECT_INGRESS or not self._ingress:
            # if direct ingress is not enabled or the replica is not an ingress replica,
            # we can just call the super method to perform the graceful shutdown.
            await super().perform_graceful_shutdown()
            return

        # set the shutting down flag to True to signal ALBs with failing health checks
        # to stop sending traffic to this replica.
        self._shutting_down = True

        # If the replica was never initialized it never served traffic, so we
        # can skip the wait period.
        if self._user_callable_initialized:
            # in order to gracefully shutdown the replica, we need to wait for the
            # requests to drain and for PROXY_MIN_DRAINING_PERIOD_S to pass.
            # this is necessary because we want to give ALB time to update its
            # target group to remove the replica from it and to mark this replica
            # as unhealthy.
            # TODO(abrar): the code below assumes that once ALB marks a replica target
            # as unhealthy, it will not send traffic to it. This is not true because
            # ALB can send traffic to a replica if all targets are unhealthy.
            # The correct way to handle is this we start the cooldown period since
            # the last request finished and wait for the cooldown period to pass.
            await asyncio.gather(
                asyncio.sleep(ANYSCALE_RAY_SERVE_DIRECT_INGRESS_MIN_DRAINING_PERIOD_S),
                self._drain_ongoing_requests(),
            )
            logger.info(
                f"Replica {self._replica_id} successfully drained ongoing requests."
            )

        await self.shutdown()
        if self._direct_ingress_http_server_task:
            self._direct_ingress_http_server_task.cancel()
        if self._direct_ingress_grpc_server_task:
            self._direct_ingress_grpc_server_task.cancel()
