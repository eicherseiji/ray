import asyncio
import concurrent.futures
import inspect
import logging
from asyncio import run_coroutine_threadsafe
from functools import wraps
from typing import Any, AsyncIterator, Callable, Coroutine, Iterator, Optional, Union

import grpc

import ray
from ray.anyscale.serve._private.serialization import RPCSerializer
from ray.exceptions import ActorUnavailableError, RayTaskError
from ray.serve._private.common import (
    OBJ_REF_NOT_SUPPORTED_ERROR,
    ReplicaQueueLengthInfo,
    RequestMetadata,
)
from ray.serve._private.constants import SERVE_LOGGER_NAME
from ray.serve._private.http_util import MessageQueue
from ray.serve._private.replica_result import ReplicaResult
from ray.serve._private.utils import generate_request_id
from ray.serve.exceptions import RequestCancelledError
from ray.serve.generated import serve_proprietary_pb2

logger = logging.getLogger(SERVE_LOGGER_NAME)


def is_running_in_asyncio_loop() -> bool:
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


class gRPCReplicaResult(ReplicaResult):
    def __init__(
        self,
        call: grpc.aio.Call,
        metadata: RequestMetadata,
        actor_id: ray.ActorID,
        loop: asyncio.AbstractEventLoop = None,
        *,
        with_rejection: bool = False,
    ):
        self._call: grpc.aio.Call = call
        self._actor_id: ray.ActorID = actor_id
        self._metadata: RequestMetadata = metadata  # Store metadata for serialization
        self._result_queue: MessageQueue = MessageQueue()
        # This is the asyncio event loop that the gRPC Call object is attached to
        self._grpc_call_loop = loop or asyncio._get_running_loop()
        self._is_streaming = metadata.is_streaming
        self._with_rejection = with_rejection
        self._rejection_response = None

        self._gen = None
        self._fut = None

        # NOTE(zcin): for now, these two concepts will be synonymous.
        # In other words, using a queue means the router is running on
        # a separate thread/event loop, and vice versa not using a queue
        # means the router is running on the main event loop, where the
        # DeploymentHandle lives.
        self._calling_from_same_loop = not metadata._on_separate_loop
        if hasattr(self._call, "__aiter__"):
            self._gen = self._call.__aiter__()
            # If the grpc call IS streaming, AND it was created on a
            # a separate loop, then use a queue to fetch the objects
            self._use_queue = metadata._on_separate_loop
        else:
            self._use_queue = False

        # Start a background task that continuously fetches from the
        # streaming grpc call. This way callbacks will actually be
        # called when the request finishes even without the user
        # explicitly consuming the response.
        self._consume_task = None
        if self._use_queue:
            self._consume_task = self._grpc_call_loop.create_task(
                self.consume_messages_from_gen()
            )

        # Keep track of in-flight requests.
        self._response_id = generate_request_id()
        request_context = ray.serve.context._get_serve_request_context()
        ray.serve.context._add_in_flight_request(
            request_context._internal_request_id, self._response_id, self
        )
        self.add_done_callback(
            lambda _: ray.serve.context._remove_in_flight_request(
                request_context._internal_request_id, self._response_id
            )
        )

    def _process_grpc_response(f: Union[Callable, Coroutine]):
        def deserialize_or_raise_error(
            grpc_response: serve_proprietary_pb2.ASGIResponse,
            metadata: RequestMetadata,
        ):
            # Create serializer with options from metadata
            serializer = RPCSerializer(
                metadata.request_serialization,
                metadata.response_serialization,
            )

            if grpc_response.is_error:
                err = serializer.loads_response(grpc_response.serialized_message)
                if isinstance(err, RayTaskError):
                    raise err.as_instanceof_cause()
                else:
                    raise err
            else:
                # If it's an HTTP request, then the proxy response generator is
                # expecting a pickled dictionary, so we return result directly
                # without deserializing. Otherwise, we deserialize the result.
                if ray.serve.context._get_serve_request_context().is_http_request:
                    return grpc_response.serialized_message
                else:
                    return serializer.loads_response(grpc_response.serialized_message)

        @wraps(f)
        def wrapper(self, *args, **kwargs):
            try:
                grpc_response = f(self, *args, **kwargs)
            except grpc.aio.AioRpcError as e:
                if e.code() == grpc.StatusCode.UNAVAILABLE:
                    raise ActorUnavailableError(
                        "Actor is unavailable.",
                        self._actor_id.binary(),
                    )
                raise
            except concurrent.futures.CancelledError:
                raise RequestCancelledError from None

            return deserialize_or_raise_error(grpc_response, self._metadata)

        @wraps(f)
        async def async_wrapper(self, *args, **kwargs):
            try:
                grpc_response = await f(self, *args, **kwargs)
            except grpc.aio.AioRpcError as e:
                if e.code() == grpc.StatusCode.UNAVAILABLE:
                    raise ActorUnavailableError(
                        "Actor is unavailable.",
                        self._actor_id.binary(),
                    )
                raise

            return deserialize_or_raise_error(grpc_response, self._metadata)

        if inspect.iscoroutinefunction(f):
            return async_wrapper
        else:
            return wrapper

    def __aiter__(self) -> AsyncIterator[Any]:
        return self

    def __iter__(self) -> Iterator[Any]:
        return self

    async def consume_messages_from_gen(self):
        try:
            async for resp in self._gen:
                self._result_queue.put_nowait(resp)
        except BaseException as e:
            self._result_queue.set_error(e)
        finally:
            self._result_queue.close()

    async def _get_internal(self):
        """Gets the result from the gRPC call object.

        If the call object is a UnaryUnaryCall, we await the call.
        Otherwise the call object is a UnaryStreamCall.
          - If the request was sent on a separate loop, then the
          streamed results are being consumed and put onto the in-memory
          queue, so we read from that queue.
          - Otherwise the request was sent on the current loop, so we
          fetch the next object from the async generator.
        """

        if self._gen is None:
            return await self._call
        elif self._use_queue:
            return await self._result_queue.get_one_message()
        else:
            return await self._gen.__anext__()

    async def get_rejection_response(self) -> Optional[ReplicaQueueLengthInfo]:
        """Get the queue length info from the replica to handle rejection."""
        assert (
            self._with_rejection
        ), "get_rejection_response() can only be called when request rejection is enabled."

        try:
            if self._rejection_response is None:
                # NOTE(edoakes): this is required for gRPC to raise an AioRpcError if something
                # goes wrong establishing the connection (for example, a bug in our code).
                await self._call.wait_for_connection()
                metadata = await self._call.initial_metadata()

                accepted = metadata.get("accepted", None)
                num_ongoing_requests = metadata.get("num_ongoing_requests", None)
                if accepted is None or num_ongoing_requests is None:
                    code = await self._call.code()
                    details = await self._call.details()
                    raise RuntimeError(f"Unexpected error ({code}): {details}.")

                self._rejection_response = ReplicaQueueLengthInfo(
                    accepted=bool(int(accepted)),
                    num_ongoing_requests=int(num_ongoing_requests),
                )

            return self._rejection_response
        except asyncio.CancelledError as e:
            # HTTP client disconnected or request was explicitly canceled.
            logger.info(
                "Cancelling request that has already been assigned to a replica."
            )
            self.cancel()
            raise e from None
        except grpc.aio.AioRpcError as e:
            # If we received an `UNAVAILABLE` grpc error, that is
            # equivalent to `RayActorError`, although we don't know
            # whether it's `ActorDiedError` or `ActorUnavailableError`.
            # Conservatively, we assume it is `ActorUnavailableError`,
            # and we raise it here so that it goes through the unified
            # code path for handling RayActorErrors.
            # The router will retry scheduling the request with the
            # cache invalidated, at which point if the actor is actually
            # dead, the router will realize through active probing.
            if not self._is_streaming:
                # In UnaryUnary calls, initial metadata is sent back with the request
                # response, so we can't determine if the request was accepted until
                # after the request is handled. If the replica crashed while handling
                # the request, we can still get initial metadata via the AioRpcError,
                # since the server sets the metadata before handling the request.
                # If there is no metadata, we know the replica was already unavailable
                # prior to the request being sent. We only raise an ActorUnavailableError
                # (and thus retry the request) if the request was rejected or if the
                # replica was already unavailable.
                metadata = e.initial_metadata()
                accepted = metadata.get("accepted", None)
                if accepted is not None and bool(int(accepted)):
                    num_ongoing_requests = metadata.get("num_ongoing_requests", None)
                    if num_ongoing_requests is None:
                        raise RuntimeError(
                            f"Unexpected error ({e.code()}): {e.details()}."
                        )

                    return ReplicaQueueLengthInfo(
                        accepted=True,
                        num_ongoing_requests=int(num_ongoing_requests),
                    )

            if e.code() == grpc.StatusCode.UNAVAILABLE:
                raise ActorUnavailableError(
                    "Actor is unavailable.",
                    self._actor_id.binary(),
                )

            raise e from None

    @_process_grpc_response
    def get(self, timeout_s: Optional[float]):
        if is_running_in_asyncio_loop():
            raise RuntimeError(
                "Sync method `get()` should not be called from within an `asyncio` "
                "event loop. Use `get_async()` instead."
            )

        if self._fut is None:
            self._fut = run_coroutine_threadsafe(
                self._get_internal(), self._grpc_call_loop
            )

        try:
            return self._fut.result(timeout=timeout_s)
        except concurrent.futures.TimeoutError:
            raise TimeoutError("Timed out waiting for result.") from None

    @_process_grpc_response
    async def get_async(self):
        if self._fut is None:
            if self._calling_from_same_loop:
                return await self._get_internal()
            else:
                self._fut = run_coroutine_threadsafe(
                    self._get_internal(), self._grpc_call_loop
                )

        return await asyncio.wrap_future(self._fut)

    @_process_grpc_response
    def __next__(self):
        if is_running_in_asyncio_loop():
            raise RuntimeError(
                "Sync method `__next__()` should not be called from within an "
                "`asyncio` event loop. Use `__anext__()` instead."
            )

        fut = run_coroutine_threadsafe(self._get_internal(), loop=self._grpc_call_loop)
        try:
            return fut.result()
        except StopAsyncIteration:
            # We need to raise the synchronous version, StopIteration
            raise StopIteration

    @_process_grpc_response
    async def __anext__(self):
        if self._calling_from_same_loop:
            return await self._get_internal()
        else:
            fut = run_coroutine_threadsafe(
                self._get_internal(), loop=self._grpc_call_loop
            )
            return await asyncio.wrap_future(fut)

    def add_done_callback(self, callback: Callable):
        self._call.add_done_callback(callback)

    def cancel(self):
        self._call.cancel()

    def to_object_ref(self, timeout_s: Optional[float]) -> ray.ObjectRef:
        raise OBJ_REF_NOT_SUPPORTED_ERROR

    async def to_object_ref_async(self) -> ray.ObjectRef:
        raise OBJ_REF_NOT_SUPPORTED_ERROR

    def to_object_ref_gen(self) -> ray.ObjectRefGenerator:
        raise OBJ_REF_NOT_SUPPORTED_ERROR
