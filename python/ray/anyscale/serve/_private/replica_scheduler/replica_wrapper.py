import asyncio
import pickle

import grpc

from ray.anyscale.serve._private.constants import (
    ANYSCALE_RAY_SERVE_REPLICA_GRPC_MAX_MESSAGE_LENGTH,
)
from ray.anyscale.serve._private.replica_result import gRPCReplicaResult
from ray.anyscale.serve._private.serialization import RPCSerializer
from ray.serve._private.common import RunningReplicaInfo
from ray.serve._private.request_router.common import PendingRequest
from ray.serve._private.request_router.replica_wrapper import (
    ActorReplicaWrapper,
    ReplicaWrapper,
    RunningReplica,
)
from ray.serve.generated import serve_proprietary_pb2, serve_proprietary_pb2_grpc


class gRPCReplicaWrapper(ReplicaWrapper):
    def __init__(self, stub, actor_id):
        self._stub = stub
        self._actor_id = actor_id
        self._loop = asyncio.get_running_loop()

    def send_request_java(self, pr: PendingRequest):
        raise RuntimeError("gRPC requests not supported for Java.")

    def send_request_python(
        self, pr: PendingRequest, *, with_rejection: bool
    ) -> gRPCReplicaResult:
        """Send the request to a Python replica."""

        # Get serialization options from request metadata
        request_serialization = pr.metadata.request_serialization
        response_serialization = pr.metadata.response_serialization

        # Get cached serializer for this request to avoid per-request instantiation overhead
        serializer = RPCSerializer.get_cached_serializer(
            request_serialization, response_serialization
        )

        asgi_request = serve_proprietary_pb2.ASGIRequest(
            pickled_request_metadata=pickle.dumps(pr.metadata),
            request_args=serializer.dumps_request(pr.args),
            request_kwargs=serializer.dumps_request(pr.kwargs),
        )
        if with_rejection and pr.metadata.is_streaming:
            # Call a separate handler that may reject the request.
            # This handler is *always* a streaming call and the first message will
            # be a system message that accepts or rejects.
            call = self._stub.HandleRequestWithRejectionStreaming(asgi_request)
        elif with_rejection and not pr.metadata.is_streaming:
            # Call a separate handler that may reject the request.
            # This handler is *always* a unary call and the first message will
            # be a system message that accepts or rejects.
            call = self._stub.HandleRequestWithRejection(asgi_request)
        elif pr.metadata.is_streaming:
            call = self._stub.HandleRequestStreaming(asgi_request)
        else:
            call = self._stub.HandleRequest(asgi_request)

        return gRPCReplicaResult(
            call,
            pr.metadata,
            self._actor_id,
            loop=self._loop,
            with_rejection=with_rejection,
        )


class AnyscaleRunningReplica(RunningReplica):
    def __init__(self, replica_info: RunningReplicaInfo):
        super().__init__(replica_info)

        # Lazily created
        self._channel = None
        self._stub = None

        # Replica wrappers
        self._actor_replica_wrapper = ActorReplicaWrapper(self._actor_handle)
        self._grpc_replica_wrapper = None

    @property
    def stub(self) -> bool:
        if self._stub is None:
            self._channel = grpc.aio.insecure_channel(
                f"{self._replica_info.node_ip}:{self._replica_info.port}",
                options=[
                    (
                        "grpc.max_receive_message_length",
                        ANYSCALE_RAY_SERVE_REPLICA_GRPC_MAX_MESSAGE_LENGTH,
                    )
                ],
            )
            self._stub = serve_proprietary_pb2_grpc.ASGIServiceStub(self._channel)

        return self._stub

    def _get_replica_wrapper(self, pr: PendingRequest) -> ReplicaWrapper:
        if self._grpc_replica_wrapper is None:
            self._grpc_replica_wrapper = gRPCReplicaWrapper(
                self.stub, self._actor_handle._actor_id
            )

        return (
            self._actor_replica_wrapper
            if pr.metadata._by_reference
            else self._grpc_replica_wrapper
        )
