import grpc

from ray.serve._private.constants import (
    RAY_SERVE_REPLICA_GRPC_MAX_MESSAGE_LENGTH,
)
from ray.serve._private.common import RunningReplicaInfo
from ray.serve._private.request_router.common import PendingRequest
from ray.serve._private.request_router.replica_wrapper import (
    ActorReplicaWrapper,
    gRPCReplicaWrapper,
    ReplicaWrapper,
    RunningReplica,
)
from ray.serve.generated import serve_pb2_grpc


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
                        RAY_SERVE_REPLICA_GRPC_MAX_MESSAGE_LENGTH,
                    )
                ],
            )
            self._stub = serve_pb2_grpc.ASGIServiceStub(self._channel)

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
