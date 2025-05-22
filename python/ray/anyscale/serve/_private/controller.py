from collections import defaultdict
import logging
from typing import Dict, List, Optional, Set, Tuple

from ray.serve._private.common import DeploymentID, RequestProtocol, RunningReplicaInfo
from ray.serve._private.controller import ServeController
from ray.serve._private.constants import (
    SERVE_LOGGER_NAME,
)
from ray.serve._private.deployment_state import DeploymentReplica
from ray.serve._private.utils import is_grpc_enabled
from ray.serve.schema import (
    Target,
    TargetGroup,
)
from ray.anyscale.serve._private.constants import (
    ANYSCALE_RAY_SERVE_ENABLE_DIRECT_INGRESS,
)
from ray.serve._private.node_port_manager import NodePortManager

logger = logging.getLogger(SERVE_LOGGER_NAME)


class AnyscaleServeController(ServeController):
    """Anyscale-specific ServeController that handles direct ingress functionality.
    This controller extends the base ServeController to support direct ingress,
    where each replica listens directly on its own ports rather than going through
    a proxy. This is useful for Kubernetes deployments where we want each replica
    to be directly accessible via the ingress controller.
    """

    async def __init__(self, *args, **kwargs):
        await super().__init__(*args, **kwargs)
        self._direct_ingress_enabled = ANYSCALE_RAY_SERVE_ENABLE_DIRECT_INGRESS
        if self._direct_ingress_enabled:
            logger.info("Direct ingress is enabled in AnyscaleServeController")

    def get_target_groups(self) -> List[TargetGroup]:
        """Get target groups for direct ingress deployments.
        This overrides the base implementation to return target groups that
        point directly to replica ports rather than proxy ports when direct
        ingress is enabled.

        Following situations are possible:
        1. Direct ingress is not enabled. In this case, we just return the
        target groups from the proxy implementation.
        2. Direct ingress is enabled and there are no applications. In this case,
        we return target groups for proxy. Serve controller is running but there
        are no applications to route traffic to.
        3. Direct ingress is enabled and there are applications. All applications
        have atleast one running replica. In this case, we return target groups
        for all applications with targets pointing to the running replicas.
        4. Direct ingress is enabled and there are applications. Some applications
        have no running replicas. In this case, for applications that have no
        running replicas, we return target groups for proxy and for applications
        that have running replicas, we return target groups for direct ingress.
        If there are multiple applications with no running replicas, we return
        only one target group for proxy.
        """
        proxy_target_groups = super().get_target_groups()
        if not self._direct_ingress_enabled:
            return proxy_target_groups

        # Get all applications and their metadata
        apps = [
            app_name
            for app_name, _ in self.application_state_manager.list_app_statuses().items()
        ]
        if not apps:
            return proxy_target_groups

        # Create target groups for each application
        target_groups = []
        atleast_one_app_has_no_running_replica = False
        for app_name in apps:
            app_target_groups = self.get_target_groups_for_app(app_name)
            if app_target_groups:
                target_groups.extend(app_target_groups)
            else:
                atleast_one_app_has_no_running_replica = True

        if atleast_one_app_has_no_running_replica:
            target_groups.extend(proxy_target_groups)
        return target_groups

    def get_running_replica_infos_for_ingress_deployment(
        self, app_name: str
    ) -> List[RunningReplicaInfo]:
        """Get running replica infos for a specific application."""
        ingress_deployment_name = (
            self.application_state_manager.get_ingress_deployment_name(app_name)
        )
        deployment_id = DeploymentID(app_name=app_name, name=ingress_deployment_name)
        return self.deployment_state_manager.get_running_replica_infos().get(
            deployment_id, []
        )

    def get_target_groups_for_app(self, app_name: str) -> List[TargetGroup]:
        """Create HTTP and gRPC target groups for a specific application."""
        route_prefix = self.application_state_manager.get_route_prefix(app_name)

        # Get running replicas for the ingress deployment
        replica_infos = self.get_running_replica_infos_for_ingress_deployment(app_name)
        if not replica_infos:
            return []

        target_groups = []

        # Create targets for each protocol
        http_targets = self._get_targets_for_protocol(
            replica_infos, RequestProtocol.HTTP
        )
        if http_targets:
            target_groups.append(
                TargetGroup(
                    protocol=RequestProtocol.HTTP,
                    route_prefix=route_prefix,
                    targets=http_targets,
                )
            )

        # Add gRPC targets if enabled
        if is_grpc_enabled(self.get_grpc_config()):
            grpc_targets = self._get_targets_for_protocol(
                replica_infos, RequestProtocol.GRPC
            )
            if grpc_targets:
                target_groups.append(
                    TargetGroup(
                        protocol=RequestProtocol.GRPC,
                        route_prefix=route_prefix,
                        targets=grpc_targets,
                    )
                )

        return target_groups

    def _get_replica_node_ip_and_port(
        self, replica_info: RunningReplicaInfo, protocol: RequestProtocol
    ) -> Tuple[str, int]:
        """Get the node IP and port for a replica based on protocol."""
        if protocol == RequestProtocol.HTTP:
            return replica_info.node_ip, self.get_http_port(replica_info)
        elif protocol == RequestProtocol.GRPC:
            return replica_info.node_ip, self.get_grpc_port(replica_info)
        else:
            raise ValueError(f"Unsupported protocol: {protocol}")

    def _get_targets_for_protocol(
        self, replica_infos: List[RunningReplicaInfo], protocol: RequestProtocol
    ) -> List[Target]:
        """Create targets for a specific protocol from a list of replicas."""
        return [
            Target(ip=ip, port=port)
            for replica_info in replica_infos
            if self.is_port_allocated(replica_info, protocol)
            for ip, port in [self._get_replica_node_ip_and_port(replica_info, protocol)]
        ]

    def _get_node_id_to_alive_replica_ids(self) -> Dict[str, Set[str]]:
        node_id_to_alive_replica_ids = defaultdict(set)
        # TODO(abrar): Expose the right APIs in the DeploymentStateManager
        # to get the alive replicas for a deployment.
        for ds in self.deployment_state_manager._deployment_states.values():
            # here we get all the replicas irrespective of their state
            # unlike in the get_running_replica_infos_for_ingress_deployment
            # where we only get the replicas that are running, because we dont
            # wish to agressively cleanup ports for replicas that are not running
            # and are in the process of being updated or are in the process of
            # being started.
            replicas: List[DeploymentReplica] = ds._replicas.get()
            for replica in replicas:
                node_id: Optional[str] = replica.actor_node_id
                if node_id is None:
                    continue
                replica_unique_id = replica.replica_id.unique_id
                node_id_to_alive_replica_ids[node_id].add(replica_unique_id)
        return node_id_to_alive_replica_ids

    async def run_control_loop_step(
        self, start_time: float, recovering_timeout: float, num_loops: int
    ):
        await super().run_control_loop_step(start_time, recovering_timeout, num_loops)

        if self._direct_ingress_enabled:
            # Clean up stale ports
            # get all alive replica ids and their node ids.
            NodePortManager.prune(self._get_node_id_to_alive_replica_ids())

    def allocate_replica_http_port(self, node_id: str, replica_id: str) -> int:
        """Allocate an HTTP port for a replica in direct ingress mode."""
        node_manager = NodePortManager.get_node_manager(node_id)
        return node_manager.allocate_http_port(replica_id)

    def allocate_replica_grpc_port(self, node_id: str, replica_id: str) -> int:
        """Allocate a gRPC port for a replica in direct ingress mode."""
        node_manager = NodePortManager.get_node_manager(node_id)
        return node_manager.allocate_grpc_port(replica_id)

    def release_replica_http_port(
        self, node_id: str, replica_id: str, port: int, block_port: bool = False
    ):
        """Release an HTTP port for a replica in direct ingress mode."""
        node_manager = NodePortManager.get_node_manager(node_id)
        node_manager.release_http_port(replica_id, port, block_port)

    def release_replica_grpc_port(
        self, node_id: str, replica_id: str, port: int, block_port: bool = False
    ):
        """Release a gRPC port for a replica in direct ingress mode."""
        node_manager = NodePortManager.get_node_manager(node_id)
        node_manager.release_grpc_port(replica_id, port, block_port)

    def get_http_port(self, replica_info: RunningReplicaInfo) -> int:
        """Get the HTTP port for a replica."""
        return NodePortManager.get_node_manager(replica_info.node_id).get_http_port(
            replica_info.replica_id.unique_id
        )

    def get_grpc_port(self, replica_info: RunningReplicaInfo) -> int:
        """Get the gRPC port for a replica."""
        return NodePortManager.get_node_manager(replica_info.node_id).get_grpc_port(
            replica_info.replica_id.unique_id
        )

    def is_port_allocated(
        self, replica_info: RunningReplicaInfo, protocol: RequestProtocol
    ) -> bool:
        """Check if the port for a replica is allocated."""
        if protocol == RequestProtocol.HTTP:
            return NodePortManager.get_node_manager(
                replica_info.node_id
            ).is_http_port_allocated(replica_info.replica_id.unique_id)
        elif protocol == RequestProtocol.GRPC:
            return NodePortManager.get_node_manager(
                replica_info.node_id
            ).is_grpc_port_allocated(replica_info.replica_id.unique_id)
        else:
            raise ValueError(f"Unsupported protocol: {protocol}")
