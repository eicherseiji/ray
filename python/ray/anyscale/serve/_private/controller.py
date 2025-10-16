import logging
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from ray.anyscale.serve._private.constants import (
    ANYSCALE_RAY_SERVE_ENABLE_DIRECT_INGRESS,
    ANYSCALE_RAY_SERVE_ENABLE_HA_PROXY,
)
from ray.serve._private.common import DeploymentID, RequestProtocol
from ray.serve._private.constants import (
    RAY_SERVE_THROUGHPUT_OPTIMIZED,
    SERVE_LOGGER_NAME,
)
from ray.serve._private.controller import ServeController
from ray.serve._private.deployment_state import DeploymentReplica
from ray.serve._private.long_poll import LongPollNamespace
from ray.serve._private.node_port_manager import NodePortManager
from ray.serve._private.utils import is_grpc_enabled
from ray.serve.config import DeploymentMode, HTTPOptions, gRPCOptions
from ray.serve.schema import (
    LoggingConfig,
    ReplicaDetails,
    Target,
    TargetGroup,
)

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
            logger.info(
                "Throughput optimized Ray Serve enabled with the following configurations:\n"
                "  • Direct ingress enabled\n"
                "  • gRPC communication enabled\n"
                "  • User code and router running in main thread (not separate)\n"
                "  • Request path log buffer size: 1000\n"
                "  • Log to stderr is disabled\n"
                "  • Garbage collector is frozen on startup\n"
            )

        self._ha_proxy_enabled = ANYSCALE_RAY_SERVE_ENABLE_HA_PROXY
        self._direct_ingress_enabled = ANYSCALE_RAY_SERVE_ENABLE_DIRECT_INGRESS
        if self._ha_proxy_enabled:
            logger.info(
                "HAProxy is enabled in AnyscaleServeController, replacing Serve proxy with HAProxy"
            )
        elif self._direct_ingress_enabled:
            logger.info(
                "Direct ingress is enabled in AnyscaleServeController, enabling proxy "
                "on head node only."
            )

            http_options.location = DeploymentMode.HeadOnly

        await super().__init__(
            http_options=http_options,
            global_logging_config=global_logging_config,
            grpc_options=grpc_options,
        )

        self._last_broadcasted_target_groups: Dict[Tuple[str, str], TargetGroup] = []

    def get_target_groups(
        self,
        app_name: Optional[str] = None,
        from_proxy_manager: bool = False,
    ) -> List[TargetGroup]:
        """Get target groups for direct ingress deployments.
        This overrides the base implementation to return target groups that
        point directly to replica ports rather than proxy ports when direct
        ingress is enabled or when called by an internal proxy manager.

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
        one target group per application with unique route prefix.
        5. HAProxy is enabled and the caller is not an internal proxy manager. In
        this case, we return target groups containing the proxies (e.g. haproxy).
        6. HAProxy is enabled and the caller is an internal proxy manager (e.g.
        haproxy manager). In this case, we return target groups containing the
        ingress replicas and possibly the Serve proxies.
        """
        proxy_target_groups = super().get_target_groups()
        if not self._direct_ingress_enabled or (
            self._ha_proxy_enabled and not from_proxy_manager
        ):
            return proxy_target_groups

        # Get all applications and their metadata
        if app_name is None:
            apps = [
                _app_name
                for _app_name, _ in self.application_state_manager.list_app_statuses().items()
            ]
        else:
            apps = [app_name]

        # TODO(landscapepainter): A better way to handle this is to write an API that can tell
        # if the ingress deployment is healthy regardless of the application status.
        apps = [
            app
            for app in apps
            if self.application_state_manager.get_route_prefix(app) is not None
        ]

        if not apps:
            # TODO: Return the http/grpc proxy on the head node if from_proxy_manager is True
            return proxy_target_groups

        # Create target groups for each application
        target_groups = []
        for app_name in apps:
            route_prefix = self.application_state_manager.get_route_prefix(app_name)
            app_target_groups = self.get_target_groups_for_app(app_name, route_prefix)
            if app_target_groups:
                target_groups.extend(app_target_groups)
            else:
                target_groups.extend(
                    self.get_target_groups_for_app_with_no_running_replicas(
                        route_prefix, app_name
                    )
                )

        return target_groups

    def get_running_replica_details_for_ingress_deployment(
        self, app_name: str
    ) -> List[ReplicaDetails]:
        """Get running replica details for a specific application."""
        ingress_deployment_name = (
            self.application_state_manager.get_ingress_deployment_name(app_name)
        )
        deployment_id = DeploymentID(app_name=app_name, name=ingress_deployment_name)
        details = self.deployment_state_manager.get_deployment_details(deployment_id)
        if not details:
            return []
        replica_details = details.replicas
        running_replica_ids = {
            replica_info.replica_id.unique_id
            for replica_info in self.deployment_state_manager.get_running_replica_infos().get(
                deployment_id, []
            )
        }
        return [
            replica_detail
            for replica_detail in replica_details
            if replica_detail.replica_id in running_replica_ids
        ]

    def get_target_groups_for_app(
        self, app_name: str, route_prefix: str
    ) -> List[TargetGroup]:
        """
        Create HTTP and gRPC target groups for a specific application.

        This function can return empty list if there are no running replicas.
        Or replicas have not fully initialized yet, where their ports are not
        allocated yet.
        """
        # Get running replicas for the ingress deployment
        replica_details = self.get_running_replica_details_for_ingress_deployment(
            app_name
        )
        if not replica_details:
            return []

        target_groups = []

        # Create targets for each protocol
        http_targets = self._get_targets_for_protocol(
            replica_details, RequestProtocol.HTTP
        )
        if http_targets:
            target_groups.append(
                TargetGroup(
                    protocol=RequestProtocol.HTTP,
                    route_prefix=route_prefix,
                    targets=http_targets,
                    app_name=app_name,
                )
            )

        # Add gRPC targets if enabled
        if is_grpc_enabled(self.get_grpc_config()):
            grpc_targets = self._get_targets_for_protocol(
                replica_details, RequestProtocol.GRPC
            )
            if grpc_targets:
                target_groups.append(
                    TargetGroup(
                        protocol=RequestProtocol.GRPC,
                        route_prefix=route_prefix,
                        targets=grpc_targets,
                        app_name=app_name,
                    )
                )

        return target_groups

    def get_target_groups_for_app_with_no_running_replicas(
        self, route_prefix: str, app_name: str
    ) -> List[TargetGroup]:
        """
        For applications that have no running replicas, we return target groups
        for proxy. This will allow applications to be discoverable via the
        proxy in situations where their replicas have scaled down to 0.
        """
        # TODO: Return the http/grpc proxy on the head node if from_proxy_manager is True
        target_groups = []
        http_targets = self.proxy_state_manager.get_targets(RequestProtocol.HTTP)
        grpc_targets = self.proxy_state_manager.get_targets(RequestProtocol.GRPC)
        if http_targets:
            target_groups.append(
                TargetGroup(
                    protocol=RequestProtocol.HTTP,
                    route_prefix=route_prefix,
                    targets=http_targets,
                    app_name=app_name,
                )
            )
        if grpc_targets:
            target_groups.append(
                TargetGroup(
                    protocol=RequestProtocol.GRPC,
                    route_prefix=route_prefix,
                    targets=grpc_targets,
                    app_name=app_name,
                )
            )
        return target_groups

    def _get_targets_for_protocol(
        self, replica_details: List[ReplicaDetails], protocol: RequestProtocol
    ) -> List[Target]:
        """Create targets for a specific protocol from a list of replicas."""
        return [
            Target(
                ip=replica_detail.node_ip,
                port=self.get_port(replica_detail, protocol),
                instance_id=replica_detail.node_instance_id,
                name=replica_detail.actor_name,
            )
            for replica_detail in replica_details
            if self.is_port_allocated(replica_detail, protocol)
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
            # Update port values for ingress replicas.
            # Non-ingress replicas are not expected to have ports allocated.
            ingress_replicas_info_list: List[
                Tuple[str, str, int, int]
            ] = self.deployment_state_manager.get_ingress_replicas_info()

            NodePortManager.update_ports(ingress_replicas_info_list)

            # Clean up stale ports
            # get all alive replica ids and their node ids.
            NodePortManager.prune(self._get_node_id_to_alive_replica_ids())

        if self._ha_proxy_enabled:
            self.broadcast_target_groups_if_changed()

    def broadcast_target_groups_if_changed(self) -> None:
        """Broadcast target groups over long poll if they have changed.

        Keeps an in-memory record of the last target groups that were broadcast
        to determine if they have changed.
        """
        target_groups: List[TargetGroup] = self.get_target_groups(
            from_proxy_manager=True,
        )

        protocol_route_to_target_group = {
            (tg.protocol, tg.route_prefix): tg for tg in target_groups
        }

        # Check if target groups have changed by comparing the mappings directly
        if self._last_broadcasted_target_groups == protocol_route_to_target_group:
            return

        self.long_poll_host.notify_changed(
            {LongPollNamespace.TARGET_GROUPS: target_groups}
        )
        self._last_broadcasted_target_groups = protocol_route_to_target_group

    def allocate_replica_port(
        self, node_id: str, replica_id: str, protocol: RequestProtocol
    ) -> int:
        """Allocate an HTTP port for a replica in direct ingress mode."""
        node_manager = NodePortManager.get_node_manager(node_id)
        return node_manager.allocate_port(replica_id, protocol)

    def release_replica_port(
        self,
        node_id: str,
        replica_id: str,
        port: int,
        protocol: RequestProtocol,
        block_port: bool = False,
    ):
        """Release an HTTP port for a replica in direct ingress mode."""
        node_manager = NodePortManager.get_node_manager(node_id)
        node_manager.release_port(replica_id, port, protocol, block_port)

    def get_port(
        self, replica_detail: ReplicaDetails, protocol: RequestProtocol
    ) -> int:
        """Get the port for a replica."""
        node_manager = NodePortManager.get_node_manager(replica_detail.node_id)
        return node_manager.get_port(replica_detail.replica_id, protocol)

    def is_port_allocated(
        self, replica_detail: ReplicaDetails, protocol: RequestProtocol
    ) -> bool:
        """Check if the port for a replica is allocated."""
        node_manager = NodePortManager.get_node_manager(replica_detail.node_id)
        return node_manager.is_port_allocated(replica_detail.replica_id, protocol)
