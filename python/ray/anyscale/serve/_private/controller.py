import logging
from typing import List, Optional

from ray.anyscale.serve._private.constants import (
    ANYSCALE_FREEZE_GC_ON_STARTUP,
    ANYSCALE_RAY_SERVE_ENABLE_HA_PROXY,
)
from ray.serve._private.common import RequestProtocol
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
from ray.serve._private.long_poll import LongPollNamespace
from ray.serve.config import HTTPOptions, gRPCOptions
from ray.serve.schema import LoggingConfig, TargetGroup

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

        self._ha_proxy_enabled = ANYSCALE_RAY_SERVE_ENABLE_HA_PROXY
        if self._ha_proxy_enabled:
            logger.info(
                "HAProxy is enabled in AnyscaleServeController, replacing Serve "
                "proxy with HAProxy."
            )

        await super().__init__(
            http_options=http_options,
            global_logging_config=global_logging_config,
            grpc_options=grpc_options,
        )

        # Ensure _direct_ingress_enabled is True when HAProxy is enabled.
        # The parent imports from OSS constants which don't have the HAProxy
        # override, so we need to set it explicitly here.
        if self._ha_proxy_enabled:
            self._direct_ingress_enabled = True

        # Initialize to None (not []) to ensure the first broadcast always happens,
        # even if target_groups is empty (e.g., route_prefix=None deployments).
        self._last_broadcasted_target_groups: Optional[List[TargetGroup]] = None

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
        # Call _get_proxy_target_groups directly instead of super().get_target_groups()
        # because the parent's get_target_groups() now includes DI logic, which would
        # return DI target groups instead of proxy target groups when DI is enabled.
        proxy_target_groups = self._get_proxy_target_groups()
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
            # When HAProxy is enabled and there are no apps, return empty target groups
            # so that all requests fall through to the default_backend (404)
            if self._ha_proxy_enabled and from_proxy_manager:
                return []

            # TODO: Return the http/grpc proxy on the head node if from_proxy_manager is True
            return proxy_target_groups

        # Create target groups for each application
        target_groups = []
        for app_name in apps:
            route_prefix = self.application_state_manager.get_route_prefix(app_name)
            # Use parent's _get_target_groups_for_app method
            app_target_groups = self._get_target_groups_for_app(app_name, route_prefix)
            if app_target_groups:
                target_groups.extend(app_target_groups)
            else:
                # This method is overridden in this class to handle HAProxy
                target_groups.extend(
                    self._get_target_groups_for_app_with_no_running_replicas(
                        route_prefix, app_name
                    )
                )

        return target_groups

    def _get_target_groups_for_app_with_no_running_replicas(
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
                    targets=[] if self._ha_proxy_enabled else http_targets,
                    app_name=app_name,
                )
            )
        if grpc_targets:
            target_groups.append(
                TargetGroup(
                    protocol=RequestProtocol.GRPC,
                    route_prefix=route_prefix,
                    targets=[] if self._ha_proxy_enabled else grpc_targets,
                    app_name=app_name,
                )
            )
        return target_groups

    async def run_control_loop_step(
        self, start_time: float, recovering_timeout: float, num_loops: int
    ):
        await super().run_control_loop_step(start_time, recovering_timeout, num_loops)

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

        # Check if target groups have changed by comparing the objects directly
        if self._last_broadcasted_target_groups == target_groups:
            return

        self.long_poll_host.notify_changed(
            {LongPollNamespace.TARGET_GROUPS: target_groups}
        )
        self._last_broadcasted_target_groups = target_groups
