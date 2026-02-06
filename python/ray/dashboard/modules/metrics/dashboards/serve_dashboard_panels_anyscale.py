# ruff: noqa: E501
"""
Ray Serve Complete Dashboard V2 - Comprehensive Monitoring Dashboard

This dashboard provides comprehensive monitoring for Ray Serve applications including:
- Serve Overview: High-level application health, QPS, latency, and errors
- Deployment Deep Dive: Per-deployment metrics, replica health, and request flow
- Autoscaling: Target vs actual replicas, autoscaling inputs and constraints
- System Metrics: Controller health, metrics pipeline, event loops, and proxy status
- Request Batching: Batch sizes, wait times, and timeout rates
- Model Multiplexing: Model cache utilization, load times, and eviction rates
- Hardware Utilization: Head node, cluster resources, and network metrics

Compatible with Grafana 7.5.17+

IMPORTANT: Panel ID Mapping for Ray Dashboard Frontend Compatibility
====================================================================
The following panel IDs are referenced by the Ray Dashboard frontend
(ServeMetricsSection.tsx) and MUST be preserved for compatibility:

| Panel ID | Frontend Reference Title          | V2 Panel Title                    |
|----------|-----------------------------------|-----------------------------------|
| 7        | QPS per application               | QPS per Application               |
| 8        | Error QPS per application         | Error QPS per Application         |
| 10       | P90 latency per deployment        | Processing Latency per Deployment (P50, P90, P99) |
| 15       | P90 latency per application       | E2E Latency (P50, P90, P99)       |
| 20       | Ongoing HTTP Requests             | Ongoing HTTP Requests             |
| 21       | Ongoing gRPC Requests             | Ongoing gRPC Requests             |
| 22       | Scheduling Tasks                  | Scheduling Tasks                  |
| 23       | Scheduling Tasks in Backoff       | Scheduling Tasks in Backoff       |
| 24       | Controller Control Loop Duration  | Control Loop Duration             |
| 25       | Number of Control Loops           | Control Loop Rate                 |

Do NOT change these IDs without updating ServeMetricsSection.tsx accordingly.
"""


from ray.dashboard.modules.metrics.dashboards.common import (
    DashboardConfig,
    GridPos,
    Panel,
    PanelTemplate,
    Row,
    Target,
)

# ==============================================================================
# SERVE OVERVIEW SECTION
# ==============================================================================

SERVE_OVERVIEW_PANELS = [
    # Row 1: Key Health Indicators (Stat Panels)
    Panel(
        id=1001,
        title="Active Nodes",
        description="Total active nodes in cluster. Note: In shared clusters (e.g., with Ray Data), this includes all nodes, not just those running Serve replicas.",
        unit="short",
        targets=[
            Target(
                expr="sum(autoscaler_active_nodes{{{global_filters}}})",
                legend="Active Nodes",
            )
        ],
        template=PanelTemplate.STAT,
        grid_pos=GridPos(0, 0, 4, 3),
        thresholds=[{"color": "blue", "value": None}],
    ),
    Panel(
        id=1002,
        title="Total QPS",
        description="Total requests per second across all applications",
        unit="reqps",
        targets=[
            Target(
                expr='(sum(rate(ray_serve_num_http_requests_total{{application=~"$Application",application!~"",route=~"$HTTP_Route",route!~"/-/.*",{global_filters}}}[5m])) or vector(0)) + (sum(rate(ray_serve_num_grpc_requests_total{{application=~"$Application",application!~"",method=~"$gRPC_Method",{global_filters}}}[5m])) or vector(0))',
                legend="Total QPS",
            )
        ],
        template=PanelTemplate.STAT,
        grid_pos=GridPos(4, 0, 4, 3),
        thresholds=[{"color": "blue", "value": None}],
    ),
    Panel(
        id=1003,
        title="Error Rate %",
        description="Percentage of requests resulting in errors",
        unit="percent",
        targets=[
            Target(
                expr='100 * ((sum(rate(ray_serve_num_http_error_requests_total{{application=~"$Application",application!~"",route=~"$HTTP_Route",route!~"/-/.*",{global_filters}}}[5m])) or vector(0)) + (sum(rate(ray_serve_num_grpc_error_requests_total{{application=~"$Application",application!~"",method=~"$gRPC_Method",{global_filters}}}[5m])) or vector(0))) / clamp_min((sum(rate(ray_serve_num_http_requests_total{{application=~"$Application",application!~"",route=~"$HTTP_Route",route!~"/-/.*",{global_filters}}}[5m])) or vector(0)) + (sum(rate(ray_serve_num_grpc_requests_total{{application=~"$Application",application!~"",method=~"$gRPC_Method",{global_filters}}}[5m])) or vector(0)), 1)',
                legend="Error Rate",
            )
        ],
        template=PanelTemplate.STAT,
        grid_pos=GridPos(8, 0, 4, 3),
        thresholds=[
            {"color": "green", "value": None},
            {"color": "yellow", "value": 1},
            {"color": "red", "value": 5},
        ],
    ),
    Panel(
        id=1004,
        title="P99 HTTP Latency",
        description="99th percentile end-to-end HTTP latency measured at the proxy (from request receipt to response sent)",
        unit="ms",
        targets=[
            Target(
                expr='histogram_quantile(0.99, sum(rate(ray_serve_http_request_latency_ms_bucket{{application=~"$Application",application!~"",route=~"$HTTP_Route",route!~"/-/.*",{global_filters}}}[5m])) by (le)) or vector(0)',
                legend="P99 HTTP Latency",
            )
        ],
        template=PanelTemplate.STAT,
        grid_pos=GridPos(12, 0, 3, 3),
        thresholds=[{"color": "blue", "value": None}],
    ),
    Panel(
        id=1005,
        title="P99 gRPC Latency",
        description="99th percentile end-to-end gRPC latency measured at the proxy (from request receipt to response sent)",
        unit="ms",
        targets=[
            Target(
                expr='histogram_quantile(0.99, sum(rate(ray_serve_grpc_request_latency_ms_bucket{{application=~"$Application",application!~"",method=~"$gRPC_Method",{global_filters}}}[5m])) by (le)) or vector(0)',
                legend="P99 gRPC Latency",
            )
        ],
        template=PanelTemplate.STAT,
        grid_pos=GridPos(15, 0, 3, 3),
        thresholds=[{"color": "blue", "value": None}],
    ),
    Panel(
        id=1006,
        title="Applications",
        description="Total number of active applications",
        unit="short",
        targets=[
            Target(
                expr='count(count by (application) (ray_serve_deployment_replica_healthy{{application=~"$Application", deployment=~"$Deployment",{global_filters}}})) or vector(0)',
                legend="Applications",
            )
        ],
        template=PanelTemplate.STAT,
        grid_pos=GridPos(18, 0, 3, 3),
        thresholds=[{"color": "blue", "value": None}],
    ),
    Panel(
        id=1007,
        title="Healthy Proxies",
        description="Number of proxies in HEALTHY state (status value 2). Note: proxy_status values are: 1=STARTING, 2=HEALTHY, 3=UNHEALTHY, 4=DRAINING, 5=DRAINED. Metrics may persist briefly after shutdown until Prometheus garbage collection.",
        unit="short",
        targets=[
            Target(
                expr="count(ray_serve_proxy_status{{{global_filters}}} == 2) or vector(0)",
                legend="Healthy Proxies",
            )
        ],
        template=PanelTemplate.STAT,
        grid_pos=GridPos(21, 0, 3, 3),
        thresholds=[{"color": "blue", "value": None}],
    ),
    # Row 2: Application Status Timeline (using Graph panel as workaround)
    Panel(
        id=1008,
        title="Application Status Timeline",
        description="Application lifecycle states over time: 0=UNKNOWN, 1=DEPLOY_FAILED, 2=UNHEALTHY, 3=NOT_STARTED, 4=DELETING, 5=DEPLOYING, 6=RUNNING. Note: Uses stepped line visualization (Grafana 7.5 compatible).",
        unit="short",
        targets=[
            Target(
                expr='ray_serve_application_status{{application=~"$Application",{global_filters}}}',
                legend="{{application}}",
            )
        ],
        template=PanelTemplate.GRAPH,
        fill=0,
        stack=False,
        grid_pos=GridPos(0, 3, 24, 10),
    ),
    # Row 3: Traffic Overview
    Panel(
        id=7,  # Referenced by Ray Dashboard frontend (ServeMetricsSection.tsx)
        title="QPS per Application",
        description="Request rate per application",
        unit="reqps",
        targets=[
            Target(
                expr='sum(rate(ray_serve_num_http_requests_total{{application=~"$Application",application!~"",route=~"$HTTP_Route",route!~"/-/.*",{global_filters}}}[5m])) by (application, route)',
                legend="HTTP: {{application}} {{route}}",
            ),
            Target(
                expr='sum(rate(ray_serve_num_grpc_requests_total{{application=~"$Application",application!~"",method=~"$gRPC_Method",{global_filters}}}[5m])) by (application, method)',
                legend="gRPC: {{application}} {{method}}",
            ),
        ],
        template=PanelTemplate.GRAPH,
        fill=1,
        stack=False,
        grid_pos=GridPos(0, 13, 12, 8),
    ),
    Panel(
        id=8,  # Referenced by Ray Dashboard frontend (ServeMetricsSection.tsx)
        title="Error QPS per Application",
        description="Error rate per application by error code",
        unit="reqps",
        targets=[
            Target(
                expr='sum(rate(ray_serve_num_http_error_requests_total{{application=~"$Application",application!~"",route=~"$HTTP_Route",route!~"/-/.*",{global_filters}}}[5m])) by (application, route, error_code)',
                legend="HTTP: {{application}} {{route}} {{error_code}}",
            ),
            Target(
                expr='sum(rate(ray_serve_num_grpc_error_requests_total{{application=~"$Application",application!~"",method=~"$gRPC_Method",{global_filters}}}[5m])) by (application, method, error_code)',
                legend="gRPC: {{application}} {{method}} {{error_code}}",
            ),
        ],
        template=PanelTemplate.GRAPH,
        fill=1,
        stack=False,
        grid_pos=GridPos(12, 13, 12, 8),
    ),
    # Row 4: Latency Overview
    Panel(
        id=15,  # Referenced by Ray Dashboard frontend (ServeMetricsSection.tsx) as "P90 latency per application"
        title="E2E Latency (P50, P90, P99)",
        description="End-to-end latency percentiles measured at the proxy. Includes queue wait time, processing time, and network overhead. Excludes client-side latency. Gap between percentiles indicates latency variance - wider gaps suggest inconsistent performance.",
        unit="ms",
        targets=[
            Target(
                expr='histogram_quantile(0.5, sum(rate(ray_serve_http_request_latency_ms_bucket{{application=~"$Application",application!~"",route=~"$HTTP_Route",route!~"/-/.*",{global_filters}}}[5m])) by (application, route, le))',
                legend="HTTP P50: {{application}} {{route}}",
            ),
            Target(
                expr='histogram_quantile(0.9, sum(rate(ray_serve_http_request_latency_ms_bucket{{application=~"$Application",application!~"",route=~"$HTTP_Route",route!~"/-/.*",{global_filters}}}[5m])) by (application, route, le))',
                legend="HTTP P90: {{application}} {{route}}",
            ),
            Target(
                expr='histogram_quantile(0.99, sum(rate(ray_serve_http_request_latency_ms_bucket{{application=~"$Application",application!~"",route=~"$HTTP_Route",route!~"/-/.*",{global_filters}}}[5m])) by (application, route, le))',
                legend="HTTP P99: {{application}} {{route}}",
            ),
            Target(
                expr='histogram_quantile(0.5, sum(rate(ray_serve_grpc_request_latency_ms_bucket{{application=~"$Application",application!~"",method=~"$gRPC_Method",{global_filters}}}[5m])) by (application, method, le))',
                legend="gRPC P50: {{application}} {{method}}",
            ),
            Target(
                expr='histogram_quantile(0.9, sum(rate(ray_serve_grpc_request_latency_ms_bucket{{application=~"$Application",application!~"",method=~"$gRPC_Method",{global_filters}}}[5m])) by (application, method, le))',
                legend="gRPC P90: {{application}} {{method}}",
            ),
            Target(
                expr='histogram_quantile(0.99, sum(rate(ray_serve_grpc_request_latency_ms_bucket{{application=~"$Application",application!~"",method=~"$gRPC_Method",{global_filters}}}[5m])) by (application, method, le))',
                legend="gRPC P99: {{application}} {{method}}",
            ),
        ],
        template=PanelTemplate.GRAPH,
        fill=0,
        stack=False,
        grid_pos=GridPos(0, 21, 24, 8),
    ),
    # Row 5: Proxy Ongoing Requests
    Panel(
        id=20,  # Referenced by Ray Dashboard frontend (ServeMetricsSection.tsx)
        title="Ongoing HTTP Requests",
        description="Current number of HTTP requests being processed by the proxy",
        unit="requests",
        targets=[
            Target(
                expr="ray_serve_num_ongoing_http_requests{{{global_filters}}}",
                legend="Ongoing HTTP Requests",
            )
        ],
        template=PanelTemplate.GRAPH,
        fill=1,
        stack=False,
        grid_pos=GridPos(0, 29, 12, 8),
    ),
    Panel(
        id=21,  # Referenced by Ray Dashboard frontend (ServeMetricsSection.tsx)
        title="Ongoing gRPC Requests",
        description="Current number of gRPC requests being processed by the proxy",
        unit="requests",
        targets=[
            Target(
                expr="ray_serve_num_ongoing_grpc_requests{{{global_filters}}}",
                legend="Ongoing gRPC Requests",
            )
        ],
        template=PanelTemplate.GRAPH,
        fill=1,
        stack=False,
        grid_pos=GridPos(12, 29, 12, 8),
    ),
]

# ==============================================================================
# DEPLOYMENT DEEP DIVE SECTION
# ==============================================================================

DEPLOYMENT_DEEP_DIVE_PANELS = [
    # Row 1: Health Indicators
    Panel(
        id=2001,
        title="Replica Health",
        description="Percentage of replicas in healthy state and serving traffic. 100% = all replicas healthy, <100% indicates some replicas are unhealthy or starting.",
        unit="percent",
        targets=[
            Target(
                expr='100 * sum(ray_serve_deployment_replica_healthy{{application=~"$Application", deployment=~"$Deployment",{global_filters}}} == 1) / clamp_min(count(ray_serve_deployment_replica_healthy{{application=~"$Application", deployment=~"$Deployment",{global_filters}}}), 1)',
                legend="Healthy %",
            )
        ],
        template=PanelTemplate.STAT,
        grid_pos=GridPos(0, 0, 5, 5),
        thresholds=[
            {"color": "red", "value": None},
            {"color": "yellow", "value": 50},
            {"color": "green", "value": 100},
        ],
    ),
    Panel(
        id=2002,
        title="Deployment QPS",
        description="Requests per second processed by deployment replicas",
        unit="reqps",
        targets=[
            Target(
                expr='sum(rate(ray_serve_deployment_request_counter_total{{application=~"$Application", deployment=~"$Deployment", replica=~"$Replica",{global_filters}}}[5m]))',
                legend="Total",
            )
        ],
        template=PanelTemplate.STAT,
        grid_pos=GridPos(5, 0, 5, 5),
        thresholds=[{"color": "blue", "value": None}],
    ),
    Panel(
        id=2003,
        title="Error Rate %",
        description="Percentage of requests resulting in exceptions",
        unit="percent",
        targets=[
            Target(
                expr='100 * sum(rate(ray_serve_deployment_error_counter_total{{application=~"$Application", deployment=~"$Deployment", replica=~"$Replica",{global_filters}}}[5m])) / clamp_min(sum(rate(ray_serve_deployment_request_counter_total{{application=~"$Application", deployment=~"$Deployment", replica=~"$Replica",{global_filters}}}[5m])), 1)',
                legend="Total",
            )
        ],
        template=PanelTemplate.STAT,
        grid_pos=GridPos(10, 0, 5, 5),
        thresholds=[
            {"color": "green", "value": None},
            {"color": "yellow", "value": 1},
            {"color": "red", "value": 5},
        ],
    ),
    Panel(
        id=2004,
        title="P99 Processing Latency",
        description="99th percentile processing latency at the replica. Time spent strictly in user code (excludes queue wait time and routing overhead). High values indicate model/logic bottleneck.",
        unit="ms",
        targets=[
            Target(
                expr='histogram_quantile(0.99, sum(rate(ray_serve_deployment_processing_latency_ms_bucket{{application=~"$Application", deployment=~"$Deployment", replica=~"$Replica",{global_filters}}}[5m])) by (le)) or vector(0)',
                legend="Total",
            )
        ],
        template=PanelTemplate.STAT,
        grid_pos=GridPos(15, 0, 4, 5),
        thresholds=[{"color": "blue", "value": None}],
    ),
    Panel(
        id=2005,
        title="Active Handles",
        description="Number of deployment handles that have processed at least one request. Each handle represents a client connection point to the deployment.",
        unit="short",
        targets=[
            Target(
                expr='count(count by (handle) (ray_serve_handle_request_counter_total{{application=~"$Application", deployment=~"$Deployment",{global_filters}}}))',
                legend="Total",
            )
        ],
        template=PanelTemplate.STAT,
        grid_pos=GridPos(19, 0, 5, 5),
        thresholds=[{"color": "blue", "value": None}],
    ),
    # Row 2: Deployment State Timeline
    Panel(
        id=2006,
        title="Deployment Status Timeline",
        description="Shows deployment lifecycle states over time: 0=UNKNOWN, 1=DEPLOY_FAILED, 2=UNHEALTHY, 3=UPDATING, 4=UPSCALING, 5=DOWNSCALING, 6=HEALTHY. Note: Uses stepped line visualization (Grafana 7.5 compatible).",
        unit="short",
        targets=[
            Target(
                expr='ray_serve_deployment_status{{application=~"$Application", deployment=~"$Deployment",{global_filters}}}',
                legend="{{application}} / {{deployment}}",
            )
        ],
        template=PanelTemplate.GRAPH,
        fill=0,
        stack=False,
        grid_pos=GridPos(0, 5, 24, 8),
    ),
    # Row 3: Replica Health Over Time (Heatmap)
    Panel(
        id=2007,
        title="Replica Health Heatmap",
        description="Heatmap showing average replica health per deployment over time. Green (100%) = all replicas healthy, Yellow (~50%) = some unhealthy, Red (0%) = all unhealthy. Each row represents a deployment.",
        unit="percent",
        targets=[
            Target(
                expr='100 * avg(ray_serve_deployment_replica_healthy{{application=~"$Application", deployment=~"$Deployment",{global_filters}}}) by (application, deployment)',
                legend="{{application}} / {{deployment}}",
            )
        ],
        template=PanelTemplate.HEATMAP,
        grid_pos=GridPos(0, 13, 24, 8),
        heatmap_color_scheme="RdYlGn",
        heatmap_color_reverse=True,
    ),
    # Row 4: Request Counts at Different Layers
    Panel(
        id=2008,
        title="DeploymentHandle Request QPS",
        description="Requests per second processed by DeploymentHandles. This metric is captured at the Handle level - the entry point when handle.remote() is called. Tracks requests from both HTTP/gRPC ingress and internal service-to-service calls.",
        unit="reqps",
        targets=[
            Target(
                expr='sum(rate(ray_serve_handle_request_counter_total{{application=~"$Application", deployment=~"$Deployment",{global_filters}}}[5m])) by (application, deployment)',
                legend="{{application}} {{deployment}}",
            )
        ],
        template=PanelTemplate.GRAPH,
        fill=1,
        stack=False,
        grid_pos=GridPos(0, 21, 8, 8),
    ),
    Panel(
        id=2009,
        title="Router Request QPS",
        description="Requests per second processed by the Router. This metric is captured at the Router level - after the handle, during request assignment. The router manages request queuing and replica selection.",
        unit="reqps",
        targets=[
            Target(
                expr='sum(rate(ray_serve_num_router_requests_total{{application=~"$Application", deployment=~"$Deployment",{global_filters}}}[5m])) by (application, deployment)',
                legend="{{application}} {{deployment}}",
            )
        ],
        template=PanelTemplate.GRAPH,
        fill=1,
        stack=False,
        grid_pos=GridPos(8, 21, 8, 8),
    ),
    Panel(
        id=2010,
        title="Replica Request QPS",
        description="Requests per second processed by deployment replicas. This metric is captured at the Replica level - the final destination where user code executes.",
        unit="reqps",
        targets=[
            Target(
                expr='sum(rate(ray_serve_deployment_request_counter_total{{application=~"$Application", deployment=~"$Deployment", replica=~"$Replica",{global_filters}}}[5m])) by (application, deployment)',
                legend="{{application}} {{deployment}}",
            )
        ],
        template=PanelTemplate.GRAPH,
        fill=1,
        stack=False,
        grid_pos=GridPos(16, 21, 8, 8),
    ),
    # Row 5: Error QPS
    Panel(
        id=2011,
        title="Error QPS per Deployment",
        description="Number of exceptions that occurred in the deployment replicas.",
        unit="reqps",
        targets=[
            Target(
                expr='sum(rate(ray_serve_deployment_error_counter_total{{application=~"$Application", deployment=~"$Deployment", replica=~"$Replica",{global_filters}}}[5m])) by (application, deployment)',
                legend="{{application}} {{deployment}}",
            )
        ],
        template=PanelTemplate.GRAPH,
        fill=1,
        stack=False,
        grid_pos=GridPos(0, 29, 24, 8),
    ),
    # Row 6: Deployment-Level Latency
    Panel(
        id=10,  # Referenced by Ray Dashboard frontend (ServeMetricsSection.tsx)
        title="Processing Latency per Deployment (P50, P90, P99)",
        description="Processing latency percentiles at replicas. Time spent strictly inside user code (model inference, business logic). Excludes queue wait time and routing overhead. Diagnostic: If this is high, your model/logic is the bottleneck. If this is low but 'E2E Latency' is high, the bottleneck is in Serve's routing or internal queuing. Gap between percentiles indicates processing variance.",
        unit="ms",
        targets=[
            Target(
                expr='histogram_quantile(0.5, sum(rate(ray_serve_deployment_processing_latency_ms_bucket{{application=~"$Application", deployment=~"$Deployment", replica=~"$Replica",{global_filters}}}[5m])) by (application, deployment, le))',
                legend="P50: {{application}} {{deployment}}",
            ),
            Target(
                expr='histogram_quantile(0.9, sum(rate(ray_serve_deployment_processing_latency_ms_bucket{{application=~"$Application", deployment=~"$Deployment", replica=~"$Replica",{global_filters}}}[5m])) by (application, deployment, le))',
                legend="P90: {{application}} {{deployment}}",
            ),
            Target(
                expr='histogram_quantile(0.99, sum(rate(ray_serve_deployment_processing_latency_ms_bucket{{application=~"$Application", deployment=~"$Deployment", replica=~"$Replica",{global_filters}}}[5m])) by (application, deployment, le))',
                legend="P99: {{application}} {{deployment}}",
            ),
        ],
        template=PanelTemplate.GRAPH,
        fill=0,
        stack=False,
        grid_pos=GridPos(0, 37, 24, 8),
    ),
    # Row 7: Router Fulfillment Time
    Panel(
        id=2015,
        title="Router Fulfillment Latency Per Deployment",
        description="Time requests spent waiting in router queue before being assigned to a replica. This includes the time to resolve the pending request's arguments. Gap between percentiles indicates routing variance.",
        unit="ms",
        targets=[
            Target(
                expr='histogram_quantile(0.5, sum(rate(ray_serve_request_router_fulfillment_time_ms_bucket{{application=~"$Application", deployment=~"$Deployment",{global_filters}}}[5m])) by (application, deployment, le))',
                legend="P50: {{application}} {{deployment}}",
            ),
            Target(
                expr='histogram_quantile(0.9, sum(rate(ray_serve_request_router_fulfillment_time_ms_bucket{{application=~"$Application", deployment=~"$Deployment",{global_filters}}}[5m])) by (application, deployment, le))',
                legend="P90: {{application}} {{deployment}}",
            ),
            Target(
                expr='histogram_quantile(0.99, sum(rate(ray_serve_request_router_fulfillment_time_ms_bucket{{application=~"$Application", deployment=~"$Deployment",{global_filters}}}[5m])) by (application, deployment, le))',
                legend="P99: {{application}} {{deployment}}",
            ),
        ],
        template=PanelTemplate.GRAPH,
        fill=0,
        stack=False,
        grid_pos=GridPos(0, 45, 24, 8),
    ),
    # Row 8: Queue & Saturation
    Panel(
        id=2018,
        title="Queued Requests at Router",
        description="Number of requests waiting in queue at the router. This is the router's view of how many requests are waiting to be assigned to a replica.",
        unit="requests",
        targets=[
            Target(
                expr='sum(ray_serve_deployment_queued_queries{{application=~"$Application", deployment=~"$Deployment",{global_filters}}}) by (application, deployment)',
                legend="{{application}} {{deployment}}",
            )
        ],
        template=PanelTemplate.GRAPH,
        fill=1,
        stack=False,
        grid_pos=GridPos(0, 53, 8, 8),
    ),
    Panel(
        id=2019,
        title="Assigned Requests (Router View)",
        description="Number of requests sent to replicas, tracked at the Router. This is the router's view of how many requests running at replicas.",
        unit="requests",
        targets=[
            Target(
                expr='sum(ray_serve_num_ongoing_requests_at_replicas{{application=~"$Application", deployment=~"$Deployment",{global_filters}}}) by (application, deployment)',
                legend="{{application}} {{deployment}}",
            )
        ],
        template=PanelTemplate.GRAPH,
        fill=1,
        stack=False,
        grid_pos=GridPos(8, 53, 8, 8),
    ),
    Panel(
        id=2020,
        title="Processing Requests (Replica View)",
        description="Number of requests currently being processed, tracked at the Replica. This is the replica's actual count of ongoing requests it is handling.",
        unit="requests",
        targets=[
            Target(
                expr='sum(ray_serve_replica_processing_queries{{application=~"$Application", deployment=~"$Deployment", replica=~"$Replica",{global_filters}}}) by (application, deployment)',
                legend="{{application}} {{deployment}}",
            )
        ],
        template=PanelTemplate.GRAPH,
        fill=1,
        stack=False,
        grid_pos=GridPos(16, 53, 8, 8),
    ),
    # Row 9: Scheduling and Load Balance Quality
    Panel(
        id=22,  # Referenced by Ray Dashboard frontend (ServeMetricsSection.tsx)
        title="Scheduling Tasks",
        description="Number of concurrent scheduling tasks in the router. Each task is responsible for assigning a pending request to an available replica. High values indicate the router is working to assign many requests.",
        unit="tasks",
        targets=[
            Target(
                expr='sum(ray_serve_num_scheduling_tasks{{application=~"$Application", deployment=~"$Deployment",{global_filters}}}) by (application, deployment)',
                legend="{{application}} {{deployment}}",
            )
        ],
        template=PanelTemplate.GRAPH,
        fill=1,
        stack=False,
        grid_pos=GridPos(0, 61, 8, 8),
    ),
    Panel(
        id=23,  # Referenced by Ray Dashboard frontend (ServeMetricsSection.tsx)
        title="Scheduling Tasks in Backoff",
        description="Number of scheduling tasks currently in exponential backoff. Tasks enter backoff when no replicas are available or all replicas are at capacity. High values indicate insufficient replica capacity.",
        unit="tasks",
        targets=[
            Target(
                expr='sum(ray_serve_num_scheduling_tasks_in_backoff{{application=~"$Application", deployment=~"$Deployment",{global_filters}}}) by (application, deployment)',
                legend="{{application}} {{deployment}}",
            )
        ],
        template=PanelTemplate.GRAPH,
        fill=1,
        stack=False,
        grid_pos=GridPos(8, 61, 8, 8),
    ),
    Panel(
        id=2023,
        title="Load Balance Quality (Max/Avg)",
        description="Max/Avg request rate ratio per deployment. Close to 1 = balanced load across replicas. Much > 1 = some replicas get more traffic (hot replica, stickiness, etc.)",
        unit="short",
        targets=[
            Target(
                expr='max(sum(rate(ray_serve_deployment_request_counter_total{{application=~"$Application", deployment=~"$Deployment", replica=~"$Replica",{global_filters}}}[1m])) by (application, deployment, replica)) by (application, deployment) / clamp_min(avg(sum(rate(ray_serve_deployment_request_counter_total{{application=~"$Application", deployment=~"$Deployment", replica=~"$Replica",{global_filters}}}[1m])) by (application, deployment, replica)) by (application, deployment), 0.001)',
                legend="{{application}} {{deployment}}",
            )
        ],
        template=PanelTemplate.GRAPH,
        fill=0,
        stack=False,
        grid_pos=GridPos(16, 61, 8, 8),
    ),
    # Row 10: Request Distribution
    Panel(
        id=2025,
        title="Request Rate Distribution Across Replicas",
        description="Request rate (req/s) for each replica over 1m window. Each line represents a replica. Helps identify hot replicas or uneven load distribution. Recommend filtering for a specific deployment to see the request rate distribution for that deployment.",
        unit="reqps",
        targets=[
            Target(
                expr='sum(rate(ray_serve_deployment_request_counter_total{{application=~"$Application", deployment=~"$Deployment", replica=~"$Replica",{global_filters}}}[1m])) by (application, deployment, replica) > 0',
                legend="{{replica}}",
            ),
        ],
        template=PanelTemplate.GRAPH,
        fill=0,
        stack=False,
        grid_pos=GridPos(0, 69, 24, 8),
    ),
]

# ==============================================================================
# AUTOSCALING & CAPACITY SECTION
# ==============================================================================

# User-facing autoscaling metrics (Row 2: Target vs Actual, Row 3: Autoscaling Inputs)
AUTOSCALING_USER_PANELS = [
    # Row 1: Target vs Actual Replicas
    Panel(
        id=3005,
        title="Target vs Actual Replicas Over Time",
        description="Desired replicas (raw autoscaling decision), Target replicas (after min/max bounds), and Actual healthy replicas over time. Gap between Desired and Target indicates min/max constraints. Gap between Target and Actual indicates autoscaling lag.",
        unit="replicas",
        targets=[
            Target(
                expr='ray_serve_autoscaling_desired_replicas{{application=~"$Application", deployment=~"$Deployment",{global_filters}}}',
                legend="Desired: {{application}} {{deployment}}",
            ),
            Target(
                expr='ray_serve_autoscaling_target_replicas{{application=~"$Application", deployment=~"$Deployment",{global_filters}}}',
                legend="Target: {{application}} {{deployment}}",
            ),
            Target(
                expr='sum(ray_serve_deployment_replica_healthy{{application=~"$Application", deployment=~"$Deployment",{global_filters}}} == 1) by (application, deployment)',
                legend="Actual: {{application}} {{deployment}}",
            ),
        ],
        template=PanelTemplate.GRAPH,
        fill=0,
        stack=False,
        grid_pos=GridPos(0, 0, 24, 8),
    ),
    # Row 2: Autoscaling Inputs
    Panel(
        id=3006,
        title="Total Requests (Autoscaler View)",
        description="Total number of requests (queued + in-flight) as seen by the autoscaler. This is the primary input to the autoscaling decision.",
        unit="requests",
        targets=[
            Target(
                expr='ray_serve_autoscaling_total_requests{{application=~"$Application", deployment=~"$Deployment",{global_filters}}}',
                legend="{{application}} {{deployment}}",
            )
        ],
        template=PanelTemplate.GRAPH,
        fill=1,
        stack=False,
        grid_pos=GridPos(0, 8, 12, 8),
    ),
    Panel(
        id=3007,
        title="Blocked by Min/Max Limits",
        description="Shows when autoscaling is constrained by replica limits. Positive = wants to scale up but hitting max_replicas. Negative = wants to scale down but hitting min_replicas. Zero = operating within bounds.",
        unit="replicas",
        targets=[
            Target(
                expr='ray_serve_autoscaling_desired_replicas{{application=~"$Application", deployment=~"$Deployment",{global_filters}}} - ray_serve_autoscaling_target_replicas{{application=~"$Application", deployment=~"$Deployment",{global_filters}}}',
                legend="Delta: {{application}} {{deployment}}",
            ),
        ],
        template=PanelTemplate.GRAPH,
        fill=0,
        stack=False,
        grid_pos=GridPos(12, 8, 12, 8),
    ),
]

# System metrics (Row 1: Status Stats, Row 4+: Metrics Pipeline Health and beyond)
AUTOSCALING_SYSTEM_PANELS = [
    # Row 1: Status Stats
    Panel(
        id=3002,
        title="Replica Startup (P99)",
        description="99th percentile time from replica creation to ready state. Includes node provisioning (if needed), runtime environment bootstrap (pip install, Docker pull, etc.), Ray actor scheduling, and actor constructor execution. Useful for debugging slow cold starts.",
        unit="ms",
        targets=[
            Target(
                expr='histogram_quantile(0.99, sum(ray_serve_replica_startup_latency_ms_bucket{{application=~"$Application", deployment=~"$Deployment",{global_filters}}}) by (le)) or vector(0)',
                legend="P99 Startup",
            )
        ],
        template=PanelTemplate.STAT,
        grid_pos=GridPos(0, 0, 6, 3),
        thresholds=[
            {"color": "green", "value": None},
            {"color": "yellow", "value": 10000},
            {"color": "red", "value": 30000},
        ],
    ),
    Panel(
        id=3003,
        title="Health Check (P99)",
        description="99th percentile health check duration. High values may indicate replicas are overloaded or have slow check_health() methods.",
        unit="ms",
        targets=[
            Target(
                expr='histogram_quantile(0.99, sum(rate(ray_serve_health_check_latency_ms_bucket{{application=~"$Application", deployment=~"$Deployment",{global_filters}}}[5m])) by (le)) or vector(0)',
                legend="P99 Health Check",
            )
        ],
        template=PanelTemplate.STAT,
        grid_pos=GridPos(6, 0, 6, 3),
        thresholds=[
            {"color": "green", "value": None},
            {"color": "yellow", "value": 100},
            {"color": "red", "value": 500},
        ],
    ),
    Panel(
        id=3001,
        title="Metrics Delay",
        description="Maximum delay for autoscaling metrics (from replicas and handles) to reach the controller. High delay means autoscaler is working with stale data.",
        unit="ms",
        targets=[
            Target(
                expr='max(avg(ray_serve_autoscaling_handle_metrics_delay_ms{{application=~"$Application", deployment=~"$Deployment",{global_filters}}}) or avg(ray_serve_autoscaling_replica_metrics_delay_ms{{application=~"$Application", deployment=~"$Deployment",{global_filters}}}))',
                legend="Metrics Delay",
            )
        ],
        template=PanelTemplate.STAT,
        grid_pos=GridPos(12, 0, 6, 3),
        thresholds=[
            {"color": "green", "value": None},
            {"color": "yellow", "value": 1000},
            {"color": "red", "value": 5000},
        ],
    ),
    Panel(
        id=3004,
        title="Long Poll (P99)",
        description="99th percentile long-poll latency. Long-poll is used to broadcast config updates from controller to routers/proxies. High latency delays autoscaling responsiveness.",
        unit="ms",
        targets=[
            Target(
                expr="histogram_quantile(0.99, sum(rate(ray_serve_long_poll_latency_ms_bucket{{{global_filters}}}[5m])) by (le)) or vector(0)",
                legend="P99 Long Poll",
            )
        ],
        template=PanelTemplate.STAT,
        grid_pos=GridPos(18, 0, 6, 3),
        thresholds=[
            {"color": "green", "value": None},
            {"color": "yellow", "value": 100},
            {"color": "red", "value": 500},
        ],
    ),
    # Row 2: Replica Startup/Shutdown/Reconfigure
    Panel(
        id=3018,
        title="Replica Startup Breakdown",
        description="Total replica startup time vs user initialization time. The gap between the lines represents system overhead (node provisioning, runtime environment bootstrap, Ray actor scheduling). User initialization is the time spent in __init__(), while total startup includes all steps from replica creation to ready state.",
        unit="ms",
        targets=[
            Target(
                expr='histogram_quantile(0.5, sum(ray_serve_replica_startup_latency_ms_bucket{{application=~"$Application", deployment=~"$Deployment",{global_filters}}}) by (application, deployment, le))',
                legend="P50 Total Startup: {{application}} {{deployment}}",
            ),
            Target(
                expr='histogram_quantile(0.5, sum(ray_serve_replica_initialization_latency_ms_bucket{{application=~"$Application", deployment=~"$Deployment",{global_filters}}}) by (application, deployment, le))',
                legend="P50 User Init: {{application}} {{deployment}}",
            ),
            Target(
                expr='histogram_quantile(0.99, sum(ray_serve_replica_startup_latency_ms_bucket{{application=~"$Application", deployment=~"$Deployment",{global_filters}}}) by (application, deployment, le))',
                legend="P99 Total Startup: {{application}} {{deployment}}",
            ),
            Target(
                expr='histogram_quantile(0.99, sum(ray_serve_replica_initialization_latency_ms_bucket{{application=~"$Application", deployment=~"$Deployment",{global_filters}}}) by (application, deployment, le))',
                legend="P99 User Init: {{application}} {{deployment}}",
            ),
        ],
        template=PanelTemplate.GRAPH,
        fill=0,
        stack=False,
        grid_pos=GridPos(0, 3, 12, 8),
    ),
    Panel(
        id=3020,
        title="Replica Reconfigure Latency",
        description="Time for a replica to complete reconfiguration. Includes both reconfigure time and one control-loop iteration, so very low values may be unreliable. Triggered by user_config updates.",
        unit="ms",
        targets=[
            Target(
                expr='histogram_quantile(0.5, sum(ray_serve_replica_reconfigure_latency_ms_bucket{{application=~"$Application", deployment=~"$Deployment",{global_filters}}}) by (application, deployment, le))',
                legend="P50: {{application}} {{deployment}}",
            ),
            Target(
                expr='histogram_quantile(0.99, sum(ray_serve_replica_reconfigure_latency_ms_bucket{{application=~"$Application", deployment=~"$Deployment",{global_filters}}}) by (application, deployment, le))',
                legend="P99: {{application}} {{deployment}}",
            ),
        ],
        template=PanelTemplate.GRAPH,
        fill=0,
        stack=False,
        grid_pos=GridPos(12, 3, 6, 8),
    ),
    Panel(
        id=3021,
        title="Replica Shutdown Duration",
        description="Time from shutdown signal to replica fully stopped. Useful for debugging slow request draining during scale-down or rolling updates. High values indicate slow graceful shutdown or long-running requests.",
        unit="ms",
        targets=[
            Target(
                expr='histogram_quantile(0.5, sum(ray_serve_replica_shutdown_duration_ms_bucket{{application=~"$Application", deployment=~"$Deployment",{global_filters}}}) by (application, deployment, le))',
                legend="P50: {{application}} {{deployment}}",
            ),
            Target(
                expr='histogram_quantile(0.99, sum(ray_serve_replica_shutdown_duration_ms_bucket{{application=~"$Application", deployment=~"$Deployment",{global_filters}}}) by (application, deployment, le))',
                legend="P99: {{application}} {{deployment}}",
            ),
        ],
        template=PanelTemplate.GRAPH,
        fill=0,
        stack=False,
        grid_pos=GridPos(18, 3, 6, 8),
    ),
    # Row 3: Replica Health
    Panel(
        id=3022,
        title="Health Check Latency",
        description="Duration of health check calls",
        unit="ms",
        targets=[
            Target(
                expr='histogram_quantile(0.5, sum(rate(ray_serve_health_check_latency_ms_bucket{{application=~"$Application", deployment=~"$Deployment",{global_filters}}}[5m])) by (application, deployment, le))',
                legend="P50: {{application}} {{deployment}}",
            ),
            Target(
                expr='histogram_quantile(0.99, sum(rate(ray_serve_health_check_latency_ms_bucket{{application=~"$Application", deployment=~"$Deployment",{global_filters}}}[5m])) by (application, deployment, le))',
                legend="P99: {{application}} {{deployment}}",
            ),
        ],
        template=PanelTemplate.GRAPH,
        fill=0,
        stack=False,
        grid_pos=GridPos(0, 11, 12, 8),
    ),
    Panel(
        id=3023,
        title="Health Check Failures",
        description="Count of failed health checks - early warning before replica marked unhealthy",
        unit="failures/s",
        targets=[
            Target(
                expr='sum(rate(ray_serve_health_check_failures_total{{application=~"$Application", deployment=~"$Deployment",{global_filters}}}[5m])) by (application, deployment)',
                legend="{{application}} {{deployment}}",
            )
        ],
        template=PanelTemplate.GRAPH,
        fill=1,
        stack=False,
        grid_pos=GridPos(12, 11, 12, 8),
    ),
    # Row 4: Replica Restarts + Proxy Status
    Panel(
        id=3024,
        title="Replica Restarts",
        description="Rate of replica restarts",
        unit="restarts/s",
        targets=[
            Target(
                expr='sum(rate(ray_serve_deployment_replica_starts_total{{application=~"$Application", deployment=~"$Deployment",{global_filters}}}[5m])) by (application, deployment)',
                legend="{{application}} {{deployment}}",
            )
        ],
        template=PanelTemplate.GRAPH,
        fill=1,
        stack=False,
        grid_pos=GridPos(0, 19, 8, 8),
    ),
    # Proxy Status Counts
    Panel(
        id=3025,
        title="Healthy Proxies",
        description="Number of proxies in HEALTHY state",
        unit="proxies",
        targets=[
            Target(
                expr="count(ray_serve_proxy_status{{{global_filters}}} == 2) or vector(0)",
                legend="Healthy",
            )
        ],
        template=PanelTemplate.STAT,
        grid_pos=GridPos(8, 19, 4, 4),
        thresholds=[{"color": "green", "value": None}],
    ),
    Panel(
        id=3026,
        title="Starting",
        description="Proxies starting up",
        unit="proxies",
        targets=[
            Target(
                expr="count(ray_serve_proxy_status{{{global_filters}}} == 1) or vector(0)",
                legend="Starting",
            )
        ],
        template=PanelTemplate.STAT,
        grid_pos=GridPos(12, 19, 2, 2),
        thresholds=[{"color": "blue", "value": None}],
    ),
    Panel(
        id=3027,
        title="Unhealthy",
        description="Proxies in unhealthy state",
        unit="proxies",
        targets=[
            Target(
                expr="count(ray_serve_proxy_status{{{global_filters}}} == 3) or vector(0)",
                legend="Unhealthy",
            )
        ],
        template=PanelTemplate.STAT,
        grid_pos=GridPos(14, 19, 2, 2),
        thresholds=[
            {"color": "green", "value": None},
            {"color": "red", "value": 1},
        ],
    ),
    Panel(
        id=3028,
        title="Draining",
        description="Proxies draining connections",
        unit="proxies",
        targets=[
            Target(
                expr="count(ray_serve_proxy_status{{{global_filters}}} == 4) or vector(0)",
                legend="Draining",
            )
        ],
        template=PanelTemplate.STAT,
        grid_pos=GridPos(12, 21, 2, 2),
        thresholds=[
            {"color": "green", "value": None},
            {"color": "yellow", "value": 1},
        ],
    ),
    Panel(
        id=3029,
        title="Drained",
        description="Proxies fully drained",
        unit="proxies",
        targets=[
            Target(
                expr="count(ray_serve_proxy_status{{{global_filters}}} == 5) or vector(0)",
                legend="Drained",
            )
        ],
        template=PanelTemplate.STAT,
        grid_pos=GridPos(14, 21, 2, 2),
        thresholds=[
            {"color": "green", "value": None},
            {"color": "orange", "value": 1},
        ],
    ),
    Panel(
        id=3030,
        title="Proxy Status Over Time",
        description="Count of proxies in each state over time",
        unit="proxies",
        targets=[
            Target(
                expr="count(ray_serve_proxy_status{{{global_filters}}} == 2) or vector(0)",
                legend="HEALTHY",
            ),
            Target(
                expr="count(ray_serve_proxy_status{{{global_filters}}} == 1) or vector(0)",
                legend="STARTING",
            ),
            Target(
                expr="count(ray_serve_proxy_status{{{global_filters}}} == 3) or vector(0)",
                legend="UNHEALTHY",
            ),
            Target(
                expr="count(ray_serve_proxy_status{{{global_filters}}} == 4) or vector(0)",
                legend="DRAINING",
            ),
            Target(
                expr="count(ray_serve_proxy_status{{{global_filters}}} == 5) or vector(0)",
                legend="DRAINED",
            ),
        ],
        template=PanelTemplate.GRAPH,
        fill=0,
        stack=False,
        grid_pos=GridPos(16, 19, 8, 8),
    ),
    # Row 5: Autoscaling Performance Metrics
    Panel(
        id=3015,
        title="Autoscaling Policy Execution Time",
        description="Time taken to execute the autoscaling policy in milliseconds. policy_scope is either 'deployment' or 'application'. High values indicate the autoscaling policy itself is slow, which can delay scaling decisions.",
        unit="ms",
        targets=[
            Target(
                expr='ray_serve_autoscaling_policy_execution_time_ms{{application=~"$Application", deployment=~"$Deployment",{global_filters}}}',
                legend="{{application}} {{deployment}} ({{policy_scope}})",
            ),
        ],
        template=PanelTemplate.GRAPH,
        fill=1,
        stack=False,
        grid_pos=GridPos(0, 27, 8, 8),
    ),
    Panel(
        id=3016,
        title="User Autoscaling Stats Latency",
        description="Time taken to execute user-defined autoscaling stats function in milliseconds. Only applicable when using custom autoscaling metrics. High values indicate the user's custom function is slow.",
        unit="ms",
        targets=[
            Target(
                expr='histogram_quantile(0.5, sum(rate(ray_serve_user_autoscaling_stats_latency_ms_bucket{{application=~"$Application", deployment=~"$Deployment",{global_filters}}}[5m])) by (application, deployment, replica, le))',
                legend="P50: {{application}} {{deployment}}",
            ),
            Target(
                expr='histogram_quantile(0.99, sum(rate(ray_serve_user_autoscaling_stats_latency_ms_bucket{{application=~"$Application", deployment=~"$Deployment",{global_filters}}}[5m])) by (application, deployment, replica, le))',
                legend="P99: {{application}} {{deployment}}",
            ),
        ],
        template=PanelTemplate.GRAPH,
        fill=0,
        stack=False,
        grid_pos=GridPos(8, 27, 8, 8),
    ),
    Panel(
        id=3017,
        title="Autoscaling Stats Collection Failures",
        description="Rate of failed attempts to collect autoscaling metrics from user-defined functions. Only applicable when using custom autoscaling metrics. Non-zero values indicate errors in user code - check replica logs for details.",
        unit="errors/s",
        targets=[
            Target(
                expr='sum(rate(ray_serve_record_autoscaling_stats_failed_total{{application=~"$Application", deployment=~"$Deployment",{global_filters}}}[5m])) by (application, deployment, exception_name)',
                legend="{{application}} {{deployment}}: {{exception_name}}",
            )
        ],
        template=PanelTemplate.GRAPH,
        fill=1,
        stack=False,
        grid_pos=GridPos(16, 27, 8, 8),
    ),
    # Row 6: Metrics Pipeline Health
    Panel(
        id=3008,
        title="Replica Metrics Delay",
        description="Time taken for replica metrics to be reported to controller",
        unit="ms",
        targets=[
            Target(
                expr='ray_serve_autoscaling_replica_metrics_delay_ms{{application=~"$Application", deployment=~"$Deployment",{global_filters}}}',
                legend="{{application}} {{deployment}}",
            )
        ],
        template=PanelTemplate.GRAPH,
        fill=1,
        stack=False,
        grid_pos=GridPos(0, 35, 12, 8),
    ),
    Panel(
        id=3009,
        title="Handle Metrics Delay",
        description="Time taken for handle metrics to be reported to controller",
        unit="ms",
        targets=[
            Target(
                expr='ray_serve_autoscaling_handle_metrics_delay_ms{{application=~"$Application", deployment=~"$Deployment",{global_filters}}}',
                legend="{{application}} {{deployment}}",
            )
        ],
        template=PanelTemplate.GRAPH,
        fill=1,
        stack=False,
        grid_pos=GridPos(12, 35, 12, 8),
    ),
    # Row 7: Routing Stats Health
    Panel(
        id=3010,
        title="Routing Stats Delay",
        description="Time for routing statistics to propagate from replicas to the controller. High values indicate network latency or controller backpressure, which can delay autoscaling decisions.",
        unit="ms",
        targets=[
            Target(
                expr='histogram_quantile(0.5, sum(rate(ray_serve_routing_stats_delay_ms_bucket{{application=~"$Application", deployment=~"$Deployment",{global_filters}}}[5m])) by (application, deployment, le))',
                legend="P50: {{application}} {{deployment}}",
            ),
            Target(
                expr='histogram_quantile(0.99, sum(rate(ray_serve_routing_stats_delay_ms_bucket{{application=~"$Application", deployment=~"$Deployment",{global_filters}}}[5m])) by (application, deployment, le))',
                legend="P99: {{application}} {{deployment}}",
            ),
        ],
        template=PanelTemplate.GRAPH,
        fill=0,
        stack=False,
        grid_pos=GridPos(0, 43, 12, 8),
    ),
    Panel(
        id=3011,
        title="Routing Stats Errors",
        description="Rate of errors when getting routing stats from replicas. error_type is 'exception' (replica raised an error) or 'timeout' (replica didn't respond in time). Non-zero values indicate communication issues between controller and replicas.",
        unit="errors/s",
        targets=[
            Target(
                expr='sum(rate(ray_serve_routing_stats_error_total{{application=~"$Application", deployment=~"$Deployment",{global_filters}}}[5m])) by (application, deployment, error_type)',
                legend="{{application}} {{deployment}}: {{error_type}}",
            )
        ],
        template=PanelTemplate.GRAPH,
        fill=1,
        stack=False,
        grid_pos=GridPos(12, 43, 12, 8),
    ),
    # Row 8: Controller Health - Control Loops
    Panel(
        id=24,  # Referenced by Ray Dashboard frontend (ServeMetricsSection.tsx) as "Controller Control Loop Duration"
        title="Control Loop Duration",
        description="Duration of the last control loop iteration in seconds. The control loop manages deployment state, replica health checks, and autoscaling decisions",
        unit="s",
        targets=[
            Target(
                expr="ray_serve_controller_control_loop_duration_s{{{global_filters}}}",
                legend="Control Loop Duration",
            )
        ],
        template=PanelTemplate.GRAPH,
        fill=1,
        stack=False,
        grid_pos=GridPos(0, 51, 12, 8),
    ),
    Panel(
        id=25,  # Referenced by Ray Dashboard frontend (ServeMetricsSection.tsx) as "Number of Control Loops"
        title="Control Loop Rate",
        description="Rate of control loops per second, grouped by controller actor ID. If the controller restarts, a new line will appear for the new actor instance.",
        unit="loops/s",
        targets=[
            Target(
                expr="sum(rate(ray_serve_controller_num_control_loops{{{global_filters}}}[1m])) by (actor_id)",
                legend="{{actor_id}}",
            )
        ],
        template=PanelTemplate.GRAPH,
        fill=1,
        stack=False,
        grid_pos=GridPos(12, 51, 12, 8),
    ),
    # Row 9: Long Poll Mechanism
    Panel(
        id=3012,
        title="Long Poll Latency",
        description="Latency of the long-poll mechanism used by Serve to broadcast configuration updates (replica lists, deployment configs) from the controller to routers and proxies. High latency delays how quickly routers learn about new replicas, impacting autoscaling responsiveness and request routing.",
        unit="ms",
        targets=[
            Target(
                expr="histogram_quantile(0.5, sum(rate(ray_serve_long_poll_latency_ms_bucket{{{global_filters}}}[5m])) by (le))",
                legend="P50",
            ),
            Target(
                expr="histogram_quantile(0.99, sum(rate(ray_serve_long_poll_latency_ms_bucket{{{global_filters}}}[5m])) by (le))",
                legend="P99",
            ),
        ],
        template=PanelTemplate.GRAPH,
        fill=0,
        stack=False,
        grid_pos=GridPos(0, 59, 8, 8),
    ),
    Panel(
        id=3013,
        title="Long Poll Pending Clients",
        description="Number of clients (routers, proxies) waiting for long-poll updates from the controller, grouped by namespace. High values indicate many components are waiting for configuration updates, which may signal controller backpressure.",
        unit="clients",
        targets=[
            Target(
                expr="sum(ray_serve_long_poll_pending_clients{{{global_filters}}}) by (namespace)",
                legend="{{namespace}}",
            )
        ],
        template=PanelTemplate.GRAPH,
        fill=1,
        stack=False,
        grid_pos=GridPos(8, 59, 8, 8),
    ),
    Panel(
        id=3014,
        title="Long Poll Transmissions",
        description="Rate of long-poll updates transmitted from the controller to clients, grouped by namespace or state. Shows the frequency of configuration updates being broadcast. High rates may indicate frequent reconfigurations or state changes.",
        unit="updates/s",
        targets=[
            Target(
                expr="sum(rate(ray_serve_long_poll_host_transmission_counter_total{{{global_filters}}}[5m])) by (namespace_or_state)",
                legend="{{namespace_or_state}}",
            )
        ],
        template=PanelTemplate.GRAPH,
        fill=1,
        stack=False,
        grid_pos=GridPos(16, 59, 8, 8),
    ),
    # Row 10: Event Loop Monitoring
    Panel(
        id=3036,
        title="Event Loop Scheduling Latency (P99)",
        description="P99 event loop scheduling delay. Measures how long the event loop was blocked beyond the expected sleep interval. <10ms=healthy, 10-50ms=acceptable under load, 50-100ms=concerning, >100ms=problematic (likely blocking I/O or CPU-bound code in async handlers)",
        unit="ms",
        targets=[
            Target(
                expr='histogram_quantile(0.99, sum(rate(ray_serve_event_loop_scheduling_latency_ms_bucket{{application=~"$Application", deployment=~"$Deployment",{global_filters}}}[5m])) by (le)) or vector(0)',
                legend="P99 Scheduling Latency",
            )
        ],
        template=PanelTemplate.STAT,
        grid_pos=GridPos(0, 67, 3, 8),
        thresholds=[
            {"color": "green", "value": None},
            {"color": "yellow", "value": 10},
            {"color": "orange", "value": 50},
            {"color": "red", "value": 100},
        ],
    ),
    Panel(
        id=3037,
        title="Event Loop Tasks",
        description="Maximum number of pending asyncio tasks across all event loops. High values may indicate task accumulation or event loop congestion",
        unit="tasks",
        targets=[
            Target(
                expr='max(ray_serve_event_loop_tasks{{application=~"$Application", deployment=~"$Deployment",{global_filters}}})',
                legend="Max Tasks",
            )
        ],
        template=PanelTemplate.STAT,
        grid_pos=GridPos(3, 67, 3, 8),
        thresholds=[{"color": "blue", "value": None}],
    ),
    Panel(
        id=3038,
        title="Event Loop Scheduling Latency by Component",
        description="Event loop scheduling delay by component and loop type. High values indicate blocking code or event loop overload.",
        unit="ms",
        targets=[
            Target(
                expr='histogram_quantile(0.5, sum(rate(ray_serve_event_loop_scheduling_latency_ms_bucket{{application=~"$Application", deployment=~"$Deployment",{global_filters}}}[5m])) by (component, loop_type, deployment, application, le))',
                legend="P50: {{component}}-{{loop_type}} {{application}} {{deployment}}",
            ),
            Target(
                expr='histogram_quantile(0.99, sum(rate(ray_serve_event_loop_scheduling_latency_ms_bucket{{application=~"$Application", deployment=~"$Deployment",{global_filters}}}[5m])) by (component, loop_type, deployment, application, le))',
                legend="P99: {{component}}-{{loop_type}} {{application}} {{deployment}}",
            ),
        ],
        template=PanelTemplate.GRAPH,
        fill=0,
        stack=False,
        grid_pos=GridPos(6, 67, 6, 8),
    ),
    Panel(
        id=3039,
        title="Event Loop Monitoring Heartbeat",
        description="Event loop monitoring iterations per second (heartbeat) summed across replicas. A drop to zero indicates all loops of this type are blocked.",
        unit="iterations/s",
        targets=[
            Target(
                expr='sum(rate(ray_serve_event_loop_monitoring_iterations_total{{application=~"$Application", deployment=~"$Deployment",{global_filters}}}[5m])) by (component, loop_type, application, deployment)',
                legend="{{component}}-{{loop_type}} {{application}} {{deployment}}",
            )
        ],
        template=PanelTemplate.GRAPH,
        fill=1,
        stack=False,
        grid_pos=GridPos(12, 67, 6, 8),
    ),
    Panel(
        id=3040,
        title="Event Loop Tasks",
        description="Number of pending asyncio tasks summed across replicas. High/growing values indicate task accumulation or event loop congestion.",
        unit="tasks",
        targets=[
            Target(
                expr='sum(ray_serve_event_loop_tasks{{application=~"$Application", deployment=~"$Deployment",{global_filters}}}) by (component, loop_type, application, deployment)',
                legend="{{component}}-{{loop_type}} {{application}} {{deployment}}",
            )
        ],
        template=PanelTemplate.GRAPH,
        fill=1,
        stack=False,
        grid_pos=GridPos(18, 67, 6, 8),
    ),
]

# ==============================================================================
# REQUEST BATCHING SECTION
# ==============================================================================

BATCHING_PANELS = [
    # Row 1: Batching Overview Stats
    Panel(
        id=4001,
        title="Median Batch Size",
        description="Median (P50) computed batch size (number of requests, or custom size if batch_size_fn is configured)",
        unit="short",
        targets=[
            Target(
                expr='histogram_quantile(0.5, sum(rate(ray_serve_actual_batch_size_bucket{{application=~"$Application", deployment=~"$Deployment",{global_filters}}}[5m])) by (le)) or vector(0)',
                legend="Median Batch Size",
            )
        ],
        template=PanelTemplate.STAT,
        grid_pos=GridPos(0, 0, 4, 3),
        thresholds=[{"color": "blue", "value": None}],
    ),
    Panel(
        id=4002,
        title="Batch Utilization %",
        description="Median (P50) batch utilization: actual_batch_size / max_batch_size * 100",
        unit="percent",
        targets=[
            Target(
                expr='histogram_quantile(0.5, sum(rate(ray_serve_batch_utilization_percent_bucket{{application=~"$Application", deployment=~"$Deployment",{global_filters}}}[5m])) by (le)) or vector(0)',
                legend="Batch Utilization",
            )
        ],
        template=PanelTemplate.STAT,
        grid_pos=GridPos(4, 0, 4, 3),
        thresholds=[{"color": "blue", "value": None}],
    ),
    Panel(
        id=4003,
        title="Batch Wait Time (P50)",
        description="Median (P50) time requests waited in the batch queue before the batch was formed and sent for processing. High values indicate batch timeout (batch_wait_timeout_s) may be too long",
        unit="ms",
        targets=[
            Target(
                expr='histogram_quantile(0.5, sum(rate(ray_serve_batch_wait_time_ms_bucket{{application=~"$Application", deployment=~"$Deployment",{global_filters}}}[5m])) by (le)) or vector(0)',
                legend="P50 Wait Time",
            )
        ],
        template=PanelTemplate.STAT,
        grid_pos=GridPos(8, 0, 4, 3),
        thresholds=[{"color": "blue", "value": None}],
    ),
    Panel(
        id=4004,
        title="Batches Processed/sec",
        description="Rate of batches executed",
        unit="batches/s",
        targets=[
            Target(
                expr='sum(rate(ray_serve_batches_processed_total{{application=~"$Application", deployment=~"$Deployment",{global_filters}}}[5m]))',
                legend="Batches/sec",
            )
        ],
        template=PanelTemplate.STAT,
        grid_pos=GridPos(12, 0, 4, 3),
        thresholds=[{"color": "blue", "value": None}],
    ),
    Panel(
        id=4005,
        title="Queue Length",
        description="Number of requests waiting in batch queue",
        unit="requests",
        targets=[
            Target(
                expr='sum(ray_serve_batch_queue_length{{application=~"$Application", deployment=~"$Deployment",{global_filters}}})',
                legend="Queue Length",
            )
        ],
        template=PanelTemplate.STAT,
        grid_pos=GridPos(16, 0, 4, 3),
        thresholds=[{"color": "blue", "value": None}],
    ),
    # Row 2: Batch Size Distribution & Utilization
    Panel(
        id=4006,
        title="Batch Size Distribution",
        description="Distribution of computed batch sizes over time. P50 (median) and P99 batch sizes grouped by application, deployment, and function. Shows if batches are reaching desired sizes and helps identify batching effectiveness.",
        unit="short",
        targets=[
            Target(
                expr='histogram_quantile(0.5, sum(rate(ray_serve_actual_batch_size_bucket{{application=~"$Application", deployment=~"$Deployment",{global_filters}}}[5m])) by (deployment, application, function_name, le))',
                legend="P50: {{application}}/{{deployment}}/{{function_name}}",
            ),
            Target(
                expr='histogram_quantile(0.99, sum(rate(ray_serve_actual_batch_size_bucket{{application=~"$Application", deployment=~"$Deployment",{global_filters}}}[5m])) by (deployment, application, function_name, le))',
                legend="P99: {{application}}/{{deployment}}/{{function_name}}",
            ),
        ],
        template=PanelTemplate.GRAPH,
        fill=0,
        stack=False,
        grid_pos=GridPos(0, 3, 12, 8),
    ),
    Panel(
        id=4007,
        title="Batch Utilization %",
        description="Median (P50) and P99 batch utilization per function. Low utilization = timeout too aggressive or low traffic",
        unit="percent",
        targets=[
            Target(
                expr='histogram_quantile(0.5, sum(rate(ray_serve_batch_utilization_percent_bucket{{application=~"$Application", deployment=~"$Deployment",{global_filters}}}[5m])) by (deployment, application, function_name, le))',
                legend="P50: {{application}} {{deployment}} {{function_name}}",
            ),
            Target(
                expr='histogram_quantile(0.99, sum(rate(ray_serve_batch_utilization_percent_bucket{{application=~"$Application", deployment=~"$Deployment",{global_filters}}}[5m])) by (deployment, application, function_name, le))',
                legend="P99: {{application}} {{deployment}} {{function_name}}",
            ),
        ],
        template=PanelTemplate.GRAPH,
        fill=0,
        stack=False,
        grid_pos=GridPos(12, 3, 12, 8),
    ),
    # Row 3: Batching Latency
    Panel(
        id=4008,
        title="Batch Wait Time",
        description="Time requests waited in the batch queue before the batch was formed and sent for processing. High P99 wait time indicates batch timeout may be too long or traffic is too sparse to form full batches",
        unit="ms",
        targets=[
            Target(
                expr='histogram_quantile(0.5, sum(rate(ray_serve_batch_wait_time_ms_bucket{{application=~"$Application", deployment=~"$Deployment",{global_filters}}}[5m])) by (deployment, application, function_name, le))',
                legend="P50: {{application}} {{deployment}} {{function_name}}",
            ),
            Target(
                expr='histogram_quantile(0.99, sum(rate(ray_serve_batch_wait_time_ms_bucket{{application=~"$Application", deployment=~"$Deployment",{global_filters}}}[5m])) by (deployment, application, function_name, le))',
                legend="P99: {{application}} {{deployment}} {{function_name}}",
            ),
        ],
        template=PanelTemplate.GRAPH,
        fill=0,
        stack=False,
        grid_pos=GridPos(0, 11, 12, 8),
    ),
    Panel(
        id=4009,
        title="Batch Execution Time",
        description="Time to execute the batch function. High values indicate slow batch processing.",
        unit="ms",
        targets=[
            Target(
                expr='histogram_quantile(0.5, sum(rate(ray_serve_batch_execution_time_ms_bucket{{application=~"$Application", deployment=~"$Deployment",{global_filters}}}[5m])) by (deployment, application, function_name, le))',
                legend="P50: {{application}} {{deployment}} {{function_name}}",
            ),
            Target(
                expr='histogram_quantile(0.99, sum(rate(ray_serve_batch_execution_time_ms_bucket{{application=~"$Application", deployment=~"$Deployment",{global_filters}}}[5m])) by (deployment, application, function_name, le))',
                legend="P99: {{application}} {{deployment}} {{function_name}}",
            ),
        ],
        template=PanelTemplate.GRAPH,
        fill=0,
        stack=False,
        grid_pos=GridPos(12, 11, 12, 8),
    ),
    # Row 4: Batching Throughput
    Panel(
        id=4010,
        title="Batches Processed per Second",
        description="Rate of batches executed. Measure batching throughput separate from request throughput.",
        unit="batches/s",
        targets=[
            Target(
                expr='sum(rate(ray_serve_batches_processed_total{{application=~"$Application", deployment=~"$Deployment",{global_filters}}}[5m])) by (deployment, application, function_name)',
                legend="{{application}} {{deployment}} {{function_name}}",
            )
        ],
        template=PanelTemplate.GRAPH,
        fill=1,
        stack=False,
        grid_pos=GridPos(0, 19, 12, 8),
    ),
    Panel(
        id=4011,
        title="Batch Queue Length",
        description="Current number of requests waiting in the batch queue. High or growing values indicate batching can't keep up with the request rate - consider increasing max_concurrent_batches or reducing batch_wait_timeout_s",
        unit="requests",
        targets=[
            Target(
                expr='sum(ray_serve_batch_queue_length{{application=~"$Application", deployment=~"$Deployment",{global_filters}}}) by (application, deployment, function_name)',
                legend="{{application}} {{deployment}} {{function_name}}",
            )
        ],
        template=PanelTemplate.GRAPH,
        fill=1,
        stack=False,
        grid_pos=GridPos(12, 19, 12, 8),
    ),
]

# ==============================================================================
# MODEL MULTIPLEXING SECTION
# ==============================================================================

MULTIPLEXING_PANELS = [
    # Row 1: Multiplexing Overview Stats
    Panel(
        id=5001,
        title="Models Loaded (Cluster-Wide)",
        description="Total number of models currently loaded across all replicas",
        unit="models",
        targets=[
            Target(
                expr='sum(ray_serve_num_multiplexed_models{{application=~"$Application", deployment=~"$Deployment",{global_filters}}})',
                legend="Models Loaded",
            )
        ],
        template=PanelTemplate.STAT,
        grid_pos=GridPos(0, 0, 6, 3),
        thresholds=[{"color": "blue", "value": None}],
    ),
    Panel(
        id=5002,
        title="Model Cache Hit Rate",
        description="Percentage of get_model() calls served by already-loaded models (cache hits). Calculated as: 100 * (1 - loads / get_model_requests).",
        unit="percent",
        targets=[
            Target(
                expr='100 * (1 - sum(rate(ray_serve_multiplexed_models_load_counter_total{{application=~"$Application", deployment=~"$Deployment",{global_filters}}}[5m])) / clamp_min(sum(rate(ray_serve_multiplexed_get_model_requests_counter_total{{application=~"$Application", deployment=~"$Deployment",{global_filters}}}[5m])), 1))',
                legend="Cache Hit Rate",
            )
        ],
        template=PanelTemplate.STAT,
        grid_pos=GridPos(6, 0, 6, 3),
        thresholds=[
            {"color": "red", "value": None},
            {"color": "yellow", "value": 50},
            {"color": "green", "value": 80},
        ],
    ),
    Panel(
        id=5003,
        title="P99 Model Load Time",
        description="99th percentile time to load a model into memory",
        unit="ms",
        targets=[
            Target(
                expr='histogram_quantile(0.99, sum(rate(ray_serve_multiplexed_model_load_latency_ms_bucket{{application=~"$Application", deployment=~"$Deployment",{global_filters}}}[5m])) by (le)) or vector(0)',
                legend="P99 Load Time",
            )
        ],
        template=PanelTemplate.STAT,
        grid_pos=GridPos(12, 0, 6, 3),
        thresholds=[{"color": "blue", "value": None}],
    ),
    Panel(
        id=5004,
        title="Model Load/Unload Rate",
        description="Combined rate of model loads and unloads per second (model churn)",
        unit="ops/s",
        targets=[
            Target(
                expr='sum(rate(ray_serve_multiplexed_models_load_counter_total{{application=~"$Application", deployment=~"$Deployment",{global_filters}}}[5m])) + sum(rate(ray_serve_multiplexed_models_unload_counter_total{{application=~"$Application", deployment=~"$Deployment",{global_filters}}}[5m]))',
                legend="Load/Unload Rate",
            )
        ],
        template=PanelTemplate.STAT,
        grid_pos=GridPos(18, 0, 6, 3),
        thresholds=[{"color": "blue", "value": None}],
    ),
    # Row 2: Models per Deployment
    Panel(
        id=5005,
        title="Loaded Models per Deployment",
        description="Number of models currently loaded per deployment",
        unit="models",
        targets=[
            Target(
                expr='sum(ray_serve_num_multiplexed_models{{application=~"$Application", deployment=~"$Deployment", replica=~"$Replica",{global_filters}}}) by (application, deployment)',
                legend="{{application}} {{deployment}}",
            )
        ],
        template=PanelTemplate.GRAPH,
        fill=1,
        stack=False,
        grid_pos=GridPos(0, 3, 8, 8),
    ),
    Panel(
        id=5006,
        title="Model Requests (Cache Hits + Misses)",
        description="Rate of get_model() calls per deployment. This counts every request for a model, regardless of whether it's already loaded (cache hit) or needs to be loaded (cache miss).",
        unit="requests/s",
        targets=[
            Target(
                expr='sum(rate(ray_serve_multiplexed_get_model_requests_counter_total{{application=~"$Application", deployment=~"$Deployment", replica=~"$Replica",{global_filters}}}[5m])) by (application, deployment)',
                legend="{{application}} {{deployment}}",
            )
        ],
        template=PanelTemplate.GRAPH,
        fill=1,
        stack=False,
        grid_pos=GridPos(8, 3, 8, 8),
    ),
    Panel(
        id=5007,
        title="Model Loads (Cache Misses)",
        description="Rate of actual model loads (cache misses). Compare with Get Model Requests to understand cache efficiency.",
        unit="loads/s",
        targets=[
            Target(
                expr='sum(rate(ray_serve_multiplexed_models_load_counter_total{{application=~"$Application", deployment=~"$Deployment", replica=~"$Replica",{global_filters}}}[5m])) by (application, deployment)',
                legend="{{application}} {{deployment}}",
            )
        ],
        template=PanelTemplate.GRAPH,
        fill=1,
        stack=False,
        grid_pos=GridPos(16, 3, 8, 8),
    ),
    # Row 3: Model Loading Performance
    Panel(
        id=5008,
        title="P99 Model Load Latency",
        description="99th percentile time to load a model into memory. High values indicate slow model loading (large models, slow storage, etc.)",
        unit="ms",
        targets=[
            Target(
                expr='histogram_quantile(0.99, sum(rate(ray_serve_multiplexed_model_load_latency_ms_bucket{{application=~"$Application", deployment=~"$Deployment",{global_filters}}}[5m])) by (application, deployment, le))',
                legend="{{application}} {{deployment}}",
            )
        ],
        template=PanelTemplate.GRAPH,
        fill=0,
        stack=False,
        grid_pos=GridPos(0, 11, 12, 8),
    ),
    Panel(
        id=5009,
        title="P99 Model Unload Latency",
        description="99th percentile time to unload a model from memory. High values may indicate cleanup issues.",
        unit="ms",
        targets=[
            Target(
                expr='histogram_quantile(0.99, sum(rate(ray_serve_multiplexed_model_unload_latency_ms_bucket{{application=~"$Application", deployment=~"$Deployment",{global_filters}}}[5m])) by (application, deployment, le))',
                legend="{{application}} {{deployment}}",
            )
        ],
        template=PanelTemplate.GRAPH,
        fill=0,
        stack=False,
        grid_pos=GridPos(12, 11, 12, 8),
    ),
    # Row 4: Cache Efficiency
    Panel(
        id=5010,
        title="Model Evictions",
        description="Rate of model evictions from the cache. High eviction rates indicate the model cache is too small for the working set.",
        unit="evictions/s",
        targets=[
            Target(
                expr='sum(rate(ray_serve_multiplexed_models_unload_counter_total{{application=~"$Application", deployment=~"$Deployment", replica=~"$Replica",{global_filters}}}[5m])) by (application, deployment)',
                legend="{{application}} {{deployment}}",
            )
        ],
        template=PanelTemplate.GRAPH,
        fill=1,
        stack=False,
        grid_pos=GridPos(0, 19, 12, 8),
    ),
    Panel(
        id=5011,
        title="Cache Hit Rate Over Time",
        description="Percentage of get_model() calls served without needing to load a model (cache hits). Low hit rate indicates high model churn or insufficient cache size relative to the working set.",
        unit="percent",
        targets=[
            Target(
                expr='100 * (1 - sum(rate(ray_serve_multiplexed_models_load_counter_total{{application=~"$Application", deployment=~"$Deployment",{global_filters}}}[5m])) by (application, deployment) / clamp_min(sum(rate(ray_serve_multiplexed_get_model_requests_counter_total{{application=~"$Application", deployment=~"$Deployment",{global_filters}}}[5m])) by (application, deployment), 1))',
                legend="{{application}} {{deployment}}",
            )
        ],
        template=PanelTemplate.GRAPH,
        fill=0,
        stack=False,
        grid_pos=GridPos(12, 19, 12, 8),
    ),
]

# ==============================================================================
# HARDWARE UTILIZATION SECTION
# ==============================================================================

HARDWARE_UTILIZATION_PANELS = [
    # Row 1: Head Node Stats - no GPU/GRAM as head typically doesn't have GPU
    Panel(
        id=6001,
        title="Head CPU",
        description="Head node CPU utilization",
        unit="percent",
        targets=[
            Target(
                expr='avg(ray_node_cpu_utilization{{IsHeadNode="true",{global_filters}}}) or vector(0)',
                legend="Head CPU",
            )
        ],
        template=PanelTemplate.GAUGE,
        grid_pos=GridPos(0, 0, 8, 5),
        min_val=0,
        max_val=100,
        thresholds=[
            {"color": "green", "value": None},
            {"color": "yellow", "value": 70},
            {"color": "red", "value": 90},
        ],
    ),
    Panel(
        id=6002,
        title="Head Memory",
        description="Head node memory utilization",
        unit="percent",
        targets=[
            Target(
                expr='sum(ray_node_mem_used{{IsHeadNode="true",{global_filters}}}) / sum(ray_node_mem_total{{IsHeadNode="true",{global_filters}}}) * 100 or vector(0)',
                legend="Head Memory",
            )
        ],
        template=PanelTemplate.GAUGE,
        grid_pos=GridPos(8, 0, 8, 5),
        min_val=0,
        max_val=100,
        thresholds=[
            {"color": "green", "value": None},
            {"color": "yellow", "value": 70},
            {"color": "red", "value": 90},
        ],
    ),
    Panel(
        id=6003,
        title="Head Disk",
        description="Head node disk utilization",
        unit="percent",
        targets=[
            Target(
                expr='sum(ray_node_disk_usage{{IsHeadNode="true",{global_filters}}}) / (sum(ray_node_disk_free{{IsHeadNode="true",{global_filters}}}) + sum(ray_node_disk_usage{{IsHeadNode="true",{global_filters}}})) * 100 or vector(0)',
                legend="Head Disk",
            )
        ],
        template=PanelTemplate.GAUGE,
        grid_pos=GridPos(16, 0, 8, 5),
        min_val=0,
        max_val=100,
        thresholds=[
            {"color": "green", "value": None},
            {"color": "yellow", "value": 70},
            {"color": "red", "value": 90},
        ],
    ),
    # Row 2: Cluster-wide Stats
    # Note: These panels show metrics across ALL nodes in the Ray cluster. In shared
    # clusters (e.g., Ray Serve + Ray Data), these include nodes that may not be
    # running Serve replicas. To filter to Serve-specific nodes, add filters like
    # instance=~"serve-.*" or use node labels/tags in your environment.
    Panel(
        id=6004,
        title="Cluster CPU",
        description="Average CPU utilization across all nodes in the cluster. Note: In shared clusters (e.g., with Ray Data), this includes all nodes, not just those running Serve replicas. Consider adding instance or node label filters to isolate Serve nodes if needed.",
        unit="percent",
        targets=[
            Target(
                expr="avg(ray_node_cpu_utilization{{{global_filters}}})",
                legend="Cluster CPU",
            )
        ],
        template=PanelTemplate.GAUGE,
        grid_pos=GridPos(0, 5, 5, 5),
        min_val=0,
        max_val=100,
        thresholds=[
            {"color": "green", "value": None},
            {"color": "yellow", "value": 70},
            {"color": "red", "value": 90},
        ],
    ),
    Panel(
        id=6005,
        title="Cluster GPU",
        description="Average GPU utilization across all GPUs in the cluster. Note: In shared clusters (e.g., with Ray Data), this includes all GPUs, not just those serving Serve workloads. Consider adding instance or node label filters to isolate Serve nodes if needed.",
        unit="percent",
        targets=[
            Target(
                expr="sum(ray_node_gpus_utilization{{{global_filters}}}) / on() (sum(autoscaler_cluster_resources{{resource='GPU',{global_filters}}}) or vector(0))",
                legend="Cluster GPU",
            )
        ],
        template=PanelTemplate.GAUGE,
        grid_pos=GridPos(5, 5, 5, 5),
        min_val=0,
        max_val=100,
        thresholds=[
            {"color": "green", "value": None},
            {"color": "yellow", "value": 70},
            {"color": "red", "value": 90},
        ],
    ),
    Panel(
        id=6006,
        title="Cluster Memory",
        description="Total memory utilization across all nodes in the cluster (sum used / sum total). Note: In shared clusters (e.g., with Ray Data), this includes all nodes, not just those running Serve replicas. Consider adding instance or node label filters to isolate Serve nodes if needed.",
        unit="percent",
        targets=[
            Target(
                expr="sum(ray_node_mem_used{{{global_filters}}}) / sum(ray_node_mem_total{{{global_filters}}}) * 100",
                legend="Cluster Memory",
            )
        ],
        template=PanelTemplate.GAUGE,
        grid_pos=GridPos(10, 5, 5, 5),
        min_val=0,
        max_val=100,
        thresholds=[
            {"color": "green", "value": None},
            {"color": "yellow", "value": 70},
            {"color": "red", "value": 90},
        ],
    ),
    Panel(
        id=6007,
        title="Cluster GRAM",
        description="Total GPU memory utilization across all GPUs in the cluster. Note: In shared clusters (e.g., with Ray Data), this includes all GPUs, not just those serving Serve workloads. Consider adding instance or node label filters to isolate Serve nodes if needed.",
        unit="percent",
        targets=[
            Target(
                expr="sum(ray_node_gram_used{{{global_filters}}}) / on() (sum(ray_node_gram_available{{{global_filters}}}) + sum(ray_node_gram_used{{{global_filters}}})) * 100",
                legend="Cluster GRAM",
            )
        ],
        template=PanelTemplate.GAUGE,
        grid_pos=GridPos(15, 5, 5, 5),
        min_val=0,
        max_val=100,
        thresholds=[
            {"color": "green", "value": None},
            {"color": "yellow", "value": 70},
            {"color": "red", "value": 90},
        ],
    ),
    Panel(
        id=6008,
        title="Cluster Disk",
        description="Total disk utilization across all nodes in the cluster. Note: In shared clusters (e.g., with Ray Data), this includes all nodes, not just those running Serve replicas. Consider adding instance or node label filters to isolate Serve nodes if needed.",
        unit="percent",
        targets=[
            Target(
                expr="sum(ray_node_disk_usage{{{global_filters}}}) / (sum(ray_node_disk_free{{{global_filters}}}) + sum(ray_node_disk_usage{{{global_filters}}})) * 100",
                legend="Cluster Disk",
            )
        ],
        template=PanelTemplate.GAUGE,
        grid_pos=GridPos(20, 5, 4, 5),
        min_val=0,
        max_val=100,
        thresholds=[
            {"color": "green", "value": None},
            {"color": "yellow", "value": 70},
            {"color": "red", "value": 90},
        ],
    ),
    # Row 3: Cluster Total Resources (capacity)
    # Note: These panels show total resources across ALL nodes in the Ray cluster.
    # In shared clusters, these include nodes not running Serve replicas.
    Panel(
        id=6009,
        title="Total CPU Cores",
        description="Total number of CPU cores available across all nodes in the cluster. Note: In shared clusters (e.g., with Ray Data), this includes all nodes, not just those running Serve replicas.",
        unit="short",
        targets=[
            Target(
                expr="sum(ray_node_cpu_count{{{global_filters}}})",
                legend="Total CPUs",
            )
        ],
        template=PanelTemplate.STAT,
        grid_pos=GridPos(0, 10, 5, 3),
        thresholds=[{"color": "blue", "value": None}],
    ),
    Panel(
        id=6010,
        title="Total GPUs",
        description="Total number of GPUs available across all nodes in the cluster. Note: In shared clusters (e.g., with Ray Data), this includes all GPUs, not just those serving Serve workloads.",
        unit="short",
        targets=[
            Target(
                expr="sum(autoscaler_cluster_resources{{resource='GPU',{global_filters}}}) or vector(0)",
                legend="Total GPUs",
            )
        ],
        template=PanelTemplate.STAT,
        grid_pos=GridPos(5, 10, 5, 3),
        thresholds=[{"color": "blue", "value": None}],
    ),
    Panel(
        id=6011,
        title="Total Memory",
        description="Total memory (RAM) available across all nodes in the cluster. Note: In shared clusters (e.g., with Ray Data), this includes all nodes, not just those running Serve replicas.",
        unit="decgbytes",
        targets=[
            Target(
                expr="sum(ray_node_mem_total{{{global_filters}}}) / 1024 / 1024 / 1024",
                legend="Total Memory",
            )
        ],
        template=PanelTemplate.STAT,
        grid_pos=GridPos(10, 10, 5, 3),
        thresholds=[{"color": "blue", "value": None}],
    ),
    Panel(
        id=6012,
        title="Total GRAM",
        description="Total GPU memory available across all GPUs in the cluster. Note: In shared clusters (e.g., with Ray Data), this includes all GPUs, not just those serving Serve workloads.",
        unit="decgbytes",
        targets=[
            Target(
                expr="(sum(ray_node_gram_available{{{global_filters}}}) + sum(ray_node_gram_used{{{global_filters}}})) / 1024 / 1024 / 1024",
                legend="Total GRAM",
            )
        ],
        template=PanelTemplate.STAT,
        grid_pos=GridPos(15, 10, 5, 3),
        thresholds=[{"color": "blue", "value": None}],
    ),
    Panel(
        id=6013,
        title="Total Disk",
        description="Total disk space available across all nodes in the cluster. Note: In shared clusters (e.g., with Ray Data), this includes all nodes, not just those running Serve replicas.",
        unit="decgbytes",
        targets=[
            Target(
                expr="(sum(ray_node_disk_free{{{global_filters}}}) + sum(ray_node_disk_usage{{{global_filters}}})) / 1024 / 1024 / 1024",
                legend="Total Disk",
            )
        ],
        template=PanelTemplate.STAT,
        grid_pos=GridPos(20, 10, 4, 3),
        thresholds=[{"color": "blue", "value": None}],
    ),
    # Row 4: Cluster Capacity Over Time
    # Note: These panels show metrics across ALL nodes in the Ray cluster. In shared
    # clusters (e.g., Ray Serve + Ray Data), these include nodes not running Serve replicas.
    Panel(
        id=6014,
        title="Node Count",
        description="Number of nodes in this cluster. Note: In shared clusters (e.g., with Ray Data), this includes all nodes, not just those running Serve replicas. Consider filtering by NodeType or instance labels to isolate Serve nodes if needed.",
        unit="nodes",
        targets=[
            Target(
                expr="sum(autoscaler_active_nodes{{{global_filters}}}) by (NodeType)",
                legend="Active: {{NodeType}}",
            ),
            Target(
                expr="sum(autoscaler_pending_nodes{{{global_filters}}}) by (NodeType)",
                legend="Pending: {{NodeType}}",
            ),
            Target(
                expr="sum(autoscaler_recently_failed_nodes{{{global_filters}}}) by (NodeType)",
                legend="Failed: {{NodeType}}",
            ),
        ],
        template=PanelTemplate.GRAPH,
        fill=1,
        stack=False,
        grid_pos=GridPos(0, 13, 8, 8),
    ),
    Panel(
        id=6015,
        title="Cluster Resource Utilization",
        description="Aggregated utilization of all physical resources across the cluster. Note: In shared clusters (e.g., with Ray Data), this includes all nodes/resources, not just those serving Serve workloads. Consider adding instance or node label filters to isolate Serve nodes if needed.",
        unit="percent",
        targets=[
            Target(
                expr="avg(ray_node_cpu_utilization{{{global_filters}}})",
                legend="CPU (physical)",
            ),
            Target(
                expr="sum(ray_node_gpus_utilization{{{global_filters}}}) / on() (sum(autoscaler_cluster_resources{{resource='GPU',{global_filters}}}) or vector(0))",
                legend="GPU (physical)",
            ),
            Target(
                expr="sum(ray_node_mem_used{{{global_filters}}}) / on() sum(ray_node_mem_total{{{global_filters}}}) * 100",
                legend="Memory (RAM)",
            ),
            Target(
                expr="sum(ray_node_gram_used{{{global_filters}}}) / on() (sum(ray_node_gram_available{{{global_filters}}}) + sum(ray_node_gram_used{{{global_filters}}})) * 100",
                legend="GRAM",
            ),
            Target(
                expr='sum(ray_object_store_memory{{{global_filters}}}) / on() sum(ray_resources{{Name="object_store_memory",{global_filters}}}) * 100',
                legend="Object Store Memory",
            ),
            Target(
                expr="sum(ray_node_disk_usage{{{global_filters}}}) / on() (sum(ray_node_disk_free{{{global_filters}}}) + sum(ray_node_disk_usage{{{global_filters}}})) * 100",
                legend="Disk",
            ),
        ],
        template=PanelTemplate.GRAPH,
        fill=0,
        stack=False,
        grid_pos=GridPos(8, 13, 8, 8),
    ),
    Panel(
        id=6016,
        title="Node Network",
        description="Network speed per node. Note: In shared clusters (e.g., with Ray Data), this includes all nodes, not just those running Serve replicas. Consider filtering by instance to isolate specific nodes.",
        unit="Bps",
        targets=[
            Target(
                expr="sum(ray_node_network_receive_speed{{{global_filters}}}) by (instance)",
                legend="Recv: {{instance}}",
            ),
            Target(
                expr="sum(ray_node_network_send_speed{{{global_filters}}}) by (instance)",
                legend="Send: {{instance}}",
            ),
        ],
        template=PanelTemplate.GRAPH,
        fill=1,
        stack=False,
        grid_pos=GridPos(16, 13, 8, 8),
    ),
    # Row 5: Hardware Utilization by Ray Component
    # Component names contain application and deployment (e.g., ray::ServeReplica:default:MainRouter)
    # so we filter by regex matching $Application and $Deployment in the Component name
    Panel(
        id=6017,
        title="CPU Usage by Component",
        description="Physical CPU usage broken down by Ray component. Ray components include system components (raylet, gcs, dashboard, agent) and Serve replica processes. Filtered by Application/Deployment selection.",
        unit="cores",
        targets=[
            Target(
                expr='sum(ray_component_cpu_percentage{{Component=~".*$Application.*$Deployment.*",{global_filters}}}) by (Component) / 100',
                legend="{{Component}}",
            ),
            Target(
                expr="sum(ray_node_cpu_count{{{global_filters}}})",
                legend="MAX",
            ),
        ],
        template=PanelTemplate.GRAPH,
        fill=1,
        stack=False,
        grid_pos=GridPos(0, 21, 12, 8),
    ),
    Panel(
        id=6018,
        title="Memory Usage by Component",
        description="Physical memory usage broken down by Ray component (RSS minus shared memory). Filtered by Application/Deployment selection.",
        unit="bytes",
        targets=[
            Target(
                expr='(sum(ray_component_rss_mb{{Component=~".*$Application.*$Deployment.*",{global_filters}}} * 1024 * 1024) by (Component)) - (sum(ray_component_mem_shared_bytes{{Component=~".*$Application.*$Deployment.*",{global_filters}}}) by (Component))',
                legend="{{Component}}",
            ),
            Target(
                expr="sum(ray_node_mem_shared_bytes{{{global_filters}}})",
                legend="shared_memory",
            ),
            Target(
                expr="sum(ray_node_mem_total{{{global_filters}}})",
                legend="MAX",
            ),
        ],
        template=PanelTemplate.GRAPH,
        fill=1,
        stack=False,
        grid_pos=GridPos(12, 21, 12, 8),
    ),
    Panel(
        id=6019,
        title="GPU Usage by Component",
        description="Physical GPU usage broken down by Ray component. Filtered by Application/Deployment selection.",
        unit="GPUs",
        targets=[
            Target(
                expr='sum(ray_component_gpu_percentage{{Component=~".*$Application.*$Deployment.*",{global_filters}}} / 100) by (Component)',
                legend="{{Component}}",
            ),
        ],
        template=PanelTemplate.GRAPH,
        fill=1,
        stack=False,
        grid_pos=GridPos(0, 29, 12, 8),
    ),
    Panel(
        id=6020,
        title="GPU Memory Usage by Component",
        description="Physical GPU memory usage broken down by Ray component. Filtered by Application/Deployment selection.",
        unit="bytes",
        targets=[
            Target(
                expr='sum(ray_component_gpu_memory_mb{{Component=~".*$Application.*$Deployment.*",{global_filters}}} * 1024 * 1024) by (Component)',
                legend="{{Component}}",
            ),
            Target(
                expr="(sum(ray_node_gram_available{{{global_filters}}}) + sum(ray_node_gram_used{{{global_filters}}})) * 1024 * 1024",
                legend="MAX",
            ),
        ],
        template=PanelTemplate.GRAPH,
        fill=1,
        stack=False,
        grid_pos=GridPos(12, 29, 12, 8),
    ),
]

# ==============================================================================
# DASHBOARD CONFIGURATION
# ==============================================================================

# Create rows for collapsible sections
serve_dashboard_rows_v2 = [
    Row(
        id=1000,
        title="Serve Overview",
        panels=SERVE_OVERVIEW_PANELS,
        collapsed=False,  # Expanded by default
    ),
    Row(
        id=2000,
        title="Deployment Deep Dive",
        panels=DEPLOYMENT_DEEP_DIVE_PANELS,
        collapsed=True,
    ),
    Row(
        id=3000,
        title="Autoscaling",
        panels=AUTOSCALING_USER_PANELS,
        collapsed=True,
    ),
    Row(
        id=3100,
        title="System Metrics",
        panels=AUTOSCALING_SYSTEM_PANELS,
        collapsed=True,
    ),
    Row(
        id=4000,
        title="Request Batching",
        panels=BATCHING_PANELS,
        collapsed=True,
    ),
    Row(
        id=5000,
        title="Model Multiplexing",
        panels=MULTIPLEXING_PANELS,
        collapsed=True,
    ),
    Row(
        id=6000,
        title="Hardware Utilization",
        panels=HARDWARE_UTILIZATION_PANELS,
        collapsed=True,
    ),
]

# Dashboard configuration
serve_dashboard_config = DashboardConfig(
    name="SERVE",
    default_uid="rayServeDashboard",
    panels=[],  # All panels are in rows
    rows=serve_dashboard_rows_v2,
    standard_global_filters=[
        'ray_io_cluster=~"$Cluster"',
    ],
    base_json_file_name="serve_grafana_dashboard_base_anyscale.json",
)

# Panel ID validation
_all_panel_ids = []
for row in serve_dashboard_rows_v2:
    for panel in row.panels:
        _all_panel_ids.append(panel.id)

_all_panel_ids.sort()

assert len(_all_panel_ids) == len(
    set(_all_panel_ids)
), f"Duplicated panel ID found. Use unique ID for each panel. IDs: {_all_panel_ids}"
