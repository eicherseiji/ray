# Copyright (2023 and onwards) Anyscale, Inc.

import os

from ray.serve._private.constants import RAY_SERVE_USE_GRPC_BY_DEFAULT

ANYSCALE_RAY_SERVE_ENABLE_PROPRIETARY_DEPLOYMENT_SCHEDULER = (
    os.environ.get("ANYSCALE_RAY_SERVE_ENABLE_PROPRIETARY_DEPLOYMENT_SCHEDULER", "1")
    == "1"
)

ANYSCALE_RAY_SERVE_DEFAULT_DRAINING_TIMEOUT_S = float(
    os.environ.get("ANYSCALE_RAY_SERVE_DEFAULT_DRAINING_TIMEOUT_S", 300.0)
)

# Default to 30 minutes
ANYSCALE_RAY_SERVE_COMPACTION_TIMEOUT_S = float(
    os.environ.get("ANYSCALE_RAY_SERVE_COMPACTION_TIMEOUT_S", 1800.0)
)

# How long to wait after deployments become stable before attempting node compaction
ANYSCALE_RAY_SERVE_NODE_COMPACTION_DELAY_S = int(
    os.environ.get("ANYSCALE_RAY_SERVE_NODE_COMPACTION_DELAY_S", "300")
)

DEFAULT_TRACING_EXPORTER_IMPORT_PATH = (
    "ray.anyscale.serve._private.tracing_utils:default_tracing_exporter"
)
# Path to tracing exporter function
# If None, then use default tracing exporter
# If empty string, then tracing is disabled
ANYSCALE_TRACING_EXPORTER_IMPORT_PATH = os.environ.get(
    "ANYSCALE_TRACING_EXPORTER_IMPORT_PATH", DEFAULT_TRACING_EXPORTER_IMPORT_PATH
)

ANYSCALE_TRACING_SAMPLING_RATIO = float(
    os.environ.get("ANYSCALE_TRACING_SAMPLING_RATIO", 0.01)
)

# Feature flag to use HAProxy.
ANYSCALE_RAY_SERVE_ENABLE_HA_PROXY = (
    os.environ.get("ANYSCALE_RAY_SERVE_ENABLE_HA_PROXY", "0") == "1"
)

# HAProxy configuration defaults
# Maximum number of concurrent connections
ANYSCALE_RAY_SERVE_HAPROXY_MAXCONN = int(
    os.environ.get("ANYSCALE_RAY_SERVE_HAPROXY_MAXCONN", "20000")
)

# Number of threads for HAProxy
ANYSCALE_RAY_SERVE_HAPROXY_NBTHREAD = int(
    os.environ.get("ANYSCALE_RAY_SERVE_HAPROXY_NBTHREAD", "4")
)

# HAProxy configuration file location
ANYSCALE_RAY_SERVE_HAPROXY_CONFIG_FILE_LOC = os.environ.get(
    "ANYSCALE_RAY_SERVE_HAPROXY_CONFIG_FILE_LOC", "/tmp/haproxy-serve/haproxy.cfg"
)

# HAProxy admin socket path
ANYSCALE_RAY_SERVE_HAPROXY_SOCKET_PATH = os.environ.get(
    "ANYSCALE_RAY_SERVE_HAPROXY_SOCKET_PATH", "/tmp/haproxy-serve/admin.sock"
)

# Enable HAProxy optimized configuration (server state persistence, etc.)
# Disabled by default to prevent test suite interference
ANYSCALE_RAY_SERVE_ENABLE_HAPROXY_OPTIMIZED_CONFIG = (
    os.environ.get("ANYSCALE_RAY_SERVE_ENABLE_HAPROXY_OPTIMIZED_CONFIG", "1") == "1"
)

# HAProxy server state path
ANYSCALE_RAY_SERVE_HAPROXY_SERVER_STATE_BASE = os.environ.get(
    "ANYSCALE_RAY_SERVE_HAPROXY_SERVER_STATE_BASE", "/tmp/haproxy-serve"
)

# HAProxy server state path
ANYSCALE_RAY_SERVE_HAPROXY_SERVER_STATE_FILE = os.environ.get(
    "ANYSCALE_RAY_SERVE_HAPROXY_SERVER_STATE_FILE", "/tmp/haproxy-serve/server-state"
)

# HAProxy hard stop after timeout
ANYSCALE_RAY_SERVE_HAPROXY_HARD_STOP_AFTER_S = int(
    os.environ.get("ANYSCALE_RAY_SERVE_HAPROXY_HARD_STOP_AFTER_S", "120")
)

# HAProxy metrics export port
ANYSCALE_RAY_SERVE_HAPROXY_METRICS_PORT = int(
    os.environ.get("ANYSCALE_RAY_SERVE_HAPROXY_METRICS_PORT", "9101")
)

# HAProxy timeout configurations (in seconds, None = no timeout)
ANYSCALE_RAY_SERVE_HAPROXY_TIMEOUT_SERVER_S = (
    int(os.environ.get("ANYSCALE_RAY_SERVE_HAPROXY_TIMEOUT_SERVER_S"))
    if os.environ.get("ANYSCALE_RAY_SERVE_HAPROXY_TIMEOUT_SERVER_S")
    else None
)

ANYSCALE_RAY_SERVE_HAPROXY_TIMEOUT_CONNECT_S = (
    int(os.environ.get("ANYSCALE_RAY_SERVE_HAPROXY_TIMEOUT_CONNECT_S"))
    if os.environ.get("ANYSCALE_RAY_SERVE_HAPROXY_TIMEOUT_CONNECT_S")
    else None
)

# HAProxy timeout client
ANYSCALE_RAY_SERVE_HAPROXY_TIMEOUT_CLIENT_S = int(
    os.environ.get("ANYSCALE_RAY_SERVE_HAPROXY_TIMEOUT_CLIENT_S", "3600")
)

# Number of consecutive failed server health checks that must occur
# before haproxy marks the server as down.
ANYSCALE_RAY_SERVE_HAPROXY_HEALTH_CHECK_FALL = int(
    os.environ.get("ANYSCALE_RAY_SERVE_HAPROXY_HEALTH_CHECK_FALL", "2")
)

# Number of consecutive successful server health checks that must occur
# before haproxy marks the server as up.
ANYSCALE_RAY_SERVE_HAPROXY_HEALTH_CHECK_RISE = int(
    os.environ.get("ANYSCALE_RAY_SERVE_HAPROXY_HEALTH_CHECK_RISE", "2")
)

# Time interval between each haproxy health check attempt. Also the
# timeout of each health check before being considered as failed.
ANYSCALE_RAY_SERVE_HAPROXY_HEALTH_CHECK_INTER = os.environ.get(
    "ANYSCALE_RAY_SERVE_HAPROXY_HEALTH_CHECK_INTER", "5s"
)

# Time interval between each haproxy health check attempt when the server is in any of the transition states: UP - transitionally DOWN or DOWN - transitionally UP
ANYSCALE_RAY_SERVE_HAPROXY_HEALTH_CHECK_FASTINTER = os.environ.get(
    "ANYSCALE_RAY_SERVE_HAPROXY_HEALTH_CHECK_FASTINTER", "250ms"
)

# Time interval between each haproxy health check attempt when the server is in the DOWN state
ANYSCALE_RAY_SERVE_HAPROXY_HEALTH_CHECK_DOWNINTER = os.environ.get(
    "ANYSCALE_RAY_SERVE_HAPROXY_HEALTH_CHECK_DOWNINTER", "250ms"
)

ANYSCALE_RAY_SERVE_REPLICA_GRPC_MAX_MESSAGE_LENGTH = int(
    # Default max message length in gRPC is 4MB, we keep that default
    os.environ.get(
        "ANYSCALE_RAY_SERVE_REPLICA_GRPC_MAX_MESSAGE_LENGTH", 4 * 1024 * 1024
    )
)

ANYSCALE_RAY_SERVE_PROXY_USE_GRPC = os.environ.get(
    "ANYSCALE_RAY_SERVE_PROXY_USE_GRPC"
) == "1" or (
    not os.environ.get("ANYSCALE_RAY_SERVE_PROXY_USE_GRPC") == "0"
    and RAY_SERVE_USE_GRPC_BY_DEFAULT
)

# Feature flag for prestarting workers in placement groups.
RAY_SERVE_PRESTART_PG_WORKERS = (
    os.environ.get("RAY_SERVE_PRESTART_PG_WORKERS", "1") == "1"
)

# How long the prestarted workers for placement groups should be kept alive without
# being used.
RAY_SERVE_PRESTART_PG_WORKERS_KEEP_ALIVE_S = int(
    os.environ.get("RAY_SERVE_PRESTART_PG_WORKERS_KEEP_ALIVE_S", "60")
)

# Feature flag to enable a limited form of direct ingress where ingress applications
# listen on port 8000 (HTTP) and 9000 (gRPC). No proxies will be started.
ANYSCALE_RAY_SERVE_ENABLE_DIRECT_INGRESS = (
    os.environ.get("ANYSCALE_RAY_SERVE_ENABLE_DIRECT_INGRESS", "0") == "1"
)
RAY_SERVE_DIRECT_INGRESS_MIN_HTTP_PORT = int(
    os.environ.get("RAY_SERVE_DIRECT_INGRESS_MIN_HTTP_PORT", "30000")
)
RAY_SERVE_DIRECT_INGRESS_MIN_GRPC_PORT = int(
    os.environ.get("RAY_SERVE_DIRECT_INGRESS_MIN_GRPC_PORT", "40000")
)
RAY_SERVE_DIRECT_INGRESS_MAX_HTTP_PORT = int(
    os.environ.get("RAY_SERVE_DIRECT_INGRESS_MAX_HTTP_PORT", "31000")
)
RAY_SERVE_DIRECT_INGRESS_MAX_GRPC_PORT = int(
    os.environ.get("RAY_SERVE_DIRECT_INGRESS_MAX_GRPC_PORT", "41000")
)
RAY_SERVE_DIRECT_INGRESS_PORT_RETRY_COUNT = int(
    os.environ.get("RAY_SERVE_DIRECT_INGRESS_PORT_RETRY_COUNT", "100")
)
# The minimum drain period for a HTTP proxy.
ANYSCALE_RAY_SERVE_DIRECT_INGRESS_MIN_DRAINING_PERIOD_S = float(
    os.environ.get("ANYSCALE_RAY_SERVE_DIRECT_INGRESS_MIN_DRAINING_PERIOD_S", "30")
)

# Feature flag to enable freezing the garbage collector on startup.
ANYSCALE_FREEZE_GC_ON_STARTUP = (
    os.environ.get("ANYSCALE_FREEZE_GC_ON_STARTUP", "0") == "1"
)

# Direct ingress must be enabled if HAProxy is enabled.
if ANYSCALE_RAY_SERVE_ENABLE_HA_PROXY:
    ANYSCALE_RAY_SERVE_ENABLE_DIRECT_INGRESS = True

# If throughput optimized Ray Serve is enabled, enable the following flags
# unless they are explicitly set.
if os.environ.get("RAY_SERVE_THROUGHPUT_OPTIMIZED", "0") == "1":
    ANYSCALE_RAY_SERVE_ENABLE_DIRECT_INGRESS = (
        os.environ.get("ANYSCALE_RAY_SERVE_ENABLE_DIRECT_INGRESS", "1") == "1"
    )
    ANYSCALE_FREEZE_GC_ON_STARTUP = (
        os.environ.get("ANYSCALE_FREEZE_GC_ON_STARTUP", "1") == "1"
    )
