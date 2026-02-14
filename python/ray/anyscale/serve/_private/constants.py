# Copyright (2023 and onwards) Anyscale, Inc.

import os

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

# Feature flag for prestarting workers in placement groups.
RAY_SERVE_PRESTART_PG_WORKERS = (
    os.environ.get("RAY_SERVE_PRESTART_PG_WORKERS", "1") == "1"
)

# How long the prestarted workers for placement groups should be kept alive without
# being used.
RAY_SERVE_PRESTART_PG_WORKERS_KEEP_ALIVE_S = int(
    os.environ.get("RAY_SERVE_PRESTART_PG_WORKERS_KEEP_ALIVE_S", "60")
)

# Feature flag to enable freezing the garbage collector on startup.
ANYSCALE_FREEZE_GC_ON_STARTUP = (
    os.environ.get("ANYSCALE_FREEZE_GC_ON_STARTUP", "0") == "1"
)

# If throughput optimized Ray Serve is enabled, enable the following flags
# unless they are explicitly set.
if os.environ.get("RAY_SERVE_THROUGHPUT_OPTIMIZED", "0") == "1":
    ANYSCALE_FREEZE_GC_ON_STARTUP = (
        os.environ.get("ANYSCALE_FREEZE_GC_ON_STARTUP", "1") == "1"
    )
