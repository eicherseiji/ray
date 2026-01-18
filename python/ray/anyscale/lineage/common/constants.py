import os

from ray.anyscale.lineage.version import __version__

# Environment variable prefix for this package
ANYSCALE_LINEAGE_ENV_PREFIX = "ANYSCALE_LINEAGE"

# URL of the GitHub repository for Anyscale lineage plugins
# This URL will appear in the lineage events, so conceal the actual repo identity
REPO_URL = "https://github.com/anyscale/lineage"

# Global OpenLineage producer for all Anyscale lineage plugins
OPENLINEAGE_PRODUCER = f"{REPO_URL}/tree/{__version__}/lineage"

# Environment variable to enable/disable lineage tracking
TRACKING_ENABLED = (
    os.getenv(f"{ANYSCALE_LINEAGE_ENV_PREFIX}_TRACKING_ENABLED", "False").lower()
    == "true"
)

# Logging configuration
# Log level can be one of: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_LEVEL = os.getenv(f"{ANYSCALE_LINEAGE_ENV_PREFIX}_LOG_LEVEL", "INFO")
# Log encoding can be one of: JSON, TEXT
LOG_ENCODING = os.getenv(f"{ANYSCALE_LINEAGE_ENV_PREFIX}_LOG_ENCODING", "JSON")
LOG_ENABLE_CONSOLE = (
    os.getenv(f"{ANYSCALE_LINEAGE_ENV_PREFIX}_LOG_ENABLE_CONSOLE", "False").lower()
    == "true"
)
# OpenLineage events log filename
LINEAGE_EVENTS_LOG_FILENAME = "lineage_events.jsonl"

# Error handling configuration
# Ignore lineage tracking errors and continue workload execution
IGNORE_ERRORS = (
    os.getenv(f"{ANYSCALE_LINEAGE_ENV_PREFIX}_IGNORE_ERRORS", "True").lower() == "true"
)
