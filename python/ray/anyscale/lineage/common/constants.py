import os

from ray.anyscale.lineage.version import __version__


# Environment variable prefix for this package
ANYSCALE_LINEAGE_ENV_PREFIX = "ANYSCALE_LINEAGE"

# URL of the GitHub repository for Anyscale lineage plugins
# This URL will appear in the lineage events, so conceal the actual repo identity
REPO_URL = "https://github.com/anyscale/lineage"

# Global OpenLineage producer for all Anyscale lineage plugins
OL_PRODUCER = f"{REPO_URL}/tree/{__version__}/lineage"

# Logging configuration
LOG_LEVEL = os.getenv(f"{ANYSCALE_LINEAGE_ENV_PREFIX}_LOG_LEVEL", "INFO")
LOG_ENCODING = os.getenv(f"{ANYSCALE_LINEAGE_ENV_PREFIX}_LOG_ENCODING", "JSON")
