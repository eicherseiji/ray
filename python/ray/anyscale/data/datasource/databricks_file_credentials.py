"""Anyscale-specific file-based Databricks credential provider.

This module provides file-based credential management for Databricks,
specifically designed to support Anyscale's credential injection mechanism.
"""

import logging
import os
from typing import Optional

from ray.data._internal.datasource.databricks_credentials import (
    DatabricksCredentialProvider,
    EnvironmentCredentialProvider,
)

logger = logging.getLogger(__name__)


class FileCredentialProvider(DatabricksCredentialProvider):
    """Credential provider that reads from an INI-style config file.

    Uses the standard Databricks CLI configuration format. The credentials file
    created by Anyscale uses the profile name "anyscale-profile", which is
    compatible with Databricks CLI/SDK when specifying profile="anyscale-profile".

    Expected INI format::

        [anyscale-profile]
        host = https://adb-12345.azuredatabricks.net
        token = dapi1234567890abcdef

    Args:
        file_path: Path to credentials file. Defaults to Anyscale path
            (computed from DATABRICKS_CREDENTIALS_PATH env var, or
            /tmp/anyscale/integrations/<ID>/databricks/credentials.cfg
            where ID is ANYSCALE_WORKSPACE_ID or ANYSCALE_JOB_ID).
        cache_duration_seconds: Cache duration before re-reading file.
            Default is 60 seconds. Set to 0 to disable caching.

    Raises:
        ValueError: If file_path is None and not in Anyscale environment.
    """

    # Environment variable names
    ENV_VAR_CREDENTIALS_PATH = "DATABRICKS_CREDENTIALS_PATH"
    ENV_VAR_ANYSCALE_WORKSPACE_ID = "ANYSCALE_WORKSPACE_ID"
    ENV_VAR_ANYSCALE_JOB_ID = "ANYSCALE_JOB_ID"

    # Anyscale credentials path configuration
    ANYSCALE_CREDENTIALS_BASE_PATH = "/tmp/anyscale/integrations"
    ANYSCALE_CREDENTIALS_FILENAME = "databricks/credentials.cfg"

    # INI config field names (standard Databricks CLI format)
    CONFIG_FIELD_TOKEN = "token"
    CONFIG_FIELD_HOST = "host"
    DEFAULT_PROFILE = "anyscale-profile"

    # Default cache duration in seconds
    DEFAULT_CACHE_DURATION_SECONDS = 60

    @staticmethod
    def _get_default_credentials_path() -> Optional[str]:
        """Get default credentials path based on Anyscale environment.

        Checks in order:
        1. DATABRICKS_CREDENTIALS_PATH env var (explicit override)
        2. Computed path: /tmp/anyscale/integrations/<ID>/databricks/credentials.cfg
           Where <ID> is ANYSCALE_WORKSPACE_ID or ANYSCALE_JOB_ID.

        Returns:
            Path to credentials file, or None if not in Anyscale environment.
        """
        explicit_path = os.environ.get(FileCredentialProvider.ENV_VAR_CREDENTIALS_PATH)
        if explicit_path:
            return explicit_path

        anyscale_id = os.environ.get(
            FileCredentialProvider.ENV_VAR_ANYSCALE_WORKSPACE_ID
        ) or os.environ.get(FileCredentialProvider.ENV_VAR_ANYSCALE_JOB_ID)
        if anyscale_id:
            return os.path.join(
                FileCredentialProvider.ANYSCALE_CREDENTIALS_BASE_PATH,
                anyscale_id,
                FileCredentialProvider.ANYSCALE_CREDENTIALS_FILENAME,
            )
        return None

    def __init__(
        self,
        file_path: Optional[str] = None,
        cache_duration_seconds: Optional[int] = None,
    ):
        if file_path is None:
            file_path = self._get_default_credentials_path()
            if file_path is None:
                raise ValueError(
                    "file_path is required when not in Anyscale environment. "
                    f"Set {self.ENV_VAR_ANYSCALE_WORKSPACE_ID} or "
                    f"{self.ENV_VAR_ANYSCALE_JOB_ID}, or provide file_path explicitly."
                )
        if cache_duration_seconds is None:
            cache_duration_seconds = self.DEFAULT_CACHE_DURATION_SECONDS

        self._file_path = file_path
        self._cache_duration_seconds = cache_duration_seconds
        self._cached_token: Optional[str] = None
        self._cached_host: Optional[str] = None
        self._cache_time: float = 0

    def _is_cache_expired(self) -> bool:
        """Check if cached credentials are expired or missing."""
        import time

        if self._cached_token is None or self._cached_host is None:
            return True
        if self._cache_duration_seconds <= 0:
            return True
        return (time.time() - self._cache_time) >= self._cache_duration_seconds

    def _load_credentials(self) -> None:
        """Load credentials from file and update cache."""
        import configparser
        import time

        try:
            config = configparser.ConfigParser()
            files_read = config.read(self._file_path)
            if not files_read:
                raise ValueError(
                    f"Credentials file not found: '{self._file_path}'. "
                    "Ensure Databricks integration is configured."
                )

            if not config.has_section(self.DEFAULT_PROFILE):
                raise ValueError(
                    f"Profile '{self.DEFAULT_PROFILE}' not found in '{self._file_path}'."
                )

            # Extract token and host into local variables first.
            # Only update cache after both are validated to avoid partial
            # cache state where token is cached but host validation fails.
            try:
                token = config.get(self.DEFAULT_PROFILE, self.CONFIG_FIELD_TOKEN)
                if not token:
                    raise ValueError(
                        f"'{self._file_path}' [{self.DEFAULT_PROFILE}] "
                        f"missing '{self.CONFIG_FIELD_TOKEN}' field."
                    )
            except configparser.NoOptionError:
                raise ValueError(
                    f"'{self._file_path}' [{self.DEFAULT_PROFILE}] "
                    f"missing '{self.CONFIG_FIELD_TOKEN}' field."
                )

            try:
                host = config.get(self.DEFAULT_PROFILE, self.CONFIG_FIELD_HOST)
                if not host:
                    raise ValueError(
                        f"'{self._file_path}' [{self.DEFAULT_PROFILE}] "
                        f"missing '{self.CONFIG_FIELD_HOST}' field."
                    )
            except configparser.NoOptionError:
                raise ValueError(
                    f"'{self._file_path}' [{self.DEFAULT_PROFILE}] "
                    f"missing '{self.CONFIG_FIELD_HOST}' field."
                )

            # Update cache atomically after both values are validated
            self._cached_token = token
            self._cached_host = host
            self._cache_time = time.time()
            logger.debug(f"Read credentials from {self._file_path}")

        except configparser.Error as e:
            raise ValueError(f"Invalid INI format in '{self._file_path}': {e}")
        except PermissionError:
            raise ValueError(f"Permission denied reading '{self._file_path}'")

    def get_token(self) -> str:
        """Get the token from credentials file."""
        if self._is_cache_expired():
            self._load_credentials()
        return self._cached_token

    def get_host(self) -> str:
        """Get the host from credentials file."""
        if self._is_cache_expired():
            self._load_credentials()
        return self._cached_host

    def invalidate(self) -> None:
        """Clear cached credentials to force re-read on next access."""
        self._cached_token = None
        self._cached_host = None
        self._cache_time = 0


def resolve_credential_provider(credential_provider=None):
    """Resolve credential provider with precedence.

    Resolution order:
    1. Explicit credential_provider argument (if provided)
    2. EnvironmentCredentialProvider (if environment variables are set)
    3. FileCredentialProvider (if Anyscale credentials file exists)

    Args:
        credential_provider: An explicit credential provider instance.
            If None, falls back to environment or file-based credentials.

    Returns:
        A DatabricksCredentialProvider instance.
    """
    if credential_provider is not None:
        return credential_provider

    # Try environment variables first
    try:
        return EnvironmentCredentialProvider()
    except ValueError:
        pass

    # Fall back to file-based credentials (Anyscale integration)
    credentials_path = FileCredentialProvider._get_default_credentials_path()
    if credentials_path and os.path.exists(credentials_path):
        return FileCredentialProvider(file_path=credentials_path)

    raise ValueError(
        "No Databricks credentials found. Set DATABRICKS_TOKEN and DATABRICKS_HOST "
        "environment variables, or configure Anyscale Databricks integration."
    )
