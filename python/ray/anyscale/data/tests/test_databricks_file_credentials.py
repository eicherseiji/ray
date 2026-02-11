"""Unit tests for Anyscale FileCredentialProvider for Databricks."""

import os
import pickle
import tempfile
from typing import Optional
from unittest import mock

import pytest

from ray.anyscale.data.datasource.databricks_file_credentials import (
    FileCredentialProvider,
    resolve_credential_provider,
)

# Test constants
TEST_TOKEN = "test_token_123"
TEST_HOST = "https://test.databricks.com"


def write_credentials_file(
    file_path: str,
    token: str = TEST_TOKEN,
    host: str = TEST_HOST,
    profile: Optional[str] = None,
) -> None:
    """Write a credentials file in INI format.

    Args:
        file_path: Path to write the credentials file.
        token: Token value to write.
        host: Host value to write.
        profile: Profile name. Defaults to FileCredentialProvider.DEFAULT_PROFILE.
    """
    if profile is None:
        profile = FileCredentialProvider.DEFAULT_PROFILE
    with open(file_path, "w") as f:
        f.write(f"[{profile}]\n")
        f.write(f"{FileCredentialProvider.CONFIG_FIELD_HOST} = {host}\n")
        f.write(f"{FileCredentialProvider.CONFIG_FIELD_TOKEN} = {token}\n")


class TestGetDefaultCredentialsPath:
    """Tests for FileCredentialProvider._get_default_credentials_path method."""

    def test_explicit_path_takes_precedence(self):
        """Test that DATABRICKS_CREDENTIALS_PATH env var takes precedence."""
        with mock.patch.dict(
            os.environ,
            {
                FileCredentialProvider.ENV_VAR_CREDENTIALS_PATH: "/custom/path/creds.cfg",
                FileCredentialProvider.ENV_VAR_ANYSCALE_WORKSPACE_ID: "ws-123",
            },
        ):
            result = FileCredentialProvider._get_default_credentials_path()
            assert result == "/custom/path/creds.cfg"

    @pytest.mark.parametrize(
        "env_var_attr,anyscale_id",
        [
            ("ENV_VAR_ANYSCALE_WORKSPACE_ID", "ws-test-123"),
            ("ENV_VAR_ANYSCALE_JOB_ID", "job-test-456"),
        ],
    )
    def test_computes_path_from_anyscale_id(self, env_var_attr, anyscale_id):
        """Test path computed from Anyscale environment variables."""
        env_var = getattr(FileCredentialProvider, env_var_attr)
        with mock.patch.dict(os.environ, {env_var: anyscale_id}, clear=True):
            result = FileCredentialProvider._get_default_credentials_path()
            expected = os.path.join(
                FileCredentialProvider.ANYSCALE_CREDENTIALS_BASE_PATH,
                anyscale_id,
                FileCredentialProvider.ANYSCALE_CREDENTIALS_FILENAME,
            )
            assert result == expected

    def test_workspace_id_takes_precedence_over_job_id(self):
        """Test ANYSCALE_WORKSPACE_ID takes precedence over ANYSCALE_JOB_ID."""
        with mock.patch.dict(
            os.environ,
            {
                FileCredentialProvider.ENV_VAR_ANYSCALE_WORKSPACE_ID: "ws-123",
                FileCredentialProvider.ENV_VAR_ANYSCALE_JOB_ID: "job-456",
            },
            clear=True,
        ):
            result = FileCredentialProvider._get_default_credentials_path()
            assert "ws-123" in result
            assert "job-456" not in result

    def test_returns_none_outside_anyscale(self):
        """Test returns None when not in Anyscale environment."""
        with mock.patch.dict(os.environ, {}, clear=True):
            assert FileCredentialProvider._get_default_credentials_path() is None


class TestFileCredentialProvider:
    """Tests for FileCredentialProvider."""

    @pytest.fixture
    def credentials_file(self):
        """Create a temporary credentials file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".cfg", delete=False) as f:
            pass  # Just create the file
        write_credentials_file(f.name, token=TEST_TOKEN, host=TEST_HOST)
        yield f.name
        os.unlink(f.name)

    def test_reads_credentials(self, credentials_file):
        """Test basic credential reading from INI file."""
        provider = FileCredentialProvider(file_path=credentials_file)
        assert provider.get_token() == TEST_TOKEN
        assert provider.get_host() == TEST_HOST

    def test_caching_prevents_repeated_reads(self, credentials_file):
        """Test that credentials are cached within TTL."""
        provider = FileCredentialProvider(
            file_path=credentials_file, cache_duration_seconds=10
        )
        assert provider.get_token() == TEST_TOKEN

        # Modify file (should not affect cached result)
        write_credentials_file(credentials_file, token="new_token", host="new_host")

        assert provider.get_token() == TEST_TOKEN

    def test_cache_disabled_reads_every_time(self, credentials_file):
        """Test that cache_duration=0 reads file on every access."""
        provider = FileCredentialProvider(
            file_path=credentials_file, cache_duration_seconds=0
        )
        assert provider.get_token() == TEST_TOKEN

        write_credentials_file(credentials_file, token="new_token", host="new_host")

        assert provider.get_token() == "new_token"

    def test_invalidate_clears_cache(self, credentials_file):
        """Test that invalidate() clears cache and forces re-read."""
        provider = FileCredentialProvider(
            file_path=credentials_file, cache_duration_seconds=300
        )
        assert provider.get_token() == TEST_TOKEN

        write_credentials_file(
            credentials_file, token="refreshed_token", host="new_host"
        )

        provider.invalidate()
        assert provider.get_token() == "refreshed_token"

    def test_missing_file_raises_error(self):
        """Test that missing file raises ValueError."""
        provider = FileCredentialProvider(file_path="/nonexistent/path/creds.cfg")
        with pytest.raises(ValueError, match="Credentials file not found"):
            provider.get_token()

    def test_invalid_ini_raises_error(self):
        """Test that invalid INI format raises ValueError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".cfg", delete=False) as f:
            # Write invalid INI format (unclosed section bracket)
            f.write("[invalid section\n")
            f.write("token = value\n")
            f.flush()
            try:
                provider = FileCredentialProvider(file_path=f.name)
                with pytest.raises(ValueError, match="Invalid INI format"):
                    provider.get_token()
            finally:
                os.unlink(f.name)

    def test_missing_profile_raises_error(self):
        """Test that missing profile section raises ValueError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".cfg", delete=False) as f:
            pass  # Just create the file
        try:
            # Write valid INI but with wrong profile name
            write_credentials_file(f.name, profile="wrong-profile")
            provider = FileCredentialProvider(file_path=f.name)
            with pytest.raises(
                ValueError,
                match=f"Profile '{FileCredentialProvider.DEFAULT_PROFILE}' not found",
            ):
                provider.get_token()
        finally:
            os.unlink(f.name)

    @pytest.mark.parametrize(
        "missing_field_attr,accessor_method",
        [
            ("CONFIG_FIELD_TOKEN", "get_token"),
            ("CONFIG_FIELD_HOST", "get_host"),
        ],
    )
    def test_missing_field_raises_error(self, missing_field_attr, accessor_method):
        """Test that missing fields raise ValueError."""
        missing_field = getattr(FileCredentialProvider, missing_field_attr)
        other_field = (
            FileCredentialProvider.CONFIG_FIELD_HOST
            if missing_field == FileCredentialProvider.CONFIG_FIELD_TOKEN
            else FileCredentialProvider.CONFIG_FIELD_TOKEN
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".cfg", delete=False) as f:
            f.write(
                f"[{FileCredentialProvider.DEFAULT_PROFILE}]\n{other_field} = some_value\n"
            )
            f.flush()
            try:
                provider = FileCredentialProvider(file_path=f.name)
                with pytest.raises(
                    ValueError, match=f"missing '{missing_field}' field"
                ):
                    getattr(provider, accessor_method)()
            finally:
                os.unlink(f.name)

    def test_requires_path_outside_anyscale(self):
        """Test that error is raised when no path and not in Anyscale env."""
        with mock.patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="file_path is required"):
                FileCredentialProvider()

    @pytest.mark.parametrize(
        "env_var_attr,anyscale_id",
        [
            ("ENV_VAR_ANYSCALE_WORKSPACE_ID", "ws-test"),
            ("ENV_VAR_ANYSCALE_JOB_ID", "job-test"),
        ],
    )
    def test_uses_anyscale_path(self, env_var_attr, anyscale_id):
        """Test that provider uses computed path from Anyscale env vars."""
        env_var = getattr(FileCredentialProvider, env_var_attr)
        with mock.patch.dict(os.environ, {env_var: anyscale_id}, clear=True):
            provider = FileCredentialProvider()
            expected = os.path.join(
                FileCredentialProvider.ANYSCALE_CREDENTIALS_BASE_PATH,
                anyscale_id,
                FileCredentialProvider.ANYSCALE_CREDENTIALS_FILENAME,
            )
            assert provider._file_path == expected

    def test_explicit_path_overrides_env(self):
        """Test that explicit file_path overrides env var."""
        with mock.patch.dict(
            os.environ,
            {FileCredentialProvider.ENV_VAR_CREDENTIALS_PATH: "/env/path.cfg"},
        ):
            provider = FileCredentialProvider(file_path="/explicit/path.cfg")
            assert provider._file_path == "/explicit/path.cfg"

    def test_is_picklable(self, credentials_file):
        """Verify FileCredentialProvider can be pickled and unpickled."""
        provider = FileCredentialProvider(file_path=credentials_file)
        unpickled = pickle.loads(pickle.dumps(provider))
        assert unpickled.get_token() == TEST_TOKEN
        assert unpickled.get_host() == TEST_HOST


class TestResolveCredentialProviderWithFileProvider:
    """Tests for resolve_credential_provider with file-based fallback."""

    def test_resolve_prefers_env_over_file(self):
        """Test that env vars take precedence over file-based credentials."""
        from ray.data._internal.datasource.databricks_credentials import (
            EnvironmentCredentialProvider,
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".cfg", delete=False) as f:
            pass  # Just create the file
        try:
            write_credentials_file(f.name, token="file_token", host="file_host")
            with mock.patch.dict(
                os.environ,
                {
                    FileCredentialProvider.ENV_VAR_CREDENTIALS_PATH: f.name,
                    "DATABRICKS_TOKEN": "env_token",
                    "DATABRICKS_HOST": "env_host",
                },
            ):
                result = resolve_credential_provider()
                assert isinstance(result, EnvironmentCredentialProvider)
                assert result.get_token() == "env_token"
        finally:
            os.unlink(f.name)

    def test_resolve_falls_back_to_file_when_env_missing(self):
        """Test fallback to FileCredentialProvider when env vars not set."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".cfg", delete=False) as f:
            pass  # Just create the file
        try:
            write_credentials_file(f.name, token="file_token", host="file_host")
            with mock.patch.dict(
                os.environ,
                {FileCredentialProvider.ENV_VAR_CREDENTIALS_PATH: f.name},
                clear=True,
            ):
                result = resolve_credential_provider()
                assert isinstance(result, FileCredentialProvider)
                assert result.get_token() == "file_token"
        finally:
            os.unlink(f.name)

    def test_resolve_raises_when_no_credentials_available(self):
        """Test that ValueError is raised when no credentials are available."""
        with mock.patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="No Databricks credentials found"):
                resolve_credential_provider()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
