"""Tests for model registry AnyscaleRestStore."""

from unittest import mock

import pytest

from ray.anyscale.lineage.mlflow_lineage.store.model_registry.rest_store import (
    AnyscaleRestStore,
)
from ray.anyscale.lineage.tests.test_constants import (
    TEST_MLFLOW_HOST_EXAMPLE,
    TEST_MLFLOW_MODEL_NAME,
    TEST_MLFLOW_S3_URI,
)


def create_mock_model_version(version="1"):
    model_version = mock.Mock()
    model_version.version = version
    return model_version


@pytest.fixture
def mock_rest_store(monkeypatch):
    mock_host_creds = mock.Mock()
    mock_host_creds.host = TEST_MLFLOW_HOST_EXAMPLE
    monkeypatch.setattr(
        "mlflow.utils.credentials.get_default_host_creds",
        lambda store_uri: mock_host_creds,
    )
    monkeypatch.setattr(
        "mlflow.store.model_registry.rest_store.RestStore.__init__",
        lambda self, func: None,
    )
    monkeypatch.setattr(
        "mlflow.store.model_registry.rest_store.RestStore.create_model_version",
        mock.Mock(return_value=create_mock_model_version()),
    )


class TestAnyscaleRestStore:
    """Tests for model registry AnyscaleRestStore."""

    def test_is_plugin_flag_set(self, mock_rest_store):
        """is_plugin flag is set to True."""
        store = AnyscaleRestStore(store_uri=f"https://{TEST_MLFLOW_HOST_EXAMPLE}")
        assert store.is_plugin is True

    def test_create_model_version_triggers_lineage(self, mock_rest_store, monkeypatch):
        """create_model_version triggers OpenLineage event emission."""
        from ray.anyscale.lineage.mlflow_lineage.store import model_registry

        model_registry.rest_store._AnyscaleRestStoreClass = None

        process_mock = mock.Mock()
        monkeypatch.setattr(
            "ray.anyscale.lineage.mlflow_lineage.store.model_registry.utils.process_and_emit_ol_events_for_model_registration",
            process_mock,
        )
        store = AnyscaleRestStore(store_uri=f"https://{TEST_MLFLOW_HOST_EXAMPLE}")

        store.create_model_version(
            name=TEST_MLFLOW_MODEL_NAME, source=TEST_MLFLOW_S3_URI
        )

        process_mock.assert_called_once()

    def test_create_model_version_catches_exceptions(
        self, mock_rest_store, monkeypatch
    ):
        """Exceptions from OpenLineage processing are caught and logged."""
        from ray.anyscale.lineage.mlflow_lineage.store import model_registry

        model_registry.rest_store._AnyscaleRestStoreClass = None

        process_mock = mock.Mock(side_effect=RuntimeError("processing failed"))
        monkeypatch.setattr(
            "ray.anyscale.lineage.mlflow_lineage.store.model_registry.utils.process_and_emit_ol_events_for_model_registration",
            process_mock,
        )
        store = AnyscaleRestStore(store_uri=f"https://{TEST_MLFLOW_HOST_EXAMPLE}")

        result = store.create_model_version(
            name=TEST_MLFLOW_MODEL_NAME, source=TEST_MLFLOW_S3_URI
        )

        assert result.version == "1"


class TestAnyscaleRestStoreFactory:
    """Tests for factory function behavior."""

    def test_factory_caches_class(self, mock_rest_store, monkeypatch):
        """Factory function caches the created class."""
        monkeypatch.setattr(
            "ray.anyscale.lineage.common.openlineage_client.AnyscaleOpenLineageClient",
            mock.Mock(return_value=mock.Mock()),
        )

        from ray.anyscale.lineage.mlflow_lineage.store import model_registry

        model_registry.rest_store._AnyscaleRestStoreClass = None

        store1 = AnyscaleRestStore(store_uri=f"https://{TEST_MLFLOW_HOST_EXAMPLE}")
        store2 = AnyscaleRestStore(store_uri="https://other.example.com")

        assert type(store1) is type(store2)
