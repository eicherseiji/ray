"""Tests for AnyscaleRestStore."""

from unittest import mock

import pytest

from ray.anyscale.lineage.mlflow_lineage.store.tracking.rest_store import (
    AnyscaleRestStore,
)
from ray.anyscale.lineage.tests.test_constants import (
    TEST_MLFLOW_HOST_EXAMPLE,
    TEST_MLFLOW_RUN_ID,
    TEST_MLFLOW_S3_URI,
)


@pytest.fixture
def mock_rest_store(monkeypatch):
    mock_host_creds = mock.Mock()
    mock_host_creds.host = TEST_MLFLOW_HOST_EXAMPLE
    monkeypatch.setattr(
        "mlflow.utils.credentials.get_default_host_creds",
        lambda store_uri: mock_host_creds,
    )
    monkeypatch.setattr(
        "mlflow.store.tracking.rest_store.RestStore.__init__",
        lambda self, func: None,
    )
    monkeypatch.setattr(
        "mlflow.store.tracking.rest_store.RestStore.record_logged_model",
        mock.Mock(),
    )


def create_mock_run(artifact_uri=TEST_MLFLOW_S3_URI):
    run = mock.Mock()
    run.info.artifact_uri = artifact_uri
    return run


def create_mock_model():
    model = mock.Mock()
    model.artifact_path = "test-model"
    model.flavors = {"python_function": {}}
    return model


class TestAnyscaleRestStore:
    """Tests for AnyscaleRestStore."""

    def test_is_plugin_flag_set(self, mock_rest_store):
        """is_plugin flag is set to True."""
        store = AnyscaleRestStore(store_uri=f"https://{TEST_MLFLOW_HOST_EXAMPLE}")
        assert store.is_plugin is True

    def test_record_logged_model_triggers_lineage(self, mock_rest_store, monkeypatch):
        """record_logged_model triggers OpenLineage event emission."""
        from ray.anyscale.lineage.mlflow_lineage.store import tracking

        # Reset cached class so monkeypatch is captured during class creation
        tracking.rest_store._AnyscaleRestStoreClass = None

        process_mock = mock.Mock()
        monkeypatch.setattr(
            "ray.anyscale.lineage.mlflow_lineage.store.tracking.utils.process_and_emit_ol_events_for_model_logging",
            process_mock,
        )
        store = AnyscaleRestStore(store_uri=f"https://{TEST_MLFLOW_HOST_EXAMPLE}")
        store.get_run = mock.Mock(return_value=create_mock_run())

        store.record_logged_model(TEST_MLFLOW_RUN_ID, create_mock_model())

        process_mock.assert_called_once()

    def test_record_logged_model_catches_exceptions(self, mock_rest_store, monkeypatch):
        """Exceptions from OpenLineage processing are caught and logged."""
        from ray.anyscale.lineage.mlflow_lineage.store import tracking

        # Reset cached class so monkeypatch is captured during class creation
        tracking.rest_store._AnyscaleRestStoreClass = None

        process_mock = mock.Mock(side_effect=RuntimeError("processing failed"))
        monkeypatch.setattr(
            "ray.anyscale.lineage.mlflow_lineage.store.tracking.utils.process_and_emit_ol_events_for_model_logging",
            process_mock,
        )
        store = AnyscaleRestStore(store_uri=f"https://{TEST_MLFLOW_HOST_EXAMPLE}")
        store.get_run = mock.Mock(return_value=create_mock_run())

        store.record_logged_model(TEST_MLFLOW_RUN_ID, create_mock_model())


class TestAnyscaleRestStoreFactory:
    """Tests for factory function behavior."""

    def test_factory_caches_class(self, mock_rest_store, monkeypatch):
        """Factory function caches the created class."""
        monkeypatch.setattr(
            "ray.anyscale.lineage.common.openlineage_client.AnyscaleOpenLineageClient",
            mock.Mock(return_value=mock.Mock()),
        )

        from ray.anyscale.lineage.mlflow_lineage.store import tracking

        tracking.rest_store._AnyscaleRestStoreClass = None

        store1 = AnyscaleRestStore(store_uri=f"https://{TEST_MLFLOW_HOST_EXAMPLE}")
        store2 = AnyscaleRestStore(store_uri="https://other.example.com")

        assert type(store1) is type(store2)
