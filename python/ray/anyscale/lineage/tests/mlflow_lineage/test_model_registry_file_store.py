"""Tests for model registry AnyscaleFileStore."""

from unittest import mock

import pytest

from ray.anyscale.lineage.mlflow_lineage.store.model_registry.file_store import (
    AnyscaleFileStore,
)
from ray.anyscale.lineage.tests.test_constants import (
    TEST_LOCAL_MLRUNS_PATH,
    TEST_MLFLOW_MODEL_NAME,
    TEST_MLFLOW_S3_URI,
)


def create_mock_model_version(version="1"):
    model_version = mock.Mock()
    model_version.version = version
    return model_version


@pytest.fixture
def mock_file_store(monkeypatch):
    monkeypatch.setattr(
        "mlflow.store.model_registry.file_store.FileStore.__init__",
        lambda self, root_directory=None: setattr(
            self, "root_directory", root_directory or "default-root"
        ),
    )
    monkeypatch.setattr(
        "mlflow.store.model_registry.file_store.FileStore.create_model_version",
        mock.Mock(return_value=create_mock_model_version()),
    )


class TestAnyscaleFileStore:
    """Tests for model registry AnyscaleFileStore."""

    def test_is_plugin_flag_set(self, mock_file_store):
        """is_plugin flag is set to True."""
        store = AnyscaleFileStore(store_uri=TEST_LOCAL_MLRUNS_PATH)
        assert store.is_plugin is True

    def test_create_model_version_triggers_lineage(self, mock_file_store, monkeypatch):
        """create_model_version triggers OpenLineage event emission."""
        from ray.anyscale.lineage.mlflow_lineage.store import model_registry

        model_registry.file_store._AnyscaleFileStoreClass = None

        process_mock = mock.Mock()
        monkeypatch.setattr(
            "ray.anyscale.lineage.mlflow_lineage.store.model_registry.utils.process_and_emit_ol_events_for_model_registration",
            process_mock,
        )
        store = AnyscaleFileStore(store_uri=TEST_LOCAL_MLRUNS_PATH)

        store.create_model_version(
            name=TEST_MLFLOW_MODEL_NAME, source=TEST_MLFLOW_S3_URI
        )

        process_mock.assert_called_once()

    def test_create_model_version_catches_exceptions(
        self, mock_file_store, monkeypatch
    ):
        """Exceptions from OpenLineage processing are caught and logged."""
        from ray.anyscale.lineage.mlflow_lineage.store import model_registry

        model_registry.file_store._AnyscaleFileStoreClass = None

        process_mock = mock.Mock(side_effect=RuntimeError("processing failed"))
        monkeypatch.setattr(
            "ray.anyscale.lineage.mlflow_lineage.store.model_registry.utils.process_and_emit_ol_events_for_model_registration",
            process_mock,
        )
        store = AnyscaleFileStore(store_uri=TEST_LOCAL_MLRUNS_PATH)

        result = store.create_model_version(
            name=TEST_MLFLOW_MODEL_NAME, source=TEST_MLFLOW_S3_URI
        )

        assert result.version == "1"


class TestAnyscaleFileStoreFactory:
    """Tests for factory function behavior."""

    def test_factory_caches_class(self, mock_file_store, monkeypatch):
        """Factory function caches the created class."""
        monkeypatch.setattr(
            "ray.anyscale.lineage.common.openlineage_client.AnyscaleOpenLineageClient",
            mock.Mock(return_value=mock.Mock()),
        )

        from ray.anyscale.lineage.mlflow_lineage.store import model_registry

        model_registry.file_store._AnyscaleFileStoreClass = None

        store1 = AnyscaleFileStore(store_uri=TEST_LOCAL_MLRUNS_PATH)
        store2 = AnyscaleFileStore(store_uri="/other/path")

        assert type(store1) is type(store2)
