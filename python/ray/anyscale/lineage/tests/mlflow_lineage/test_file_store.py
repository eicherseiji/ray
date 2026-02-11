"""Tests for AnyscaleFileStore."""

from unittest import mock

import pytest

from ray.anyscale.lineage.mlflow_lineage.store.tracking.file_store import (
    AnyscaleFileStore,
)
from ray.anyscale.lineage.tests.test_constants import (
    TEST_LOCAL_MLRUNS_PATH,
    TEST_MLFLOW_RUN_ID,
    TEST_MLFLOW_S3_URI,
)


@pytest.fixture
def mock_file_store(monkeypatch):
    monkeypatch.setattr(
        "mlflow.store.tracking.file_store.FileStore.__init__",
        lambda self, root_directory=None, artifact_root_uri=None: setattr(
            self, "root_directory", root_directory or "default-root"
        ),
    )
    monkeypatch.setattr(
        "mlflow.store.tracking.file_store.FileStore.record_logged_model",
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


class TestAnyscaleFileStore:
    """Tests for AnyscaleFileStore."""

    def test_is_plugin_flag_set(self, mock_file_store):
        """is_plugin flag is set to True."""
        store = AnyscaleFileStore(store_uri=TEST_LOCAL_MLRUNS_PATH)
        assert store.is_plugin is True

    def test_record_logged_model_triggers_lineage(self, mock_file_store, monkeypatch):
        """record_logged_model triggers OpenLineage event emission."""
        from ray.anyscale.lineage.mlflow_lineage.store import tracking

        # Reset cached class so monkeypatch is captured during class creation
        tracking.file_store._AnyscaleFileStoreClass = None

        process_mock = mock.Mock()
        monkeypatch.setattr(
            "ray.anyscale.lineage.mlflow_lineage.store.tracking.utils.process_and_emit_ol_events_for_model_logging",
            process_mock,
        )
        store = AnyscaleFileStore(store_uri=TEST_LOCAL_MLRUNS_PATH)
        store.get_run = mock.Mock(return_value=create_mock_run())

        store.record_logged_model(TEST_MLFLOW_RUN_ID, create_mock_model())

        process_mock.assert_called_once()

    def test_record_logged_model_catches_exceptions(self, mock_file_store, monkeypatch):
        """Exceptions from OpenLineage processing are caught and logged."""
        from ray.anyscale.lineage.mlflow_lineage.store import tracking

        # Reset cached class so monkeypatch is captured during class creation
        tracking.file_store._AnyscaleFileStoreClass = None

        process_mock = mock.Mock(side_effect=RuntimeError("processing failed"))
        monkeypatch.setattr(
            "ray.anyscale.lineage.mlflow_lineage.store.tracking.utils.process_and_emit_ol_events_for_model_logging",
            process_mock,
        )
        store = AnyscaleFileStore(store_uri=TEST_LOCAL_MLRUNS_PATH)
        store.get_run = mock.Mock(return_value=create_mock_run())

        store.record_logged_model(TEST_MLFLOW_RUN_ID, create_mock_model())


class TestAnyscaleFileStoreFactory:
    """Tests for factory function behavior."""

    def test_factory_caches_class(self, mock_file_store, monkeypatch):
        """Factory function caches the created class."""
        monkeypatch.setattr(
            "ray.anyscale.lineage.common.openlineage_client.AnyscaleOpenLineageClient",
            mock.Mock(return_value=mock.Mock()),
        )

        from ray.anyscale.lineage.mlflow_lineage.store import tracking

        tracking.file_store._AnyscaleFileStoreClass = None

        store1 = AnyscaleFileStore(store_uri=TEST_LOCAL_MLRUNS_PATH)
        store2 = AnyscaleFileStore(store_uri="/other/path")

        assert type(store1) is type(store2)
