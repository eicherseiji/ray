from unittest import mock

import pytest

from ray.anyscale.lineage.common.exceptions import (
    AnyscaleLineageMLflowError,
)
from ray.anyscale.lineage.mlflow_lineage.store.tracking.file_store import (
    AnyscaleFileStore,
)
from ray.anyscale.lineage.tests.test_constants import (
    MLFLOW_ARTIFACT_REPO_PREFIX,
    MLFLOW_TRACKING_STORE_FILE_PREFIX,
    TEST_LOCAL_ARTIFACTS_PATH,
    TEST_LOCAL_MLRUNS_PATH,
    TEST_MLFLOW_MODEL_NAME,
    TEST_MLFLOW_RUN_ID,
)


def test_anyscale_file_store_initializes_client(monkeypatch) -> None:
    monkeypatch.setattr(
        "mlflow.store.tracking.file_store.FileStore.__init__",
        lambda self, root_directory=None, artifact_root_uri=None: setattr(
            self, "root_directory", root_directory or "default-root"
        ),
    )
    super_record = mock.Mock()
    monkeypatch.setattr(
        "mlflow.store.tracking.file_store.FileStore.record_logged_model",
        super_record,
    )

    process_mock = mock.Mock()
    monkeypatch.setattr(
        "ray.anyscale.lineage.mlflow_lineage.store.tracking.utils.process_and_emit_ol_events_for_model_logging",
        process_mock,
    )

    store = AnyscaleFileStore(store_uri=TEST_LOCAL_MLRUNS_PATH)

    mock_run = mock.Mock()
    store.get_run = mock.Mock(return_value=mock_run)

    assert store.host == TEST_LOCAL_MLRUNS_PATH
    assert hasattr(store, "ol_client")
    assert store.is_plugin is True

    store.record_logged_model(TEST_MLFLOW_RUN_ID, TEST_MLFLOW_MODEL_NAME)

    super_record.assert_called_once_with(TEST_MLFLOW_RUN_ID, TEST_MLFLOW_MODEL_NAME)
    process_mock.assert_called_once()
    call_args = process_mock.call_args
    assert call_args.kwargs["mlflow_host"] == TEST_LOCAL_MLRUNS_PATH
    assert call_args.kwargs["run"] == mock_run
    assert call_args.kwargs["mlflow_model"] == TEST_MLFLOW_MODEL_NAME


def test_anyscale_file_store_strips_plugin_prefix(monkeypatch) -> None:
    """Test that AnyscaleFileStore replaces the plugin prefix with 'file'."""
    monkeypatch.setattr(
        "mlflow.store.tracking.file_store.FileStore.__init__",
        lambda self, root_directory=None, artifact_root_uri=None: setattr(
            self, "root_directory", root_directory or "default-root"
        ),
    )

    mock_client = mock.Mock()
    monkeypatch.setattr(
        "ray.anyscale.lineage.common.openlineage_client.AnyscaleOpenLineageClient",
        mock.Mock(return_value=mock_client),
    )

    store = AnyscaleFileStore(
        store_uri=f"{MLFLOW_TRACKING_STORE_FILE_PREFIX}{TEST_LOCAL_MLRUNS_PATH}"
    )

    assert store.host == f"file:{TEST_LOCAL_MLRUNS_PATH}"


def test_anyscale_file_store_strips_artifact_repo_prefix(monkeypatch) -> None:
    """Test that AnyscaleFileStore strips the artifact repo prefix from artifact_uri."""

    def mock_init(self, root_directory=None, artifact_root_uri=None):
        self.root_directory = root_directory or "default-root"
        self.artifact_uri = artifact_root_uri

    monkeypatch.setattr(
        "mlflow.store.tracking.file_store.FileStore.__init__",
        mock_init,
    )

    mock_client = mock.Mock()
    monkeypatch.setattr(
        "ray.anyscale.lineage.common.openlineage_client.AnyscaleOpenLineageClient",
        mock.Mock(return_value=mock_client),
    )

    store = AnyscaleFileStore(
        store_uri=TEST_LOCAL_MLRUNS_PATH,
        artifact_uri=f"{MLFLOW_ARTIFACT_REPO_PREFIX}{TEST_LOCAL_ARTIFACTS_PATH}",
    )

    assert store.artifact_uri == TEST_LOCAL_ARTIFACTS_PATH


def test_anyscale_file_store_factory_caches_class(monkeypatch) -> None:
    """Test that the factory function caches the created class."""
    monkeypatch.setattr(
        "mlflow.store.tracking.file_store.FileStore.__init__",
        lambda self, root_directory=None, artifact_root_uri=None: setattr(
            self, "root_directory", root_directory or "default-root"
        ),
    )

    mock_client = mock.Mock()
    monkeypatch.setattr(
        "ray.anyscale.lineage.common.openlineage_client.AnyscaleOpenLineageClient",
        mock.Mock(return_value=mock_client),
    )

    from ray.anyscale.lineage.mlflow_lineage.store import tracking

    tracking.file_store._AnyscaleFileStoreClass = None

    store1 = AnyscaleFileStore(store_uri=TEST_LOCAL_MLRUNS_PATH)
    store2 = AnyscaleFileStore(store_uri="/tmp/other_path")

    assert type(store1) is type(store2)
    assert store1.__class__.__name__ == "AnyscaleFileStore"


def test_anyscale_file_store_record_logged_model_catches_exceptions(
    monkeypatch,
) -> None:
    """Test that record_logged_model catches and wraps exceptions from OpenLineage processing."""
    monkeypatch.setattr(
        "mlflow.store.tracking.file_store.FileStore.__init__",
        lambda self, root_directory=None, artifact_root_uri=None: setattr(
            self, "root_directory", root_directory or "default-root"
        ),
    )

    super_record = mock.Mock()
    monkeypatch.setattr(
        "mlflow.store.tracking.file_store.FileStore.record_logged_model",
        super_record,
    )

    process_mock = mock.Mock(side_effect=RuntimeError("OpenLineage processing failed"))
    monkeypatch.setattr(
        "ray.anyscale.lineage.mlflow_lineage.store.tracking.utils.process_and_emit_ol_events_for_model_logging",
        process_mock,
    )

    # Reset the cached class to ensure monkeypatch takes effect
    from ray.anyscale.lineage.mlflow_lineage.store import tracking

    tracking.file_store._AnyscaleFileStoreClass = None

    store = AnyscaleFileStore(store_uri=TEST_LOCAL_MLRUNS_PATH)

    mock_run = mock.Mock()
    store.get_run = mock.Mock(return_value=mock_run)

    mock_model = mock.Mock()

    with pytest.raises(
        AnyscaleLineageMLflowError, match="OpenLineage processing failed"
    ) as exc_info:
        store.record_logged_model(TEST_MLFLOW_RUN_ID, mock_model)

    assert isinstance(exc_info.value.__cause__, RuntimeError)

    super_record.assert_called_once_with(TEST_MLFLOW_RUN_ID, mock_model)
