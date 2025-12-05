from unittest import mock

import pytest

from ray.anyscale.lineage.common.exceptions import (
    AnyscaleLineageMLflowError,
)
from ray.anyscale.lineage.mlflow_lineage.store.tracking.rest_store import (
    AnyscaleRestStore,
)
from ray.anyscale.lineage.tests.test_constants import (
    TEST_MLFLOW_HOST_EXAMPLE,
    TEST_MLFLOW_MODEL_NAME,
    TEST_MLFLOW_RUN_ID,
)


def test_anyscale_rest_store_initializes_client(monkeypatch) -> None:
    mock_host_creds = mock.Mock()
    mock_host_creds.host = TEST_MLFLOW_HOST_EXAMPLE
    get_host_creds = mock.Mock(return_value=mock_host_creds)
    monkeypatch.setattr(
        "mlflow.utils.credentials.get_default_host_creds",
        lambda store_uri: get_host_creds(),
    )

    monkeypatch.setattr(
        "mlflow.store.tracking.rest_store.RestStore.__init__",
        lambda self, func: None,
    )

    super_record = mock.Mock()
    monkeypatch.setattr(
        "mlflow.store.tracking.rest_store.RestStore.record_logged_model",
        super_record,
    )

    process_mock = mock.Mock()
    monkeypatch.setattr(
        "ray.anyscale.lineage.mlflow_lineage.store.tracking.utils.process_and_emit_ol_events_for_model_logging",
        process_mock,
    )

    store = AnyscaleRestStore(store_uri=f"https://{TEST_MLFLOW_HOST_EXAMPLE}")

    mock_run = mock.Mock()
    store.get_run = mock.Mock(return_value=mock_run)

    assert store.host == TEST_MLFLOW_HOST_EXAMPLE
    assert hasattr(store, "ol_client")
    assert store.is_plugin is True

    store.record_logged_model(TEST_MLFLOW_RUN_ID, TEST_MLFLOW_MODEL_NAME)

    super_record.assert_called_once_with(TEST_MLFLOW_RUN_ID, TEST_MLFLOW_MODEL_NAME)
    process_mock.assert_called_once()
    call_args = process_mock.call_args
    assert call_args.kwargs["mlflow_host"] == TEST_MLFLOW_HOST_EXAMPLE
    assert call_args.kwargs["run"] == mock_run
    assert call_args.kwargs["mlflow_model"] == TEST_MLFLOW_MODEL_NAME


def test_anyscale_rest_store_factory_caches_class(monkeypatch) -> None:
    """Test that the factory function caches the created class."""
    mock_host_creds = mock.Mock()
    mock_host_creds.host = TEST_MLFLOW_HOST_EXAMPLE
    get_host_creds = mock.Mock(return_value=mock_host_creds)
    monkeypatch.setattr(
        "mlflow.utils.credentials.get_default_host_creds",
        lambda store_uri: get_host_creds(),
    )

    monkeypatch.setattr(
        "mlflow.store.tracking.rest_store.RestStore.__init__",
        lambda self, func: None,
    )

    mock_client = mock.Mock()
    monkeypatch.setattr(
        "ray.anyscale.lineage.common.openlineage_client.AnyscaleOpenLineageClient",
        mock.Mock(return_value=mock_client),
    )

    from ray.anyscale.lineage.mlflow_lineage.store import tracking

    tracking.rest_store._AnyscaleRestStoreClass = None

    store1 = AnyscaleRestStore(store_uri=f"https://{TEST_MLFLOW_HOST_EXAMPLE}")
    store2 = AnyscaleRestStore(store_uri="https://other-host.example.com")

    assert type(store1) is type(store2)
    assert store1.__class__.__name__ == "AnyscaleRestStore"


def test_anyscale_rest_store_record_logged_model_catches_exceptions(
    monkeypatch,
) -> None:
    """Test that record_logged_model catches and wraps exceptions from OpenLineage processing."""
    from ray.anyscale.lineage.mlflow_lineage import utils as mlflow_utils

    # Set IGNORE_ERRORS to False to ensure exceptions are raised
    monkeypatch.setattr(mlflow_utils, "IGNORE_ERRORS", False)

    mock_host_creds = mock.Mock()
    mock_host_creds.host = TEST_MLFLOW_HOST_EXAMPLE
    get_host_creds = mock.Mock(return_value=mock_host_creds)
    monkeypatch.setattr(
        "mlflow.utils.credentials.get_default_host_creds",
        lambda store_uri: get_host_creds(),
    )

    monkeypatch.setattr(
        "mlflow.store.tracking.rest_store.RestStore.__init__",
        lambda self, func: None,
    )

    super_record = mock.Mock()
    monkeypatch.setattr(
        "mlflow.store.tracking.rest_store.RestStore.record_logged_model",
        super_record,
    )

    process_mock = mock.Mock(side_effect=RuntimeError("OpenLineage processing failed"))
    monkeypatch.setattr(
        "ray.anyscale.lineage.mlflow_lineage.store.tracking.utils.process_and_emit_ol_events_for_model_logging",
        process_mock,
    )

    # Reset the cached class to ensure monkeypatch takes effect
    from ray.anyscale.lineage.mlflow_lineage.store import tracking

    tracking.rest_store._AnyscaleRestStoreClass = None

    store = AnyscaleRestStore(store_uri=f"https://{TEST_MLFLOW_HOST_EXAMPLE}")

    mock_run = mock.Mock()
    store.get_run = mock.Mock(return_value=mock_run)

    mock_model = mock.Mock()

    with pytest.raises(
        AnyscaleLineageMLflowError, match="OpenLineage processing failed"
    ) as exc_info:
        store.record_logged_model(TEST_MLFLOW_RUN_ID, mock_model)

    assert isinstance(exc_info.value.__cause__, RuntimeError)

    super_record.assert_called_once_with(TEST_MLFLOW_RUN_ID, mock_model)
