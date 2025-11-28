import pytest


def _ensure_file_extensions_attributes():
    """Ensure Ray Data datasource classes have _FILE_EXTENSIONS attributes.

    Some Ray Data datasource classes use _FUTURE_FILE_EXTENSIONS instead of
    _FILE_EXTENSIONS. This function backfills _FILE_EXTENSIONS for compatibility
    with our test utilities.
    """
    from ray.anyscale.lineage.ray_lineage.data.utils import (
        get_file_format_datasinks,
        get_file_format_datasources,
    )

    # Only file format datasources/datasinks have file extensions
    all_classes = []
    all_classes.extend(get_file_format_datasources())
    all_classes.extend(get_file_format_datasinks())

    for cls in all_classes:
        if not hasattr(cls, "_FILE_EXTENSIONS"):
            fallback = getattr(cls, "_FUTURE_FILE_EXTENSIONS", [])
            cls._FILE_EXTENSIONS = fallback


@pytest.fixture(autouse=True)
def reset_logging_state():
    """Reset logging state before each test.

    Since main.py configures logging at module level, we need to reset
    the logging state for test isolation.
    """
    from ray.anyscale.lineage.common import logging as logging_module

    # Reset logging state before test
    logging_module.reset_logging()

    yield

    # Reset logging state after test
    logging_module.reset_logging()


@pytest.fixture
def patch_facet_constructors(monkeypatch):
    """Patch facet constructor methods for testing dataset constructors.

    This fixture mocks the RayDataFacetConstructor methods to return
    simplified facet data for testing purposes.

    Note: This fixture is opt-in to avoid interfering with error-handling tests.
    Tests that need mocked facet constructors should explicitly request this fixture.
    """
    from ray.anyscale.lineage.ray_lineage.data import facet_constructor

    monkeypatch.setattr(
        facet_constructor.RayDataFacetConstructor,
        "construct_dataset_type_dataset_facet",
        lambda dataset_type: {"dataset_type": dataset_type},
    )
    monkeypatch.setattr(
        facet_constructor.RayDataFacetConstructor,
        "construct_datasource_dataset_facet",
        lambda uri=None, name=None: {"datasource": uri or name},
    )
    monkeypatch.setattr(
        facet_constructor.RayDataFacetConstructor,
        "construct_file_format_dataset_facet",
        lambda format: {"file_format": format},
    )
    monkeypatch.setattr(
        facet_constructor.RayDataFacetConstructor,
        "construct_schema_dataset_facet",
        lambda fields: {"schema": fields},
    )
    monkeypatch.setattr(
        facet_constructor.RayDataFacetConstructor,
        "construct_ownership_dataset_facet",
        lambda: {"owner": []},
    )


@pytest.fixture(autouse=True)
def patch_dataset_naming(monkeypatch):
    """Patch dataset naming resolution for testing.

    This fixture mocks the dataset naming resolution functions for
    file_format and list_files modules to return simplified values.
    """
    from ray.anyscale.lineage.ray_lineage.data.dataset_constructor import (
        file_format,
        list_files,
    )

    monkeypatch.setattr(
        file_format,
        "resolve_dataset_naming_type_and_attributes",
        lambda path: ("type", {"value": path}),
    )
    monkeypatch.setattr(
        file_format,
        "resolve_ol_dataset_namespace_and_name",
        lambda dataset_type, **attrs: ("namespace", attrs["value"]),
    )
    monkeypatch.setattr(
        list_files,
        "resolve_dataset_naming_type_and_attributes",
        lambda path: ("type", {"value": path}),
    )
    monkeypatch.setattr(
        list_files,
        "resolve_ol_dataset_namespace_and_name",
        lambda dataset_type, **attrs: ("namespace", attrs["value"]),
    )


# Initialize file extensions when module is loaded
_ensure_file_extensions_attributes()
