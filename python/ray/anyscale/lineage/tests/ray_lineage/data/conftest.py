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


# Initialize file extensions when module is loaded
_ensure_file_extensions_attributes()
