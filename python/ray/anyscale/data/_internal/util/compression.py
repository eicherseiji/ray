from typing import Optional


def infer_compression(path: str) -> Optional[str]:
    import pyarrow as pa

    compression = None
    try:
        # Try to detect compression codec from path.
        compression = pa.Codec.detect(path).name
    except (ValueError, TypeError):
        # Arrow's compression inference on the file path doesn't work for Snappy, so we double-check ourselves.
        import pathlib

        suffix = pathlib.Path(path).suffix
        if suffix and suffix[1:] == "snappy":
            compression = "snappy"
    return compression
