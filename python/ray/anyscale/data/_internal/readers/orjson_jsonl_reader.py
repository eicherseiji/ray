from typing import Any, Dict, Optional

from ray.data.datasource.partitioning import Partitioning

from .line_delimited_file_reader import LineDelimitedFileReader


def decode_fn(data: bytes) -> Dict[str, Any]:
    import orjson

    obj = orjson.loads(data)

    # NOTE: These semantics are designed to match the behavior of `pandas.read_json` and
    # the OSS implementation. (The semantics of `pandas.read_json` aren't documented,
    # but this is the behavior that I've observed.)
    if isinstance(obj, list):
        return {str(i): obj[i] for i in range(len(obj))}
    elif isinstance(obj, dict):
        return obj
    elif isinstance(obj, str):
        return {"0": obj}
    else:
        raise NotImplementedError(f"Unsupported JSON type: {type(obj)}")


class OrjsonJSONLReader(LineDelimitedFileReader):
    """Reader for newline‑delimited JSON (*JSONL*) files using orjson."""

    def __init__(
        self,
        *,
        include_paths: bool = False,
        partitioning: Optional[Partitioning] = None,
        open_args: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            include_paths=include_paths,
            partitioning=partitioning,
            open_args=open_args,
            decode_fn=decode_fn,
        )
