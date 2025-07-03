from typing import Any, Dict

from .text_reader import TextReader


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


class OrjsonJSONLReader(TextReader):
    def __init__(self, *args, **kwargs):
        super().__init__(decode_fn=decode_fn, drop_empty_lines=True, *args, **kwargs)
