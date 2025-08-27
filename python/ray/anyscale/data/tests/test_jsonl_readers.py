import pyarrow as pa
import pytest

from ray.anyscale.data._internal.readers import OrjsonJSONLReader, PandasJSONLReader
from ray.anyscale.data.tests.utils import batches_to_rows


@pytest.mark.parametrize(
    "text, expected_rows",
    [
        # Basic objects
        ('{"col": "spam"}\n{"col": "ham"}', [{"col": "spam"}, {"col": "ham"}]),
        # Objects with mismatched columns
        (
            '{"col1": "spam"}\n{"col2": "ham"}',
            [{"col1": "spam", "col2": None}, {"col1": None, "col2": "ham"}],
        ),
        # Objects with mismatched types
        (
            '{"col": "spam"}\n{"col": 0}',
            [{"col": "spam"}, {"col": 0}],
        ),
        # Nested objects
        ('{"col": {"inner": "spam"}}', [{"col": {"inner": "spam"}}]),
        # Strings
        ('"spam"', [{"0": "spam"}]),
        # Lists
        ('["spam", "ham"]', [{"0": "spam", "1": "ham"}]),
    ],
)
@pytest.mark.parametrize("reader_factory", [OrjsonJSONLReader, PandasJSONLReader])
def test_basic(text, expected_rows, reader_factory):
    reader = reader_factory()
    stream = pa.BufferReader(text.encode())
    filename = "test.json"

    batches = reader.read_stream(stream, filename)

    rows = batches_to_rows(batches)
    assert rows == expected_rows


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
