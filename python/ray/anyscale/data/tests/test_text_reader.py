import pyarrow as pa
import pytest

from ray.anyscale.data._internal.readers import TextReader
from ray.anyscale.data.tests.utils import batches_to_rows


@pytest.mark.parametrize(
    "text, expected_rows",
    [
        # Empty text.
        ("", []),
        # Single line.
        ("line1", [{"text": "line1"}]),
        # Multiple lines.
        ("line1\nline2", [{"text": "line1"}, {"text": "line2"}]),
        # Trailing newline.
        ("line1\n", [{"text": "line1"}]),
    ],
)
def test_basic(text, expected_rows):
    reader = TextReader()
    stream = pa.BufferReader(text.encode())

    batches = reader.read_stream(stream, "test.txt")

    assert batches_to_rows(batches) == expected_rows


@pytest.mark.parametrize(
    "text, expected_rows, drop_empty_lines",
    [
        ("\nline1\n\nline2\n", [{"text": "line1"}, {"text": "line2"}], True),
        (
            "\nline1\n\nline2\n",
            [
                {"text": ""},
                {"text": "line1"},
                {"text": ""},
                {"text": "line2"},
                # No trailing newline.
            ],
            False,
        ),
    ],
)
def test_drop_empty_lines(text, expected_rows, drop_empty_lines):
    reader = TextReader(drop_empty_lines=drop_empty_lines)
    stream = pa.BufferReader(text.encode())

    batches = reader.read_stream(stream, "test.txt")

    assert batches_to_rows(batches) == expected_rows


@pytest.mark.parametrize("encoding", ["ascii", "utf-8", "utf-16"])
def test_encoding(encoding):
    reader = TextReader(encoding=encoding)
    stream = pa.BufferReader("spam".encode(encoding))

    batches = reader.read_stream(stream, "test.txt")

    assert batches_to_rows(batches) == [{"text": "spam"}]


@pytest.mark.parametrize(
    "text, expected_rows, chunk_size",
    [
        # Chunk larger than text.
        ("line1\nline2", [{"text": "line1"}, {"text": "line2"}], 12),
        # Chunk smaller than text.
        ("line1\nline2", [{"text": "line1"}, {"text": "line2"}], 1),
        # Chunk breaks exactly at newline
        ("line1\nline2", [{"text": "line1"}, {"text": "line2"}], 6),
    ],
)
def test_chunk_size(text, expected_rows, chunk_size):
    reader = TextReader(chunk_size=chunk_size)
    stream = pa.BufferReader(text.encode())

    batches = reader.read_stream(stream, "test.txt")

    assert batches_to_rows(batches) == expected_rows


def test_decode_fn():
    # This decode function just returns the raw bytes.
    reader = TextReader(decode_fn=lambda data: {"data": data})
    stream = pa.BufferReader(b"spam")

    batches = reader.read_stream(stream, "test.txt")

    assert batches_to_rows(batches) == [{"data": b"spam"}]


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
