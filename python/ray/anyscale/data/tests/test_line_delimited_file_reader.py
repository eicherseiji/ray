from pathlib import Path
from typing import Any, Dict, List

import pyarrow.fs as pa_fs
import pytest

from ray.anyscale.data._internal.readers import LineDelimitedFileReader
from ray.anyscale.data.tests.utils import batches_to_rows
from ray.data._internal.util import RetryingPyFileSystem


class TestLineDelimitedFileReader:
    """Test cases for LineDelimitedFileReader."""

    def _read_file_stream(
        self, reader: LineDelimitedFileReader, file_path: Path
    ) -> List[Dict[str, Any]]:
        """Read a file using the reader's open_input_source and read_stream methods."""
        # Create a local filesystem and wrap it with retrying functionality
        local_fs = pa_fs.LocalFileSystem()
        filesystem = RetryingPyFileSystem.wrap(
            local_fs, retryable_errors=[]  # No retryable errors for local filesystem
        )

        # Open the file using the reader's open_input_source method
        file_obj = reader.open_input_source(str(file_path), filesystem=filesystem)

        # Read the stream
        batches = reader.read_stream(file_obj, str(file_path))
        rows = batches_to_rows(batches)

        return rows

    # ============================================================================
    # 1. Basic Functionality Tests
    # ============================================================================

    @pytest.mark.parametrize(
        "content, expected_rows",
        [
            # Empty file
            ("", []),
            # Single line
            ("hello world", [{"text": "hello world"}]),
            # Multiple lines
            (
                "line1\nline2\nline3",
                [{"text": "line1"}, {"text": "line2"}, {"text": "line3"}],
            ),
            # Trailing newline
            ("line1\nline2\n", [{"text": "line1"}, {"text": "line2"}]),
            # No trailing newline
            ("line1\nline2", [{"text": "line1"}, {"text": "line2"}]),
            # Single line with trailing newline
            ("hello\n", [{"text": "hello"}]),
        ],
    )
    def test_basic_functionality(
        self, content: str, expected_rows: List[Dict[str, Any]], tmp_path
    ):
        """Test basic line reading functionality."""
        reader = LineDelimitedFileReader()
        file_path = tmp_path / "test.txt"
        with open(file_path, "w") as file:
            file.write(content)

        rows = self._read_file_stream(reader, file_path)
        assert rows == expected_rows

    @pytest.mark.parametrize(
        "content, drop_empty_lines, expected_rows",
        [
            # Empty lines with drop_empty_lines=False
            (
                "line1\n\nline2",
                False,
                [{"text": "line1"}, {"text": ""}, {"text": "line2"}],
            ),
            # Empty lines with drop_empty_lines=True
            ("line1\n\nline2", True, [{"text": "line1"}, {"text": "line2"}]),
            # Multiple empty lines
            ("line1\n\n\nline2", True, [{"text": "line1"}, {"text": "line2"}]),
            # Empty lines at start and end
            ("\nline1\nline2\n", True, [{"text": "line1"}, {"text": "line2"}]),
            # Only empty lines
            ("\n\n\n", True, []),
            # Whitespace-only lines
            ("line1\n   \nline2", True, [{"text": "line1"}, {"text": "line2"}]),
        ],
    )
    def test_empty_lines_handling(
        self,
        content: str,
        drop_empty_lines: bool,
        expected_rows: List[Dict[str, Any]],
        tmp_path,
    ):
        """Test handling of empty lines with drop_empty_lines parameter."""
        reader = LineDelimitedFileReader(drop_empty_lines=drop_empty_lines)
        file_path = tmp_path / "test.txt"
        with open(file_path, "w") as file:
            file.write(content)

        rows = self._read_file_stream(reader, file_path)
        assert rows == expected_rows

    # ============================================================================
    # 2. Chunk Reading Tests (Seekable Files)
    # ============================================================================

    @pytest.mark.parametrize(
        "content, chunk_start, chunk_end, expected_rows",
        [
            # Complete chunk with aligned boundaries
            (
                "line1\nline2\nline3",
                0,
                15,
                [{"text": "line1"}, {"text": "line2"}, {"text": "line3"}],
            ),
            # Chunk starting in middle of line
            ("line1\nline2\nline3", 2, 10, [{"text": "line2"}]),
            # Chunk ending in middle of line
            ("line1\nline2\nline3", 0, 8, [{"text": "line1"}, {"text": "line2"}]),
            # Chunk with partial line at start and end
            ("line1\nline2\nline3", 3, 9, [{"text": "line2"}]),
            # Empty chunk
            ("line1\nline2\nline3", 10, 10, []),
            # Chunk beyond file end
            ("line1\nline2", 20, 30, []),
        ],
    )
    @pytest.mark.parametrize("buffer_size", [1024, 8192, 8 << 20])  # 1KB, 8KB, 8MB
    def test_chunk_reading(
        self,
        content: str,
        chunk_start: int,
        chunk_end: int,
        expected_rows: List[Dict[str, Any]],
        buffer_size: int,
        tmp_path,
    ):
        """Test reading specific chunks of seekable files with different buffer sizes."""
        reader = LineDelimitedFileReader(buffer_size=buffer_size)
        file_path = tmp_path / "test.txt"
        with open(file_path, "w") as file:
            file.write(content)

        local_fs = pa_fs.LocalFileSystem()
        filesystem = RetryingPyFileSystem.wrap(local_fs, retryable_errors=[])
        file_obj = reader.open_input_source(str(file_path), filesystem=filesystem)

        metadata = {
            "chunk_byte_start_idx": chunk_start,
            "chunk_byte_end_idx": chunk_end,
        }

        batches = reader.read_stream(file_obj, str(file_path), metadata)
        rows = batches_to_rows(batches)

        assert rows == expected_rows

    @pytest.mark.parametrize("buffer_size", [1024, 8192, 8 << 20])  # 1KB, 8KB, 8MB
    def test_chunk_reading_no_metadata(self, buffer_size: int, tmp_path):
        """Test chunk reading when metadata is None with different buffer sizes."""
        content = "line1\nline2\nline3"
        reader = LineDelimitedFileReader(buffer_size=buffer_size)
        file_path = tmp_path / "test.txt"
        with open(file_path, "w") as file:
            file.write(content)

        rows = self._read_file_stream(reader, file_path)

        # Should read the entire file when metadata is None
        assert len(rows) > 0
        assert all("text" in row for row in rows)

    @pytest.mark.parametrize("buffer_size", [1024, 8192, 8 << 20])  # 1KB, 8KB, 8MB
    def test_chunk_reading_larger_than_buffer(self, buffer_size: int, tmp_path):
        """Test chunk reading when chunk size is larger than buffer size."""
        # Create content with many lines to test chunk larger than buffer
        lines = [f"line{i:03}" for i in range(1000)]  # 1000 lines
        content = "\n".join(lines)

        reader = LineDelimitedFileReader(buffer_size=buffer_size)
        file_path = tmp_path / "test.txt"
        with open(file_path, "w") as file:
            file.write(content)

        local_fs = pa_fs.LocalFileSystem()
        filesystem = RetryingPyFileSystem.wrap(local_fs, retryable_errors=[])
        file_obj = reader.open_input_source(str(file_path), filesystem=filesystem)

        # Create a chunk that's larger than the buffer size
        num_bytes_per_line = 8  # line (4) + line number (3) + \n (1)
        num_lines_in_chunk = 500
        chunk_size = num_bytes_per_line * num_lines_in_chunk
        metadata = {"chunk_byte_start_idx": 0, "chunk_byte_end_idx": chunk_size}

        batches = reader.read_stream(file_obj, str(file_path), metadata)
        rows = batches_to_rows(batches)

        # Should read all lines that start within the chunk
        assert len(rows) == num_lines_in_chunk
        assert all(f"line{i:03}" in row["text"] for i, row in enumerate(rows))

    @pytest.mark.parametrize("buffer_size", [1024, 8192, 8 << 20])  # 1KB, 8KB, 8MB
    def test_chunk_reading_partial_larger_than_buffer(self, buffer_size: int, tmp_path):
        """Test chunk reading when chunk size is larger than buffer size and chunk doesn't start at beginning."""
        # Create content with many lines
        lines = [f"line{i:03}" for i in range(1000)]  # 1000 lines
        content = "\n".join(lines)

        reader = LineDelimitedFileReader(buffer_size=buffer_size)
        file_path = tmp_path / "test.txt"
        with open(file_path, "w") as file:
            file.write(content)

        local_fs = pa_fs.LocalFileSystem()
        filesystem = RetryingPyFileSystem.wrap(local_fs, retryable_errors=[])
        file_obj = reader.open_input_source(str(file_path), filesystem=filesystem)

        # Start chunk in the middle of the file (e.g., after 100 lines)
        # Each line is roughly 6 bytes (5 chars + newline), so 100 lines ≈ 600 bytes
        num_bytes_per_line = 8  # line (4) + line number (3) + \n (1)
        num_lines_in_chunk = 500
        chunk_size = num_bytes_per_line * num_lines_in_chunk
        start_line_number = 200
        start_pos = num_bytes_per_line * start_line_number  # start 200 lines into file
        metadata = {
            "chunk_byte_start_idx": start_pos,
            "chunk_byte_end_idx": start_pos + chunk_size,
        }

        batches = reader.read_stream(file_obj, str(file_path), metadata)
        rows = batches_to_rows(batches)

        # Should read lines that start within the chunk range
        assert len(rows) == num_lines_in_chunk
        assert all(
            f"line{(i + start_line_number):03}" in row["text"]
            for i, row in enumerate(rows)
        )

    # ============================================================================
    # 3. Streaming Tests (Non-seekable Files via Compression)
    # ============================================================================

    @pytest.mark.parametrize(
        "content, expected_rows",
        [
            # Empty file
            ("", []),
            # Single line
            ("hello world", [{"text": "hello world"}]),
            # Multiple lines
            (
                "line1\nline2\nline3",
                [{"text": "line1"}, {"text": "line2"}, {"text": "line3"}],
            ),
            # Trailing newline
            ("line1\nline2\n", [{"text": "line1"}, {"text": "line2"}]),
        ],
    )
    @pytest.mark.parametrize("buffer_size", [1024, 8192, 8 << 20])  # 1KB, 8KB, 8MB
    def test_streaming_basic(
        self,
        content: str,
        expected_rows: List[Dict[str, Any]],
        buffer_size: int,
        tmp_path,
    ):
        """Test fallback for basic streaming functionality with compressed (non-seekable) files."""
        reader = LineDelimitedFileReader(buffer_size=buffer_size)
        file_path = tmp_path / "test.gz"

        # Create compressed file
        import gzip

        with gzip.open(file_path, "wt") as f:
            f.write(content)

        rows = self._read_file_stream(reader, file_path)
        assert rows == expected_rows

    @pytest.mark.parametrize(
        "content, drop_empty_lines, expected_rows",
        [
            # Empty lines with drop_empty_lines=True
            ("line1\n\nline2", True, [{"text": "line1"}, {"text": "line2"}]),
            # Empty lines with drop_empty_lines=False
            (
                "line1\n\nline2",
                False,
                [{"text": "line1"}, {"text": ""}, {"text": "line2"}],
            ),
        ],
    )
    @pytest.mark.parametrize("buffer_size", [1024, 8192, 8 << 20])  # 1KB, 8KB, 8MB
    def test_streaming_empty_lines(
        self,
        content: str,
        drop_empty_lines: bool,
        expected_rows: List[Dict[str, Any]],
        buffer_size: int,
        tmp_path,
    ):
        """Test streaming with empty lines handling."""
        reader = LineDelimitedFileReader(
            drop_empty_lines=drop_empty_lines, buffer_size=buffer_size
        )
        file_path = tmp_path / "test.gz"

        # Create compressed file
        import gzip

        with gzip.open(file_path, "wt") as f:
            f.write(content)

        rows = self._read_file_stream(reader, file_path)
        assert rows == expected_rows

    def test_streaming_large_content(self, tmp_path):
        """Test streaming with content larger than buffer size."""
        # Create content larger than the default 8MB buffer
        large_line = "x" * 1000
        lines = [large_line] * 10000  # ~10MB of content
        content = "\n".join(lines)

        reader = LineDelimitedFileReader()
        file_path = tmp_path / "test.gz"

        # Create compressed file
        import gzip

        with gzip.open(file_path, "wt") as f:
            f.write(content)

        rows = self._read_file_stream(reader, file_path)

        # Should read all lines
        assert len(rows) == 10000
        assert all(row["text"] == large_line for row in rows)

    def test_compression_fallback(self, tmp_path):
        """Test that compressed files fall back to streaming (non-seekable) reading."""
        content = "line1\nline2\nline3"
        reader = LineDelimitedFileReader()

        # Create both uncompressed and compressed versions
        uncompressed_path = tmp_path / "test.txt"
        with open(uncompressed_path, "w") as f:
            f.write(content)

        compressed_path = tmp_path / "test.gz"
        import gzip

        with gzip.open(compressed_path, "wt") as f:
            f.write(content)

        # Read uncompressed file (should use seekable reading)
        uncompressed_rows = self._read_file_stream(reader, uncompressed_path)

        # Read compressed file (should use streaming reading)
        compressed_rows = self._read_file_stream(reader, compressed_path)

        # Results should be the same
        assert uncompressed_rows == compressed_rows
        assert len(uncompressed_rows) == 3

    # ============================================================================
    # 4. Delimiter and Encoding Tests
    # ============================================================================

    def test_multibyte_delimiter(self, tmp_path):
        """Test handling of multibyte delimiters (Windows CRLF)."""
        content = "hello\r\nworld\r\ntest\r\n"
        expected_rows = [{"text": "hello"}, {"text": "world"}, {"text": "test"}]

        reader = LineDelimitedFileReader(delimiter=b"\r\n")
        file_path = tmp_path / "test.txt"
        with open(file_path, "w") as file:
            file.write(content)

        rows = self._read_file_stream(reader, file_path)
        assert rows == expected_rows

    def test_json_decode_function(self, tmp_path):
        """Test with JSON decode function from JSONLReader."""
        from ray.anyscale.data._internal.readers import OrjsonJSONLReader

        # Use the decode function from JSONLReader
        jsonl_reader = OrjsonJSONLReader()
        json_decode_fn = jsonl_reader._decode_fn

        content = '{"name": "alice", "age": 30}\n{"name": "bob", "age": 25}\n'
        reader = LineDelimitedFileReader(decode_fn=json_decode_fn)
        file_path = tmp_path / "test.jsonl"
        with open(file_path, "w") as file:
            file.write(content)

        rows = self._read_file_stream(reader, file_path)

        expected = [{"name": "alice", "age": 30}, {"name": "bob", "age": 25}]
        assert rows == expected

    # ============================================================================
    # 5. Edge Case Tests
    # ============================================================================

    @pytest.mark.parametrize("buffer_size", [1024, 1 << 20])  # 1KB, 1MB
    @pytest.mark.parametrize(
        "chunk_start_idx",
        [
            0,
            1000,
        ],
    )
    @pytest.mark.parametrize(
        "chunk_end_idx",
        [
            1024,
            8192,
        ],
    )
    def test_single_long_line(
        self, buffer_size: int, chunk_start_idx: int, chunk_end_idx: int, tmp_path
    ):
        """Test reading a very long line that starts at the beginning of the file."""
        # Create a line that's much longer than the buffer size
        long_line = "x" * (4 << 20)  # 4MB
        content = long_line + "\n"  # Single very long line

        reader = LineDelimitedFileReader(buffer_size=buffer_size)
        file_path = tmp_path / "test.txt"
        with open(file_path, "w") as file:
            file.write(content)

        local_fs = pa_fs.LocalFileSystem()
        filesystem = RetryingPyFileSystem.wrap(local_fs, retryable_errors=[])
        file_obj = reader.open_input_source(str(file_path), filesystem=filesystem)

        # Test chunk starting at beginning of file
        metadata = {
            "chunk_byte_start_idx": chunk_start_idx,
            "chunk_byte_end_idx": chunk_end_idx,
        }

        batches = reader.read_stream(file_obj, str(file_path), metadata)
        rows = batches_to_rows(batches)

        # Should read the entire long line since it starts within the chunk
        if chunk_start_idx == 0:
            assert len(rows) == 1
            assert rows[0]["text"] == long_line
        else:
            assert len(rows) == 0

    @pytest.mark.parametrize("buffer_size", [1024, 1 << 20])  # 1KB, 1MB
    @pytest.mark.parametrize(
        "chunk_start_idx",
        [
            0,  # start of file
            1000,  # middle of file before long line
            500 * 8 + 200,  # middle of long line (500 lines * 8 bytes/line + 200 bytes)
        ],
    )
    @pytest.mark.parametrize(
        "chunk_end_idx",
        [
            4200,  # middle of long line (500 lines * 8 bytes/line + 200 bytes)
            500 * 8
            + (4 << 20)
            + 1
            + 200,  # middle of file after long line (500 lines * 8 bytes/line + 4MB + 1 byte + 200 bytes)
            1000 * 8
            + (4 << 20)
            + 1,  # end of file after long line (1000 lines * 8 bytes/line + 4MB + 1 byte)
        ],
    )
    def test_mixed_line_lengths(
        self, buffer_size: int, chunk_start_idx: int, chunk_end_idx: int, tmp_path
    ):
        """Test reading a file with a mix of very long lines and short lines."""
        # Create content with short lines, then a very long line, then more short lines
        short_lines_before = [f"line{i:03}" for i in range(500)]  # 500 short lines
        long_line = "x" * (4 << 20)  # 4MB
        short_lines_after = [
            f"line{i + 500:03}" for i in range(500)
        ]  # 500 more short lines
        content = "\n".join(short_lines_before + [long_line] + short_lines_after)

        reader = LineDelimitedFileReader(buffer_size=buffer_size)
        file_path = tmp_path / "test.txt"
        with open(file_path, "w") as file:
            file.write(content)

        local_fs = pa_fs.LocalFileSystem()
        filesystem = RetryingPyFileSystem.wrap(local_fs, retryable_errors=[])
        file_obj = reader.open_input_source(str(file_path), filesystem=filesystem)

        metadata = {
            "chunk_byte_start_idx": chunk_start_idx,
            "chunk_byte_end_idx": chunk_end_idx,
        }

        batches = reader.read_stream(file_obj, str(file_path), metadata)
        rows = batches_to_rows(batches)

        # Calculate positions of the long line in the file
        # Each short line before is 7 characters + 1 newline = 8 bytes
        long_line_start = 500 * 8  # 500 short lines * 8 bytes each
        long_line_end = long_line_start + len(long_line)  # start + 4MB

        # Determine expected content based on chunk boundaries
        expected_short_lines_before = 0
        expected_short_lines_after = 0
        should_include_long_line = False

        # Count short lines before the long line that should be included
        if chunk_start_idx < long_line_start:
            # Chunk starts before the long line
            lines_before_start = max(0, chunk_start_idx // 8)
            lines_before_end = min(500, chunk_end_idx // 8)
            expected_short_lines_before = max(0, lines_before_end - lines_before_start)

        # Check if long line should be included
        # Long line is included if its start position is within the chunk
        if long_line_start >= chunk_start_idx and long_line_start < chunk_end_idx:
            should_include_long_line = True

        # Count short lines after the long line that should be included
        short_lines_after_start = long_line_end + 1  # +1 for newline after long line
        if chunk_end_idx > short_lines_after_start:
            # Chunk extends past the long line
            lines_after_start_idx = max(
                0, (chunk_start_idx - short_lines_after_start) // 8
            )
            lines_after_end_idx = min(
                500, (chunk_end_idx - short_lines_after_start) // 8
            )
            if chunk_start_idx < short_lines_after_start:
                lines_after_start_idx = 0
            expected_short_lines_after = max(
                0, lines_after_end_idx - lines_after_start_idx
            )

        # Verify the results
        total_expected_lines = (
            expected_short_lines_before
            + (1 if should_include_long_line else 0)
            + expected_short_lines_after
        )
        assert (
            len(rows) == total_expected_lines
        ), f"Expected {total_expected_lines} lines, got {len(rows)}"

        # Check that the long line is present when expected
        long_line_found = any(row["text"] == long_line for row in rows)
        assert (
            long_line_found == should_include_long_line
        ), f"Long line presence mismatch: expected {should_include_long_line}, found {long_line_found}"

        # Verify short lines content
        short_lines_in_rows = [row["text"] for row in rows if row["text"] != long_line]
        expected_total_short_lines = (
            expected_short_lines_before + expected_short_lines_after
        )
        assert (
            len(short_lines_in_rows) == expected_total_short_lines
        ), f"Expected {expected_total_short_lines} short lines, got {len(short_lines_in_rows)}"
