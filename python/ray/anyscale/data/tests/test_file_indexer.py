import math
import os
from unittest.mock import MagicMock, patch
import pytest
import pyarrow as pa

from ray.data.context import DataContext
from ray.anyscale.data._internal.file_indexer import (
    LineDelimitedFileChunker,
    WholeFileChunker,
    NonSamplingFileIndexer,
    LineDelimitedFileChunkMetadata,
    create_chunk_metadata,
)
from ray.data._internal.util import RetryingPyFileSystem
import pyarrow.fs as pa_fs


class TestFileChunkers:
    """Test cases for file chunking functionality."""

    @pytest.mark.parametrize(
        "file_size,chunk_size,expected_chunk_boundaries",
        [
            (1024, 256 * 1024 * 1024, [(0, 1024)]),  # Small file -> 1 chunk
            (
                256 * 1024 * 1024,
                256 * 1024 * 1024,
                [(0, 256 * 1024 * 1024)],
            ),  # Exact chunk size -> 1 chunk
            (
                512 * 1024 * 1024,
                256 * 1024 * 1024,
                [(0, 256 * 1024 * 1024), (256 * 1024 * 1024, 512 * 1024 * 1024)],
            ),  # 2x chunk size -> 2 chunks
            (
                640 * 1024 * 1024,
                256 * 1024 * 1024,
                [
                    (0, 256 * 1024 * 1024),
                    (256 * 1024 * 1024, 512 * 1024 * 1024),
                    (512 * 1024 * 1024, 640 * 1024 * 1024),
                ],
            ),  # 2.5x chunk size -> 3 chunks
            (0, 256 * 1024 * 1024, []),  # Empty file -> no chunks
        ],
    )
    def test_line_delimited_chunker(
        self, file_size, chunk_size, expected_chunk_boundaries
    ):
        """Test LineDelimitedFileChunker with various file sizes."""
        chunker = LineDelimitedFileChunker()
        chunker._CHUNK_BYTE_SIZE = chunk_size

        chunks = list(chunker.generate_chunk_metadatas("test.txt", file_size))

        assert len(chunks) == len(expected_chunk_boundaries)

        # Verify chunk boundaries and total size
        total_size = 0
        for i, (metadata, chunk_size) in enumerate(chunks):
            expected_start, expected_end = expected_chunk_boundaries[i]
            expected_size = expected_end - expected_start

            assert metadata["chunk_byte_start_idx"] == expected_start
            assert metadata["chunk_byte_end_idx"] == expected_end
            assert chunk_size == expected_size
            total_size += chunk_size

        assert total_size == file_size

    @pytest.mark.parametrize(
        "file_size", [0, 1024, 256 * 1024 * 1024, 1024 * 1024 * 1024]
    )
    def test_whole_file_chunker(self, file_size):
        """Test WholeFileChunker always produces a single chunk."""
        chunker = WholeFileChunker()
        chunks = list(chunker.generate_chunk_metadatas("test.txt", file_size))

        assert len(chunks) == 1
        metadata, chunk_size = chunks[0]
        assert metadata is None
        assert chunk_size == file_size


class TestChunkMetadata:
    """Test chunk metadata creation and validation."""

    def test_create_chunk_metadata_success(self):
        """Test successful creation of chunk metadata."""
        metadata = create_chunk_metadata(
            LineDelimitedFileChunkMetadata,
            chunk_byte_start_idx=0,
            chunk_byte_end_idx=1024,
        )

        assert metadata["chunk_byte_start_idx"] == 0
        assert metadata["chunk_byte_end_idx"] == 1024

    def test_create_chunk_metadata_validation(self):
        """Test metadata validation with missing and extra keys."""
        # Missing required key
        with pytest.raises(ValueError, match="Missing required keys"):
            create_chunk_metadata(
                LineDelimitedFileChunkMetadata, chunk_byte_start_idx=0
            )

        # Extra key
        with pytest.raises(ValueError, match="Unexpected keys"):
            create_chunk_metadata(
                LineDelimitedFileChunkMetadata,
                chunk_byte_start_idx=0,
                chunk_byte_end_idx=1024,
                extra_key="not_allowed",
            )


class TestNonSamplingFileIndexerWithChunking:
    """Test NonSamplingFileIndexer with chunking functionality."""

    @pytest.mark.parametrize(
        "chunker_class,expected_chunks_per_file",
        [
            (
                LineDelimitedFileChunker,
                lambda file_size, chunk_size: math.ceil(file_size / chunk_size),
            ),
            (WholeFileChunker, lambda file_size, chunk_size: 1),
        ],
    )
    def test_file_indexer_chunking(
        self, tmp_path, chunker_class, expected_chunks_per_file
    ):
        """Test NonSamplingFileIndexer with different chunkers."""
        # Create test files of different sizes
        file_paths = []
        file_sizes = []
        for i in range(3):
            file_path = tmp_path / f"test_{i}.txt"
            content = f"line_{i}_" + "x" * (1024 * (i + 1))  # 1KB, 2KB, 3KB files
            with open(file_path, "w") as f:
                f.write(content)
            file_paths.append(str(file_path))
            file_sizes.append(file_path.stat().st_size)

        # Create chunker
        chunker = chunker_class()
        if hasattr(chunker, "_CHUNK_BYTE_SIZE"):
            chunker._CHUNK_BYTE_SIZE = 1024  # 1KB chunks for testing

        # Create indexer
        indexer = NonSamplingFileIndexer(
            ignore_missing_paths=False, file_chunker=chunker
        )

        # List files
        paths_column = pa.array(file_paths)
        local_fs = pa_fs.LocalFileSystem()
        filesystem = RetryingPyFileSystem.wrap(local_fs, retryable_errors=[])

        manifests = list(indexer.list_files(paths_column, filesystem=filesystem))

        # Collect all entries
        all_paths = []
        all_sizes = []
        all_chunk_metadatas = []

        for manifest in manifests:
            all_paths.extend(manifest.paths)
            all_sizes.extend(manifest.file_sizes)
            all_chunk_metadatas.extend(manifest.file_chunk_metadatas)

        # Calculate expected total chunks
        expected_total_chunks = 0
        for file_size in file_sizes:
            chunk_size = getattr(chunker, "_CHUNK_BYTE_SIZE", file_size)
            expected_total_chunks += expected_chunks_per_file(file_size, chunk_size)

        assert len(all_paths) == expected_total_chunks

    def test_file_indexer_skips_zero_size_files(self, tmp_path):
        """Test that zero-size files are skipped even with chunking."""
        # Create regular and zero-size files
        regular_file = tmp_path / "regular.txt"
        with open(regular_file, "w") as f:
            f.write("content")

        zero_file = tmp_path / "zero.txt"
        with open(zero_file, "w") as f:
            pass  # Empty file

        # Test with chunking enabled
        chunker = LineDelimitedFileChunker()
        indexer = NonSamplingFileIndexer(
            ignore_missing_paths=False, file_chunker=chunker
        )

        paths_column = pa.array([str(regular_file), str(zero_file)])
        local_fs = pa_fs.LocalFileSystem()
        filesystem = RetryingPyFileSystem.wrap(local_fs, retryable_errors=[])

        manifests = list(indexer.list_files(paths_column, filesystem=filesystem))

        all_paths = []
        for manifest in manifests:
            all_paths.extend(manifest.paths)

        # Should only have the regular file
        assert len(all_paths) == 1
        assert str(regular_file) in all_paths
        assert str(zero_file) not in all_paths

    @patch("ray.anyscale.data.api.read_api.NonSamplingFileIndexer")
    def test_disable_large_file_chunking(
        self, mock_file_indexer_class, restore_data_context
    ):
        """Test that disable_large_file_chunking forces WholeFileChunker."""
        mock_file_indexer = MagicMock()
        mock_file_indexer_class.return_value = mock_file_indexer

        from ray.anyscale.data._internal.readers import LineDelimitedFileReader
        from ray.anyscale.data.api.read_api import read_files

        def mock_call_read_files():
            read_files(
                paths=["test.txt"],
                reader=LineDelimitedFileReader(),
                file_chunker=LineDelimitedFileChunker(),
                filesystem=None,
                columns=None,
                partition_filter=None,
                ignore_missing_paths=False,
                file_extensions=None,
                shuffle=None,
                concurrency=None,
                ray_remote_args=None,
            )

        # Test with no context, - should use passed LineDelimitedFileChunker
        mock_call_read_files()
        assert (
            type(mock_file_indexer_class.call_args.kwargs["file_chunker"])
            is LineDelimitedFileChunker
        )

        # Test with disable_large_file_chunking=True - should use WholeFileChunker
        ctx = DataContext.get_current()
        ctx.disable_large_file_chunking = True
        mock_call_read_files()
        assert (
            type(mock_file_indexer_class.call_args.kwargs["file_chunker"])
            is WholeFileChunker
        )

        # Test with disable_large_file_chunking=False - should use passed LineDelimitedFileChunker
        ctx.disable_large_file_chunking = False
        mock_call_read_files()
        assert (
            type(mock_file_indexer_class.call_args.kwargs["file_chunker"])
            is LineDelimitedFileChunker
        )

    @patch("ray.anyscale.data.api.read_api.NonSamplingFileIndexer")
    @pytest.mark.parametrize("env_var_value", ["0", "1"])
    def test_disable_large_file_chunking_with_env_var(
        self, mock_file_indexer_class, env_var_value
    ):
        mock_file_indexer = MagicMock()
        mock_file_indexer_class.return_value = mock_file_indexer

        from ray.anyscale.data._internal.readers import LineDelimitedFileReader
        from ray.anyscale.data.api.read_api import read_files

        def mock_call_read_files():
            read_files(
                paths=["test.txt"],
                reader=LineDelimitedFileReader(),
                file_chunker=LineDelimitedFileChunker(),
                filesystem=None,
                columns=None,
                partition_filter=None,
                ignore_missing_paths=False,
                file_extensions=None,
                shuffle=None,
                concurrency=None,
                ray_remote_args=None,
            )

        os.environ["RAY_TURBO_DISABLE_LARGE_FILE_CHUNKING"] = env_var_value
        mock_call_read_files()
        expected_chunker_class = (
            WholeFileChunker if env_var_value == "True" else LineDelimitedFileChunker
        )
        assert (
            type(mock_file_indexer_class.call_args.kwargs["file_chunker"])
            is expected_chunker_class
        )
        del os.environ["RAY_TURBO_DISABLE_LARGE_FILE_CHUNKING"]


class TestChunkingIntegration:
    """Integration tests for chunking + reading functionality."""

    @pytest.mark.parametrize(
        "chunk_size", [1024, 4 * 1024, 16 * 1024]
    )  # 1KB, 4KB, 16KB chunks
    def test_chunked_vs_whole_file_reading(self, tmp_path, chunk_size):
        """Test that chunked reading produces same result as whole file reading."""
        from ray.anyscale.data._internal.readers import LineDelimitedFileReader
        from ray.anyscale.data.tests.utils import batches_to_rows

        # Create test file (~11KB total)
        lines = [f"line_{i:04d}_" + "x" * 100 for i in range(100)]
        content = "\n".join(lines)

        file_path = tmp_path / "test_file.txt"
        with open(file_path, "w") as f:
            f.write(content)

        # Read whole file
        reader = LineDelimitedFileReader()
        local_fs = pa_fs.LocalFileSystem()
        filesystem = RetryingPyFileSystem.wrap(local_fs, retryable_errors=[])

        file_obj = reader.open_input_source(str(file_path), filesystem=filesystem)
        whole_file_batches = reader.read_stream(file_obj, str(file_path))
        whole_file_rows = batches_to_rows(whole_file_batches)
        file_obj.close()

        # Read using chunking
        chunker = LineDelimitedFileChunker()
        chunker._CHUNK_BYTE_SIZE = chunk_size

        chunked_rows = []
        for chunk_metadata, _ in chunker.generate_chunk_metadatas(
            str(file_path), file_path.stat().st_size
        ):
            file_obj = reader.open_input_source(str(file_path), filesystem=filesystem)
            batches = reader.read_stream(file_obj, str(file_path), chunk_metadata)
            rows = batches_to_rows(batches)
            chunked_rows.extend(rows)
            file_obj.close()

        # Results should be identical
        assert len(whole_file_rows) == len(chunked_rows)
        assert whole_file_rows == chunked_rows


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
