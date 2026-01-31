"""Unit tests for generate_and_copy_images.py."""

import os
import sys
from unittest import mock

import pytest
from generate_and_copy_images import (
    PYTHON_VERSIONS_RAY,
    _build_repo_mappings,
    _generate_llm_mappings,
    _generate_regular_mappings,
    _generate_slim_mappings,
    _get_slim_platform,
    _platform_to_img_type_code,
    _print_summary,
    _python_version_to_code,
    get_py_version_code,
    main,
)


class TestPlatformConversion:
    """Tests for platform conversion functions."""

    def test_cpu_always_returns_cpu(self):
        """CPU platform always returns 'cpu'."""
        assert _platform_to_img_type_code("cpu") == "cpu"
        assert _get_slim_platform("cpu") == "cpu"

    @pytest.mark.parametrize(
        "platform,expected",
        [
            ("cu11.8.0-cudnn8", "cu118"),
            ("cu12.1.0-cudnn8", "cu121"),
            ("cu12.8.1-cudnn9", "cu128"),
            ("cu10.2.0-cudnn7", "cu102"),
            ("cu13.0.0-cudnn9", "cu130"),
        ],
    )
    def test_cuda_platform_extracts_major_minor(self, platform, expected):
        """CUDA platform cu{major}.{minor}.{patch}-cudnn{N} returns cu{major}{minor}."""
        result = _platform_to_img_type_code(platform)
        assert result == expected

    @pytest.mark.parametrize(
        "platform,expected",
        [
            ("cu11.8.0-cudnn8", "cu11.8.0"),
            ("cu12.1.0-cudnn9", "cu12.1.0"),
            ("cu12.8.1-cudnn", "cu12.8.1"),
        ],
    )
    def test_slim_platform_removes_cudnn_suffix(self, platform, expected):
        """Slim platform removes -cudnn{N} suffix."""
        result = _get_slim_platform(platform)
        assert result == expected
        assert "-cudnn" not in result

    @pytest.mark.parametrize(
        "platform",
        [
            "cu11.8.0",
            "cu12.1.0",
            "cu13.0.0",
        ],
    )
    def test_slim_platform_without_cudnn_unchanged(self, platform):
        """Platform without -cudnn suffix is unchanged."""
        result = _get_slim_platform(platform)
        assert result == platform

    @pytest.mark.parametrize("platform", ["cu11", "cu", "invalid"])
    def test_single_version_part_returns_unknown(self, platform):
        """Platform with single version part returns 'unknown'."""
        assert _platform_to_img_type_code(platform) == "unknown"


class TestPythonVersionConversion:
    """Tests for Python version conversion functions."""

    @pytest.mark.parametrize(
        "version,expected",
        [
            ("3.10", "py310"),
            ("3.11", "py311"),
            ("3.12", "py312"),
        ],
    )
    def test_python_version_format(self, version, expected):
        """Python version {major}.{minor} returns 'py{major}{minor}'."""
        result = _python_version_to_code(version)
        assert result == expected
        assert "." not in result

    def test_get_py_version_code_validates_against_allowed_versions(self):
        """get_py_version_code only returns valid codes for allowed versions."""
        for py_version in PYTHON_VERSIONS_RAY:
            result = get_py_version_code(py_version)
            assert result.startswith("py")
            assert result != "unknown"

    @pytest.mark.parametrize("version", ["2.7", "invalid", "abc", ""])
    def test_invalid_versions_return_unknown(self, version):
        """Invalid Python versions return 'unknown'."""
        result = get_py_version_code(version)
        assert result == "unknown"


class TestGenerateRegularMappings:
    """Tests for _generate_regular_mappings function."""

    def test_generates_mappings(self):
        """Test regular mappings are generated."""
        rayci_build_id = "abc123"
        mappings = _generate_regular_mappings(rayci_build_id)

        assert len(mappings) > 0
        for mapping in mappings:
            assert ":" in mapping
            source, dest = mapping.split(":", 1)
            assert source.startswith(rayci_build_id)
            assert dest.startswith("turbonightly-")

    def test_mapping_format(self):
        """Test mappings have correct format."""
        rayci_build_id = "testbuild"
        mappings = _generate_regular_mappings(rayci_build_id)

        for mapping in mappings:
            source, dest = mapping.split(":", 1)
            # Source should be: {build_id}-{py_code}-{img_code}
            assert "-py3" in source
            # Dest should be: turbonightly-{py_code}-{img_code}
            assert dest.startswith("turbonightly-py3")


class TestGenerateSlimMappings:
    """Tests for _generate_slim_mappings function."""

    def test_generates_slim_mappings(self):
        """Test slim mappings are generated."""
        rayci_build_id = "abc123"
        mappings = _generate_slim_mappings(rayci_build_id)

        assert len(mappings) > 0
        for mapping in mappings:
            source, dest = mapping.split(":", 1)
            assert "slim" in source
            assert "slim" in dest

    def test_slim_mapping_format(self):
        """Test slim mappings have correct format."""
        rayci_build_id = "testbuild"
        mappings = _generate_slim_mappings(rayci_build_id)

        for mapping in mappings:
            source, dest = mapping.split(":", 1)
            # Source should be: {build_id}-slim-{py_code}-{img_code}
            assert source.startswith(f"{rayci_build_id}-slim-")
            # Dest should be: turbonightly-slim-{py_code}-{img_code}
            assert dest.startswith("turbonightly-slim-")


class TestGenerateLlmMappings:
    """Tests for _generate_llm_mappings function."""

    def test_generates_llm_mappings(self):
        """Test LLM mappings are generated."""
        rayci_build_id = "abc123"
        mappings = _generate_llm_mappings(rayci_build_id)

        # LLM currently generates 1 mapping for py311
        assert len(mappings) >= 1
        for mapping in mappings:
            source, dest = mapping.split(":", 1)
            assert "py311" in source
            assert "py311" in dest


class TestBuildRepoMappings:
    """Tests for _build_repo_mappings function."""

    def test_builds_ray_mappings(self):
        """Test ray mappings are built correctly."""
        regular_mappings = ["tag1:dest1", "tag2:dest2"]
        llm_mappings = []

        result = _build_repo_mappings(regular_mappings, llm_mappings)

        assert "ray" in result
        assert result["ray"] == "tag1:dest1,tag2:dest2"
        assert "ray-llm" not in result

    def test_builds_llm_mappings(self):
        """Test LLM mappings are built correctly."""
        regular_mappings = []
        llm_mappings = ["llm1:dest1"]

        result = _build_repo_mappings(regular_mappings, llm_mappings)

        assert "ray-llm" in result
        assert result["ray-llm"] == "llm1:dest1"
        assert "ray" not in result

    def test_builds_both_mappings(self):
        """Test both ray and LLM mappings are built."""
        regular_mappings = ["tag1:dest1"]
        llm_mappings = ["llm1:dest1"]

        result = _build_repo_mappings(regular_mappings, llm_mappings)

        assert "ray" in result
        assert "ray-llm" in result

    def test_empty_mappings(self):
        """Test empty mappings return empty dict."""
        result = _build_repo_mappings([], [])
        assert result == {}


class TestPrintSummary:
    """Tests for _print_summary function."""

    def test_all_success(self, capsys):
        """Test summary with all successful results."""
        results = [
            {"env": "env1", "success": True},
            {"env": "env2", "success": True},
        ]

        # Should not exit since all succeeded
        _print_summary(results, 2)

        captured = capsys.readouterr()
        assert "2/2 environments succeeded" in captured.out
        assert "0 failed" in captured.out

    def test_partial_failure(self, capsys):
        """Test summary with partial failures exits with code 1."""
        results = [
            {"env": "env1", "success": True},
            {"env": "env2", "success": False, "error": "Connection failed"},
        ]

        with pytest.raises(SystemExit) as exc_info:
            _print_summary(results, 2)

        assert exc_info.value.code == 1

        captured = capsys.readouterr()
        assert "1/2 environments succeeded" in captured.out
        assert "1 failed" in captured.out


class TestMain:
    """Tests for main function."""

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_missing_rayci_build_id_exits(self):
        """Test main exits when RAYCI_BUILD_ID is not set."""
        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
