"""Unit tests for anyscale_image_copy_job.py."""

import sys
from unittest import mock

import pytest

# Mock ray before importing the module
sys.modules["ray"] = mock.MagicMock()

import anyscale_image_copy_job  # noqa: E402
from anyscale_image_copy_job import (  # noqa: E402
    DEFAULT_REPO,
    ECR_REGISTRY,
    SOURCE_REPO_MAP,
    ecr_login,
    parse_mappings,
)


class TestParseMappings:
    """Tests for parse_mappings function."""

    @pytest.mark.parametrize(
        "source,dest",
        [
            ("tag1", "dest1"),
            ("a1b2c3-py311-cu118", "turbonightly-py311-cu118"),
            ("abc_def.123", "xyz_456.789"),
            ("UPPER", "lower"),
            ("a", "b"),
        ],
    )
    def test_single_valid_mapping_parses_correctly(self, source, dest):
        """Valid source:dest mappings should parse to (source, dest)."""
        result = parse_mappings(f"{source}:{dest}")
        assert result == [(source, dest)]

    @pytest.mark.parametrize(
        "mappings,expected",
        [
            ("a:b,c:d", [("a", "b"), ("c", "d")]),
            (
                "tag1:dest1,tag2:dest2,tag3:dest3",
                [("tag1", "dest1"), ("tag2", "dest2"), ("tag3", "dest3")],
            ),
            ("x:y", [("x", "y")]),
        ],
    )
    def test_multiple_mappings_all_preserved(self, mappings, expected):
        """All mappings in a comma-separated list should be preserved."""
        result = parse_mappings(mappings)
        assert result == expected

    @pytest.mark.parametrize(
        "mapping_str,expected",
        [
            ("  tag1:dest1  ", [("tag1", "dest1")]),
            ("  a:b  ,  c:d  ", [("a", "b"), ("c", "d")]),
            (" x : y ", [("x ", " y")]),  # Note: internal whitespace preserved
        ],
    )
    def test_whitespace_around_mappings(self, mapping_str, expected):
        """Whitespace around mappings should be handled."""
        result = parse_mappings(mapping_str)
        assert result == expected

    @pytest.mark.parametrize(
        "source,dest_with_colons",
        [
            ("source", "dest:with:colons"),
            ("a", "b:c:d:e"),
            ("tag", "registry:5000/image:latest"),
        ],
    )
    def test_dest_can_contain_colons(self, source, dest_with_colons):
        """Dest tag can contain colons - only splits on first colon."""
        result = parse_mappings(f"{source}:{dest_with_colons}")
        assert result == [(source, dest_with_colons)]

    @pytest.mark.parametrize(
        "invalid_tag",
        [
            "no_colon_here",
            "invalid",
            "abc123",
        ],
    )
    def test_mapping_without_colon_raises_error(self, invalid_tag):
        """Strings without colons should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid format"):
            parse_mappings(invalid_tag)

    @pytest.mark.parametrize(
        "mapping_str",
        [
            "tag1:dest1,,tag2:dest2",
            "tag1:dest1,,,tag2:dest2",
            ",tag1:dest1,tag2:dest2,",
        ],
    )
    def test_empty_mappings_filtered_out(self, mapping_str):
        """Empty mappings (consecutive commas) should be filtered out."""
        result = parse_mappings(mapping_str)
        assert ("tag1", "dest1") in result
        assert ("tag2", "dest2") in result

    def test_empty_string_returns_empty_list(self):
        """Empty string should return empty list."""
        assert parse_mappings("") == []

    def test_realistic_image_tags(self):
        """Test with realistic image tag formats."""
        mappings = "a1b2c3-py311-cu118:turbonightly-py311-cu118,a1b2c3-slim-py310-cpu:turbonightly-slim-py310-cpu"
        result = parse_mappings(mappings)
        assert result == [
            ("a1b2c3-py311-cu118", "turbonightly-py311-cu118"),
            ("a1b2c3-slim-py310-cpu", "turbonightly-slim-py310-cpu"),
        ]


class TestSourceRepoMap:
    """Tests for SOURCE_REPO_MAP configuration."""

    def test_ray_maps_to_runtime(self):
        """ray dest repo should map to runtime source repo."""
        assert SOURCE_REPO_MAP["ray"] == "runtime"

    def test_ray_llm_maps_to_runtime_llm(self):
        """ray-llm dest repo should map to runtime-llm source repo."""
        assert SOURCE_REPO_MAP["ray-llm"] == "runtime-llm"

    def test_default_repo_has_mapping(self):
        """DEFAULT_REPO should have a mapping in SOURCE_REPO_MAP."""
        assert DEFAULT_REPO in SOURCE_REPO_MAP


class TestEcrLogin:
    """Tests for ecr_login function."""

    @mock.patch.object(anyscale_image_copy_job, "_ensure_crane_installed")
    @mock.patch.object(anyscale_image_copy_job.subprocess, "check_call")
    def test_ecr_login_calls_bash_with_aws_and_crane(
        self, mock_check_call, mock_ensure_crane
    ):
        """Test ECR login calls bash with correct AWS and crane commands."""
        mock_check_call.return_value = 0

        ecr_login()

        mock_ensure_crane.assert_called_once()
        mock_check_call.assert_called_once()
        call_args = mock_check_call.call_args[0][0]
        assert call_args[0] == "/bin/bash"
        assert call_args[1] == "-elic"
        assert "aws ecr get-login-password" in call_args[2]
        assert "crane auth login" in call_args[2]
        assert ECR_REGISTRY in call_args[2]


class TestMainArgParsing:
    """Tests for main function argument parsing."""

    @mock.patch.object(anyscale_image_copy_job, "ray")
    @mock.patch.object(anyscale_image_copy_job, "ecr_login")
    @mock.patch.object(anyscale_image_copy_job, "process_repo_mappings")
    def test_main_with_images_arg(self, mock_process, mock_ecr, mock_ray):
        """Test main with positional images argument."""
        test_args = ["prog", "tag1:dest1,tag2:dest2"]
        with mock.patch.object(sys, "argv", test_args):
            anyscale_image_copy_job.main()

        mock_ray.init.assert_called_once()
        mock_ecr.assert_called_once()
        mock_process.assert_called_once()
        args, kwargs = mock_process.call_args
        assert args[0] == DEFAULT_REPO
        assert args[1] == [("tag1", "dest1"), ("tag2", "dest2")]

    @mock.patch.object(anyscale_image_copy_job, "ray")
    @mock.patch.object(anyscale_image_copy_job, "ecr_login")
    @mock.patch.object(anyscale_image_copy_job, "process_repo_mappings")
    def test_main_with_custom_repo(self, mock_process, mock_ecr, mock_ray):
        """Test main with custom repo argument."""
        test_args = ["prog", "--repo", "ray-llm", "tag1:dest1"]
        with mock.patch.object(sys, "argv", test_args):
            anyscale_image_copy_job.main()

        args, kwargs = mock_process.call_args
        assert args[0] == "ray-llm"

    @mock.patch.object(anyscale_image_copy_job, "ray")
    @mock.patch.object(anyscale_image_copy_job, "ecr_login")
    @mock.patch.object(anyscale_image_copy_job, "process_repo_mappings")
    def test_main_with_json_arg(self, mock_process, mock_ecr, mock_ray):
        """Test main with JSON argument for multiple repos."""
        json_arg = '{"ray": "tag1:dest1", "ray-llm": "tag2:dest2"}'
        test_args = ["prog", "--json", json_arg]
        with mock.patch.object(sys, "argv", test_args):
            anyscale_image_copy_job.main()

        # Should be called twice, once for each repo
        assert mock_process.call_count == 2

    @mock.patch.object(anyscale_image_copy_job, "ray")
    @mock.patch.object(anyscale_image_copy_job, "ecr_login")
    def test_main_with_invalid_json(self, mock_ecr, mock_ray):
        """Test main exits with invalid JSON."""
        test_args = ["prog", "--json", "invalid json"]
        with mock.patch.object(sys, "argv", test_args):
            with pytest.raises(SystemExit) as exc_info:
                anyscale_image_copy_job.main()
            assert exc_info.value.code == 1

    @mock.patch.object(anyscale_image_copy_job, "ray")
    @mock.patch.object(anyscale_image_copy_job, "ecr_login")
    def test_main_no_args_exits(self, mock_ecr, mock_ray):
        """Test main exits when no arguments provided."""
        test_args = ["prog"]
        with mock.patch.object(sys, "argv", test_args):
            with pytest.raises(SystemExit):
                anyscale_image_copy_job.main()


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
