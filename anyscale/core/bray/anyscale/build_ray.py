import os
import subprocess
import json
import requests
import random
import string
from typing import Tuple

import anyscale
from anyscale.workspace.models import WorkspaceConfig
from anyscale.compute_config.models import (
    ComputeConfig,
    HeadNodeConfig,
)

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

_BRAY_PREFIX = "bray"
_ANYSCALE_HOST = os.environ.get("ANYSCALE_HOST", "https://console.anyscale.com")
_WORKSPACE_URL = f"{_ANYSCALE_HOST}/cld_kvedZWag2qA8i5BjxUevf5i7/prj_g7p6lsu6r8g7garwbxifppyz23/workspaces"
_WORKSPACE_IMAGE = "anyscale/ray:2.46.0-slim-py312"
_WORKSPACE_CLOUD = "anyscale_v2_default_cloud"
_WORKSPACE_INSTANCE_TYPE = "m7i.2xlarge"
_WORKSPACE_IDLE_TERMINATION_MINUTES = 120

_RAY_CHECKOUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ray")
_BAZEL_CACHE_PORT = 9095
_BAZEL_CACHE_SERVER = f"http://localhost:{_BAZEL_CACHE_PORT}"
_BAZEL_CACHE_S3_BUCKET = "core-bazel-cache"
_BUILD_ARTIFACT_TARBALL = "ray-opt.tgz"


def log(message: str) -> None:
    """Log a message to the console."""
    print(f"\033[32m{message}\033[0m")


def _get_aws_credentials() -> Tuple[str, str, str]:
    """
    Get a short-lived AWS credentials from the current Anyscale job. These credentials
    are used to connect the local bazel cache with the S3 storage.
    """
    identity = json.loads(
        subprocess.run(
            ["aws", "sts", "get-caller-identity"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )
    account_id = identity["Account"]
    role = identity["Arn"].split("/")[1]
    credentials = json.loads(
        subprocess.run(
            [
                "aws",
                "sts",
                "assume-role",
                "--role-arn",
                f"arn:aws:iam::{account_id}:role/{role}",
                # The unique name of the session is not important, but it is required.
                "--role-session-name",
                "ray-cache-server",
                # Lifespan of the temporary credentials in seconds. 1 hour is the maximum.
                "--duration-seconds",
                "3600",
                "--output",
                "json",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )
    return (
        credentials["Credentials"]["AccessKeyId"],
        credentials["Credentials"]["SecretAccessKey"],
        credentials["Credentials"]["SessionToken"],
    )


def _run_build_cache_server(
    access_key_id: str, secret_access_key: str, session_token: str
):
    """Run the bazel-remote cache server locally and connect it to the S3 storage."""
    subprocess.Popen(
        [
            "bazel-remote",
            f"--s3.bucket={_BAZEL_CACHE_S3_BUCKET}",
            "--host=0.0.0.0",
            f"--port={_BAZEL_CACHE_PORT}",
            f"--grpc_port={_BAZEL_CACHE_PORT + 1}",
            "--dir=/tmp/ray-checkout-cache",
            "--max_size=100",
            "--s3.auth_method=access_key",
            "--s3.endpoint=s3.amazonaws.com",
            f"--s3.access_key_id={access_key_id}",
            f"--s3.secret_access_key={secret_access_key}",
            f"--s3.session_token={session_token}",
        ],
    )

    # Wait for the server to start
    while True:
        try:
            response = requests.get(f"{_BAZEL_CACHE_SERVER}/status")
            if response.status_code == 200:
                print("Cache server is running.")
                break
        except requests.exceptions.RequestException:
            pass


def _build_ray():
    """Build the Ray package using bazel."""
    process = subprocess.Popen(
        ["bazel", "build", "--remote_cache", _BAZEL_CACHE_SERVER, "//:ray_pkg"],
        cwd=_RAY_CHECKOUT_DIR,
    )
    return process.wait() == 0


def _upload_build_artifacts():
    """
    Upload the build artifacts to S3. We do this by creating a tarball of the python
    directory and uploading it to S3. The tarball is unique per build name.
    """
    subprocess.run(
        ["tar", "-czf", _BUILD_ARTIFACT_TARBALL, "python"],
        cwd=_RAY_CHECKOUT_DIR,
        check=True,
    )
    build_name = os.environ["RAY_REMOTE_BUILD_NAME"]
    subprocess.run(
        [
            "aws",
            "s3",
            "cp",
            _BUILD_ARTIFACT_TARBALL,
            f"s3://{_BAZEL_CACHE_S3_BUCKET}/{build_name}/{_BUILD_ARTIFACT_TARBALL}",
        ],
        cwd=_RAY_CHECKOUT_DIR,
        check=True,
    )


def _run_tests():
    """
    Run the tests specified in the RAY_REMOTE_TESTS environment variable. If no tests are
    specified, this function does nothing.
    """
    tests = os.environ.get("RAY_REMOTE_TESTS", "").strip()
    if not tests:
        log("No tests to run.")
        return

    log(f"Running tests: {tests}")
    test_list = tests.split()
    process = subprocess.Popen(
        [
            "bazel",
            "test",
            "--config=ci",
            "--test_output",
            "streamed",
            "--remote_cache",
            _BAZEL_CACHE_SERVER,
            f"--test_env=PYTHONPATH={_RAY_CHECKOUT_DIR}/python:/home/ray/anaconda3/lib/python3.12/site-packages",
            *test_list,
        ],
        cwd=_RAY_CHECKOUT_DIR,
    )
    if process.wait() != 0:
        raise RuntimeError("Tests failed")


def _get_base_commit(ray_base_commit: str, is_ray_turbo: bool) -> str:
    if is_ray_turbo:
        return ray_base_commit[:6]
    ray_commit_map = json.loads(
        subprocess.check_output(
            [
                "aws",
                "s3",
                "cp",
                f"s3://{_BAZEL_CACHE_S3_BUCKET}/ray_commit_map.json",
                "-",
            ],
            text=True,
        )
    )
    return ray_commit_map.get("ray_to_ray_turbo_commit_map", {}).get(
        ray_base_commit,
        ray_commit_map["ray_turbo_head"],
    )[:6]


def _create_workspace(build_name: str, ray_base_commit: str, is_ray_turbo: bool):
    """
    Create an Anyscale workspace to host the build artifacts.
    Args:
        build_name: Unique identifier for the build job.
        working_dir: Directory containing the Ray source code and build scripts.
        ray_dir: Directory containing the Ray source code.
    """
    try:
        workspace_id = anyscale.workspace.get(
            name=f"{_BRAY_PREFIX}-{build_name}",
        ).id
    except ValueError:
        workspace_id = anyscale.workspace.create(
            WorkspaceConfig(
                name=f"{_BRAY_PREFIX}-{build_name}",
                image_uri=_WORKSPACE_IMAGE,
                compute_config=ComputeConfig(
                    cloud=_WORKSPACE_CLOUD,
                    head_node=HeadNodeConfig(instance_type=_WORKSPACE_INSTANCE_TYPE),
                    worker_nodes=[],
                ),
                idle_termination_minutes=_WORKSPACE_IDLE_TERMINATION_MINUTES,
            ),
        )
    anyscale.workspace.start(id=workspace_id)
    log(f"Workspace started: {_WORKSPACE_URL}/{workspace_id}")

    # Instruction to rebuild the workspace
    log("Go to your workspace and rebuild with the following Dockerfile:")
    with open(f"{_CURRENT_DIR}/bray.Dockerfile", "r") as infile:
        content = infile.read()
        content = content.replace(
            "__REMOTE_RANDOM_ECHO__",
            "".join(random.choices(string.ascii_letters + string.digits, k=8)),
        )
        content = content.replace("__REMOTE_BUILD_NAME__", build_name)
        content = content.replace(
            "__REMOTE_BUILD_COMMIT_BASE__",
            _get_base_commit(ray_base_commit, is_ray_turbo),
        )
        log(content)


def main():
    _run_build_cache_server(*_get_aws_credentials())
    is_success = _build_ray()
    if not is_success:
        raise RuntimeError("Build failed")
    _upload_build_artifacts()
    _run_tests()
    no_workspace = os.environ.get("NO_WORKSPACE", "0") == "1"
    if no_workspace:
        log("Skipping workspace creation.")
        return

    log("Creating workspace...")
    _create_workspace(
        os.environ["RAY_REMOTE_BUILD_NAME"],
        os.environ["RAY_BASE_COMMIT"],
        os.environ.get("IS_RAY_TURBO", "0") == "1",
    )


if __name__ == "__main__":
    main()
