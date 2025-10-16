import os
import subprocess
import tempfile
from typing import Tuple

import click

import anyscale
from anyscale.compute_config.models import (
    ComputeConfig,
    HeadNodeConfig,
)
from anyscale.job.models import JobConfig

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

_BRAY_PREFIX = "bray"

_BUILD_JOB_IMAGE = "anyscale/image/ray-build-env:3"
_BUILD_JOB_CLOUD = "anyscale_v2_default_cloud"
_BUILD_JOB_INSTANCE_TYPE = "m7a.12xlarge"


def _is_ray_turbo(ray_dir: str) -> bool:
    """
    Check if the Ray directory is a Ray Turbo directory.

    Args:
        ray_dir: Directory containing the Ray source code.

    Returns:
        True if the directory is a Ray Turbo directory, False otherwise.
    """
    return os.path.exists(os.path.join(ray_dir, ".UPSTREAM"))


def log(message: str) -> None:
    """Log a message to the console."""
    print(f"\033[32m{message}\033[0m")


def build_ray(
    build_name: str,
    working_dir: str,
    ray_dir: str,
    tests: Tuple[str],
    no_workspace: bool,
) -> None:
    """
    Submit a job to build Ray on Anyscale.

    Args:
        build_name: Unique identifier for the build job.
        working_dir: Directory containing the Ray source code and build scripts.
    """
    build_base_commit = (
        subprocess.check_output(["git", "merge-base", "master", "HEAD"], cwd=ray_dir)
        .strip()
        .decode("utf-8")
    )
    job_config = JobConfig(
        name=f"{_BRAY_PREFIX}-build-{build_name}",
        entrypoint="python build_ray.py",
        working_dir=working_dir,
        excludes=["ray/.git"],
        image_uri=_BUILD_JOB_IMAGE,
        compute_config=ComputeConfig(
            cloud=_BUILD_JOB_CLOUD,
            head_node=HeadNodeConfig(instance_type=_BUILD_JOB_INSTANCE_TYPE),
        ),
        env_vars={
            "RAY_REMOTE_BUILD_NAME": build_name,
            "RAY_BASE_COMMIT": build_base_commit,
            "RAY_REMOTE_TESTS": " ".join(list(tests)),
            "IS_RAY_TURBO": "1" if _is_ray_turbo(working_dir) else "0",
            "NO_WORKSPACE": "1" if no_workspace else "0",
        },
        max_retries=0,
    )
    id = anyscale.job.submit(job_config)

    # Stream the logs of the job. This will block until the job is finished. We are
    # using the CLI instead of the Python API to stream the logs.
    process = subprocess.Popen(
        ["anyscale", "job", "logs", "--follow", "--job-id", id],
    )
    process.wait()
    anyscale.job.wait(id=id)


@click.command(
    help=(
        "Build Ray on Anyscale using a remote job. This script then creates an "
        "anyscale workspace with the build artifacts produced by the job. "
    )
)
@click.option(
    "--build-name", required=True, help="Unique identifier for the build job."
)
@click.option(
    "--ray-dir", required=True, help="Directory containing the Ray code to be built."
)
@click.option(
    "--test",
    multiple=True,
    help=(
        "List of tests to run after building Ray, e.g. --test=TEST1 --test=TEST2. "
        "Note that TEST1/TEST2 is the bazel target, e.g. //python/ray/tests:test01 or "
        "//src/ray/core:test01. If not provided, no tests will be run."
    ),
)
@click.option(
    "--no-workspace",
    is_flag=True,
    default=False,
    help="Do not create a workspace to host the build artifacts.",
)
def main(
    build_name: str,
    ray_dir: str,
    test: Tuple[str],
    no_workspace: bool,
) -> None:
    with tempfile.TemporaryDirectory() as working_dir:
        # Copy the Ray source code and the build scripts to the temporary directory
        log("Wrapping local ray ...")
        os.symlink(ray_dir, f"{working_dir}/ray")
        os.system(f"cp -r {_CURRENT_DIR}/anyscale/* {working_dir}")

        # Submit the build job
        log("Build ray remotely ...")
        build_ray(build_name, working_dir, ray_dir, test, no_workspace)


if __name__ == "__main__":
    main()
