import json
import os
import subprocess
import tempfile
from dataclasses import asdict, dataclass

import click

import anyscale
from anyscale.compute_config.models import (
    ComputeConfig,
    HeadNodeConfig,
)
from anyscale.job.models import JobConfig

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

_BRAY_PREFIX = "bray"

_BUILD_JOB_CLOUD = "anyscale_v2_default_cloud"
_BUILD_JOB_INSTANCE_TYPE = "m5.xlarge"
_COMMIT_MAP_LENGTH = 1000
_COMMIT_MAP_FILE_NAME = "ray_commit_map.json"


@dataclass
class RayCommitMap:
    """
    A class to represent a mapping from Ray commit to Ray Turbo commit.
    """

    ray_turbo_head: str
    ray_to_ray_turbo_commit_map: dict


def log(message: str) -> None:
    """Log a message to the console."""
    print(f"\033[32m{message}\033[0m")


def upload_ray_commit_map(working_dir: str) -> None:
    """
    Submit a job to upload the Ray commit map to S3.

    Args:
        ray_commit_map: A RayCommitMap object containing the mapping.
    """
    job_config = JobConfig(
        name=f"{_BRAY_PREFIX}-upload-ray-commit-map",
        entrypoint="python populate_ray_commit_map.py",
        working_dir=working_dir,
        compute_config=ComputeConfig(
            cloud=_BUILD_JOB_CLOUD,
            head_node=HeadNodeConfig(instance_type=_BUILD_JOB_INSTANCE_TYPE),
        ),
        env_vars={
            "RAY_COMMIT_MAP_FILE_NAME": _COMMIT_MAP_FILE_NAME,
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


def compute_ray_commit_map(
    ray_turbo_dir: str, ray_dir: str, working_dir: str
) -> RayCommitMap:
    """
    Compute a mapping from Ray commit to Ray Turbo commit.

    Args:
        ray_turbo_dir: Directory containing the Ray code to be built.

    Returns:
        A RayCommitMap object containing the mapping.
    """
    commit_map = RayCommitMap(
        ray_turbo_head="",
        ray_to_ray_turbo_commit_map={},
    )
    ray_commits = (
        subprocess.check_output(
            ["git", "log", "-n", f"{2 * _COMMIT_MAP_LENGTH}", "--pretty=format:%H"],
            cwd=ray_dir,
        )
        .strip()
        .decode("utf-8")
    ).split("\n")
    ray_turbo_commits = (
        subprocess.check_output(
            ["git", "log", "-n", f"{_COMMIT_MAP_LENGTH}", "--pretty=format:%H"],
            cwd=ray_turbo_dir,
        )
        .strip()
        .decode("utf-8")
    ).split("\n")
    ray_turbo_commits.reverse()
    for commit in ray_turbo_commits:
        if commit not in ray_commits:
            commit_map.ray_turbo_head = commit
            continue
        if commit_map.ray_turbo_head:
            commit_map.ray_to_ray_turbo_commit_map[commit] = commit_map.ray_turbo_head

    with open(f"{working_dir}/{_COMMIT_MAP_FILE_NAME}", "w") as f:
        json.dump(asdict(commit_map), f)


@click.command(
    help=("Compute a map from Ray commit to Ray Turbo commit. Upload the result to S3.")
)
@click.option(
    "--ray-turbo-dir",
    required=True,
    help="Directory containing the Ray code to be built.",
)
@click.option(
    "--ray-dir",
    required=True,
    help="Directory containing the Ray source code.",
)
def main(ray_turbo_dir: str, ray_dir: str) -> None:
    with tempfile.TemporaryDirectory() as working_dir:
        # Compute the Ray commit map
        log("Computing Ray commit map ...")
        compute_ray_commit_map(ray_turbo_dir, ray_dir, working_dir)

        # Copy the Ray source code and the build scripts to the temporary directory
        log("Constructing working directory ...")
        os.system(f"cp -r {_CURRENT_DIR}/anyscale/* {working_dir}")

        # Upload the Ray commit map
        log("Uploading Ray commit map ...")
        upload_ray_commit_map(working_dir)


if __name__ == "__main__":
    main()
