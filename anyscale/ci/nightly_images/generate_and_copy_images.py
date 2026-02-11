"""Generate and copy container images from ECR to local registry."""

import json
import multiprocessing
import os
import subprocess
import sys
from typing import Any

# NOTE: Do NOT import anyscale at module level!
# The SDK caches credentials at import time.
# We import it inside _copy_to_environment AFTER setting ANYSCALE_HOST and ANYSCALE_CLI_TOKEN.

# Copied from ci/ray_ci/docker_container.py; please keep in sync
PLATFORMS_RAY = [
    "cpu",
    "cu11.7.1-cudnn8",
    "cu11.8.0-cudnn8",
    "cu12.1.1-cudnn8",
    "cu12.3.2-cudnn9",
    "cu12.4.1-cudnn",
    "cu12.5.1-cudnn",
    "cu12.6.3-cudnn",
    "cu12.8.1-cudnn",
    "cu12.9.1-cudnn",
]
PYTHON_VERSIONS_RAY = ["3.10", "3.11", "3.12"]


def _platform_to_img_type_code(platform: str) -> str:
    """Convert platform string to image type code (e.g., 'cu11.8.0-cudnn8' -> 'cu118')."""
    if platform == "cpu":
        return "cpu"
    # Split by '.' and take first two parts: 'cu11.8.0-cudnn8' -> ['cu11', '8', '0-cudnn8'] -> 'cu118'
    versions = platform.split(".")
    if len(versions) >= 2:
        return f"{versions[0]}{versions[1]}"
    return "unknown"


def _get_slim_platform(platform: str) -> str:
    """Convert regular platform to slim platform by removing '-cudnn*' suffix."""
    if platform == "cpu":
        return "cpu"
    # Remove '-cudnn*' suffix: 'cu11.8.0-cudnn8' -> 'cu11.8.0'
    if "-cudnn" in platform:
        return platform.split("-cudnn")[0]
    return platform


def get_img_type_code(base_type: str, is_slim: bool) -> str:
    """Map base type to image type code."""
    return _platform_to_img_type_code(base_type)


def _python_version_to_code(py_version: str) -> str:
    """Convert Python version to version code (e.g., '3.10' -> 'py310')."""
    return f"py{py_version.replace('.', '')}"


def get_py_version_code(py_version: str) -> str:
    """Map Python version to version code."""
    if py_version not in PYTHON_VERSIONS_RAY:
        return "unknown"
    return _python_version_to_code(py_version)


def _generate_regular_mappings(rayci_build_id: str) -> list[str]:
    """Generate tag mappings for regular images (amd64)."""
    mappings = []
    for py in PYTHON_VERSIONS_RAY:
        for base in PLATFORMS_RAY:
            py_code = get_py_version_code(py)
            img_code = get_img_type_code(base, False)
            if img_code != "unknown" and py_code != "unknown":
                source_tag = f"{rayci_build_id}-{py_code}-{img_code}"
                dest_tag = f"turbonightly-{py_code}-{img_code}"
                mappings.append(f"{source_tag}:{dest_tag}")
    return mappings


def _generate_slim_mappings(rayci_build_id: str) -> list[str]:
    """Generate tag mappings for slim images (amd64)."""
    mappings = []
    slim_bases = [_get_slim_platform(platform) for platform in PLATFORMS_RAY]
    for py in PYTHON_VERSIONS_RAY:
        for base in slim_bases:
            py_code = get_py_version_code(py)
            img_code = get_img_type_code(base, True)
            if img_code != "unknown" and py_code != "unknown":
                source_tag = f"{rayci_build_id}-slim-{py_code}-{img_code}"
                dest_tag = f"turbonightly-slim-{py_code}-{img_code}"
                mappings.append(f"{source_tag}:{dest_tag}")
    return mappings


def _generate_llm_mappings(rayci_build_id: str) -> list[str]:
    """Generate tag mappings for LLM images (amd64)."""
    mappings = []
    py_code = get_py_version_code("3.11")
    img_code = get_img_type_code("cu12.8.1-cudnn", False)
    if img_code != "unknown" and py_code != "unknown":
        source_tag = f"{rayci_build_id}-{py_code}-{img_code}"
        dest_tag = f"turbonightly-{py_code}-{img_code}"
        mappings.append(f"{source_tag}:{dest_tag}")
    return mappings


def _build_repo_mappings(
    regular_mappings: list[str], llm_mappings: list[str]
) -> dict[str, str]:
    """Build repository mappings dictionary from image mappings."""
    repo_mappings = {}
    if regular_mappings:
        mappings_str = ",".join(regular_mappings)
        print(f"Will copy regular and slim images with mappings: {mappings_str}")
        repo_mappings["ray"] = mappings_str

    if llm_mappings:
        llm_mappings_str = ",".join(llm_mappings)
        print(f"Will copy LLM images with mappings: {llm_mappings_str}")
        repo_mappings["ray-llm"] = llm_mappings_str

    return repo_mappings


def _get_environments() -> list[dict[str, str]]:
    """Get list of environments to copy images to."""
    return [
        {
            "name": "predeploy",
            "host": "https://console.predeploy.anyscale.dev",
            "secret_arn": "arn:aws:secretsmanager:us-west-2:830883877497:secret:anyscale_cli_token_ci_predeploy-nkT72j",
        },
        {
            "name": "staging",
            "host": "https://console.anyscale-staging.com",
            "secret_arn": "arn:aws:secretsmanager:us-west-2:830883877497:secret:anyscale_cli_token_ci_staging-yc6NSV",
        },
        {
            "name": "production",
            "host": "https://console.anyscale.com",
            "secret_arn": "arn:aws:secretsmanager:us-west-2:830883877497:secret:anyscale_cli_token_ci_production-NgDlqA",
        },
    ]


def _copy_to_environment(
    env: dict[str, str], rayci_build_id: str, repo_mappings: dict[str, str]
) -> dict[str, Any]:
    """Copy all images to a specific environment by launching a single job.

    Uses the anyscale Python SDK. This function should be called from a separate
    process (via multiprocessing) so that each environment gets its own isolated
    environment variables for ANYSCALE_HOST and ANYSCALE_CLI_TOKEN.
    """
    env_name = env["name"]
    job_name = f"copy-images-{env_name}-{rayci_build_id}"

    print(f"\n{'='*60}")
    print(f"Processing environment: {env_name}")
    print(f"{'='*60}")

    try:
        # Fetch ANYSCALE_CLI_TOKEN for this environment
        anyscale_token = fetch_anyscale_token(env["secret_arn"])

        if not anyscale_token:
            error_msg = f"Error: Failed to fetch ANYSCALE_CLI_TOKEN for {env_name}"
            print(error_msg, file=sys.stderr)
            return {"env": env_name, "success": False, "error": error_msg}

        # Create JSON string for all repo mappings
        repo_mappings_json = json.dumps(repo_mappings)

        print(f"Submitting Anyscale job for {job_name}...")

        # Configure credentials for this environment BEFORE importing anyscale
        # This works because:
        # 1. Each environment runs in its own process (via multiprocessing)
        # 2. We import anyscale AFTER setting env vars so it picks up the right credentials
        os.environ["ANYSCALE_HOST"] = env["host"]
        os.environ["ANYSCALE_CLI_TOKEN"] = anyscale_token

        # Import anyscale AFTER setting environment variables
        # The SDK reads ANYSCALE_HOST and ANYSCALE_CLI_TOKEN at import time
        import anyscale.job

        # Get the working directory (same directory as this script)
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(current_script_dir, "anyscale_image_copy_job.py")
        script_path = os.path.abspath(script_path)
        working_dir = os.path.dirname(script_path)

        if not os.path.exists(script_path):
            raise FileNotFoundError(
                f"Script not found at {script_path}. "
                "Make sure anyscale_image_copy_job.py is in the same directory as this script."
            )

        # Create job configuration using anyscale SDK
        # crane binary is downloaded by anyscale_image_copy_job.py on each node
        job_config = anyscale.job.JobConfig(
            name=job_name,
            entrypoint=f"python3 anyscale_image_copy_job.py --json '{repo_mappings_json}'",
            working_dir=working_dir,
            cloud="anyscale_v2_default_cloud",
            max_retries=1,
        )

        # Submit the job using the SDK
        submitted_job_id = anyscale.job.submit(config=job_config)
        print(f"Job submitted with ID: {submitted_job_id}")

        # Wait for the job to complete
        anyscale.job.wait(
            id=submitted_job_id,
            state=anyscale.job.JobState.SUCCEEDED,
            timeout_s=3600,  # 1 hour timeout
        )

        success_msg = f"Successfully copied all images for {env_name}"
        print(success_msg)
        return {"env": env_name, "success": True}
    except Exception as e:
        error_msg = f"Error copying images for {env_name}: {e}"
        print(error_msg, file=sys.stderr)
        return {"env": env_name, "success": False, "error": error_msg}


def _copy_worker(
    env: dict[str, str],
    rayci_build_id: str,
    repo_mappings: dict[str, str],
    result_queue: multiprocessing.Queue,
) -> None:
    """Worker function for multiprocessing. Runs _copy_to_environment and puts result in queue."""
    result = _copy_to_environment(env, rayci_build_id, repo_mappings)
    result_queue.put(result)


def _run_parallel_copies(
    environments: list[dict[str, str]],
    rayci_build_id: str,
    repo_mappings: dict[str, str],
) -> list[dict[str, Any]]:
    """Run copies for all environments in parallel using multiprocessing.

    We use multiprocessing instead of threading because:
    1. Each environment needs different ANYSCALE_HOST and ANYSCALE_CLI_TOKEN
    2. The anyscale SDK reads credentials from os.environ at import/call time
    3. Multiprocessing gives each worker its own copy of environment variables
    """
    print(f"\n{'='*60}")
    print("Starting parallel copies to all environments...")
    print(f"{'='*60}")

    # Use multiprocessing for environment isolation
    result_queue: multiprocessing.Queue = multiprocessing.Queue()
    processes = []

    for env in environments:
        p = multiprocessing.Process(
            target=_copy_worker,
            args=(env, rayci_build_id, repo_mappings, result_queue),
        )
        processes.append((p, env["name"]))
        p.start()

    # Collect results
    results = []
    for p, env_name in processes:
        p.join()
        if p.exitcode != 0:
            results.append(
                {
                    "env": env_name,
                    "success": False,
                    "error": f"Process exited with code {p.exitcode}",
                }
            )
        else:
            try:
                result = result_queue.get_nowait()
                results.append(result)
            except Exception:
                results.append(
                    {
                        "env": env_name,
                        "success": False,
                        "error": "Failed to get result from process",
                    }
                )

    return results


def _print_summary(results: list[dict[str, Any]], num_environments: int) -> None:
    """Print copy summary."""
    print(f"\n{'='*60}")
    print("Copy Summary")
    print(f"{'='*60}")
    success_count = sum(1 for r in results if r.get("success"))
    failed_count = len(results) - success_count

    for result in results:
        status = "✓" if result.get("success") else "✗"
        env_name = result.get("env", "unknown")
        print(f"{status} {env_name}: ", end="")
        if result.get("success"):
            print("Success")
        else:
            print(f"Failed - {result.get('error', 'Unknown error')}")

    print(
        f"\nTotal: {success_count}/{num_environments} environments succeeded, {failed_count} failed"
    )

    # Exit with error code if any failed
    if failed_count > 0:
        sys.exit(1)


def fetch_anyscale_token(secret_arn: str) -> str:
    """Fetch ANYSCALE_CLI_TOKEN from AWS Secrets Manager."""
    role_arn = "arn:aws:iam::830883877497:role/buildkite-deploy-to-premerge"

    print(f"Assuming IAM role {role_arn}...")
    try:
        # Assume the IAM role
        assume_role_result = subprocess.run(
            [
                "aws",
                "sts",
                "assume-role",
                "--role-arn",
                role_arn,
                "--role-session-name",
                "buildkite-image-copy",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        # Parse the credentials from the assume-role response
        assume_role_data = json.loads(assume_role_result.stdout)
        credentials = assume_role_data["Credentials"]

        # Set environment variables for AWS credentials
        aws_env = os.environ.copy()
        aws_env["AWS_ACCESS_KEY_ID"] = credentials["AccessKeyId"]
        aws_env["AWS_SECRET_ACCESS_KEY"] = credentials["SecretAccessKey"]
        aws_env["AWS_SESSION_TOKEN"] = credentials["SessionToken"]

        print(f"Fetching ANYSCALE_CLI_TOKEN from AWS Secrets Manager ({secret_arn})...")
        # Fetch the secret using the assumed role credentials
        result = subprocess.run(
            [
                "aws",
                "secretsmanager",
                "get-secret-value",
                "--secret-id",
                secret_arn,
                "--region",
                "us-west-2",
                "--query",
                "SecretString",
                "--output",
                "text",
            ],
            env=aws_env,
            capture_output=True,
            text=True,
            check=True,
        )
        token = result.stdout.strip()
        return token
    except subprocess.CalledProcessError as e:
        print(
            f"Error: Failed to fetch ANYSCALE_CLI_TOKEN from AWS Secrets Manager: {e}",
            file=sys.stderr,
        )
        if e.stderr:
            print(f"Error details: {e.stderr}", file=sys.stderr)
        raise
    except (KeyError, json.JSONDecodeError) as e:
        print(f"Error: Failed to parse assume-role response: {e}", file=sys.stderr)
        raise


def main():
    # Ensure RAYCI_BUILD_ID is set
    rayci_build_id = os.environ.get("RAYCI_BUILD_ID")
    if not rayci_build_id:
        print("Error: RAYCI_BUILD_ID is not set", file=sys.stderr)
        sys.exit(1)

    # Generate tag mappings
    regular_mappings = _generate_regular_mappings(rayci_build_id)
    slim_mappings = _generate_slim_mappings(rayci_build_id)
    llm_mappings = _generate_llm_mappings(rayci_build_id)

    # Combine regular and slim mappings (both go to "runtime" repo)
    all_runtime_mappings = regular_mappings + slim_mappings

    # Validate we have mappings to process
    if not all_runtime_mappings and not llm_mappings:
        print("No image mappings generated", file=sys.stderr)
        sys.exit(1)

    # Build repository mappings
    repo_mappings = _build_repo_mappings(all_runtime_mappings, llm_mappings)

    # Get environments and run copies
    environments = _get_environments()
    results = _run_parallel_copies(environments, rayci_build_id, repo_mappings)

    # Print summary and exit with appropriate code
    _print_summary(results, len(environments))


if __name__ == "__main__":
    main()
