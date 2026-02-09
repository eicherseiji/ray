#!/usr/bin/env python3
"""Copy container images from ECR to a local registry using crane."""

import argparse
import json
import os
import subprocess
import sys

import ray

ECR_REGISTRY = "830883877497.dkr.ecr.us-west-2.amazonaws.com"
LOCAL_REGISTRY = "localhost:5555"
DEFAULT_REPO = "ray"

# Mapping from destination repo names to source (ECR) repo names
# Source images are in ECR under "runtime"/"runtime-llm", but we copy them
# to local registry under "ray"/"ray-llm"
SOURCE_REPO_MAP = {
    "ray": "runtime",
    "ray-llm": "runtime-llm",
}

# Crane binary location and version
CRANE_VERSION = "v0.20.2"
CRANE_PATH = "/tmp/crane"


def _ensure_crane_installed():
    """Download crane binary if not already present."""
    if os.path.exists(CRANE_PATH) and os.access(CRANE_PATH, os.X_OK):
        return
    print(f"Downloading crane {CRANE_VERSION}...")
    download_cmd = (
        f'curl -sL "https://github.com/google/go-containerregistry/releases/download/'
        f'{CRANE_VERSION}/go-containerregistry_Linux_x86_64.tar.gz" | tar -xzf - -C /tmp crane'
    )
    subprocess.check_call(["/bin/bash", "-c", download_cmd])
    os.chmod(CRANE_PATH, 0o755)
    print("crane downloaded successfully")


def ecr_login():
    """Authenticate to ECR using crane."""
    _ensure_crane_installed()
    print("Logging into ECR...")
    cmd = (
        f"aws ecr get-login-password --region us-west-2 | "
        f"{CRANE_PATH} auth login -u AWS --password-stdin {ECR_REGISTRY}"
    )
    subprocess.check_call(["/bin/bash", "-elic", cmd])
    print("ECR login successful")


@ray.remote
def copy_image(source_image: str, dest_image: str):
    """Copy an image using crane."""
    # Ensure crane is installed on this worker node
    _ensure_crane_installed()

    # Authenticate to ECR on each worker node before copying
    # ECR credentials are node-specific and not shared across Ray workers
    try:
        ecr_login()
    except Exception as e:
        print(f"Warning: ECR login failed on worker: {e}", file=sys.stderr)
        # Continue anyway - might already be authenticated

    print(f"Copying {source_image} -> {dest_image}")
    subprocess.run([CRANE_PATH, "cp", source_image, dest_image], check=True)
    print(f"Successfully copied {dest_image}")
    return dest_image


def parse_mappings(mappings: str) -> list[tuple[str, str]]:
    """Parse comma-separated image tag mappings string into a list of tuples.

    Args:
        mappings: Comma-separated string of image tag mappings in format
            "source_tag:dest_tag,source_tag:dest_tag". Each mapping specifies
            a source tag from ECR and the destination tag to use in the local registry.
            Example: "a1b2c3-py311-cu118:nightly-py311-cu118,a1b2c3-slim-py310-cpu:nightly-slim-py310-cpu"

    Returns:
        List of (source_tag, dest_tag) tuples.

    Raises:
        ValueError: If any mapping is in an invalid format.
    """
    image_mappings = []
    for mapping in mappings.split(","):
        mapping = mapping.strip()
        if not mapping:
            continue
        if ":" not in mapping:
            raise ValueError(
                f"Invalid format '{mapping}'. Expected 'source_tag:dest_tag'"
            )
        source_tag, dest_tag = mapping.split(":", 1)
        image_mappings.append((source_tag, dest_tag))
    return image_mappings


def _copy_images(
    dest_repo: str, image_mappings: list[tuple[str, str]]
) -> list[tuple[str, str]]:
    """Copy images from ECR to local registry in parallel.

    Args:
        dest_repo: Destination repository name (e.g., "ray", "ray-llm")
        image_mappings: List of (source_tag, dest_tag) tuples

    Returns:
        List of (dest_tag, dest_image) tuples for successfully copied images.

    Raises:
        SystemExit: If any image copy fails.
    """
    # Get source repo from mapping, or use dest_repo if not mapped
    source_repo = SOURCE_REPO_MAP.get(dest_repo, dest_repo)

    print(f"\nCopying {len(image_mappings)} images...")
    print(f"  Source: {ECR_REGISTRY}/anyscale/{source_repo}")
    print(f"  Dest:   {LOCAL_REGISTRY}/anyscale/{dest_repo}")

    # Copy all images in parallel using Ray
    copy_tasks = []
    for source_tag, dest_tag in image_mappings:
        source_image = f"{ECR_REGISTRY}/anyscale/{source_repo}:{source_tag}"
        dest_image = f"{LOCAL_REGISTRY}/anyscale/{dest_repo}:{dest_tag}"
        copy_tasks.append((copy_image.remote(source_image, dest_image), dest_tag))

    # Wait for all copies to complete
    dest_images = []
    failed_copies = []
    for copy_ref, dest_tag in copy_tasks:
        try:
            ray.get(copy_ref)
            dest_image = f"{LOCAL_REGISTRY}/anyscale/{dest_repo}:{dest_tag}"
            dest_images.append((dest_tag, dest_image))
        except Exception as e:
            error_msg = f"Failed to copy image for tag {dest_tag}: {e}"
            print(error_msg, file=sys.stderr)
            failed_copies.append((dest_tag, str(e)))

    if failed_copies:
        print(
            f"Error: {len(failed_copies)} image(s) failed to copy for repository '{dest_repo}'",
            file=sys.stderr,
        )
        for dest_tag, error in failed_copies:
            print(f"  - {dest_tag}: {error}", file=sys.stderr)
        sys.exit(1)

    print(f"All images for repository '{dest_repo}' copied successfully!")
    return dest_images


def process_repo_mappings(
    dest_repo: str,
    image_mappings: list[tuple[str, str]],
):
    """Process image mappings for a specific repository.

    Args:
        dest_repo: Destination repository name (e.g., "ray", "ray-llm").
            Source repo is determined via SOURCE_REPO_MAP.
        image_mappings: List of (source_tag, dest_tag) tuples specifying
            image tag mappings from ECR to local registry.
    """
    if not image_mappings:
        return

    source_repo = SOURCE_REPO_MAP.get(dest_repo, dest_repo)
    print(f"\nProcessing {len(image_mappings)} images: {source_repo} -> {dest_repo}...")

    # Copy images from ECR to local registry
    _copy_images(dest_repo, image_mappings)


def main():
    parser = argparse.ArgumentParser(
        description="Copy container images from ECR to a local registry using crane.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s f493b0-slim-py311-cpu:turbonightly-slim-py311-cpu
  %(prog)s tag1:dest1,tag2:dest2,tag3:dest3
  %(prog)s --json '{"ray": "tag1:dest1,tag2:dest2", "ray-llm": "tag3:dest3"}'
        """,
    )
    parser.add_argument(
        "images",
        metavar="MAPPINGS",
        nargs="?",
        help="Comma-separated image tag mappings in format 'source:dest,source:dest'",
    )
    parser.add_argument(
        "--repo",
        default=DEFAULT_REPO,
        help=f"Destination repository name (default: {DEFAULT_REPO}). Source repo is auto-mapped (ray->runtime, ray-llm->runtime-llm).",
    )
    parser.add_argument(
        "--json",
        help='JSON object mapping repo names to comma-separated mappings, e.g. \'{"ray": "tag1:dest1", "ray-llm": "tag2:dest2"}\'',
    )
    args = parser.parse_args()

    ray.init(ignore_reinit_error=True)

    ecr_login()

    if args.json:
        # JSON mode: process multiple repositories
        try:
            repo_mappings = json.loads(args.json)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON format: {e}", file=sys.stderr)
            sys.exit(1)

        for repo, mappings in repo_mappings.items():
            image_mappings = parse_mappings(mappings)
            process_repo_mappings(repo, image_mappings)
    elif args.images:
        # single repository with mappings
        image_mappings = parse_mappings(args.images)
        process_repo_mappings(args.repo, image_mappings)
    else:
        parser.error("Either 'images' argument or '--json' option must be provided")

    print("\nAll images copied successfully!")

    ray.shutdown()


if __name__ == "__main__":
    main()
