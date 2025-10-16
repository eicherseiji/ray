import os
import subprocess

BAZEL_CACHE_S3_BUCKET = "core-bazel-cache"
RAY_COMMIT_MAP_NAME = "ray_commit_map.json"


def main():
    subprocess.run(
        [
            "aws",
            "s3",
            "cp",
            os.environ["RAY_COMMIT_MAP_FILE_NAME"],
            f"s3://{BAZEL_CACHE_S3_BUCKET}/{RAY_COMMIT_MAP_NAME}",
        ],
        check=True,
    )
    print("Upload successful.")


if __name__ == "__main__":
    main()
