#!/bin/bash

set -euo pipefail

export RELEASE_QUEUE_DEFAULT="rayturbo_small_queue"
RAY_WANT_COMMIT_IN_IMAGE="$(cat .UPSTREAM)"
export RAY_WANT_COMMIT_IN_IMAGE
export RELEASE_AWS_BUCKET="runtime-release-test-artifacts"
# Get build ID from environment variables
BUILD_ID="${RAYCI_BUILD_ID:-}"

if [[ -z "${BUILD_ID}" ]]; then
    if [[ -n "${BUILDKITE_BUILD_ID:-}" ]]; then
        # Generate SHA256 hash of BUILDKITE_BUILD_ID and take first 8 chars
        BUILD_ID=$(echo -n "${BUILDKITE_BUILD_ID}" | sha256sum | cut -c1-8)
    fi
fi

export RAYCI_BUILD_ID="${BUILD_ID}"
echo "RAYCI_BUILD_ID: ${RAYCI_BUILD_ID}"

aws ecr get-login-password --region us-west-2 | \
    docker login --username AWS --password-stdin 830883877497.dkr.ecr.us-west-2.amazonaws.com
echo "--- Generate custom build steps"

curl -sSfLo /tmp/bazelisk https://github.com/bazelbuild/bazelisk/releases/download/v1.19.0/bazelisk-linux-amd64
chmod +x /tmp/bazelisk

# Keep in sync with test-init.sh
/tmp/bazelisk run //release:custom_image_build_and_test_init -- \
    --test-collection-file release/release_runtime_tests.yaml \
    --test-collection-file release/release_data_tests.yaml \
    --test-collection-file release/release_daft_tests.yaml \
    --test-collection-file release/release_tests.yaml \
    --run-jailed-tests \
    --global-config runtime_config.yaml \
    --run-unstable-tests \
    --custom-build-jobs-output-file .buildkite/release/custom_build_jobs.rayci.yaml \
    --test-jobs-output-file .buildkite/release/release_tests.json

echo "--- Upload steps"

curl -sSfL "https://raw.githubusercontent.com/ray-project/rayci/stable/run_rayci.sh" > /tmp/run_rayci.sh
/bin/bash /tmp/run_rayci.sh -upload -config anyscale/ci/config-release.yaml
buildkite-agent pipeline upload .buildkite/release/release_tests.json
