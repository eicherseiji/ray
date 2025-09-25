#!/bin/bash

set -euo pipefail

export RELEASE_QUEUE_DEFAULT="rayturbo_small_queue"
RAY_WANT_COMMIT_IN_IMAGE="$(cat .UPSTREAM)"
export RAY_WANT_COMMIT_IN_IMAGE
export RELEASE_AWS_BUCKET="runtime-release-test-artifacts"

echo "--- Generate custom build steps"

curl -sSfLo /tmp/bazelisk https://github.com/bazelbuild/bazelisk/releases/download/v1.19.0/bazelisk-linux-amd64
chmod +x /tmp/bazelisk

# Keep in sync with test-init.sh
/tmp/bazelisk run //release:custom_byod_build_init -- \
    --test-collection-file release/release_runtime_tests.yaml \
    --test-collection-file release/release_data_tests.yaml \
    --test-collection-file release/release_daft_tests.yaml \
    --test-collection-file release/release_tests.yaml \
    --run-jailed-tests \
    --global-config runtime_config.yaml \
    --run-unstable-tests

echo "--- Upload steps"

curl -sSfL "https://raw.githubusercontent.com/ray-project/rayci/stable/run_rayci.sh" > /tmp/run_rayci.sh
/bin/bash /tmp/run_rayci.sh -upload -config anyscale/ci/config-release.yaml
