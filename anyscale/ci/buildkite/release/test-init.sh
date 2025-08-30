#!/bin/bash

set -euo pipefail

if [[ ${BUILDKITE_COMMIT} == "HEAD" ]]; then
    BUILDKITE_COMMIT="$(git rev-parse HEAD)"
    export BUILDKITE_COMMIT
fi

aws ecr get-login-password --region us-west-2 | \
    docker login --username AWS --password-stdin 830883877497.dkr.ecr.us-west-2.amazonaws.com

# Login to GCP.
bash release/gcloud_docker_login.sh release/aws2gce_runtime_iam.json
export PATH="${PWD}/google-cloud-sdk/bin:${PATH}"

if [[ "${AUTOMATIC:-0}" == "1" && "${BUILDKITE_BRANCH}" == "master" ]]; then
    export REPORT_TO_RAY_TEST_DB=1
fi
export RELEASE_QUEUE_DEFAULT="rayturbo_small_queue"
RAY_WANT_COMMIT_IN_IMAGE="$(cat .UPSTREAM)"
export RAY_WANT_COMMIT_IN_IMAGE
export RELEASE_AWS_BUCKET="runtime-release-test-artifacts"

curl -sSfL -o /tmp/bazel https://github.com/bazelbuild/bazelisk/releases/download/v1.19.0/bazelisk-linux-amd64
chmod +x /tmp/bazel
/tmp/bazel run //release:build_pipeline -- \
    --test-collection-file release/release_runtime_tests.yaml \
    --test-collection-file release/release_data_tests.yaml \
    --test-collection-file release/release_tests.yaml \
    --run-jailed-tests \
    --global-config runtime_config.yaml \
    --run-unstable-tests \
    | buildkite-agent pipeline upload
