#!/bin/bash

set -euo pipefail

# Check if .UPSTREAM file's content is the merge-base of current commit
# and upstream branch.

UPSTREAM_COMMIT="$(cat .UPSTREAM)"
if [[ "${UPSTREAM_COMMIT}" == "" ]]; then
    echo "No upstream commit found" >/dev/stderr
    exit 1
fi

BRANCH_NAME="master" # Default to master if no branch is specified
if [[ "${BUILDKITE_PULL_REQUEST_BASE_BRANCH:-}" != "" ]]; then
    # If this is a pull request, use the base branch.
    BRANCH_NAME="${BUILDKITE_PULL_REQUEST_BASE_BRANCH:-}"
elif [[ "${BUILDKITE_BRANCH:-}" != "" && "${BUILDKITE_BRANCH}" =~ ^releases/.* ]]; then
    # If this is a release branch, use the branch name.
    BRANCH_NAME="${BUILDKITE_BRANCH:-}"
fi

if [[ "${TMP_DIR:-}" == "" ]]; then
    TMP_DIR="$(mktemp -d)"
fi

# Clone into a temp directory to avoid polluting the current directory.
git clone . "${TMP_DIR}/rayturbo" >/dev/null 2>&1

echo "--- Fetching upstream branch ${BRANCH_NAME}"

git -C "${TMP_DIR}/rayturbo" fetch "https://github.com/ray-project/ray.git" "${BRANCH_NAME}"
MERGE_BASE="$(git -C "${TMP_DIR}/rayturbo" merge-base HEAD FETCH_HEAD)"

rm -rf "${TMP_DIR}"

if [[ "${MERGE_BASE}" != "${UPSTREAM_COMMIT}" ]]; then
    echo "ERROR: Merge base is ${MERGE_BASE}, but upstream commit in .UPSTREAM is ${UPSTREAM_COMMIT}" >/dev/stderr
    exit 1
fi
