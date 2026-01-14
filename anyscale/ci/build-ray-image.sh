#!/bin/bash

set -euo pipefail

PY_VERSION="${1:-3.8}"
IMG_TYPE="${2:-cpu}"
BASE_TYPE="${3:-ray}"
IS_SLIM="${4:-0}"

LAYER_TYPE="${LAYER_TYPE:-base}"
REPO_SERIES_NAME="${REPO_SERIES_NAME:-runtime}"

source anyscale/ci/setup-env.sh

IMAGE_PREFIX="${RAYCI_BUILD_ID}"

UPSTREAM_COMMIT="$(cat .UPSTREAM)"
if [[ "${UPSTREAM_COMMIT}" == "" ]]; then
    echo "No upstream commit found" >/dev/stderr
    exit 1
fi

UPSTREAM_BRANCH="master"
if [[ "${RAY_VERSION}" != "3.0.0.dev0" ]]; then
    UPSTREAM_BRANCH="releases/${RAY_VERSION}"
fi

# TODO(aslonnie): add some graceful wait for the wheel from upstream to be built.
# Normally at thist point, the wheel should have already been built, but there
# is no hard guarantee.
OSS_WHEEL_URL_PREFIX="https://ray-wheels.s3.us-west-2.amazonaws.com/${UPSTREAM_BRANCH}/${UPSTREAM_COMMIT}/"

if [[ "${BASE_TYPE}" == "ray" ]]; then
    RAYTURBO_REPO="830883877497.dkr.ecr.us-west-2.amazonaws.com/anyscale/${REPO_SERIES_NAME}"
elif [[ "${BASE_TYPE}" == "ray-llm" ]]; then
    RAYTURBO_REPO="830883877497.dkr.ecr.us-west-2.amazonaws.com/anyscale/${REPO_SERIES_NAME}-llm"
elif [[ "${BASE_TYPE}" == "ray-ml" ]]; then
    RAYTURBO_REPO="830883877497.dkr.ecr.us-west-2.amazonaws.com/anyscale/${REPO_SERIES_NAME}-ml"
else
    echo "Unknown base type: ${BASE_TYPE}" >/dev/stderr
    exit 1
fi

if [[ "$MACHTYPE" =~ ^x86_64 ]]; then
    HOSTTYPE="x86_64"
    ARCH_SUFFIX=""
elif [[ "$MACHTYPE" =~ ^aarch64 ]]; then
    HOSTTYPE="aarch64"
    ARCH_SUFFIX="-aarch64"
else
    echo "Unsupported architecture $MACHTYPE" >/dev/stderr
    exit 1
fi

if [[ "${PY_VERSION}" == "3.9" ]]; then
    PY_VERSION_CODE="py39"
    WHEEL_PYTHON_CODE="cp39-cp39"
elif [[ "${PY_VERSION}" == "3.10" ]]; then
    PY_VERSION_CODE="py310"
    WHEEL_PYTHON_CODE="cp310-cp310"
elif [[ "${PY_VERSION}" == "3.11" ]]; then
    PY_VERSION_CODE="py311"
    WHEEL_PYTHON_CODE="cp311-cp311"
elif [[ "${PY_VERSION}" == "3.12" ]]; then
    PY_VERSION_CODE="py312"
    WHEEL_PYTHON_CODE="cp312-cp312"
else
    echo "Unknown python version code: ${PY_VERSION}" >/dev/stderr
    exit 1
fi

WHEEL_FILE="ray-${RAY_VERSION}-${WHEEL_PYTHON_CODE}-manylinux2014_${HOSTTYPE}.whl"
CPP_WHEEL_FILE="ray_cpp-${RAY_VERSION}-py3-none-manylinux2014_${HOSTTYPE}.whl"

if [[ "${IS_SLIM}" == "1" ]]; then
    if [[ "${IMG_TYPE}" == "cpu" ]]; then
        IMG_TYPE_CODE=cpu
    elif [[ "${IMG_TYPE}" == "cu11.7.1" ]]; then
        IMG_TYPE_CODE="cu117"
    elif [[ "${IMG_TYPE}" == "cu11.8.0" ]]; then
        IMG_TYPE_CODE="cu118"
    elif [[ "${IMG_TYPE}" == "cu12.1.1" ]]; then
        IMG_TYPE_CODE="cu121"
    elif [[ "${IMG_TYPE}" == "cu12.3.2" ]]; then
        IMG_TYPE_CODE="cu123"
    elif [[ "${IMG_TYPE}" == "cu12.4.1" ]]; then
        IMG_TYPE_CODE="cu124"
    elif [[ "${IMG_TYPE}" == "cu12.5.1" ]]; then
        IMG_TYPE_CODE="cu125"
    elif [[ "${IMG_TYPE}" == "cu12.6.3" ]]; then
        IMG_TYPE_CODE="cu126"
    elif [[ "${IMG_TYPE}" == "cu12.8.1" ]]; then
        IMG_TYPE_CODE="cu128"
    elif [[ "${IMG_TYPE}" == "cu12.9.1" ]]; then
        IMG_TYPE_CODE="cu129"
    else
        echo "Unknown image type: ${IMG_TYPE}" >/dev/stderr
        exit 1
    fi
else
    if [[ "${IMG_TYPE}" == "cpu" ]]; then
        IMG_TYPE_CODE=cpu
    elif [[ "${IMG_TYPE}" == "cu11.7.1-cudnn8" ]]; then
        IMG_TYPE_CODE="cu117"
    elif [[ "${IMG_TYPE}" == "cu11.8.0-cudnn8" ]]; then
        IMG_TYPE_CODE="cu118"
    elif [[ "${IMG_TYPE}" == "cu12.1.1-cudnn8" ]]; then
        IMG_TYPE_CODE="cu121"
    elif [[ "${IMG_TYPE}" == "cu12.3.2-cudnn9" ]]; then
        IMG_TYPE_CODE="cu123"
    elif [[ "${IMG_TYPE}" == "cu12.4.1-cudnn" ]]; then
        IMG_TYPE_CODE="cu124"
    elif [[ "${IMG_TYPE}" == "cu12.5.1-cudnn" ]]; then
        IMG_TYPE_CODE="cu125"
    elif [[ "${IMG_TYPE}" == "cu12.6.3-cudnn" ]]; then
        IMG_TYPE_CODE="cu126"
    elif [[ "${IMG_TYPE}" == "cu12.8.1-cudnn" ]]; then
        IMG_TYPE_CODE="cu128"
    elif [[ "${IMG_TYPE}" == "cu12.9.1-cudnn" ]]; then
        IMG_TYPE_CODE="cu129"
    else
        echo "Unknown image type: ${IMG_TYPE}" >/dev/stderr
        exit 1
    fi
fi

function docker_push_as {
    local SRC_IMG="$1"
    local DEST_IMG="$2"
    docker tag "${SRC_IMG}" "${DEST_IMG}"
    docker push "${DEST_IMG}"
    if [[ "${BUILDKITE:-}" == "true" && "${IMG_ANNOTATE:-}" == "true" ]]; then
        buildkite-agent annotate --style=info \
            --context="${PY_VERSION_CODE}-images" --append "${DEST_IMG}<br/>"
    fi
}

function docker_push {
    local IMG="$1"
    docker push "${IMG}"
    if [[ "${BUILDKITE:-}" == "true" && "${IMG_ANNOTATE:-}" == "true" ]]; then
        buildkite-agent annotate --style=info \
            --context="${PY_VERSION_CODE}-images" --append "${IMG}<br/>"
    fi
}

if [[ "${PUSH_COMMIT_TAGS:-}" == "" ]]; then
    if [[ "${BUILDKITE_BRANCH:-}" =~ ^(master|releases/) ]]; then
        PUSH_COMMIT_TAGS="true"
    else
        PUSH_COMMIT_TAGS="false"
    fi
fi

BUILD_TMP="$(mktemp -d)"

mkdir -p "${BUILD_TMP}/oss-whl"
mkdir -p "${BUILD_TMP}/runtime-whl"

FULL_COMMIT="$(git rev-parse HEAD)"


####
echo "--- Fetch wheel and base image"
####

echo "OSS wheel: ${OSS_WHEEL_URL_PREFIX}${WHEEL_FILE}"
curl -sfL "${OSS_WHEEL_URL_PREFIX}${WHEEL_FILE}" -o "${BUILD_TMP}/oss-whl/${WHEEL_FILE}"
curl -sfL "${OSS_WHEEL_URL_PREFIX}${CPP_WHEEL_FILE}" -o "${BUILD_TMP}/oss-whl/${CPP_WHEEL_FILE}"

aws s3 cp "${S3_TEMP}/${WHEEL_FILE}" "${BUILD_TMP}/runtime-whl/${WHEEL_FILE}"

if [[ "${IS_SLIM}" == "1" ]]; then
    readonly BASE_IMG="${RAYCI_WORK_REPO}:${IMAGE_PREFIX}-${BASE_TYPE}-slim-py${PY_VERSION}-${IMG_TYPE}-${LAYER_TYPE}${ARCH_SUFFIX}"
else
    readonly BASE_IMG="${RAYCI_WORK_REPO}:${IMAGE_PREFIX}-${BASE_TYPE}-py${PY_VERSION}-${IMG_TYPE}-${LAYER_TYPE}${ARCH_SUFFIX}"
fi

aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin "${RUNTIME_ECR}"

docker pull "${BASE_IMG}"

if [[ "${BUILDKITE:-}" == "true" ]]; then
    rm -rf /artifact-mount/sitepkg
    mkdir -p /artifact-mount/sitepkg/ray-oss
    mkdir -p /artifact-mount/sitepkg/ray-opt
fi

# Everything is prepared, starts building now.

export DOCKER_BUILDKIT=1

if [[ "${IS_SLIM}" == "1" ]]; then
    BUILD_TAG="${IMAGE_PREFIX}-slim-${PY_VERSION_CODE}-${IMG_TYPE_CODE}${ARCH_SUFFIX}"
    # The site package tarball name cannot be changed from min to slim because it is
    # already a coded convention in Anyscale product, and we worry that changing it
    # might break some users. We will keep the min name for now.
    SITEPKG_TGZ="${BASE_TYPE}-${PY_VERSION_CODE}-${IMG_TYPE_CODE}-min${ARCH_SUFFIX}.tar.gz"
else
    BUILD_TAG="${IMAGE_PREFIX}-${PY_VERSION_CODE}-${IMG_TYPE_CODE}${ARCH_SUFFIX}"
    SITEPKG_TGZ="${BASE_TYPE}-${PY_VERSION_CODE}-${IMG_TYPE_CODE}${ARCH_SUFFIX}.tar.gz"
fi
RAY_IMG="${RAYTURBO_REPO}:${BUILD_TAG}"


####
echo "--- Step 1: Build OSS site package tarball"
####

CONTEXT_TMP="$(mktemp -d)"

mkdir -p "${CONTEXT_TMP}/.whl"
cp "${BUILD_TMP}/oss-whl/${WHEEL_FILE}" "${CONTEXT_TMP}/.whl/${WHEEL_FILE}"
cp "${BUILD_TMP}/oss-whl/${CPP_WHEEL_FILE}" "${CONTEXT_TMP}/.whl/${CPP_WHEEL_FILE}"
cp python/requirements_compiled.txt "${CONTEXT_TMP}/."
cp anyscale/docker/Dockerfile.sitepkg "${CONTEXT_TMP}/Dockerfile"

(
    cd "${CONTEXT_TMP}"
    tar --mtime="UTC 2020-01-01" --sort=name -c -f - . \
        | docker build --progress=plain \
            --build-arg FULL_BASE_IMAGE="${BASE_IMG}" \
            --build-arg RAY_VERSION="${RAY_VERSION}" \
            --build-arg WHEEL_PATH=".whl/${WHEEL_FILE}" \
            --build-arg RAY_MOD_DATE="2020-01-01" \
            --output="${BUILD_TMP}" --target=final -f Dockerfile -
)

mv "${BUILD_TMP}/ray.tgz" "${BUILD_TMP}/ray-oss.tgz"

aws s3 cp "${BUILD_TMP}/ray-oss.tgz" "${S3_TEMP}/ray-oss/${PY_VERSION_CODE}/${SITEPKG_TGZ}"
if [[ "${BUILDKITE:-}" == "true" ]]; then
    cp "${BUILD_TMP}/ray-oss.tgz" "/artifact-mount/sitepkg/ray-oss/${SITEPKG_TGZ}"
fi


####
echo "--- Step 2: Build Runtime site package tarball"
####

# Only need to overwrite the wheel
cp "${BUILD_TMP}/runtime-whl/${WHEEL_FILE}" "${CONTEXT_TMP}/.whl/${WHEEL_FILE}"
rm "${CONTEXT_TMP}/.whl/${CPP_WHEEL_FILE}" # And removes the ray-cpp wheel.

# Runtime uses a later date, this will force pyc file recompile after
# extraction.
(
    cd "${CONTEXT_TMP}"
    tar --mtime="UTC 2023-01-01" --sort=name -c -f - . \
        | docker build --progress=plain \
            --build-arg FULL_BASE_IMAGE="${BASE_IMG}" \
            --build-arg RAY_VERSION="${RAY_VERSION}" \
            --build-arg WHEEL_PATH=".whl/${WHEEL_FILE}" \
            --build-arg RAY_MOD_DATE="2023-01-01" \
            --output="${BUILD_TMP}" --target=final -f Dockerfile -
)

mv "${BUILD_TMP}/ray.tgz" "${BUILD_TMP}/ray-opt.tgz"

aws s3 cp "${BUILD_TMP}/ray-opt.tgz" "${S3_TEMP}/ray-opt/${PY_VERSION_CODE}/${SITEPKG_TGZ}"
if [[ "${BUILDKITE:-}" == "true" ]]; then
    cp "${BUILD_TMP}/ray-opt.tgz" "/artifact-mount/sitepkg/ray-opt/${SITEPKG_TGZ}"
fi

# Cleanup sitepkg build context.
rm -rf "${CONTEXT_TMP}"


####
echo "--- Step 3: Build ${RAY_IMG}"
####

CONTEXT_TMP="$(mktemp -d)"

mkdir -p "${CONTEXT_TMP}/.whl"

cp "${BUILD_TMP}/runtime-whl/${WHEEL_FILE}" "${CONTEXT_TMP}/.whl/${WHEEL_FILE}"
cp anyscale/docker/Dockerfile.ray "${CONTEXT_TMP}/Dockerfile"
cp anyscale/docker/runtime-requirements.txt "${CONTEXT_TMP}/runtime-requirements.txt"
cp python/requirements_compiled.txt "${CONTEXT_TMP}/."
cp anyscale/docker/NOTICE "${CONTEXT_TMP}/."
cp anyscale/docker/ray-prestart "${CONTEXT_TMP}/."
cp LICENSE.runtime "${CONTEXT_TMP}/LICENSE"
aws s3 cp "${S3_TEMP}/download_anyscale_data${ARCH_SUFFIX}" "${CONTEXT_TMP}/download_anyscale_data"
chmod +x "${CONTEXT_TMP}/download_anyscale_data"

# Build anyscaleruntime wheel from source.
pip wheel --no-deps -w "${CONTEXT_TMP}/.whl" ./anyscaleruntime
ANYSCALERUNTIME_WHEEL_FILE=$(basename "${CONTEXT_TMP}/.whl"/anyscaleruntime*.whl)

# If RAY_INLINE_SITE_PKG is not set, then inline the site package if it is a dev build.
if [[ "${RAY_INLINE_SITE_PKG:-}" == "" ]]; then
    if [[ "${RAY_RELEASE_BUILD:-}" != "true" ]]; then
        RAY_INLINE_SITE_PKG="true"
    else
        RAY_INLINE_SITE_PKG="false"
    fi
fi

# Must keep this consistent with anyscale/ci/upload-rayturbo-artifacts.sh
if [[ "${RAY_INLINE_SITE_PKG}" != "true" ]]; then
  if [[ "${IS_SLIM}" == "1" ]]; then
    ANYSCALE_PRESTART_DATA_PATH="common/ray-opt/${RAY_VERSION}/${FULL_COMMIT}/ray-opt-${PY_VERSION_CODE}-min${ARCH_SUFFIX}.tar.gz"
  else
    ANYSCALE_PRESTART_DATA_PATH="common/ray-opt/${RAY_VERSION}/${FULL_COMMIT}/ray-opt-${PY_VERSION_CODE}${ARCH_SUFFIX}.tar.gz"
  fi
else
    ANYSCALE_PRESTART_DATA_PATH=""  # stub an empty label
fi

# Generates a version stamp file.
{
    echo "#!/bin/bash"
    echo ": \${ANYSCALE_PY_VERSION_CODE:=${PY_VERSION_CODE}}"
    echo ": \${ANYSCALE_RAY_VERSION:=${RAY_VERSION}}"
    echo ": \${ANYSCALE_RAY_COMMIT:=${FULL_COMMIT}}"
    echo ": \${ANYSCALE_RAY_MINIMIZED:=${IS_SLIM}}"
    echo "export ANYSCALE_PY_VERSION_CODE ANYSCALE_RAY_VERSION ANYSCALE_RAY_COMMIT ANYSCALE_RAY_MINIMIZED"
} > "${CONTEXT_TMP}/version-envs.sh"

# We place in the oss site package.
cp "${BUILD_TMP}/ray-oss.tgz" "${CONTEXT_TMP}/ray-oss.tgz"

if [[ "${RAY_INLINE_SITE_PKG}" == "true" ]]; then
    # In dev builds, we copy in the runtime site package, so that we do not
    # need to upload a dev version of site package to org data S3.
    cp "${BUILD_TMP}/ray-opt.tgz" "${CONTEXT_TMP}/ray-opt.tgz"
fi

(
    cd "${CONTEXT_TMP}"
    tar --mtime="UTC 2023-10-01" --sort=name -c -f - . \
        | docker build --progress=plain \
            --build-arg FULL_BASE_IMAGE="${BASE_IMG}" \
            --build-arg WHEEL_PATH=".whl/${WHEEL_FILE}" \
            --build-arg ANYSCALERUNTIME_WHEEL_PATH=".whl/${ANYSCALERUNTIME_WHEEL_FILE}" \
            --build-arg RAY_VERSION="${RAY_VERSION}" \
            --label io.ray.ray-version="${RAY_VERSION}" \
            --label io.ray.ray-commit="${UPSTREAM_COMMIT}" \
            --label com.anyscale.ray.rayturbo-commit="${FULL_COMMIT}" \
            --label com.anyscale.ray.prestart-data-path="${ANYSCALE_PRESTART_DATA_PATH}" \
            -t "${RAY_IMG}" -f Dockerfile -
)

rm -rf "${CONTEXT_TMP}"

####
echo "--- Pushing images"
####

IMG_ANNOTATE=true docker_push "${RAY_IMG}"

if [[ "${IMG_TYPE_CODE}" == "${ML_CUDA_VERSION}" && "${IS_SLIM}" != "1" ]]; then
    BUILD_GPU_TAG="${IMAGE_PREFIX}-${PY_VERSION_CODE}-gpu${ARCH_SUFFIX}"
    IMG_ANNOTATE=true docker_push_as "${RAY_IMG}" "${RAYTURBO_REPO}:${BUILD_GPU_TAG}"
fi

if [[ "${PUSH_COMMIT_TAGS}" == "true" ]]; then
    SHORT_COMMIT="${FULL_COMMIT:0:6}"  # Use 6 chars to be consistent with Ray upstream
    # During branch cut, do not modify ray version in this script
    if [[ "${RAY_RELEASE_BUILD:-}" == "true" ]]; then
        SHORT_COMMIT="${RAY_VERSION}.${SHORT_COMMIT}"
    fi

    if [[ "${IS_SLIM}" == "1" ]]; then
        COMMIT_TAG="${SHORT_COMMIT}-slim-${PY_VERSION_CODE}-${IMG_TYPE_CODE}${ARCH_SUFFIX}"
    else
        COMMIT_TAG="${SHORT_COMMIT}-${PY_VERSION_CODE}-${IMG_TYPE_CODE}${ARCH_SUFFIX}"
    fi

    IMG_ANNOTATE=true docker_push_as "${RAY_IMG}" "${RAYTURBO_REPO}:${COMMIT_TAG}"

    if [[ "${IMG_TYPE_CODE}" == "${ML_CUDA_VERSION}" ]]; then
        COMMIT_GPU_TAG="${SHORT_COMMIT}-${PY_VERSION_CODE}-gpu${ARCH_SUFFIX}"
        IMG_ANNOTATE=true docker_push_as "${RAY_IMG}" "${RAYTURBO_REPO}:${COMMIT_GPU_TAG}"
    fi
fi
