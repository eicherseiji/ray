#!/bin/bash

set -euo pipefail

source anyscale/ci/setup-env.sh

PY_VERSION_CODES=(py39 py310 py311 py312)
SUM_FILES=(sums-min.txt sums-aarch64.txt sums-default.txt)

tmp_dir="$(mktemp -d)"

# Download and check ray-oss site package files.
for ray_type in ray-oss ray-opt ; do
    for py_version_code in "${PY_VERSION_CODES[@]}"; do
        mkdir -p "${tmp_dir}/${ray_type}/${py_version_code}"
        (
            cd "${tmp_dir}/${ray_type}/${py_version_code}"

            aws s3 sync "${S3_TEMP}/${ray_type}/${py_version_code}" .

            for sums_file in "${SUM_FILES[@]}"; do
                # Create empty sum files.
                : > "${sums_file}"
            done

            for file in *.tar.gz; do
                if [[ "${file}" == *-min.tar.gz ]]; then
                    sha256sum "${file}" | tee -a sums-min.txt
                elif [[ "${file}" == *-aarch64.tar.gz ]]; then
                    sha256sum "${file}" | tee -a sums-aarch64.txt
                else
                    sha256sum "${file}" | tee -a sums-default.txt
                fi
            done

            for sums_file in "${SUM_FILES[@]}"; do
                count="$(awk '{print $1}' "${sums_file}" | sort | uniq | wc -l)"
                if [[ "${count}" != "1" ]]; then
                    echo "Digest mismatch for ${ray_type} ${py_version_code} in file: ${sums_file})" >/dev/stderr
                    exit 1
                fi
            done
        )
        echo "Successfully checked ${ray_type} ${py_version_code}"
        rm -rf "${tmp_dir:?}/${ray_type:?}/${py_version_code:?}"
    done
done

# Cleanup temp dir.
rm -rf "${tmp_dir}"
