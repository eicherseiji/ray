#!/bin/bash

set -euo pipefail

echo "Installing anyscaleruntime with --no-deps..."
pip install ./anyscaleruntime --no-deps

echo "Verifying anyscaleruntime can be imported..."
python -c "import anyscaleruntime" || {
    echo "Failed to import anyscaleruntime" >/dev/stderr
    exit 1
}

echo "Checking version is set..."
pip show anyscaleruntime | grep "^Version:" || {
    echo "Failed to get anyscaleruntime version" >/dev/stderr
    exit 1
}

echo "Successfully checked anyscaleruntime package installation"
