#!/bin/bash
#
# This script installs the bray command-line tool for Anyscale.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Installing bray..."
sudo mkdir -p /usr/local/bin/anyscale
sudo cp -rf "${SCRIPT_DIR}"/* /usr/local/bin/anyscale
sudo mv /usr/local/bin/anyscale/bray /usr/local/bin/bray

# Clean up the installation script
sudo rm /usr/local/bin/anyscale/install_bray.sh
sudo rm /usr/local/bin/anyscale/uninstall_bray.sh
echo "Bray installed successfully. You can now use the 'bray' command."
