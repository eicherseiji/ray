#!/bin/bash

set -euo pipefail

echo "Uninstalling bray..."
sudo rm -f /usr/local/bin/bray
sudo rm -rf /usr/local/bin/anyscale
echo "Bray uninstalled successfully."
