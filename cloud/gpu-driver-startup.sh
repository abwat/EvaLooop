#!/bin/bash
# Official Google Compute Engine installer pattern for Linux GPU VMs.
set -eu

INSTALL_DIR=/opt/google/cuda-installer
MARKER="$INSTALL_DIR/cuda_installation"

if test -f "$MARKER"; then
  exit 0
fi

mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"
curl -fSsL -O https://storage.googleapis.com/compute-gpu-installation-us/installer/latest/cuda_installer.pyz
python3 cuda_installer.pyz install_driver
touch "$MARKER"
