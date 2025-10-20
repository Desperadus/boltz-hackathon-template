#!/usr/bin/env bash
set -euo pipefail

# ==============================
# P2Rank Installation Script
# ==============================
# This script installs P2Rank (v2.5) in /opt/p2rank,
# exactly as done in the Dockerfile.
# Requires: wget, tar, and sudo privileges for /opt installation.
# ==============================

P2RANK_VERSION="2.5"
P2RANK_URL="https://github.com/rdk/p2rank/releases/download/${P2RANK_VERSION}/p2rank_${P2RANK_VERSION}.tar.gz"
INSTALL_DIR="/opt/p2rank"

echo "🚀 Installing P2Rank v${P2RANK_VERSION}..."

# Ensure required tools are present
if ! command -v wget >/dev/null 2>&1; then
  echo "❌ wget not found. Installing wget..."
  sudo apt-get update && sudo apt-get install -y wget
fi

if ! command -v tar >/dev/null 2>&1; then
  echo "❌ tar not found. Installing tar..."
  sudo apt-get update && sudo apt-get install -y tar
fi

# Create install directory
echo "📂 Creating ${INSTALL_DIR}..."
sudo mkdir -p "${INSTALL_DIR}"

# Download and extract
echo "📦 Downloading P2Rank..."
wget -q "${P2RANK_URL}" -O /tmp/p2rank.tar.gz

echo "🧩 Extracting..."
sudo tar -xzf /tmp/p2rank.tar.gz -C "${INSTALL_DIR}" --strip-components=1

# Cleanup
rm -f /tmp/p2rank.tar.gz

# Add P2Rank to PATH (optional persistent setup)
if ! grep -q "/opt/p2rank" ~/.bashrc; then
  echo "🛠 Adding P2Rank to PATH in ~/.bashrc..."
  echo 'export PATH="/opt/p2rank:$PATH"' >>~/.bashrc
  echo "✅ PATH updated. Restart your shell or run: source ~/.bashrc"
fi

echo "✅ P2Rank v${P2RANK_VERSION} installed successfully at ${INSTALL_DIR}"
echo "👉 To test, run: ${INSTALL_DIR}/prank.sh"
