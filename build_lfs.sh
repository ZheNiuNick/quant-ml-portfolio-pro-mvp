#!/bin/bash
set -e

# Install Git LFS and pull LFS files before build
echo "Installing Git LFS..."
git lfs install

echo "Pulling Git LFS files..."
git lfs pull

echo "Git LFS files pulled successfully!"

