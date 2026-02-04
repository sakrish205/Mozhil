#!/usr/bin/env bash
# exit on error
set -o errexit
set -x

echo "Starting build process..."

# Install python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install ffmpeg for pydub
echo "Installing FFmpeg..."
mkdir -p bin
curl -L https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz | tar -xJ -C bin --strip-components 1

# Ensure bin is in PATH for the rest of the build if needed
export PATH=$PATH:$(pwd)/bin

echo "Build process complete!"
