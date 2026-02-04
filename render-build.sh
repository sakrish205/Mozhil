#!/usr/bin/env bash
# exit on error
set -o errexit

# Install python dependencies
pip install -r requirements.txt

# Install ffmpeg for pydub
# Render uses a Debian-based environment, so we can use apt-get
# Note: In some Render environments, you might need to use a different approach
# but adding it to the build command or using a custom runtime is common.
# However, many Render environments actually have ffmpeg pre-installed
# or allow it through build scripts.

# If apt-get is permitted in the build environment:
# apt-get update && apt-get install -y ffmpeg

# Alternatively, using a static binary if apt-get is restricted:
mkdir -p bin
curl -L https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz | tar -xJ -C bin --strip-components 1
export PATH=$PATH:$(pwd)/bin
