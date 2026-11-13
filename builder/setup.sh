#!/bin/bash

set -euo pipefail

apt-get update
apt-get upgrade -y
apt-get install -y software-properties-common
add-apt-repository -y ppa:deadsnakes/ppa
apt-get update
apt-get install -y --no-install-recommends \
    wget ffmpeg libsm6 libxext6 git curl libgl1 libglib2.0-0 libgoogle-perftools-dev \
    python3.11-dev python3.11-venv python3-html5lib python3-apt
apt-get autoremove -y
apt-get clean -y
rm -rf /var/lib/apt/lists/*

python3.11 -m venv /opt/venv

# Pinned sd-scripts fork (must match builder/requirements.txt commit).
git clone https://github.com/anifusion/sd-scripts.git
cd sd-scripts
git checkout 6afdac3dc5ddc37f684af8a5c1c79d39995e61bb

# Cache models
#wget https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors -P /model_cache
#wget https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned.safetensors -P /model_cache
