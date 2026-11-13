# Base image. Keep its CUDA version aligned with the torch, xformers and bitsandbytes wheels
# in builder/requirements.txt (torch 2.15 bundles CUDA 13.2 libraries).
FROM nvidia/cuda:13.2.1-cudnn-runtime-ubuntu22.04@sha256:1f6d3723d8650a1f34dd2df34fd70a3195ada805c332a99f08e0d961e9e96561

# Use bash shell with pipefail option
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive
# Keep pip, python3, accelerate and child processes on the same interpreter.
ENV PATH="/opt/venv/bin:${PATH}"

# Set the working directory
WORKDIR /

# Install system packages, clone repo, and cache models
COPY builder/setup.sh /setup.sh
RUN bash /setup.sh

# Install Python dependencies (Worker Template)
COPY builder/requirements.txt /requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -c "import sys; assert sys.version_info[:2] == (3, 11)" && \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install --upgrade -r /requirements.txt --no-cache-dir && \
    python3 -m pip check && \
    rm /requirements.txt

COPY checks /checks
RUN PYTHONDONTWRITEBYTECODE=1 python3 /checks/runtime_compatibility.py --sd-scripts-root /sd-scripts

# Pre-cache SDXL CLIP tokenizers so training never needs HuggingFace Hub access
RUN python3 -c "\
from transformers import CLIPTokenizer; \
CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14').save_pretrained('/tokenizer_cache/openai_clip-vit-large-patch14'); \
CLIPTokenizer.from_pretrained('laion/CLIP-ViT-bigG-14-laion2B-39B-b160k').save_pretrained('/tokenizer_cache/laion_CLIP-ViT-bigG-14-laion2B-39B-b160k')"

ENV HF_HOME=/runpod-volume/hf-cache

# Add src files (Worker Template)
ADD src /sd-scripts

RUN PYTHONDONTWRITEBYTECODE=1 python3 /checks/worker_compatibility.py

WORKDIR /sd-scripts

CMD ["/sd-scripts/start.sh"]
