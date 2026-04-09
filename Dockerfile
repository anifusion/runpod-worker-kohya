# Base image
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Use bash shell with pipefail option
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory
WORKDIR /

# Install system packages, clone repo, and cache models
COPY builder/setup.sh /setup.sh
RUN bash /setup.sh

# Install Python dependencies (Worker Template)
COPY builder/requirements.txt /requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install --upgrade -r /requirements.txt --no-cache-dir && \
    pip install --force-reinstall -v "numpy==1.25.2" && \
    rm /requirements.txt

# Pre-cache SDXL CLIP tokenizers so training never needs HuggingFace Hub access
RUN python3 -c "\
from transformers import CLIPTokenizer; \
CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14').save_pretrained('/tokenizer_cache/openai_clip-vit-large-patch14'); \
CLIPTokenizer.from_pretrained('laion/CLIP-ViT-bigG-14-laion2B-39B-b160k').save_pretrained('/tokenizer_cache/laion_CLIP-ViT-bigG-14-laion2B-39B-b160k')"

ENV HF_HOME=/runpod-volume/hf-cache

# Add src files (Worker Template)
ADD src /sd-scripts

WORKDIR /sd-scripts

CMD ["/sd-scripts/start.sh"]
