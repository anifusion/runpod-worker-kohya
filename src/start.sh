#!/usr/bin/env bash

# Use libtcmalloc for better memory management
#TCMALLOC="$(ldconfig -p | grep -Po "libtcmalloc.so.\d" | head -n 1)"
#export LD_PRELOAD="${TCMALLOC}"

# Serve the API and don't shutdown the container
echo "runpod-worker-kohya: Starting RunPod Handler"

echo "runpod-worker-kohya: Verifying tokenizer cache..."
python3 -c "
from transformers import CLIPTokenizer
CLIPTokenizer.from_pretrained('/tokenizer_cache/openai_clip-vit-large-patch14', local_files_only=True)
CLIPTokenizer.from_pretrained('/tokenizer_cache/laion_CLIP-ViT-bigG-14-laion2B-39B-b160k', local_files_only=True)
print('Tokenizer cache OK')
" || { echo 'FATAL: tokenizer cache missing or corrupt'; exit 1; }

if [ "$SERVE_API_LOCALLY" == "true" ]; then
    python3 -u ./handler.py --rp_serve_api --rp_api_host=0.0.0.0
else
    python3 -u ./handler.py
fi
