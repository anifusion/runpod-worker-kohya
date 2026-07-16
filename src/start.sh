#!/usr/bin/env bash

# Use libtcmalloc for better memory management
#TCMALLOC="$(ldconfig -p | grep -Po "libtcmalloc.so.\d" | head -n 1)"
#export LD_PRELOAD="${TCMALLOC}"

# Serve the API and don't shutdown the container
echo "runpod-worker-kohya: Starting RunPod Handler"

# Best-effort volume cache for tools that read HF_HOME; handler.py writes its own /tmp config per job.
ACCEL_DIR="${HF_HOME:-$HOME/.cache/huggingface}/accelerate"
ACCEL_CONFIG="$ACCEL_DIR/default_config.yaml"
if mkdir -p "$ACCEL_DIR" 2>/dev/null; then
  if cat > "$ACCEL_CONFIG" <<'EOF'
compute_environment: LOCAL_MACHINE
distributed_type: 'NO'
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF
  then
    if ! grep -q '^compute_environment:' "$ACCEL_CONFIG"; then
      echo "runpod-worker-kohya: WARN invalid accelerate config at $ACCEL_CONFIG (handler uses /tmp fallback)"
    fi
  else
    echo "runpod-worker-kohya: WARN could not write accelerate config to $ACCEL_CONFIG (handler uses /tmp fallback)"
  fi
else
  echo "runpod-worker-kohya: WARN could not create $ACCEL_DIR (handler uses /tmp fallback)"
fi

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
