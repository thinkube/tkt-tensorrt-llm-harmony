#!/bin/bash
set -e

MODEL_ID="${MODEL_ID:?MODEL_ID environment variable is required}"

echo "=== TensorRT-LLM Inference Server Startup ==="
echo "Model: ${MODEL_ID}"

# Fix LD_LIBRARY_PATH to include NVIDIA cuda_nvrtc libraries
# Required for TensorRT-LLM bindings to find libnvrtc.so.12
export LD_LIBRARY_PATH="/usr/local/lib/python3.12/dist-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH}"
echo "LD_LIBRARY_PATH updated to include cuda_nvrtc"

# Configure HuggingFace to use local models only (no network access)
# Models are stored in MLflow artifacts on JuiceFS at /mlflow-models/artifacts/{run_id}/artifacts/model
export HF_HUB_OFFLINE=1
export HF_HOME="/mlflow-models"

# Prevent memory spikes during model loading on unified memory (DGX Spark)
# Serializes weight loading instead of parallel loading
export TRT_LLM_DISABLE_LOAD_WEIGHTS_IN_PARALLEL=1

echo "HuggingFace offline mode enabled (HF_HUB_OFFLINE=1)"
echo "Models will be loaded from MLflow artifacts on JuiceFS"
echo "Serialized weight loading enabled (TRT_LLM_DISABLE_LOAD_WEIGHTS_IN_PARALLEL=1)"

# Create extra options file for guided decoding
cat > /tmp/extra_llm_api_options.yaml << 'EOF'
guided_decoding_backend: xgrammar
EOF

# Start server (handles MLflow query, trtllm-serve subprocess, and admin endpoints)
exec python3 server.py
