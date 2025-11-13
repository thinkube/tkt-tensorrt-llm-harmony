#!/bin/bash
set -e

echo "=== TensorRT-LLM Inference Server Startup ==="
echo "Model: ${MODEL_ID}"
echo "Engine directory: ${ENGINE_DIR}"

# Check if engine already exists
ENGINE_PATH="${ENGINE_DIR}/$(echo ${MODEL_ID} | tr '/' '_')"

if [ -d "${ENGINE_PATH}" ] && [ -f "${ENGINE_PATH}/config.json" ]; then
    echo "Engine found at ${ENGINE_PATH}, skipping conversion"
else
    echo "Engine not found, building from HuggingFace model..."
    echo "This will take 30+ minutes depending on model size"

    mkdir -p "${ENGINE_PATH}"

    # Convert HuggingFace checkpoint to TensorRT-LLM format
    python /app/tensorrt_llm/examples/llama/convert_checkpoint.py \
        --model_dir "${CACHE_DIR}/models--${MODEL_ID/\//_}" \
        --output_dir "${ENGINE_PATH}/checkpoint" \
        --dtype float16 \
        --tp_size 1

    # Build TensorRT engine
    trtllm-build \
        --checkpoint_dir "${ENGINE_PATH}/checkpoint" \
        --output_dir "${ENGINE_PATH}" \
        --gemm_plugin auto \
        --max_batch_size 8 \
        --max_input_len 2048 \
        --max_output_len 2048

    echo "Engine built successfully at ${ENGINE_PATH}"
fi

# Start TensorRT-LLM server in background
echo "Starting TensorRT-LLM OpenAI-compatible server on port 8000..."
trtllm-serve "${ENGINE_PATH}" --port 8000 --host 0.0.0.0 &

# Wait for server to be ready
echo "Waiting for TensorRT-LLM server to start..."
for i in {1..30}; do
    if curl -s http://localhost:8000/v1/models >/dev/null 2>&1; then
        echo "TensorRT-LLM server is ready"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "ERROR: TensorRT-LLM server failed to start"
        exit 1
    fi
    sleep 2
done

# Start Gradio UI
echo "Starting Gradio UI on port 7860..."
exec python3 server.py
