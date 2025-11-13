# TensorRT-LLM Inference Server Template

# ⚠️ Under Development - Not Ready for Use

NVIDIA TensorRT-LLM optimized inference template with Gradio UI and NVFP4 support for Blackwell GPUs (DGX Spark GB10).

## Features

- **TensorRT-LLM Engine**: NVIDIA's optimized inference engine with automatic model conversion
- **NVFP4 Support**: Native 4-bit floating point quantization for Blackwell architecture (to be tested)
- **OpenAI-Compatible API**: Standard API on port 8000 for easy integration
- **Gradio UI**: Interactive web interface with Thinkube theming on port 7860
- **Automatic Engine Building**: Converts HuggingFace models to optimized TensorRT engines on first run

## Supported Models (Expected)

- Mistral, Mixtral MOE, GPT-OSS, DeepSeek V3
- Most HuggingFace causal language models

## Template Parameters

- `model_id`: HuggingFace model ID (e.g., `mistralai/Mistral-7B-Instruct-v0.2`)

## Secrets

- `HF_TOKEN`: HuggingFace API token for accessing gated models

## License

Apache License 2.0 - See [LICENSE](LICENSE)

## Copyright

Copyright 2025 Alejandro Martínez Corriá
