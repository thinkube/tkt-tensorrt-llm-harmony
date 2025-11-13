# TensorRT-LLM Inference Server Template

# ⚠️ Under Development - Not Ready for Use

NVIDIA TensorRT-LLM optimized inference template with Gradio UI and NVFP4 support for Blackwell GPUs (DGX Spark GB10).

## Features

- **Pre-Quantized Models**: Uses NVIDIA's ready-to-serve models with NVFP4/FP8/MXFP4 quantization
- **TensorRT-LLM Engine**: NVIDIA's optimized inference engine for maximum performance
- **NVFP4 Support**: Native 4-bit floating point quantization for Blackwell architecture
- **OpenAI-Compatible API**: Standard API on port 8355 for easy integration
- **Gradio UI**: Interactive web interface with Thinkube theming on port 7860
- **No Conversion Needed**: Direct serving of pre-quantized models from HuggingFace

## Supported Models

### GPT-OSS (MXFP4)
- openai/gpt-oss-20b
- openai/gpt-oss-120b

### Llama Models (FP4/FP8)
- nvidia/Llama-3.1-8B-Instruct-FP8
- nvidia/Llama-3.1-8B-Instruct-FP4
- nvidia/Llama-3.3-70B-Instruct-FP4
- nvidia/Llama-3_3-Nemotron-Super-49B-v1_5-FP8
- nvidia/Llama-4-Scout-17B-16E-Instruct-FP4

### Qwen Models (FP4/FP8)
- nvidia/Qwen3-8B-FP8, nvidia/Qwen3-8B-FP4
- nvidia/Qwen3-14B-FP8, nvidia/Qwen3-14B-FP4
- nvidia/Qwen3-32B-FP4
- nvidia/Qwen3-30B-A3B-FP4
- nvidia/Qwen2.5-VL-7B-Instruct-FP8, nvidia/Qwen2.5-VL-7B-Instruct-FP4
- nvidia/Qwen3-235B-A22B-FP4 (requires two DGX Sparks)

### Phi Models (FP4/FP8)
- nvidia/Phi-4-multimodal-instruct-FP8, nvidia/Phi-4-multimodal-instruct-FP4
- nvidia/Phi-4-reasoning-plus-FP8, nvidia/Phi-4-reasoning-plus-FP4

## Template Parameters

- `model_id`: Select from dropdown of pre-quantized models

## Secrets

- `HF_TOKEN`: HuggingFace API token for accessing gated models

## License

Apache License 2.0 - See [LICENSE](LICENSE)

## Copyright

Copyright 2025 Alejandro Martínez Corriá
