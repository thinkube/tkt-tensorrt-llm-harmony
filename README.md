# tkt-tensorrt-llm-harmony

The NVIDIA TensorRT-LLM inference server behind the LLM Gateway. It is the
`tensorrt` optional component of Thinkube.

## What it does

`server.py` is a FastAPI app on port 7860. It runs `trtllm-serve` (PyTorch
backend) as a subprocess on `127.0.0.1:8355` and passes requests to it.

- **Model weights come from MLflow.** The server finds the latest version of
  the model in the MLflow Model Registry and serves it from
  `/mlflow-models/artifacts/<experiment>/<run>/artifacts/model` on shared
  storage. Hugging Face runs offline (`HF_HUB_OFFLINE=1`). The models are
  NVIDIA's pre-quantized checkpoints (NVFP4, FP4, FP8, MXFP4), served as
  they are, with no conversion step.
- **It starts idle.** When `MODEL_ID` is set, it loads that model at start,
  with `STOP_TOKENS`, `REASONING_FORMAT`, `TOOL_USE` and
  `MAX_CONTEXT_LENGTH` from the environment.
- **It switches models in place.** `POST /admin/switch-model` stops the
  running `trtllm-serve`, starts it with the new model, and rolls back to
  the previous model if the new one does not start.
- **It sets `trtllm-serve` options per model**: xgrammar guided decoding,
  chunked prefill, a maximum sequence length (`max_context_length`, else
  `TRTLLM_DEFAULT_MAX_SEQ_LEN`, default 8192), and an FP8 KV cache with
  Mamba cache settings for `nemotron_h` models. Reasoning formats starting
  with `nano` add `--reasoning_parser nano-v3`.
- **It gives an OpenAI-compatible API.** `/v1/chat/completions` accepts
  tools, streaming, and a `batch` list of requests. `trtllm-serve` parses
  the gpt-oss harmony format and returns the answer, the reasoning and the
  tool calls as separate fields.
- **It has a chat page.** A Gradio chat UI at `/`.

On arm64 the base image is `tensorrt-llm-spark` (DGX Spark, Blackwell
sm_121a, NVFP4 and MXFP4). On amd64 it is the standard TensorRT-LLM
release. Weights load one after another
(`TRT_LLM_DISABLE_LOAD_WEIGHTS_IN_PARALLEL=1`) to avoid memory spikes on
unified memory.

### Endpoints

| Method | Path | What it does |
|---|---|---|
| GET | `/health` | `idle`, `starting`, `switching`, `healthy`, or HTTP 503 |
| GET | `/admin/current-model` | model id and path, status, uptime, stop tokens, reasoning format, tool use |
| POST | `/admin/switch-model` | body: `model_id` (required), `stop_tokens`, `reasoning_format`, `tool_use`, `max_context_length` |
| GET | `/admin/status` | status, model id, process id, `ready`, uptime, last error |
| POST | `/v1/chat/completions` | OpenAI chat completions; a `batch` list runs several at once |
| POST | `/v1/batch/completions` | a `requests` list, run concurrently |
| GET | `/v1/models` | the loaded model |
| GET | `/` | Gradio chat page |

## How it reaches a user

It is the `tensorrt` optional component of
[Thinkube](https://github.com/thinkube/thinkube). It is installed and
removed from the Optional Components page in thinkube-control. The
Templates page refuses it. The install deploys it with no pods
(`replicas: 0`, `gateway_managed: true` in `thinkube.yaml`); the LLM Gateway
creates a pod on a node when a model is loaded there. It is not installed on
its own.

## Supported Models

These are the models the model catalogue
([thinkube-metadata `models.json`](https://github.com/thinkube/thinkube-metadata/blob/main/models.json))
marks for TensorRT-LLM. A model that is not in the catalogue cannot be
mirrored or served.

### GPT-OSS (MXFP4)
- openai/gpt-oss-20b
- openai/gpt-oss-120b

### Llama and Nemotron Models (FP4/FP8/NVFP4)
- nvidia/Llama-3_3-Nemotron-Super-49B-v1_5-FP8
- nvidia/Llama-3_3-Nemotron-Super-49B-v1_5-NVFP4
- nvidia/Llama-4-Scout-17B-16E-Instruct-FP4
- nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4
- nvidia/NVIDIA-Nemotron-Nano-9B-v2-NVFP4

### Qwen Models (FP4/FP8)
- nvidia/Qwen3-8B-FP8, nvidia/Qwen3-8B-FP4
- nvidia/Qwen3-14B-FP8, nvidia/Qwen3-14B-FP4
- nvidia/Qwen3-32B-FP4
- nvidia/Qwen3-30B-A3B-FP4
- nvidia/Qwen2.5-VL-7B-Instruct-FP8, nvidia/Qwen2.5-VL-7B-Instruct-FP4
- nvidia/Qwen3-235B-A22B-FP4 (requires two DGX Sparks)

### Gemma Models (NVFP4)
- nvidia/Gemma-4-31B-IT-NVFP4

### Phi Models (FP4/FP8)
- nvidia/Phi-4-multimodal-instruct-FP8, nvidia/Phi-4-multimodal-instruct-FP4
- nvidia/Phi-4-reasoning-plus-FP8, nvidia/Phi-4-reasoning-plus-FP4

## Working on it

| File | What it is |
|---|---|
| `server.py` | the FastAPI app, the `trtllm-serve` subprocess, the Gradio page |
| `entrypoint.sh` | sets offline Hugging Face mode and serial weight loading, starts `server.py` |
| `thinkube_theme.py` | the Gradio theme |
| `Containerfile` | the image, on `tensorrt-llm-base`; adds the Nemotron reasoning parser |
| `requirements.txt` | Python packages; TensorRT-LLM comes from the base image |
| `thinkube.yaml` | the component deployment: one GPU, port 7860 |
| `manifest.yaml` | the template metadata |

## License

MIT. Code generated from this template is yours: no attribution required, and you may license the app you build however you choose. See [LICENSE](LICENSE).

Copyright Alejandro Martínez Corriá and the Thinkube contributors
