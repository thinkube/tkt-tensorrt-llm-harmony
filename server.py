#!/usr/bin/env python3
"""
TensorRT-LLM Inference Server with Gradio UI and Admin Management Endpoints

Manages the trtllm-serve subprocess lifecycle, proxies inference requests,
and exposes /admin/* endpoints for model switching by thinkube-control.
"""

import os
import json
import time
import logging
import asyncio
import subprocess
import threading
import atexit

import httpx
import requests
import urllib3
import gradio as gr
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn
from transformers import AutoTokenizer
from thinkube_theme import create_thinkube_theme, THINKUBE_CSS

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

APP_NAME = os.environ.get("APP_NAME", "tensorrt-llm")
APP_TITLE = os.environ.get("APP_TITLE", APP_NAME)

TRTLLM_BACKEND_URL = "http://127.0.0.1:8355"

app = FastAPI(title=f"{APP_NAME} TensorRT-LLM Server")

tokenizer = None
http_client = None
sync_http_client = None


# ============================================================================
# MLflow Model Resolution
# ============================================================================

def query_mlflow_model_path(model_id: str) -> str:
    """Query MLflow to resolve model_id to a local artifact path on JuiceFS."""
    token_url = os.environ['MLFLOW_KEYCLOAK_TOKEN_URL']
    token_response = requests.post(
        token_url,
        data={
            'grant_type': 'password',
            'client_id': os.environ['MLFLOW_KEYCLOAK_CLIENT_ID'],
            'client_secret': os.environ['MLFLOW_CLIENT_SECRET'],
            'username': os.environ['MLFLOW_AUTH_USERNAME'],
            'password': os.environ['MLFLOW_AUTH_PASSWORD'],
            'scope': 'openid'
        },
        verify=False,
        timeout=30
    )
    token_response.raise_for_status()
    access_token = token_response.json()['access_token']

    model_name = model_id.replace('/', '-')
    mlflow_url = os.environ.get('MLFLOW_TRACKING_URI', 'http://mlflow.mlflow.svc.cluster.local:5000')

    response = requests.get(
        f"{mlflow_url}/api/2.0/mlflow/model-versions/search",
        params={'filter': f"name='{model_name}'"},
        headers={'Authorization': f'Bearer {access_token}'},
        verify=False,
        timeout=30
    )
    response.raise_for_status()

    versions = response.json().get('model_versions', [])
    if not versions:
        raise ValueError(f"Model {model_name} not found in MLflow registry")

    latest = max(versions, key=lambda v: int(v['version']))
    run_id = latest['run_id']
    logger.info(f"Found model version {latest['version']} with run_id: {run_id}")

    run_response = requests.get(
        f"{mlflow_url}/api/2.0/mlflow/runs/get",
        params={'run_id': run_id},
        headers={'Authorization': f'Bearer {access_token}'},
        verify=False,
        timeout=30
    )
    run_response.raise_for_status()
    experiment_id = run_response.json()['run']['info']['experiment_id']
    logger.info(f"Experiment ID: {experiment_id}")

    model_path = f'/mlflow-models/artifacts/{experiment_id}/{run_id}/artifacts/model'

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    logger.info(f"Resolved {model_id} -> {model_path}")
    return model_path


# ============================================================================
# trtllm-serve Subprocess Management
# ============================================================================

class TrtllmBackend:
    """Manages the trtllm-serve subprocess lifecycle."""

    def __init__(self):
        self.process: subprocess.Popen | None = None
        self.model_id: str | None = None
        self.model_path: str | None = None
        self.status: str = "stopped"
        self.start_time: float | None = None
        self.error: str | None = None
        self._switch_lock = threading.Lock()
        self.stop_tokens: list[str] = []
        self.reasoning_format: str | None = None
        self.tool_use: bool = False

    @property
    def uptime_seconds(self) -> int:
        if self.start_time and self.status == "serving":
            return int(time.time() - self.start_time)
        return 0

    def start(self, model_path: str, model_id: str, wait_timeout: int = 600,
              max_context_length: int | None = None) -> bool:
        """Start trtllm-serve with the given model. Blocks until ready or timeout."""
        self.model_path = model_path
        self.model_id = model_id
        self.status = "starting"
        self.error = None

        self._write_extra_options(max_context_length)
        logger.info(f"Starting trtllm-serve for {model_id} from {model_path}"
                     + (f" (max_seq_len={max_context_length})" if max_context_length else ""))

        self.process = subprocess.Popen(
            [
                "trtllm-serve", model_path,
                "--backend", "pytorch",
                "--extra_llm_api_options", "/tmp/extra_llm_api_options.yaml",
                "--host", "127.0.0.1",
                "--port", "8355",
                "--log_level", "info",
            ]
        )

        logger.info(f"trtllm-serve started with PID {self.process.pid}")

        if self._wait_for_ready(wait_timeout):
            self.status = "serving"
            self.start_time = time.time()
            logger.info(f"trtllm-serve is ready, serving {model_id}")
            return True
        else:
            self.status = "error"
            self.error = "trtllm-serve failed to start within timeout"
            logger.error(self.error)
            self._kill_process()
            return False

    def stop(self):
        """Stop the current trtllm-serve process."""
        self._kill_process()
        self.status = "stopped"

    def _kill_process(self):
        if self.process and self.process.poll() is None:
            logger.info(f"Stopping trtllm-serve (PID {self.process.pid})")
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                logger.warning("trtllm-serve did not stop gracefully, killing")
                self.process.kill()
                self.process.wait()
        self.process = None

    def _wait_for_ready(self, timeout: int = 600) -> bool:
        """Poll trtllm-serve health endpoint until ready."""
        deadline = time.time() + timeout
        check_count = 0
        while time.time() < deadline:
            try:
                r = httpx.get(f"{TRTLLM_BACKEND_URL}/health", timeout=5)
                if r.status_code == 200:
                    return True
            except Exception:
                pass
            if self.process is None or self.process.poll() is not None:
                rc = self.process.returncode if self.process else "N/A"
                logger.error(f"trtllm-serve exited with code {rc}")
                return False
            check_count += 1
            if check_count % 10 == 0:
                logger.info(f"Waiting for trtllm-serve... ({check_count * 5}s elapsed)")
            time.sleep(5)
        return False

    def switch_model(self, new_model_id: str, metadata: dict | None = None,
                     max_context_length: int | None = None) -> dict:
        """Switch to a different model. Blocks until complete. Thread-safe."""
        with self._switch_lock:
            return self._do_switch(new_model_id, metadata, max_context_length)

    def _do_switch(self, new_model_id: str, metadata: dict | None = None,
                   max_context_length: int | None = None) -> dict:
        previous_model = self.model_id
        previous_path = self.model_path
        previous_stop_tokens = self.stop_tokens
        previous_reasoning_format = self.reasoning_format
        previous_tool_use = self.tool_use
        switch_start = time.time()

        if new_model_id == self.model_id and self.status == "serving":
            if metadata:
                self._apply_metadata(metadata)
            return {
                "previous_model": previous_model,
                "current_model": self.model_id,
                "status": "serving",
                "switch_time_seconds": 0
            }

        # Resolve new model path via MLflow
        try:
            new_model_path = query_mlflow_model_path(new_model_id)
        except Exception as e:
            return {
                "previous_model": previous_model,
                "current_model": previous_model,
                "status": self.status,
                "error": f"Failed to resolve {new_model_id}: {e}"
            }

        self.status = "switching"

        # Stop current backend
        self._kill_process()

        # Start new backend
        if self.start(new_model_path, new_model_id, max_context_length=max_context_length):
            if metadata:
                self._apply_metadata(metadata)

            global tokenizer
            try:
                tokenizer = AutoTokenizer.from_pretrained(new_model_path)
                logger.info(f"Tokenizer reloaded for {new_model_id}")
            except Exception as e:
                logger.warning(f"Failed to reload tokenizer: {e}")

            return {
                "previous_model": previous_model,
                "current_model": new_model_id,
                "status": "serving",
                "switch_time_seconds": round(time.time() - switch_start, 1)
            }

        # Rollback to previous model and metadata
        logger.warning(f"Failed to start {new_model_id}, rolling back to {previous_model}")
        self.stop_tokens = previous_stop_tokens
        self.reasoning_format = previous_reasoning_format
        self.tool_use = previous_tool_use
        if previous_path and self.start(previous_path, previous_model):
            return {
                "previous_model": previous_model,
                "current_model": previous_model,
                "status": "serving",
                "error": f"Failed to load {new_model_id}: backend failed to start. Rolled back to {previous_model}"
            }

        return {
            "previous_model": previous_model,
            "current_model": previous_model,
            "status": "error",
            "error": f"Failed to load {new_model_id} and failed to rollback to {previous_model}"
        }

    def _write_extra_options(self, max_context_length: int | None = None):
        """Write extra_llm_api_options.yaml with optional max_seq_len."""
        lines = ["guided_decoding_backend: xgrammar"]
        if max_context_length:
            lines.append(f"max_seq_len: {max_context_length}")
        with open("/tmp/extra_llm_api_options.yaml", "w") as f:
            f.write("\n".join(lines) + "\n")

    def _apply_metadata(self, metadata: dict):
        if "stop_tokens" in metadata:
            self.stop_tokens = metadata["stop_tokens"]
            logger.info(f"Stop tokens updated: {self.stop_tokens}")
        if "reasoning_format" in metadata:
            self.reasoning_format = metadata["reasoning_format"]
            logger.info(f"Reasoning format updated: {self.reasoning_format}")
        if "tool_use" in metadata:
            self.tool_use = metadata["tool_use"]
            logger.info(f"Tool use updated: {self.tool_use}")


backend = TrtllmBackend()
atexit.register(backend.stop)


# ============================================================================
# Initialization
# ============================================================================

def initialize():
    """Initialize HTTP clients (and tokenizer if a model is loaded)."""
    global tokenizer, http_client, sync_http_client

    http_client = httpx.AsyncClient(
        base_url=TRTLLM_BACKEND_URL,
        timeout=httpx.Timeout(300.0, connect=10.0),
    )

    sync_http_client = httpx.Client(
        base_url=TRTLLM_BACKEND_URL,
        timeout=httpx.Timeout(300.0, connect=10.0),
    )

    if backend.model_path:
        print(f"Loading tokenizer for {backend.model_id}...")
        tokenizer = AutoTokenizer.from_pretrained(backend.model_path)
        print(f"Tokenizer loaded, HTTP clients initialized")
    else:
        print("Starting in idle mode — no model loaded")


# ============================================================================
# Gradio UI
# ============================================================================

def generate_response(message: str, history: list, temperature: float = 0.7, max_tokens: int = 512):
    """Generate response via trtllm-serve backend with harmony chat template"""
    global sync_http_client

    if backend.status != "serving":
        yield "No model loaded. Use the thinkube control panel to load a model."
        return

    logger.info(f"=== Gradio generate_response ===")
    logger.info(f"message type: {type(message)}, value: {message!r}")
    logger.info(f"history type: {type(history)}, length: {len(history) if history else 0}")
    if history:
        logger.info(f"history[0] type: {type(history[0]) if history else 'N/A'}")
        logger.info(f"history sample: {history[:2]}")

    # Build messages list from history
    # Gradio 6.x sends: {"role": "user", "content": [{"type": "text", "text": "..."}], "metadata": None, "options": None}
    # trtllm-serve expects: {"role": "user", "content": "..."}
    messages = []
    if history:
        for item in history:
            if isinstance(item, dict):
                role = item.get("role", "user")
                content = item.get("content", "")
                # Normalize content: Gradio 6.x uses [{"type": "text", "text": "..."}]
                if isinstance(content, list):
                    text_parts = []
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                        elif isinstance(block, str):
                            text_parts.append(block)
                    content = "".join(text_parts)
                if content:
                    messages.append({"role": role, "content": content})
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                user_msg, assistant_msg = item
                if user_msg:
                    messages.append({"role": "user", "content": str(user_msg)})
                if assistant_msg:
                    messages.append({"role": "assistant", "content": str(assistant_msg)})
            else:
                logger.warning(f"Unknown history item format: {type(item)}, {item!r}")

    messages.append({"role": "user", "content": str(message)})

    logger.info(f"Final messages to send: {messages}")

    response = sync_http_client.post(
        "/v1/chat/completions",
        json={
            "model": backend.model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": 0.9,
            **({"stop": backend.stop_tokens} if backend.stop_tokens else {}),
        }
    )
    if response.status_code != 200:
        logger.error(f"Backend error {response.status_code}: {response.text[:2000]}")
    response.raise_for_status()
    result = response.json()

    msg = result["choices"][0]["message"]
    content = msg.get("content") or ""
    yield content


thinkube_theme = create_thinkube_theme()

demo = gr.ChatInterface(
    generate_response,
    title=APP_TITLE,
    description="Chat with the loaded model (powered by TensorRT-LLM with NVFP4)",
    examples=[
        ["Hello! How are you?", 0.7, 512],
        ["Can you explain quantum computing in simple terms?", 0.7, 512],
        ["Write a Python function to calculate fibonacci numbers", 0.7, 512],
    ],
    analytics_enabled=False,
    additional_inputs=[
        gr.Slider(0.1, 2.0, value=0.7, label="Temperature"),
        gr.Slider(64, 2048, value=512, label="Max Tokens"),
    ],
)


# ============================================================================
# Health Check
# ============================================================================

@app.get("/health")
async def health_check():
    if backend.status == "stopped" and backend.model_id is None:
        return {"status": "idle", "model": None, "engine": "trtllm-serve"}
    if backend.status not in ("serving",):
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "backend_status": backend.status,
                "model": backend.model_id,
                "error": backend.error,
            }
        )
    try:
        response = await http_client.get("/health")
        if response.status_code == 200:
            return {
                "status": "healthy",
                "model": backend.model_id,
                "model_path": backend.model_path,
                "engine": "trtllm-serve (MXFP4)",
                "backend": {"status": "healthy"}
            }
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "model": backend.model_id,
                "engine": "trtllm-serve (MXFP4)",
                "backend": {"status": "unhealthy"}
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )


# ============================================================================
# Admin Management Endpoints (cluster-internal only, not exposed via HTTPRoute)
# ============================================================================

_admin_switch_lock = asyncio.Lock()


@app.get("/admin/current-model")
async def admin_current_model():
    return JSONResponse({
        "model_id": backend.model_id,
        "model_path": backend.model_path,
        "status": backend.status,
        "engine": "trtllm-serve",
        "uptime_seconds": backend.uptime_seconds,
        "stop_tokens": backend.stop_tokens,
        "reasoning_format": backend.reasoning_format,
        "tool_use": backend.tool_use,
    })


@app.post("/admin/switch-model")
async def admin_switch_model(request: Request):
    if _admin_switch_lock.locked():
        return JSONResponse(
            status_code=409,
            content={"error": "Model switch already in progress"}
        )

    body = await request.json()
    new_model_id = body.get("model_id")
    if not new_model_id:
        return JSONResponse(
            status_code=400,
            content={"error": "model_id is required"}
        )

    if backend.status == "starting":
        return JSONResponse(
            status_code=409,
            content={"error": "Backend is still starting, cannot switch models yet"}
        )

    metadata = {}
    if "stop_tokens" in body:
        metadata["stop_tokens"] = body["stop_tokens"]
    if "reasoning_format" in body:
        metadata["reasoning_format"] = body["reasoning_format"]
    if "tool_use" in body:
        metadata["tool_use"] = body["tool_use"]

    max_context_length = body.get("max_context_length")

    async with _admin_switch_lock:
        result = await asyncio.to_thread(
            backend.switch_model, new_model_id, metadata or None, max_context_length
        )

    if result.get("status") == "error":
        return JSONResponse(status_code=500, content=result)
    return JSONResponse(content=result)


@app.get("/admin/status")
async def admin_status():
    return JSONResponse({
        "status": backend.status,
        "model_id": backend.model_id,
        "pid": backend.process.pid if backend.process and backend.process.poll() is None else None,
        "ready": backend.status == "serving",
        "uptime_seconds": backend.uptime_seconds,
        "error": backend.error,
    })


# ============================================================================
# OpenAI-Compatible API Endpoints (for LiteLLM integration)
# ============================================================================
# Note: trtllm-serve handles harmony format parsing internally and returns:
# - message.content: final user-facing response
# - message.reasoning: chain-of-thought (analysis channel)
# - message.tool_calls: list of tool calls [{id, type, function: {name, arguments}}]


async def _handle_batch_completions(body: dict, batch_requests: list):
    """
    Handle batch completions via trtllm-serve backend.

    Processes requests sequentially through the backend (backend handles batching internally).

    Request format (via /v1/chat/completions):
    {
        "batch": [
            {"messages": [...]},
            {"messages": [...]}
        ],
        "max_tokens": 512,      # Shared defaults (can be overridden per request)
        "temperature": 0.7,
        "top_p": 0.9
    }

    Response format:
    {
        "object": "batch.completion",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "..."}, "finish_reason": "stop"},
            {"index": 1, "message": {"role": "assistant", "content": "..."}, "finish_reason": "stop"}
        ],
        "usage": {...},
        "batch_info": {"count": N, "processing_time_ms": X}
    }
    """

    start_time = time.time()

    default_temperature = body.get('temperature', 0.7)
    default_max_tokens = body.get('max_tokens', 512)
    default_top_p = body.get('top_p', 0.9)
    include_reasoning = body.get('include_reasoning', False)

    total_requests = len(batch_requests)
    logger.info(f"=== Batch Request via /v1/chat/completions === total={total_requests}")

    if total_requests == 0:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "Empty batch", "type": "invalid_request"}}
        )

    async def process_single_request(idx: int, req: dict):
        messages = req.get('messages', [])
        temperature = req.get('temperature', default_temperature)
        max_tokens = req.get('max_tokens', default_max_tokens)
        top_p = req.get('top_p', default_top_p)

        response = await http_client.post(
            "/v1/chat/completions",
            json={
                "model": backend.model_id,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                **({"stop": backend.stop_tokens} if backend.stop_tokens else {}),
            }
        )
        response.raise_for_status()
        result = response.json()

        msg = result["choices"][0]["message"]
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning")
        finish_reason = result["choices"][0].get("finish_reason", "stop")

        choice = {
            "index": idx,
            "message": {
                "role": "assistant",
                "content": content
            },
            "finish_reason": finish_reason
        }

        if include_reasoning and reasoning:
            choice["reasoning"] = reasoning

        return choice

    tasks = [process_single_request(i, req) for i, req in enumerate(batch_requests)]
    all_choices = await asyncio.gather(*tasks)

    all_choices = sorted(all_choices, key=lambda x: x["index"])

    processing_time_ms = int((time.time() - start_time) * 1000)
    logger.info(f"Batch completed: {total_requests} prompts, {processing_time_ms}ms ({processing_time_ms/total_requests:.0f}ms/prompt)")

    return JSONResponse({
        "id": f"batch-{int(time.time() * 1000)}",
        "object": "batch.completion",
        "created": int(time.time()),
        "model": backend.model_id,
        "choices": all_choices,
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        },
        "batch_info": {
            "count": total_requests,
            "processing_time_ms": processing_time_ms,
            "avg_time_per_prompt_ms": processing_time_ms / total_requests
        }
    })


@app.post("/v1/chat/completions")
async def openai_chat_completions(request: Request):
    """OpenAI-compatible chat completions endpoint via trtllm-serve

    Supports both single and batch requests:
    - Single: {"messages": [...], "max_tokens": 512, ...}
    - Batch:  {"batch": [{"messages": [...]}, {"messages": [...]}], "max_tokens": 512, ...}

    Also supports OpenAI-compatible tool calling:
    - tools: [{"type": "function", "function": {...}}]
    - Returns tool_calls in response when model wants to call a function

    Proxies to trtllm-serve backend and applies harmony format parsing.
    """
    if backend.status != "serving":
        return JSONResponse(status_code=503, content={"error": {"message": "No model loaded", "type": "server_error"}})
    try:
        body = await request.json()

        batch_requests = body.get('batch')
        if batch_requests and isinstance(batch_requests, list):
            return await _handle_batch_completions(body, batch_requests)

        messages = body.get('messages', [])
        tools = body.get('tools', None)
        temperature = body.get('temperature', 0.7)
        max_tokens = body.get('max_tokens', 512)
        stream = body.get('stream', False)
        top_p = body.get('top_p', 0.9)
        include_reasoning = body.get('include_reasoning', False)

        logger.info(f"=== API Request ===")
        logger.info(f"max_tokens: {max_tokens}, temperature: {temperature}, top_p: {top_p}")
        logger.info(f"Messages: {len(messages)}")
        if tools:
            logger.info(f"Tools: {[t.get('function', {}).get('name') for t in tools if t.get('type') == 'function']}")

        backend_payload = {
            "model": backend.model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            **({"stop": backend.stop_tokens} if backend.stop_tokens else {}),
        }
        if tools:
            backend_payload["tools"] = tools

        response = await http_client.post("/v1/chat/completions", json=backend_payload)
        response.raise_for_status()
        result = response.json()

        msg = result["choices"][0]["message"]
        response_text = msg.get("content") or ""
        reasoning_text = msg.get("reasoning") if include_reasoning else None
        tool_calls = msg.get("tool_calls") or None
        if tool_calls is not None and len(tool_calls) == 0:
            tool_calls = None
        finish_reason = result["choices"][0].get("finish_reason", "stop")

        logger.info(f"=== Response from trtllm-serve ===")
        logger.info(f"Content length (chars): {len(response_text)}")
        logger.info(f"Reasoning: {reasoning_text is not None}")
        logger.info(f"Tool calls: {len(tool_calls) if tool_calls else 0}")
        logger.info(f"Finish reason: {finish_reason}")
        if tool_calls:
            logger.info(f"Tool call names: {[tc['function']['name'] for tc in tool_calls]}")

        completion_id = f"chatcmpl-{int(time.time() * 1000)}"
        created = int(time.time())

        if stream:
            async def generate_stream():
                message_content = {"role": "assistant"}
                if tool_calls:
                    message_content["tool_calls"] = tool_calls
                    if response_text:
                        message_content["content"] = response_text
                else:
                    message_content["content"] = response_text

                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": backend.model_id,
                    "choices": [{
                        "index": 0,
                        "delta": message_content,
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(chunk)}\n\n"

                final_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": backend.model_id,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "tool_calls" if tool_calls else "stop"
                    }]
                }
                yield f"data: {json.dumps(final_chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                generate_stream(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
            )

        message = {"role": "assistant"}
        if tool_calls:
            message["tool_calls"] = tool_calls
            message["content"] = response_text if response_text else None
            api_finish_reason = "tool_calls"
        else:
            message["content"] = response_text
            api_finish_reason = "stop"

        response_data = {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": backend.model_id,
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": api_finish_reason
            }],
            "usage": result.get("usage", {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            })
        }

        if reasoning_text:
            response_data["reasoning"] = reasoning_text

        return JSONResponse(response_data)

    except Exception as e:
        logger.error(f"Error in chat completions: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(e), "type": "internal_error"}}
        )


@app.get("/v1/models")
async def openai_models():
    """OpenAI-compatible models endpoint - returns available LLM model"""
    if backend.model_id is None:
        return JSONResponse({"object": "list", "data": []})
    return JSONResponse({
        "object": "list",
        "data": [{
            "id": backend.model_id,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "thinkube",
            "permission": [],
            "root": backend.model_id,
            "parent": None
        }]
    })


# ============================================================================
# Batch Completions Endpoint (for parallel data generation)
# ============================================================================

@app.post("/v1/batch/completions")
async def batch_completions(request: Request):
    """
    Batch completions endpoint for parallel inference via trtllm-serve.

    Accepts multiple prompts and processes them concurrently.
    Backend handles GPU batching internally.

    Request format:
    {
        "requests": [
            {"messages": [...], "max_tokens": 512, "temperature": 0.7},
            {"messages": [...], "max_tokens": 512, "temperature": 0.7},
            ...
        ]
    }

    Response format:
    {
        "responses": [
            {"content": "...", "finish_reason": "stop"},
            {"content": "...", "finish_reason": "stop"},
            ...
        ],
        "batch_size": N,
        "processing_time_ms": X
    }
    """

    try:
        start_time = time.time()
        body = await request.json()
        requests_list = body.get('requests', [])

        if not requests_list:
            return JSONResponse(
                status_code=400,
                content={"error": {"message": "No requests provided", "type": "invalid_request"}}
            )

        total_requests = len(requests_list)
        logger.info(f"=== Batch Request === total={total_requests}")

        async def process_single_request(idx: int, req: dict):
            messages = req.get('messages', [])
            temperature = req.get('temperature', 0.7)
            max_tokens = req.get('max_tokens', 512)
            top_p = req.get('top_p', 0.9)

            response = await http_client.post(
                "/v1/chat/completions",
                json={
                    "model": backend.model_id,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "top_p": top_p,
                    **({"stop": backend.stop_tokens} if backend.stop_tokens else {}),
                }
            )
            response.raise_for_status()
            result = response.json()

            msg = result["choices"][0]["message"]
            content = msg.get("content") or ""
            reasoning = msg.get("reasoning")
            finish_reason = result["choices"][0].get("finish_reason", "stop")

            return {
                "index": idx,
                "content": content,
                "finish_reason": finish_reason,
                "reasoning": reasoning
            }

        tasks = [process_single_request(i, req) for i, req in enumerate(requests_list)]
        results = await asyncio.gather(*tasks)

        results = sorted(results, key=lambda x: x["index"])
        all_responses = [{k: v for k, v in r.items() if k != "index"} for r in results]

        processing_time_ms = int((time.time() - start_time) * 1000)
        logger.info(f"Batch completed: {total_requests} prompts, {processing_time_ms}ms total ({processing_time_ms/total_requests:.0f}ms/prompt)")

        return JSONResponse({
            "responses": all_responses,
            "batch_size": total_requests,
            "processing_time_ms": processing_time_ms,
            "avg_time_per_prompt_ms": processing_time_ms / total_requests
        })

    except Exception as e:
        logger.error(f"Batch error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(e), "type": "internal_error"}}
        )


# Mount Gradio app with Thinkube theme and favicon (Gradio 6.x: theme/css go in mount_gradio_app)
app = gr.mount_gradio_app(
    app,
    demo,
    path="/",
    theme=thinkube_theme,
    css=THINKUBE_CSS,
    favicon_path="/app/icons/tk_ai.png"
)

if __name__ == "__main__":
    logger.info("Starting in idle mode — waiting for model load via /admin/switch-model")
    initialize()

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=7860,
        log_level="info",
    )
