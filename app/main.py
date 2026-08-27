"""serving-stack: the FastAPI service (week 2, GPU + CPU fallback).

The same application can run on either CUDA or CPU.

Run it:

    uvicorn main:app --host 0.0.0.0 --port 8000

Model: Qwen/Qwen2.5-0.5B-Instruct
"""

from __future__ import annotations

import os
import time
import uuid

import torch
from fastapi import FastAPI, Header, HTTPException
from transformers import AutoModelForCausalLM, AutoTokenizer

from schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    HealthResponse,
    ModelCard,
    ModelList,
    ResponseMessage,
    Usage,
)


MODEL_ID = os.environ.get(
    "MODEL_ID",
    "Qwen/Qwen2.5-0.5B-Instruct",
)

# W2D5: API key for protecting /v1/* endpoints.
# If empty, the API stays open for backwards compatibility.
API_KEY = os.environ.get("API_KEY", "")

# W2D5: Maximum number of tokens a client may generate.
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "256"))

app = FastAPI(title="serving-stack", version="wk2")


# ---------------------------------------------------------------------------
# W2D5: API key protection
# ---------------------------------------------------------------------------

if not API_KEY:
    print(
        "WARNING: API_KEY is not set. "
        "The /v1 API is running WITHOUT authentication."
    )


def require_api_key(
    authorization: str | None = Header(default=None),
) -> None:
    """Require Bearer authentication for /v1/* endpoints."""

    # Keep old labs compatible when API_KEY is not configured.
    if not API_KEY:
        return

    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(
            status_code=401,
            detail="Unauthorized",
        )


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

# W2D4:
# Use CUDA when it is available.
# Otherwise fall back to CPU.
device = "cuda" if torch.cuda.is_available() else "cpu"

# float16 is appropriate for GPU inference.
# CPU keeps float32.
dtype = torch.float16 if device == "cuda" else torch.float32

print(f"loading {MODEL_ID} on {device} ...")


# ---------------------------------------------------------------------------
# Load model once at startup
# ---------------------------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
)

model.to(device)
model.eval()

print(f"model ready on {device}")


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness and readiness endpoint."""

    # W2D5:
    # /health intentionally stays open so health probes do not need a key.
    return HealthResponse(
        status="ok",
        model=MODEL_ID,
    )


# ---------------------------------------------------------------------------
# GET /v1/models
# ---------------------------------------------------------------------------

@app.get("/v1/models", response_model=ModelList)
def list_models(
    authorization: str | None = Header(default=None),
) -> ModelList:
    """List the model served by this API."""

    # W2D5: protect /v1/* with the API key.
    require_api_key(authorization)

    card = ModelCard(
        id=MODEL_ID,
        created=int(time.time()),
    )

    return ModelList(data=[card])


# ---------------------------------------------------------------------------
# POST /v1/chat/completions
# ---------------------------------------------------------------------------

@app.post(
    "/v1/chat/completions",
    response_model=ChatCompletionResponse,
)
def chat_completions(
    req: ChatCompletionRequest,
    authorization: str | None = Header(default=None),
) -> ChatCompletionResponse:
    """Generate an OpenAI-compatible chat completion."""

    # W2D5: protect /v1/* with the API key.
    require_api_key(authorization)

    # W2D5:
    # Silently clamp a client's requested max_tokens to the server limit.
    max_tokens = min(req.max_tokens, MAX_TOKENS)

    # Newer transformers versions can return a BatchEncoding.
    # Ask for a dictionary and extract the input_ids tensor.
    encoded = tokenizer.apply_chat_template(
        [m.model_dump() for m in req.messages],
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    # Extract the real tensor and move it to the same device as the model.
    input_ids = encoded["input_ids"].to(device)

    prompt_tokens = input_ids.shape[1]

    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            do_sample=req.temperature > 0,
            temperature=req.temperature if req.temperature > 0 else None,
        )

    new_tokens = out[0][prompt_tokens:]

    completion_tokens = len(new_tokens)

    text = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    )

    finish_reason = (
        "length"
        if completion_tokens >= max_tokens
        else "stop"
    )

    return ChatCompletionResponse(
        id="chatcmpl-" + uuid.uuid4().hex,
        created=int(time.time()),
        model=req.model,
        choices=[
            Choice(
                message=ResponseMessage(
                    role="assistant",
                    content=text,
                ),
                finish_reason=finish_reason,
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


@app.post("/v1/embeddings")
def embeddings(
    payload: dict,
    authorization: str | None = Header(default=None),
):
    """GPU-only embeddings endpoint."""

    require_api_key(authorization)

    if device != "cuda":
        raise HTTPException(
            status_code=400,
            detail=(
                "Embeddings require a GPU-backed instance; "
                "this instance is running in CPU-fallback mode."
            ),
        )

    return {
        "vector": [0.1] * 8,
        "device_used": device,
    }