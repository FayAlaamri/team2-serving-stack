"""OpenAI-compatible request and response shapes.

This file is GIVEN, complete. The contract's teeth are not the exercise: your
job is to fill in the routes in main.py so they read these requests and return
these responses. Do not weaken these models. The Agentic AI cohort's client
(and the openai Python client) expects exactly these field names.

The `tools`/`tool_choice` fields are accepted from day 1 (the contract says a
consumer's payload always validates) and go unused until the tool-calling
engine at tier 1.
"""
"""Pydantic schemas for the OpenAI-compatible serving API."""

from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------
# Chat messages
# ---------------------------------------------------------

class ChatMessage(BaseModel):
    """One message sent to the chat completion endpoint."""

    role: Literal["system", "user", "assistant"]
    content: str


# ---------------------------------------------------------
# Chat completion request
# ---------------------------------------------------------

class ChatCompletionRequest(BaseModel):
    """The body of POST /v1/chat/completions."""

    model: str

    # At least one message must exist.
    messages: List[ChatMessage] = Field(..., min_length=1)

    # Generation controls.
    max_tokens: int = Field(default=256, ge=1, le=4096)

    temperature: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
    )

    stream: bool = False

    # Accepted for OpenAI compatibility.
    # They are not used by our model yet.
    tools: Optional[List[dict]] = None
    tool_choice: Optional[Union[str, dict]] = None

    @field_validator("messages")
    @classmethod
    def last_message_must_be_user_or_system(cls, v):
        """
        Reject a conversation whose final message is from the assistant.

        The model should be asked to respond to a user/system turn,
        rather than continue an already-finished assistant turn.
        """

        if v and v[-1].role == "assistant":
            raise ValueError(
                "the last message must be from 'user' or 'system', "
                "not 'assistant'"
            )

        return v


# ---------------------------------------------------------
# Chat completion response
# ---------------------------------------------------------

class ResponseMessage(BaseModel):
    """The assistant message inside a completion choice."""

    role: Literal["assistant"] = "assistant"
    content: str


class Choice(BaseModel):
    """One completion choice."""

    index: int = 0
    message: ResponseMessage
    finish_reason: Literal["stop", "length"] = "stop"


class Usage(BaseModel):
    """Token accounting."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    """The body returned by POST /v1/chat/completions."""

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage


# ---------------------------------------------------------
# /v1/models
# ---------------------------------------------------------

class ModelCard(BaseModel):
    """One model returned by GET /v1/models."""

    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str = "aidc"


class ModelList(BaseModel):
    """The body returned by GET /v1/models."""

    object: Literal["list"] = "list"
    data: List[ModelCard]


# ---------------------------------------------------------
# /health
# ---------------------------------------------------------

class HealthResponse(BaseModel):
    """The body returned by GET /health."""

    status: Literal["ok"] = "ok"
    model: str
