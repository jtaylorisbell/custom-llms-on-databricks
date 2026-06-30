"""Thin proxy translating the Databricks chat contract into vLLM LoRA selection.

Databricks Model Serving's gateway sets the OpenAI ``model`` field to the *endpoint*
name, but vLLM selects a LoRA adapter via that same ``model`` field. We bridge the
two with a tiny FastAPI app that sits in front of the vLLM OpenAI server:

* ``GET /health`` — reports healthy **only** once the upstream vLLM ``/health`` is
  ready, so Model Serving does not route traffic before the engine can answer.
* ``POST /v1/chat/completions`` — reads the requested adapter from
  ``custom_inputs.adapter`` (the ``llm/v1/chat`` caller contract), validates it
  against an allowlist, strips ``custom_inputs``, rewrites ``model`` to the adapter
  name vLLM expects, and forwards to the upstream OpenAI server. As a convenience
  for local testing it also accepts a top-level ``{adapter, prompt}`` body and
  converts ``prompt`` into ``messages=[{role: "user", content: prompt}]``.
  ``stream: true`` requests are passed through as a Server-Sent Events stream.

The upstream HTTP call is injectable (``http_client``) so the app can be unit
tested with a fake upstream and no running vLLM.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

__all__ = [
    "UpstreamClient",
    "UpstreamResponse",
    "create_proxy_app",
]

# Server-Sent Events media type used by the OpenAI streaming protocol.
_SSE_MEDIA_TYPE = "text/event-stream"


@dataclass(frozen=True)
class UpstreamResponse:
    """A non-streaming response captured from the upstream vLLM server."""

    status_code: int
    content: bytes
    media_type: str | None = None


class UpstreamClient(Protocol):
    """The upstream operations the proxy needs from the vLLM OpenAI server.

    Abstracted as a Protocol so tests can inject a fake (no running vLLM). The
    default implementation, :class:`_HttpxUpstream`, is backed by ``httpx``.
    """

    async def health(self) -> bool:
        """Return ``True`` only when the upstream ``/health`` endpoint is ready."""
        ...

    async def chat_completions(self, body: dict[str, Any]) -> UpstreamResponse:
        """Forward a (non-streaming) chat completion request and return the reply."""
        ...

    def stream_chat_completions(self, body: dict[str, Any]) -> AsyncIterator[bytes]:
        """Forward a streaming chat completion request, yielding raw SSE bytes."""
        ...


class _HttpxUpstream:
    """Default :class:`UpstreamClient` backed by an ``httpx.AsyncClient``."""

    def __init__(self, base_url: str, *, client: httpx.AsyncClient | None = None) -> None:
        self._base_url = base_url.rstrip("/")
        # No timeout: chat completions (especially streaming) can run for a while.
        self._client = client if client is not None else httpx.AsyncClient(timeout=None)

    @property
    def _chat_url(self) -> str:
        return f"{self._base_url}/v1/chat/completions"

    async def health(self) -> bool:
        try:
            response = await self._client.get(f"{self._base_url}/health")
        except httpx.HTTPError:
            return False
        return response.status_code == 200

    async def chat_completions(self, body: dict[str, Any]) -> UpstreamResponse:
        response = await self._client.post(self._chat_url, json=body)
        return UpstreamResponse(
            status_code=response.status_code,
            content=response.content,
            media_type=response.headers.get("content-type"),
        )

    async def stream_chat_completions(self, body: dict[str, Any]) -> AsyncIterator[bytes]:
        async with self._client.stream("POST", self._chat_url, json=body) as response:
            async for chunk in response.aiter_bytes():
                yield chunk


def _translate(body: Mapping[str, Any], allowlist: frozenset[str]) -> dict[str, Any]:
    """Translate an inbound request body into the body to forward to vLLM.

    Resolves the requested adapter (from ``custom_inputs.adapter`` or, for local
    testing, a top-level ``adapter``), validates it against ``allowlist``, strips
    ``custom_inputs``/``adapter``/``prompt``, applies the ``{adapter, prompt}``
    convenience, and sets ``model`` to the adapter name vLLM expects.

    Raises:
        HTTPException: 400 if the body is not an object, or the adapter is missing
            or not in the allowlist.
    """
    if not isinstance(body, Mapping):
        raise HTTPException(status_code=400, detail="request body must be a JSON object")

    forwarded: dict[str, Any] = dict(body)

    # Strip the proxy-only fields so they never reach vLLM (which would reject them).
    custom_inputs = forwarded.pop("custom_inputs", None)
    top_level_adapter = forwarded.pop("adapter", None)
    prompt = forwarded.pop("prompt", None)

    # {adapter, prompt} convenience: synthesize chat messages when none were given.
    if "messages" not in forwarded and prompt is not None:
        forwarded["messages"] = [{"role": "user", "content": prompt}]

    # custom_inputs.adapter is the primary contract; top-level adapter is the
    # local-testing convenience fallback.
    adapter: Any = None
    if isinstance(custom_inputs, Mapping):
        adapter = custom_inputs.get("adapter")
    if adapter is None:
        adapter = top_level_adapter

    if adapter is None:
        raise HTTPException(
            status_code=400,
            detail="missing adapter: set custom_inputs.adapter (or a top-level adapter)",
        )
    if adapter not in allowlist:
        raise HTTPException(
            status_code=400,
            detail=f"unknown adapter {adapter!r}; allowed adapters: {sorted(allowlist)}",
        )

    forwarded["model"] = adapter
    return forwarded


def create_proxy_app(
    adapter_names: list[str],
    vllm_base_url: str,
    *,
    http_client: UpstreamClient | None = None,
) -> FastAPI:
    """Build the proxy FastAPI app.

    Args:
        adapter_names: Allowlist of adapter names that may be requested. Any other
            adapter is rejected with HTTP 400.
        vllm_base_url: Base URL of the upstream vLLM OpenAI server (e.g.
            ``http://127.0.0.1:8081``). Used to build the default HTTP client.
        http_client: Optional upstream client override. Defaults to an
            ``httpx``-backed client pointed at ``vllm_base_url``. Injecting a fake
            here makes the app unit-testable without a running vLLM.

    Returns:
        A configured :class:`fastapi.FastAPI` application.
    """
    allowlist = frozenset(adapter_names)
    upstream: UpstreamClient = (
        http_client if http_client is not None else _HttpxUpstream(vllm_base_url)
    )

    app = FastAPI(title="multi-adapter-serving proxy")

    @app.get("/health")
    async def health() -> Response:
        # Gate readiness on the upstream engine so traffic is only routed once vLLM
        # (with all LoRA modules loaded) can actually answer.
        if await upstream.health():
            return JSONResponse({"status": "healthy"})
        return JSONResponse({"status": "unavailable"}, status_code=503)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        try:
            raw = await request.json()
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="invalid JSON body") from exc

        forwarded = _translate(raw, allowlist)

        if forwarded.get("stream"):
            return StreamingResponse(
                upstream.stream_chat_completions(forwarded),
                media_type=_SSE_MEDIA_TYPE,
            )

        result = await upstream.chat_completions(forwarded)
        return Response(
            content=result.content,
            status_code=result.status_code,
            media_type=result.media_type,
        )

    return app
