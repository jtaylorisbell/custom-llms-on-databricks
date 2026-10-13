"""Thin proxy translating the Databricks chat contract into vLLM LoRA selection.

Databricks Model Serving's gateway sets the OpenAI ``model`` field to the *endpoint*
name, but vLLM selects a LoRA adapter via that same ``model`` field. We bridge the
two with a tiny FastAPI app that sits in front of the vLLM OpenAI server:

* ``GET /health`` — reports healthy **only** once the upstream vLLM ``/health`` is
  ready, so Model Serving does not route traffic before the engine can answer.
* ``POST /invocations`` (the route Model Serving calls) and ``POST
  /v1/chat/completions`` — read the requested adapter from ``custom_inputs.adapter``
  (the ``llm/v1/chat`` caller contract), validate it against an allowlist, strip
  ``custom_inputs``, rewrite ``model`` to the adapter name vLLM expects, and forward
  to the upstream OpenAI server. A request without an adapter goes to the base
  model. As a convenience for local testing it also accepts a top-level ``{adapter,
  prompt}`` body and converts ``prompt`` into ``messages=[{role: "user", content:
  prompt}]``. ``stream: true`` requests are passed through as a Server-Sent Events
  stream.
* Any other ``GET`` (``/v1/models``, ``/metrics``) is forwarded to vLLM verbatim.

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
    "UpstreamStream",
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


class UpstreamStream(Protocol):
    """An opened upstream streaming response.

    Exposes the *pre-stream* status code and content-type (available as soon as the
    response head is received) so the proxy can decide whether to pass the response
    through as an SSE stream (2xx) or surface an upstream error verbatim (non-2xx),
    instead of mislabeling a failure as a successful event-stream.
    """

    @property
    def status_code(self) -> int: ...

    @property
    def media_type(self) -> str | None: ...

    async def aread(self) -> bytes:
        """Read and return the full response body (used for error passthrough)."""
        ...

    def aiter_bytes(self) -> AsyncIterator[bytes]:
        """Iterate the response body in chunks (used for the SSE passthrough)."""
        ...

    async def aclose(self) -> None:
        """Release the underlying upstream connection."""
        ...


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

    async def open_chat_stream(self, body: dict[str, Any]) -> UpstreamStream:
        """Open a streaming chat completion and return the response handle.

        The handle's ``status_code``/``media_type`` reflect the upstream response
        head, so the caller can surface pre-stream errors instead of streaming them.
        """
        ...

    async def forward_get(
        self, path: str, params: Mapping[str, Any] | None = None
    ) -> UpstreamResponse:
        """Proxy an arbitrary GET (e.g. ``/v1/models``, ``/metrics``) to vLLM verbatim."""
        ...


class _HttpxUpstreamStream:
    """:class:`UpstreamStream` backed by an open ``httpx`` streaming response."""

    def __init__(self, response: httpx.Response) -> None:
        self._response = response

    @property
    def status_code(self) -> int:
        return self._response.status_code

    @property
    def media_type(self) -> str | None:
        content_type: str | None = self._response.headers.get("content-type")
        return content_type

    async def aread(self) -> bytes:
        return await self._response.aread()

    def aiter_bytes(self) -> AsyncIterator[bytes]:
        return self._response.aiter_bytes()

    async def aclose(self) -> None:
        await self._response.aclose()


class _HttpxUpstream:
    """Default :class:`UpstreamClient` backed by an ``httpx.AsyncClient``."""

    def __init__(self, base_url: str, *, client: httpx.AsyncClient | None = None) -> None:
        self._base_url = base_url.rstrip("/")
        # No timeout: chat completions (especially streaming) can run for a while. No keep-alive:
        # vLLM closes idle connections after 5s, and reusing one just as it closes fails the
        # request with ReadError (seen on Model Serving's probes). Loopback connects are cheap.
        self._client = (
            client
            if client is not None
            else httpx.AsyncClient(timeout=None, limits=httpx.Limits(max_keepalive_connections=0))
        )

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

    async def open_chat_stream(self, body: dict[str, Any]) -> UpstreamStream:
        # Send with stream=True so the response head (status + content-type) is
        # available before the body is consumed; the caller closes the handle.
        request = self._client.build_request("POST", self._chat_url, json=body)
        response = await self._client.send(request, stream=True)
        return _HttpxUpstreamStream(response)

    async def forward_get(
        self, path: str, params: Mapping[str, Any] | None = None
    ) -> UpstreamResponse:
        response = await self._client.get(
            f"{self._base_url}/{path.lstrip('/')}", params=params
        )
        return UpstreamResponse(
            status_code=response.status_code,
            content=response.content,
            media_type=response.headers.get("content-type"),
        )


def _translate(
    body: Mapping[str, Any], allowlist: frozenset[str], base_model_name: str
) -> dict[str, Any]:
    """Translate an inbound request body into the body to forward to vLLM.

    Resolves the requested adapter (from ``custom_inputs.adapter`` or, for local
    testing, a top-level ``adapter``), validates it against ``allowlist``, strips
    ``custom_inputs``/``adapter``/``prompt``, applies the ``{adapter, prompt}``
    convenience, and sets ``model`` to the adapter name vLLM expects. A request that
    names no adapter is routed to ``base_model_name``, so plain chat callers (AI
    Playground, ``ai_query``, readiness checks) work unchanged.

    Raises:
        HTTPException: 400 if the body is not an object, or the adapter is malformed
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
        # Overwrite rather than drop: the gateway may have filled ``model`` with the
        # endpoint name, which vLLM would reject as an unknown model.
        forwarded["model"] = base_model_name
        return forwarded
    # Guard the allowlist membership test: a non-string adapter (list/dict) is
    # unhashable and would raise TypeError -> HTTP 500. A malformed adapter is a
    # client error, so reject it (and empty strings) with 400 per the contract.
    if not isinstance(adapter, str) or not adapter:
        raise HTTPException(
            status_code=400,
            detail=f"adapter must be a non-empty string, got {adapter!r}",
        )
    if adapter not in allowlist:
        raise HTTPException(
            status_code=400,
            detail=f"unknown adapter {adapter!r}; allowed adapters: {sorted(allowlist)}",
        )

    forwarded["model"] = adapter
    return forwarded


def _is_success(status_code: int) -> bool:
    return 200 <= status_code < 300


async def _stream_or_error(upstream: UpstreamClient, body: dict[str, Any]) -> Response:
    """Open the upstream stream; pass 2xx through as SSE, surface errors verbatim.

    A vLLM pre-stream failure (e.g. 4xx/5xx before any tokens) must not be returned
    as a 200 event-stream. We inspect the upstream response head: non-2xx responses
    are returned verbatim (status, body, content-type); only a 2xx is streamed.
    """
    stream = await upstream.open_chat_stream(body)

    if not _is_success(stream.status_code):
        try:
            error_body = await stream.aread()
        finally:
            await stream.aclose()
        return Response(
            content=error_body,
            status_code=stream.status_code,
            media_type=stream.media_type,
        )

    async def _passthrough() -> AsyncIterator[bytes]:
        # Keep the upstream connection open while FastAPI drains the body, then
        # close it. Mid-stream failures after headers are sent flow through as-is.
        try:
            async for chunk in stream.aiter_bytes():
                yield chunk
        finally:
            await stream.aclose()

    return StreamingResponse(
        _passthrough(),
        status_code=stream.status_code,
        media_type=stream.media_type or _SSE_MEDIA_TYPE,
    )


def create_proxy_app(
    adapter_names: list[str],
    vllm_base_url: str,
    *,
    base_model_name: str = "base",
    http_client: UpstreamClient | None = None,
) -> FastAPI:
    """Build the proxy FastAPI app.

    Args:
        adapter_names: Allowlist of adapter names that may be requested. Any other
            adapter is rejected with HTTP 400.
        vllm_base_url: Base URL of the upstream vLLM OpenAI server (e.g.
            ``http://127.0.0.1:8081``). Used to build the default HTTP client.
        base_model_name: vLLM ``--served-model-name`` of the base weights; requests
            that name no adapter are routed here.
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

    # Databricks Model Serving forwards llm/v1/chat requests to POST /invocations
    # (the MLflow/SageMaker scoring path that native vLLM also serves); the OpenAI
    # path is kept for direct clients. Both share this handler.
    @app.post("/invocations")
    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        try:
            raw = await request.json()
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="invalid JSON body") from exc

        forwarded = _translate(raw, allowlist, base_model_name)

        if forwarded.get("stream"):
            return await _stream_or_error(upstream, forwarded)

        result = await upstream.chat_completions(forwarded)
        return Response(
            content=result.content,
            status_code=result.status_code,
            media_type=result.media_type,
        )

    # Forward any other GET (e.g. /v1/models, /metrics) straight to vLLM. Databricks
    # Model Serving's readiness probe requests /v1/models on llm/v1/chat endpoints;
    # without this passthrough it 404s and the endpoint never reaches READY. Declared
    # last so the explicit /health route above still wins.
    @app.get("/{path:path}")
    async def passthrough(path: str, request: Request) -> Response:
        result = await upstream.forward_get(path, dict(request.query_params))
        return Response(
            content=result.content,
            status_code=result.status_code,
            media_type=result.media_type,
        )

    return app
