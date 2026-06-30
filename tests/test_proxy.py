"""Unit tests for the multi-LoRA proxy.

These use FastAPI's ``TestClient`` with a fake injected upstream, so they run with
no GPU, no vLLM, and no network.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from multi_adapter_serving.serving.proxy import UpstreamResponse, create_proxy_app


class FakeStream:
    """In-memory :class:`UpstreamStream` handle with a controllable response head."""

    def __init__(
        self,
        *,
        status_code: int,
        media_type: str | None,
        chunks: list[bytes],
        error_body: bytes,
    ) -> None:
        self._status_code = status_code
        self._media_type = media_type
        self._chunks = chunks
        self._error_body = error_body
        self.closed = False

    @property
    def status_code(self) -> int:
        return self._status_code

    @property
    def media_type(self) -> str | None:
        return self._media_type

    async def aread(self) -> bytes:
        return self._error_body

    async def aiter_bytes(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


class FakeUpstream:
    """In-memory stand-in for the vLLM OpenAI server.

    Records the bodies the proxy forwards so tests can assert the translation, and
    lets each test control health, the unary response, and the streamed response
    head/body.
    """

    def __init__(
        self,
        *,
        healthy: bool = True,
        response_body: dict[str, Any] | None = None,
        status_code: int = 200,
        stream_status: int = 200,
        stream_media_type: str | None = "text/event-stream",
        stream_chunks: list[bytes] | None = None,
        stream_error_body: bytes = b"",
    ) -> None:
        self.healthy = healthy
        self.response_body = (
            response_body if response_body is not None else {"id": "cmpl-1"}
        )
        self.status_code = status_code
        self.stream_status = stream_status
        self.stream_media_type = stream_media_type
        self.stream_chunks = (
            stream_chunks
            if stream_chunks is not None
            else [b'data: {"x": 1}\n\n', b"data: [DONE]\n\n"]
        )
        self.stream_error_body = stream_error_body
        self.chat_calls: list[dict[str, Any]] = []
        self.stream_calls: list[dict[str, Any]] = []
        self.opened_streams: list[FakeStream] = []

    async def health(self) -> bool:
        return self.healthy

    async def chat_completions(self, body: dict[str, Any]) -> UpstreamResponse:
        self.chat_calls.append(body)
        return UpstreamResponse(
            status_code=self.status_code,
            content=json.dumps(self.response_body).encode("utf-8"),
            media_type="application/json",
        )

    async def open_chat_stream(self, body: dict[str, Any]) -> FakeStream:
        self.stream_calls.append(body)
        handle = FakeStream(
            status_code=self.stream_status,
            media_type=self.stream_media_type,
            chunks=self.stream_chunks,
            error_body=self.stream_error_body,
        )
        self.opened_streams.append(handle)
        return handle


def make_client(fake: FakeUpstream, adapters: list[str] | None = None) -> TestClient:
    app = create_proxy_app(
        adapters if adapters is not None else ["ner", "code"],
        "http://vllm.invalid:8081",
        http_client=fake,
    )
    return TestClient(app)


def test_adapter_moved_from_custom_inputs_to_model_and_stripped() -> None:
    fake = FakeUpstream()
    client = make_client(fake)

    resp = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "custom_inputs": {"adapter": "ner"},
            "temperature": 0.2,
        },
    )

    assert resp.status_code == 200
    assert len(fake.chat_calls) == 1
    forwarded = fake.chat_calls[0]
    # Adapter is moved into the OpenAI `model` field vLLM selects on...
    assert forwarded["model"] == "ner"
    # ...and the proxy-only custom_inputs is stripped before forwarding.
    assert "custom_inputs" not in forwarded
    # Other fields are preserved untouched.
    assert forwarded["messages"] == [{"role": "user", "content": "hi"}]
    assert forwarded["temperature"] == 0.2


def test_response_body_passed_through_unchanged() -> None:
    body = {"id": "cmpl-xyz", "choices": [{"message": {"content": "yo"}}]}
    fake = FakeUpstream(response_body=body)
    client = make_client(fake)

    resp = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "custom_inputs": {"adapter": "code"},
        },
    )

    assert resp.status_code == 200
    assert resp.json() == body


def test_unknown_adapter_returns_400_and_does_not_forward() -> None:
    fake = FakeUpstream()
    client = make_client(fake, adapters=["ner"])

    resp = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "custom_inputs": {"adapter": "does-not-exist"},
        },
    )

    assert resp.status_code == 400
    assert fake.chat_calls == []


def test_missing_adapter_returns_400() -> None:
    fake = FakeUpstream()
    client = make_client(fake)

    resp = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )

    assert resp.status_code == 400
    assert fake.chat_calls == []


def test_top_level_adapter_and_prompt_become_messages() -> None:
    fake = FakeUpstream()
    client = make_client(fake)

    resp = client.post("/v1/chat/completions", json={"adapter": "code", "prompt": "hello"})

    assert resp.status_code == 200
    forwarded = fake.chat_calls[0]
    assert forwarded["model"] == "code"
    assert forwarded["messages"] == [{"role": "user", "content": "hello"}]
    # The convenience fields are not forwarded to vLLM.
    assert "adapter" not in forwarded
    assert "prompt" not in forwarded


def test_health_healthy_only_when_upstream_ready() -> None:
    ready = make_client(FakeUpstream(healthy=True))
    not_ready = make_client(FakeUpstream(healthy=False))

    ok = ready.get("/health")
    assert ok.status_code == 200
    assert ok.json() == {"status": "healthy"}

    down = not_ready.get("/health")
    assert down.status_code == 503


def test_streaming_passthrough() -> None:
    chunks = [b'data: {"a": 1}\n\n', b'data: {"a": 2}\n\n', b"data: [DONE]\n\n"]
    fake = FakeUpstream(stream_chunks=chunks)
    client = make_client(fake)

    resp = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "custom_inputs": {"adapter": "ner"},
            "stream": True,
        },
    )

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    # Bytes are passed through verbatim and in order.
    assert resp.content == b"".join(chunks)
    # Streaming used the stream path, not the unary path, with the translated body.
    assert fake.chat_calls == []
    assert len(fake.stream_calls) == 1
    assert fake.stream_calls[0]["model"] == "ner"
    assert "custom_inputs" not in fake.stream_calls[0]
    # The upstream stream handle is closed once the body is drained.
    assert fake.opened_streams[0].closed is True


@pytest.mark.parametrize(
    "adapter",
    [["ner"], {"name": "ner"}, 123, "", None],
    ids=["list", "dict", "number", "empty-string", "null"],
)
def test_malformed_adapter_returns_400_not_500(adapter: Any) -> None:
    # A non-string/empty/missing adapter must be a client error (400). Before the
    # fix, list/dict adapters raised TypeError at the allowlist check -> HTTP 500.
    fake = FakeUpstream()
    client = make_client(fake)

    resp = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "custom_inputs": {"adapter": adapter},
        },
    )

    assert resp.status_code == 400
    assert fake.chat_calls == []
    assert fake.stream_calls == []


def test_streaming_upstream_4xx_returned_verbatim_not_sse() -> None:
    # A vLLM pre-stream 4xx must surface as that status + body, not a 200 SSE.
    fake = FakeUpstream(
        stream_status=400,
        stream_media_type="application/json",
        stream_error_body=b'{"error": "bad request"}',
    )
    client = make_client(fake)

    resp = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "custom_inputs": {"adapter": "ner"},
            "stream": True,
        },
    )

    assert resp.status_code == 400
    assert resp.content == b'{"error": "bad request"}'
    assert resp.headers["content-type"].startswith("application/json")
    assert not resp.headers["content-type"].startswith("text/event-stream")
    assert fake.opened_streams[0].closed is True


def test_streaming_upstream_5xx_returned_verbatim_not_sse() -> None:
    # A vLLM pre-stream 5xx must surface as that status + body, not a 200 SSE.
    fake = FakeUpstream(
        stream_status=503,
        stream_media_type="application/json",
        stream_error_body=b'{"error": "engine overloaded"}',
    )
    client = make_client(fake)

    resp = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "custom_inputs": {"adapter": "code"},
            "stream": True,
        },
    )

    assert resp.status_code == 503
    assert resp.content == b'{"error": "engine overloaded"}'
    assert resp.headers["content-type"].startswith("application/json")
    assert not resp.headers["content-type"].startswith("text/event-stream")
    assert fake.opened_streams[0].closed is True
