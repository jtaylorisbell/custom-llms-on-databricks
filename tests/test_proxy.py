"""Unit tests for the multi-LoRA proxy.

These use FastAPI's ``TestClient`` with a fake injected upstream, so they run with
no GPU, no vLLM, and no network.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi.testclient import TestClient

from multi_adapter_serving.serving.proxy import UpstreamResponse, create_proxy_app


class FakeUpstream:
    """In-memory stand-in for the vLLM OpenAI server.

    Records the bodies the proxy forwards so tests can assert the translation, and
    lets each test control health, the unary response, and the streamed chunks.
    """

    def __init__(
        self,
        *,
        healthy: bool = True,
        response_body: dict[str, Any] | None = None,
        status_code: int = 200,
        stream_chunks: list[bytes] | None = None,
    ) -> None:
        self.healthy = healthy
        self.response_body = (
            response_body if response_body is not None else {"id": "cmpl-1"}
        )
        self.status_code = status_code
        self.stream_chunks = (
            stream_chunks
            if stream_chunks is not None
            else [b'data: {"x": 1}\n\n', b"data: [DONE]\n\n"]
        )
        self.chat_calls: list[dict[str, Any]] = []
        self.stream_calls: list[dict[str, Any]] = []

    async def health(self) -> bool:
        return self.healthy

    async def chat_completions(self, body: dict[str, Any]) -> UpstreamResponse:
        self.chat_calls.append(body)
        return UpstreamResponse(
            status_code=self.status_code,
            content=json.dumps(self.response_body).encode("utf-8"),
            media_type="application/json",
        )

    async def stream_chat_completions(self, body: dict[str, Any]) -> AsyncIterator[bytes]:
        self.stream_calls.append(body)
        for chunk in self.stream_chunks:
            yield chunk


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
