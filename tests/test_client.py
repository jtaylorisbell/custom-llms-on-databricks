"""Unit tests for the user-facing endpoint client (HTTP invoker mocked).

No network, no Databricks workspace. The default invoker's auth is verified by
mocking ``databricks.sdk.WorkspaceClient`` and asserting OAuth-via-profile usage with
no personal access token.
"""

from __future__ import annotations

from typing import Any

import pytest

from multi_adapter_serving.client import build_payload, query


def test_build_payload_shape() -> None:
    payload = build_payload("ner", "Hello", temperature=0.2, max_tokens=64)

    assert payload == {
        "messages": [{"role": "user", "content": "Hello"}],
        "custom_inputs": {"adapter": "ner"},
        "temperature": 0.2,
        "max_tokens": 64,
    }


def test_build_payload_rejects_reserved_kwargs() -> None:
    # A caller must not clobber the keys this module owns.
    with pytest.raises(ValueError, match="reserved payload keys"):
        build_payload("ner", "hi", messages=[{"role": "user", "content": "x"}])
    with pytest.raises(ValueError, match="reserved payload keys"):
        build_payload("ner", "hi", custom_inputs={"adapter": "other"})


def test_query_rejects_reserved_kwargs_before_calling_client() -> None:
    calls: list[Any] = []

    def spy_client(endpoint_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        calls.append(payload)
        return {}

    with pytest.raises(ValueError, match="reserved payload keys"):
        query("ep", "ner", "hi", client=spy_client, custom_inputs={"adapter": "evil"})

    # Rejected before the endpoint was ever invoked.
    assert calls == []


def test_query_sends_exact_payload_and_returns_text() -> None:
    captured: dict[str, Any] = {}

    def fake_client(endpoint_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        captured["endpoint_name"] = endpoint_name
        captured["payload"] = payload
        return {"choices": [{"message": {"role": "assistant", "content": "hi there"}}]}

    result = query(
        "my-endpoint",
        "code",
        "Write a function",
        client=fake_client,
        temperature=0.7,
        max_tokens=128,
    )

    # Endpoint + exact payload shape: messages + custom_inputs.adapter + passthrough.
    assert captured["endpoint_name"] == "my-endpoint"
    assert captured["payload"] == {
        "messages": [{"role": "user", "content": "Write a function"}],
        "custom_inputs": {"adapter": "code"},
        "temperature": 0.7,
        "max_tokens": 128,
    }
    # Assistant text extracted + raw response preserved.
    assert result.text == "hi there"
    assert result.raw == {
        "choices": [{"message": {"role": "assistant", "content": "hi there"}}]
    }


def test_query_no_kwargs_minimal_payload() -> None:
    captured: dict[str, Any] = {}

    def fake_client(endpoint_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        captured["payload"] = payload
        return {"choices": []}

    result = query("ep", "ner", "ping", client=fake_client)

    assert captured["payload"] == {
        "messages": [{"role": "user", "content": "ping"}],
        "custom_inputs": {"adapter": "ner"},
    }
    # No extractable choice -> empty text, but the raw response is still returned.
    assert result.text == ""
    assert result.raw == {"choices": []}


def test_default_invoker_uses_oauth_profile_and_no_pat(monkeypatch: Any) -> None:
    recorded: dict[str, Any] = {}

    class FakeApiClient:
        def do(self, method: str, path: str, *, body: dict[str, Any]) -> dict[str, Any]:
            recorded["method"] = method
            recorded["path"] = path
            recorded["body"] = body
            return {"choices": [{"message": {"content": "ok"}}]}

    class FakeWorkspaceClient:
        def __init__(self, **kwargs: Any) -> None:
            recorded["ws_kwargs"] = kwargs
            self.api_client = FakeApiClient()

    import databricks.sdk

    monkeypatch.setattr(databricks.sdk, "WorkspaceClient", FakeWorkspaceClient)

    result = query("my-endpoint", "code", "hi", profile="fevm-aws")

    # OAuth via the named profile, never a PAT.
    assert recorded["ws_kwargs"] == {"profile": "fevm-aws"}
    assert "token" not in recorded["ws_kwargs"]
    assert "auth_type" not in recorded["ws_kwargs"]
    # Posts the chat payload to the endpoint's invocations path.
    assert recorded["method"] == "POST"
    assert recorded["path"] == "/serving-endpoints/my-endpoint/invocations"
    assert recorded["body"]["custom_inputs"]["adapter"] == "code"
    assert recorded["body"]["messages"] == [{"role": "user", "content": "hi"}]
    assert result.text == "ok"
