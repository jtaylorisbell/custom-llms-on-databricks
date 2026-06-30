"""User-facing client for the multi-LoRA endpoint: ``{adapter, prompt}`` in, text out.

This is the two-argument interface the whole project exists to provide: pick an
adapter by its logical name and send a prompt. :func:`query` builds the Databricks
``llm/v1/chat`` request body — the adapter travels in ``custom_inputs.adapter`` (the
contract the serving proxy reads to select the LoRA) — and invokes the endpoint over
OAuth.

The wire payload is exactly::

    {
        "messages": [{"role": "user", "content": <prompt>}],
        "custom_inputs": {"adapter": <adapter>},
        **chat_kwargs,            # e.g. temperature, max_tokens, stream
    }

The HTTP invoker is injectable (``client``) so this can be unit tested with no
Databricks workspace. The default invoker authenticates with OAuth via the Databricks
SDK config profile — never a personal access token.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

__all__ = [
    "EndpointInvoker",
    "QueryResult",
    "build_payload",
    "query",
]


class EndpointInvoker(Protocol):
    """Invokes a serving endpoint with a JSON payload and returns the parsed reply.

    Abstracted as a Protocol so tests can inject a fake (no network). The default
    implementation posts to ``/serving-endpoints/{name}/invocations`` via the
    Databricks SDK (OAuth).
    """

    def __call__(self, endpoint_name: str, payload: dict[str, Any]) -> dict[str, Any]: ...


@dataclass(frozen=True)
class QueryResult:
    """The result of a :func:`query` call."""

    # The assistant's reply text (first choice), or "" if none could be extracted.
    text: str
    # The full, raw JSON response from the endpoint.
    raw: dict[str, Any]


# Payload keys this module owns; callers may not override them via chat_kwargs,
# otherwise the guaranteed {messages, custom_inputs.adapter} shape could be clobbered.
_RESERVED_PAYLOAD_KEYS = ("messages", "custom_inputs")


def build_payload(adapter: str, prompt: str, **chat_kwargs: Any) -> dict[str, Any]:
    """Build the ``llm/v1/chat`` request body for ``adapter`` + ``prompt``.

    The adapter is carried in ``custom_inputs.adapter`` (the proxy contract); any
    extra OpenAI-style chat parameters are passed through at the top level.

    Raises:
        ValueError: If ``chat_kwargs`` contains a reserved key (``messages`` or
            ``custom_inputs``), which would clobber the constructed payload shape.
    """
    clobbered = [key for key in _RESERVED_PAYLOAD_KEYS if key in chat_kwargs]
    if clobbered:
        raise ValueError(
            f"chat_kwargs may not override reserved payload keys: {clobbered}; "
            "pass the prompt/adapter via the positional arguments instead"
        )
    return {
        "messages": [{"role": "user", "content": prompt}],
        "custom_inputs": {"adapter": adapter},
        **chat_kwargs,
    }


def query(
    endpoint_name: str,
    adapter: str,
    prompt: str,
    *,
    client: EndpointInvoker | None = None,
    profile: str | None = None,
    **chat_kwargs: Any,
) -> QueryResult:
    """Query the multi-LoRA endpoint with an ``{adapter, prompt}`` pair.

    Args:
        endpoint_name: Name of the Databricks Model Serving endpoint.
        adapter: Logical adapter (LoRA) name to route to, sent as
            ``custom_inputs.adapter``.
        prompt: The user prompt; wrapped into a single ``user`` chat message.
        client: Injectable endpoint invoker. Defaults to a Databricks SDK invoker
            (OAuth via ``profile``; never a PAT).
        profile: Databricks config profile for the default invoker's OAuth. If
            ``None``, the SDK's default auth resolution is used.
        **chat_kwargs: Extra chat parameters passed through verbatim (e.g.
            ``temperature``, ``max_tokens``, ``stream``).

    Returns:
        A :class:`QueryResult` with the assistant ``text`` and the ``raw`` response.
    """
    # Build (and validate) the payload first, so a bad kwarg fails before we
    # construct a workspace client / attempt any auth.
    payload = build_payload(adapter, prompt, **chat_kwargs)
    invoker = client if client is not None else _default_invoker(profile)
    raw = invoker(endpoint_name, payload)
    return QueryResult(text=_extract_text(raw), raw=raw)


def _extract_text(raw: Mapping[str, Any]) -> str:
    """Pull the assistant reply text out of an ``llm/v1/chat`` response.

    Tolerant of shape: returns the first choice's ``message.content`` (or ``text``),
    falling back to ``""`` when the response does not look like a chat completion.
    """
    if not isinstance(raw, Mapping):
        return ""
    choices = raw.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, Mapping):
        return ""
    message = first.get("message")
    if isinstance(message, Mapping):
        content = message.get("content")
        if isinstance(content, str):
            return content
    text = first.get("text")
    return text if isinstance(text, str) else ""


def _default_invoker(profile: str | None) -> EndpointInvoker:
    """Build the default OAuth invoker backed by the Databricks SDK.

    Posts the payload to ``/serving-endpoints/{name}/invocations`` using the
    workspace client's authenticated API client (OAuth via ``profile``).
    """
    from databricks.sdk import WorkspaceClient

    workspace = WorkspaceClient(profile=profile)

    def _invoke(endpoint_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = workspace.api_client.do(
            "POST",
            f"/serving-endpoints/{endpoint_name}/invocations",
            body=payload,
        )
        if not isinstance(response, dict):
            raise TypeError(
                f"unexpected non-JSON response from endpoint {endpoint_name!r}: "
                f"{type(response).__name__}"
            )
        return response

    return _invoke
