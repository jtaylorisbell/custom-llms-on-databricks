"""Smoke-test every adapter on a deployed multi-LoRA endpoint.

:func:`smoke_test` walks every adapter declared in the config, sends each a single
prompt through the endpoint, and records whether the call succeeded plus a short
snippet of the reply (or the error). It is deliberately forgiving: one adapter
failing does not abort the run, so the returned report shows the status of *all*
adapters at once.

The query function is injectable (``query_fn``) so the run can be unit tested with no
Databricks workspace. The default delegates to :func:`multi_adapter_serving.client.query`.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

from multi_adapter_serving.client import QueryResult
from multi_adapter_serving.config import Config

__all__ = [
    "DEFAULT_PROMPT",
    "QueryFn",
    "smoke_test",
]

logger = logging.getLogger(__name__)

# A neutral prompt that any chat adapter should be able to answer.
DEFAULT_PROMPT = "Hello! In one short sentence, what can you help with?"

# Number of characters of reply text retained in the per-adapter snippet.
_SNIPPET_CHARS = 200


class QueryFn(Protocol):
    """Callable that queries one adapter on the endpoint and returns its reply."""

    def __call__(self, endpoint_name: str, adapter: str, prompt: str) -> QueryResult: ...


def smoke_test(
    config: Config,
    *,
    query_fn: QueryFn | None = None,
    prompt: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Query every adapter once and report pass/fail per adapter.

    Args:
        config: Validated configuration (supplies the endpoint name and the full
            list of adapters to exercise).
        query_fn: Injectable per-adapter query callable. Defaults to one backed by
            :func:`multi_adapter_serving.client.query` (OAuth via the config profile).
        prompt: Prompt to send to every adapter. Defaults to :data:`DEFAULT_PROMPT`.

    Returns:
        A mapping of adapter name to a result dict. On success:
        ``{"ok": True, "snippet": <reply snippet>}``; on failure:
        ``{"ok": False, "error": <error message>}``.
    """
    run = query_fn if query_fn is not None else _default_query_fn(config)
    text_prompt = prompt if prompt is not None else DEFAULT_PROMPT
    endpoint_name = config.serving.endpoint_name

    results: dict[str, dict[str, Any]] = {}
    for adapter in config.adapters:
        name = adapter.name
        try:
            result = run(endpoint_name, name, text_prompt)
        except Exception as exc:  # report any failure, never abort the whole run.
            logger.warning("Smoke test FAILED for adapter %r: %s", name, exc)
            results[name] = {"ok": False, "error": str(exc)}
            continue
        snippet = _snippet(result)
        logger.info("Smoke test PASSED for adapter %r: %s", name, snippet)
        results[name] = {"ok": True, "snippet": snippet}
    return results


def _snippet(result: QueryResult) -> str:
    """A short, single-line snippet of the reply text for the report."""
    text = getattr(result, "text", "") or ""
    collapsed = " ".join(str(text).split())
    if len(collapsed) <= _SNIPPET_CHARS:
        return collapsed
    return collapsed[:_SNIPPET_CHARS].rstrip() + "..."


def _default_query_fn(config: Config) -> QueryFn:
    """Build the default query function backed by the OAuth client."""
    from multi_adapter_serving.client import query

    profile = config.databricks.profile

    def _run(endpoint_name: str, adapter: str, prompt: str) -> QueryResult:
        return query(endpoint_name, adapter, prompt, profile=profile)

    return _run
