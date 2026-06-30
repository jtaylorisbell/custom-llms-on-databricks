"""Unit tests for the per-adapter smoke test (query function mocked)."""

from __future__ import annotations

from typing import Any

from multi_adapter_serving.client import QueryResult
from multi_adapter_serving.config import Config
from multi_adapter_serving.smoke_test import DEFAULT_PROMPT, smoke_test


def make_config(**overrides: Any) -> Config:
    data: dict[str, Any] = {
        "source_type": "huggingface",
        "base_model": {"ref": "org/base-model"},
        "adapters": [
            {"name": "ner", "ref": "org/ner"},
            {"name": "code", "ref": "org/code"},
            {"name": "sql", "ref": "org/sql"},
        ],
        "unity_catalog": {
            "catalog": "classic_stable_h0vpq7_catalog",
            "schema": "default",
            "model_name": "multi_lora_vllm",
            "volume": "weights",
        },
        "serving": {
            "endpoint_name": "multi-lora-vllm",
            "workload_type": "GPU_MEDIUM",
            "workload_size": "Small",
            "scale_to_zero": True,
        },
        "vllm": {"max_loras": 4, "max_lora_rank": 16, "max_model_len": 8192},
        "databricks": {"profile": "fevm-aws"},
    }
    data.update(overrides)
    return Config.model_validate(data)


def test_iterates_every_adapter_and_aggregates_with_a_failure() -> None:
    calls: list[tuple[str, str, str]] = []

    def fake_query_fn(endpoint_name: str, adapter: str, prompt: str) -> QueryResult:
        calls.append((endpoint_name, adapter, prompt))
        if adapter == "code":
            raise RuntimeError("upstream 500")
        return QueryResult(text=f"reply from {adapter}", raw={"ok": True})

    results = smoke_test(make_config(), query_fn=fake_query_fn, prompt="ping")

    # Every adapter was queried, against the configured endpoint, with the prompt.
    assert [c[1] for c in calls] == ["ner", "code", "sql"]
    assert {c[0] for c in calls} == {"multi-lora-vllm"}
    assert {c[2] for c in calls} == {"ping"}

    # Result aggregates pass/fail for all adapters; one failing adapter is captured.
    assert set(results) == {"ner", "code", "sql"}
    assert results["ner"] == {"ok": True, "snippet": "reply from ner"}
    assert results["sql"] == {"ok": True, "snippet": "reply from sql"}
    assert results["code"]["ok"] is False
    assert "upstream 500" in results["code"]["error"]


def test_default_prompt_used_when_none() -> None:
    seen: list[str] = []

    def fake_query_fn(endpoint_name: str, adapter: str, prompt: str) -> QueryResult:
        seen.append(prompt)
        return QueryResult(text="hi", raw={})

    smoke_test(make_config(), query_fn=fake_query_fn)

    assert seen and all(p == DEFAULT_PROMPT for p in seen)


def test_snippet_is_collapsed_and_truncated() -> None:
    long_reply = "word " * 100  # 500 chars with whitespace/newlines collapsed

    def fake_query_fn(endpoint_name: str, adapter: str, prompt: str) -> QueryResult:
        return QueryResult(text="  multi\n  line\t reply  ", raw={}) if adapter == "ner" \
            else QueryResult(text=long_reply, raw={})

    results = smoke_test(make_config(), query_fn=fake_query_fn)

    # Whitespace collapsed to single spaces, trimmed.
    assert results["ner"]["snippet"] == "multi line reply"
    # Long replies are truncated with an ellipsis.
    assert results["code"]["snippet"].endswith("...")
    assert len(results["code"]["snippet"]) <= 203  # 200 chars + "..."
