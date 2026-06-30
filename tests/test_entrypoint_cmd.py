"""Unit tests for the entrypoint-command builder (pure, no Databricks/vLLM)."""

from __future__ import annotations

from typing import Any

import pytest

from multi_adapter_serving.config import Config
from multi_adapter_serving.serving.entrypoint_cmd import (
    BASE_SERVED_MODEL_NAME,
    PROXY_ENTRYPOINT_MODULE,
    PROXY_PORT,
    VLLM_HOST,
    VLLM_PORT,
    build_entrypoint_command,
)

UC_PREFIX = "/Volumes/classic_stable_h0vpq7_catalog/default/weights/multi_lora_vllm"


def make_config(**overrides: Any) -> Config:
    data: dict[str, Any] = {
        "source_type": "huggingface",
        "base_model": {"ref": "org/base-model"},
        "adapters": [
            {"name": "ner", "ref": "org/ner"},
            {"name": "code", "ref": "org/code"},
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
        "vllm": {"max_loras": 2, "max_lora_rank": 16, "max_model_len": 8192, "dtype": "half"},
        "databricks": {"profile": "fevm-aws"},
    }
    data.update(overrides)
    return Config.model_validate(data)


def make_manifest() -> dict[str, str]:
    return {
        "base": f"{UC_PREFIX}/base",
        "ner": f"{UC_PREFIX}/adapters/ner",
        "code": f"{UC_PREFIX}/adapters/code",
    }


def _value_after(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def test_command_runs_the_proxy_launcher_module() -> None:
    command = build_entrypoint_command(make_config(), make_manifest())
    assert command[:4] == ["python", "-u", "-m", PROXY_ENTRYPOINT_MODULE]


def test_hosts_and_ports() -> None:
    command = build_entrypoint_command(make_config(), make_manifest())
    # Public proxy binds 0.0.0.0:8080; vLLM stays on loopback.
    assert _value_after(command, "--host") == "0.0.0.0"
    assert _value_after(command, "--port") == str(PROXY_PORT) == "8080"
    assert _value_after(command, "--vllm-host") == VLLM_HOST == "127.0.0.1"
    assert _value_after(command, "--vllm-port") == str(VLLM_PORT) == "8081"


def test_base_model_path_from_manifest() -> None:
    manifest = make_manifest()
    command = build_entrypoint_command(make_config(), manifest)
    assert _value_after(command, "--base-model") == manifest["base"]
    assert _value_after(command, "--served-model-name") == BASE_SERVED_MODEL_NAME == "base"


def test_vllm_knobs_come_from_config() -> None:
    config = make_config(
        vllm={"max_loras": 4, "max_lora_rank": 32, "max_model_len": 8192, "dtype": "bfloat16"}
    )
    command = build_entrypoint_command(config, make_manifest())
    assert _value_after(command, "--max-loras") == "4"
    assert _value_after(command, "--max-lora-rank") == "32"
    assert _value_after(command, "--dtype") == "bfloat16"


def test_lora_modules_are_name_equals_ucpath_pairs_in_config_order() -> None:
    manifest = make_manifest()
    command = build_entrypoint_command(make_config(), manifest)
    pairs = command[command.index("--lora-modules") + 1 :]
    assert pairs == [
        f"ner={manifest['ner']}",
        f"code={manifest['code']}",
    ]


def test_missing_base_in_manifest_raises() -> None:
    manifest = make_manifest()
    del manifest["base"]
    with pytest.raises(KeyError, match="base"):
        build_entrypoint_command(make_config(), manifest)


def test_missing_adapter_path_in_manifest_raises() -> None:
    manifest = make_manifest()
    del manifest["ner"]
    with pytest.raises(KeyError, match="ner"):
        build_entrypoint_command(make_config(), manifest)
