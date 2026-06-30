"""Unit tests for model authoring + registration (MLflow fully mocked)."""

from __future__ import annotations

import shlex
from typing import Any

import pytest

from multi_adapter_serving import model as model_mod
from multi_adapter_serving.config import Config
from multi_adapter_serving.model import CHAT_TASK, ENV_PACK, MultiLoRAChatModel, log_and_register
from multi_adapter_serving.serving.entrypoint_cmd import build_entrypoint_command

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


class FakeModelInfo:
    def __init__(self, model_uri: str) -> None:
        self.model_uri = model_uri


class FakeModelVersion:
    def __init__(self, version: str) -> None:
        self.version = version


class FakePyfunc:
    def __init__(self) -> None:
        self.log_model_calls: list[dict[str, Any]] = []

    def log_model(self, **kwargs: Any) -> FakeModelInfo:
        self.log_model_calls.append(kwargs)
        return FakeModelInfo("runs:/abc123/model")


class FakeMlflow:
    def __init__(self) -> None:
        self.pyfunc = FakePyfunc()
        self.register_calls: list[dict[str, Any]] = []

    def register_model(self, model_uri: str, name: str, *, env_pack: str) -> FakeModelVersion:
        self.register_calls.append(
            {"model_uri": model_uri, "name": name, "env_pack": env_pack}
        )
        return FakeModelVersion("7")


def test_log_and_register_assembles_metadata_entrypoint_envpack_and_uc_name() -> None:
    fake = FakeMlflow()
    config = make_config()
    manifest = make_manifest()

    name, version = log_and_register(
        config, manifest, mlflow_module=fake, chat_model=object()
    )

    # --- log_model: metadata carries the chat task + the built entrypoint command ---
    assert len(fake.pyfunc.log_model_calls) == 1
    call = fake.pyfunc.log_model_calls[0]
    metadata = call["metadata"]
    assert metadata["task"] == CHAT_TASK == "llm/v1/chat"
    expected_entrypoint = shlex.join(build_entrypoint_command(config, manifest))
    assert metadata["entrypoint"] == expected_entrypoint

    # --- log_model: both proxy modules are attached as code paths ---
    code_paths = call["code_paths"]
    assert any(p.endswith("/serving/proxy.py") for p in code_paths)
    assert any(p.endswith("/serving/proxy_entrypoint.py") for p in code_paths)

    # --- register_model: env_pack + Unity Catalog three-level name ---
    assert len(fake.register_calls) == 1
    reg = fake.register_calls[0]
    assert reg["model_uri"] == "runs:/abc123/model"
    assert reg["env_pack"] == ENV_PACK == "databricks_model_serving"
    assert reg["name"] == "classic_stable_h0vpq7_catalog.default.multi_lora_vllm"

    # --- return value ---
    assert name == "classic_stable_h0vpq7_catalog.default.multi_lora_vllm"
    assert version == 7


def test_default_python_model_is_a_chatmodel_and_default_artifact_name() -> None:
    fake = FakeMlflow()

    log_and_register(make_config(), make_manifest(), mlflow_module=fake)

    call = fake.pyfunc.log_model_calls[0]
    assert isinstance(call["python_model"], MultiLoRAChatModel)
    assert call["name"] == "model"


def test_predict_is_not_the_serving_path() -> None:
    chat_model = MultiLoRAChatModel()
    with pytest.raises(NotImplementedError):
        chat_model.predict(None, [], None)  # type: ignore[arg-type]


def test_uc_name_uses_db_schema_alias() -> None:
    fake = FakeMlflow()
    config = make_config(
        unity_catalog={
            "catalog": "cat",
            "schema": "sch",
            "model_name": "mname",
            "volume": "weights",
        }
    )

    name, _ = log_and_register(config, make_manifest(), mlflow_module=fake)

    assert name == "cat.sch.mname"
    assert model_mod.log_and_register is log_and_register  # module import sanity
