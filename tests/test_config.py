"""Unit tests for the configuration schema and loader."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from multi_adapter_serving.config import Config, load_config


def valid_config_dict() -> dict[str, Any]:
    """Return a fresh, fully valid config dict for mutation in tests."""
    return {
        "source_type": "huggingface",
        "base_model": {"ref": "org/base-model"},
        "adapters": [
            {"name": "adapter_a", "ref": "org/adapter-a"},
            {"name": "adapter_b", "ref": "org/adapter-b"},
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
        "vllm": {
            "max_loras": 4,
            "max_lora_rank": 16,
            "max_model_len": 8192,
        },
        "databricks": {"profile": "fevm-aws"},
    }


def write_yaml(tmp_path: Path, data: Any, name: str = "config.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_load_valid_config(tmp_path: Path) -> None:
    path = write_yaml(tmp_path, valid_config_dict())
    config = load_config(path)

    assert isinstance(config, Config)
    assert config.source_type == "huggingface"
    assert config.base_model.ref == "org/base-model"
    assert [a.name for a in config.adapters] == ["adapter_a", "adapter_b"]
    assert [a.ref for a in config.adapters] == ["org/adapter-a", "org/adapter-b"]
    # `schema` YAML key is exposed under the `db_schema` python attribute.
    assert config.unity_catalog.catalog == "classic_stable_h0vpq7_catalog"
    assert config.unity_catalog.db_schema == "default"
    assert config.unity_catalog.model_name == "multi_lora_vllm"
    assert config.unity_catalog.volume == "weights"
    assert config.serving.endpoint_name == "multi-lora-vllm"
    assert config.serving.scale_to_zero is True
    assert config.vllm.max_loras == 4
    assert config.vllm.max_lora_rank == 16
    assert config.vllm.max_model_len == 8192
    assert config.databricks.profile == "fevm-aws"


def test_load_valid_local_source(tmp_path: Path) -> None:
    data = valid_config_dict()
    data["source_type"] = "local"
    config = load_config(write_yaml(tmp_path, data))
    assert config.source_type == "local"


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "does-not-exist.yaml")


def test_non_mapping_root_raises(tmp_path: Path) -> None:
    path = write_yaml(tmp_path, ["not", "a", "mapping"])
    with pytest.raises(ValueError, match="must be a YAML mapping"):
        load_config(path)


def test_invalid_source_type(tmp_path: Path) -> None:
    data = valid_config_dict()
    data["source_type"] = "s3"
    with pytest.raises(ValidationError):
        load_config(write_yaml(tmp_path, data))


def test_duplicate_adapter_names(tmp_path: Path) -> None:
    data = valid_config_dict()
    data["adapters"] = [
        {"name": "dup", "ref": "org/a"},
        {"name": "dup", "ref": "org/b"},
    ]
    with pytest.raises(ValidationError, match="unique"):
        load_config(write_yaml(tmp_path, data))


def test_empty_adapters_list(tmp_path: Path) -> None:
    data = valid_config_dict()
    data["adapters"] = []
    with pytest.raises(ValidationError):
        load_config(write_yaml(tmp_path, data))


def test_empty_adapter_name(tmp_path: Path) -> None:
    data = valid_config_dict()
    data["adapters"][0]["name"] = "   "
    with pytest.raises(ValidationError):
        load_config(write_yaml(tmp_path, data))


def test_max_loras_must_be_positive(tmp_path: Path) -> None:
    data = valid_config_dict()
    data["vllm"]["max_loras"] = 0
    with pytest.raises(ValidationError):
        load_config(write_yaml(tmp_path, data))


def test_max_lora_rank_must_be_positive(tmp_path: Path) -> None:
    data = valid_config_dict()
    data["vllm"]["max_lora_rank"] = -1
    with pytest.raises(ValidationError):
        load_config(write_yaml(tmp_path, data))


def test_missing_required_section(tmp_path: Path) -> None:
    data = valid_config_dict()
    del data["base_model"]
    with pytest.raises(ValidationError):
        load_config(write_yaml(tmp_path, data))


def test_unknown_field_rejected(tmp_path: Path) -> None:
    data = valid_config_dict()
    data["unexpected"] = "boom"
    with pytest.raises(ValidationError):
        load_config(write_yaml(tmp_path, data))


def test_empty_base_model_ref(tmp_path: Path) -> None:
    data = valid_config_dict()
    data["base_model"]["ref"] = ""
    with pytest.raises(ValidationError):
        load_config(write_yaml(tmp_path, data))


def test_adapter_name_and_ref_are_stripped(tmp_path: Path) -> None:
    data = copy.deepcopy(valid_config_dict())
    data["adapters"][0]["name"] = "  adapter_a  "
    data["adapters"][0]["ref"] = "  org/adapter-a  "
    config = load_config(write_yaml(tmp_path, data))
    assert config.adapters[0].name == "adapter_a"
    assert config.adapters[0].ref == "org/adapter-a"


def test_vllm_dtype_defaults_to_half(tmp_path: Path) -> None:
    data = valid_config_dict()
    assert "dtype" not in data["vllm"]
    config = load_config(write_yaml(tmp_path, data))
    assert config.vllm.dtype == "half"


def test_vllm_dtype_explicit_valid(tmp_path: Path) -> None:
    data = valid_config_dict()
    data["vllm"]["dtype"] = "bfloat16"
    config = load_config(write_yaml(tmp_path, data))
    assert config.vllm.dtype == "bfloat16"


def test_vllm_dtype_invalid_raises(tmp_path: Path) -> None:
    data = valid_config_dict()
    data["vllm"]["dtype"] = "int8"
    with pytest.raises(ValidationError):
        load_config(write_yaml(tmp_path, data))


def test_shipped_example_config_loads() -> None:
    example = Path(__file__).resolve().parents[1] / "config.example.yaml"
    config = load_config(example)
    assert isinstance(config, Config)
    assert config.source_type == "huggingface"
    assert config.base_model.ref == "mistralai/Mistral-7B-Instruct-v0.2"
    assert [a.name for a in config.adapters] == ["ner", "code"]
    assert config.vllm.dtype == "half"
