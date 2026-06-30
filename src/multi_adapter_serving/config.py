"""Configuration schema and loader for multi-adapter (multi-LoRA) serving.

This module defines the **shared configuration contract** that every track in the
project depends on. The pydantic models below mirror, field for field, the shape
of ``config.yaml`` (see ``config.example.yaml`` at the repo root). Do not rename
fields without coordinating across tracks.

The public entry point is :func:`load_config`, which reads a YAML file and returns
a validated :class:`Config`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "Adapter",
    "BaseModelSpec",
    "Config",
    "DatabricksConfig",
    "ServingConfig",
    "SourceType",
    "UnityCatalogConfig",
    "VLLMConfig",
    "load_config",
]

# Allowed values for ``source_type``. Kept as a module-level alias so other tracks
# can import the literal type rather than re-declaring it.
SourceType = Literal["huggingface", "local"]


class BaseModelSpec(BaseModel):
    """Reference to the base model weights."""

    model_config = ConfigDict(extra="forbid")

    ref: str = Field(
        ...,
        min_length=1,
        description="HF repo id or local/UC-Volume path to the base model.",
    )


class Adapter(BaseModel):
    """A single LoRA adapter."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        ...,
        min_length=1,
        description="Logical name used in the {adapter} request field; must be unique.",
    )
    ref: str = Field(
        ...,
        min_length=1,
        description="HF repo id or local/UC-Volume path to the adapter weights.",
    )

    @field_validator("name", "ref")
    @classmethod
    def _strip_non_empty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must be a non-empty string")
        return stripped


class UnityCatalogConfig(BaseModel):
    """Unity Catalog destination for staged weights and the registered model."""

    # ``schema`` shadows a deprecated ``BaseModel`` attribute, so it is exposed under
    # the python attribute ``db_schema`` while keeping the YAML key ``schema``.
    # ``model_name`` collides with pydantic's protected ``model_`` namespace, which we
    # disable here.
    model_config = ConfigDict(
        extra="forbid",
        protected_namespaces=(),
        populate_by_name=True,
    )

    catalog: str = Field(..., min_length=1)
    db_schema: str = Field(..., min_length=1, alias="schema")
    model_name: str = Field(..., min_length=1)
    volume: str = Field(..., min_length=1)


class ServingConfig(BaseModel):
    """Databricks Model Serving endpoint settings."""

    model_config = ConfigDict(extra="forbid")

    endpoint_name: str = Field(..., min_length=1)
    workload_type: str = Field(..., min_length=1)
    workload_size: str = Field(..., min_length=1)
    scale_to_zero: bool


class VLLMConfig(BaseModel):
    """vLLM engine settings relevant to multi-LoRA serving."""

    model_config = ConfigDict(extra="forbid")

    max_loras: int = Field(..., gt=0)
    max_lora_rank: int = Field(..., gt=0)
    max_model_len: int = Field(..., gt=0)


class DatabricksConfig(BaseModel):
    """Databricks authentication settings (OAuth via a config profile)."""

    model_config = ConfigDict(extra="forbid")

    profile: str = Field(..., min_length=1)


class Config(BaseModel):
    """Top-level configuration object — the shared contract across tracks."""

    model_config = ConfigDict(extra="forbid")

    source_type: SourceType
    base_model: BaseModelSpec
    adapters: list[Adapter] = Field(..., min_length=1)
    unity_catalog: UnityCatalogConfig
    serving: ServingConfig
    vllm: VLLMConfig
    databricks: DatabricksConfig

    @field_validator("adapters")
    @classmethod
    def _unique_adapter_names(cls, adapters: list[Adapter]) -> list[Adapter]:
        names = [a.name for a in adapters]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"adapter names must be unique; duplicates: {duplicates}")
        return adapters


def load_config(path: str | Path) -> Config:
    """Load and validate a config YAML file.

    Args:
        path: Path to a ``config.yaml`` file.

    Returns:
        A validated :class:`Config`.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file does not contain a YAML mapping.
        pydantic.ValidationError: If the contents violate the schema.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if not isinstance(raw, dict):
        kind = type(raw).__name__
        raise ValueError(f"Config root must be a YAML mapping, got {kind}")

    return Config.model_validate(raw)
