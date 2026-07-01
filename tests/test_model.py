"""Unit tests for model authoring + registration.

Most tests mock MLflow entirely. ``test_log_and_register_logs_to_local_file_store``
deliberately exercises the *real* ``mlflow.pyfunc.log_model`` path against a temp
file store (regression for the live register failure) — see its docstring.
"""

from __future__ import annotations

import shlex
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mlflow
import pytest
from mlflow.models import Model

from multi_adapter_serving import model as model_mod
from multi_adapter_serving.config import Config
from multi_adapter_serving.model import (
    CHAT_SIGNATURE,
    CHAT_TASK,
    ENV_PACK,
    MultiLoRAChatModel,
    log_and_register,
)
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
        # Record the URIs log_and_register points MLflow at (default: Databricks + UC).
        # Offline: no network — these just capture what would be configured live.
        self.tracking_uri: str | None = None
        self.registry_uri: str | None = None

    def set_tracking_uri(self, uri: str) -> None:
        self.tracking_uri = uri

    def set_registry_uri(self, uri: str) -> None:
        self.registry_uri = uri

    def register_model(
        self, model_uri: str, name: str, **kwargs: Any
    ) -> FakeModelVersion:
        # Capture kwargs verbatim so tests can assert env_pack is *omitted* off-runtime
        # (real mlflow.register_model defaults env_pack to None when not passed).
        self.register_calls.append({"model_uri": model_uri, "name": name, "kwargs": kwargs})
        return FakeModelVersion("7")


def test_log_and_register_assembles_metadata_entrypoint_envpack_and_uc_name() -> None:
    fake = FakeMlflow()
    config = make_config()
    manifest = make_manifest()

    name, version = log_and_register(
        config,
        manifest,
        mlflow_module=fake,
        chat_model=object(),
        # Simulate a Databricks serverless runtime so env_pack is applied.
        runtime_detector=lambda: True,
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
    assert reg["kwargs"]["env_pack"] == ENV_PACK == "databricks_model_serving"
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


def test_log_and_register_logs_to_local_file_store(tmp_path: Path) -> None:
    """The REAL mlflow log path must succeed offline, without calling predict().

    Regression: the model was previously an MLflow ``ChatModel``, so
    ``mlflow.pyfunc.log_model`` validated it by running the intentionally-raising
    ``predict()`` against a default input example ("Predicting on input example to
    validate output" -> ``NotImplementedError``), which aborted the live register
    step. The fix logs a plain pyfunc model with an explicit ``llm/v1/chat``
    signature and no input example.

    This test drives the real ``mlflow.pyfunc.log_model`` against a temp file store
    (no Databricks / no UC / no network). Only the UC-touching ``register_model``
    call is stubbed, so the test stays fully offline while proving the model logs.
    Before the fix this raised ``NotImplementedError`` at log time; after the fix it
    logs successfully with the chat serving contract intact.

    ``uri_configurator`` is overridden to a no-op so ``log_and_register`` does NOT point
    MLflow at Databricks + Unity Catalog: this test has already set a tmp file store as
    the tracking+registry URI below, and that local store must stay in effect to keep the
    real log path offline.
    """
    mlflow.set_tracking_uri(f"file:{tmp_path / 'mlruns'}")

    register_calls: list[dict[str, Any]] = []

    def fake_register_model(
        model_uri: str, name: str, **kwargs: Any
    ) -> FakeModelVersion:
        register_calls.append({"model_uri": model_uri, "name": name, "kwargs": kwargs})
        return FakeModelVersion("1")

    # Real mlflow.pyfunc (local file store) + a guarded, offline register_model.
    local_mlflow = SimpleNamespace(pyfunc=mlflow.pyfunc, register_model=fake_register_model)

    config = make_config()
    manifest = make_manifest()

    with mlflow.start_run():
        # Simulate the local-CLI case (no Databricks runtime): env_pack must be omitted
        # so this real register path would succeed offline.
        name, version = log_and_register(
            config,
            manifest,
            mlflow_module=local_mlflow,
            runtime_detector=lambda: False,
            # Keep offline: do NOT switch to Databricks/UC; the tmp file store set above stays.
            uri_configurator=lambda _m, _profile: None,
        )

    # --- register stayed offline (guarded) and returned our fake version ---
    assert len(register_calls) == 1
    assert "env_pack" not in register_calls[0]["kwargs"]
    assert name == "classic_stable_h0vpq7_catalog.default.multi_lora_vllm"
    assert version == 1

    # --- the model actually logged; the llm/v1/chat serving contract is intact ---
    logged = Model.load(mlflow.artifacts.download_artifacts(register_calls[0]["model_uri"]))
    assert logged.metadata["task"] == CHAT_TASK == "llm/v1/chat"
    expected_entrypoint = shlex.join(build_entrypoint_command(config, manifest))
    assert logged.metadata["entrypoint"] == expected_entrypoint
    assert logged.signature == CHAT_SIGNATURE


def test_env_pack_omitted_when_not_on_databricks_runtime() -> None:
    """Regression: off a Databricks runtime, register must OMIT env_pack.

    The live failure was ``register_model(..., env_pack='databricks_model_serving')``
    raising ``Failed to parse databricks runtime version 'None'``: env_pack packs the
    *current* Python env as the serving image and needs a Databricks serverless runtime,
    which is absent on a local machine. The fix omits env_pack off-runtime so register
    succeeds locally and still returns a usable UC version (Model Serving builds the env
    at deploy time from the logged requirements). This uses the REAL default detector —
    the test process is not a Databricks runtime — so it guards the exact live scenario.
    """
    fake = FakeMlflow()

    name, version = log_and_register(make_config(), make_manifest(), mlflow_module=fake)

    assert len(fake.register_calls) == 1
    reg = fake.register_calls[0]
    # env_pack is not passed at all -> real mlflow does a plain, serving-valid register.
    assert "env_pack" not in reg["kwargs"]
    # register still yields a usable (UC name, version) reference for the deploy step.
    assert name == "classic_stable_h0vpq7_catalog.default.multi_lora_vllm"
    assert version == 7


def test_env_pack_applied_on_databricks_serverless_runtime() -> None:
    """On a Databricks serverless runtime, register applies the express env_pack."""
    fake = FakeMlflow()

    name, version = log_and_register(
        make_config(),
        make_manifest(),
        mlflow_module=fake,
        runtime_detector=lambda: True,
    )

    assert len(fake.register_calls) == 1
    reg = fake.register_calls[0]
    assert reg["kwargs"]["env_pack"] == ENV_PACK == "databricks_model_serving"
    assert name == "classic_stable_h0vpq7_catalog.default.multi_lora_vllm"
    assert version == 7


def test_register_path_targets_unity_catalog_registry() -> None:
    """Regression: register must target the Unity Catalog registry, not a local one.

    The live failure logged/registered into a LOCAL sqlite tracking+registry store (no
    ``set_registry_uri`` / ``databricks-uc`` anywhere), so the three-level UC name became a
    plain local model name and ``deploy`` failed with ``ResourceDoesNotExist: Registered
    model ... does not exist`` — the version never landed in Unity Catalog. The fix points
    MLflow's tracking + registry at Databricks + UC, scoped to ``config.databricks.profile``,
    BEFORE logging/registering.

    Fully offline: the injected fake records the URIs and stubs ``register_model``; no
    Databricks and no network. This uses the DEFAULT ``uri_configurator`` — the exact live
    code path — asserting it targets ``databricks-uc://<profile>`` and not a local registry.
    """
    fake = FakeMlflow()
    config = make_config()  # databricks.profile == "fevm-aws"

    log_and_register(config, make_manifest(), mlflow_module=fake)

    # Registry -> Unity Catalog (not a local sqlite/file registry); tracking -> the
    # Databricks workspace; both scoped to the config's OAuth profile.
    assert fake.registry_uri == "databricks-uc://fevm-aws"
    assert fake.registry_uri.startswith("databricks-uc")
    assert fake.tracking_uri == "databricks://fevm-aws"
