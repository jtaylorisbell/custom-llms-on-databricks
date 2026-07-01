"""Unit tests for model authoring + registration.

Most tests mock MLflow entirely. ``test_log_and_register_logs_to_local_file_store``
deliberately exercises the *real* ``mlflow.pyfunc.log_model`` path against a temp
file store (regression for the live register failure) — see its docstring.
"""

from __future__ import annotations

import shlex
from collections.abc import Callable
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
    ModelVersionResolutionError,
    MultiLoRAChatModel,
    latest_ready_model_version,
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


def offline_configure_tracking(
    user_name: str = "svc-user@example.com",
) -> Callable[[Any, Config], None]:
    """A ``configure_tracking`` override that runs the REAL URI+experiment logic offline.

    Delegates to the production ``_configure_databricks_tracking`` but injects a stub user
    resolver, so no real ``WorkspaceClient`` is built and nothing touches the network. Lets
    the mocked tests exercise the actual tracking setup (URIs + workspace experiment) while
    staying fully offline.
    """

    def _configure(mlflow: Any, config: Config) -> None:
        # Pin the LOCAL branch (runtime_detector -> False) so these mocked tests
        # deterministically exercise the profile-scoped URIs + workspace experiment,
        # regardless of where the suite runs.
        model_mod._configure_databricks_tracking(
            mlflow,
            config,
            user_resolver=lambda _profile: user_name,
            runtime_detector=lambda: False,
        )

    return _configure


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
        # Record what log_and_register points MLflow at before logging (default: Databricks
        # + UC URIs and a workspace experiment). Offline: no network — these just capture
        # what would be configured live.
        self.tracking_uri: str | None = None
        self.registry_uri: str | None = None
        self.experiment_path: str | None = None

    def set_tracking_uri(self, uri: str) -> None:
        self.tracking_uri = uri

    def set_registry_uri(self, uri: str) -> None:
        self.registry_uri = uri

    def set_experiment(
        self, experiment_name: str | None = None, experiment_id: str | None = None
    ) -> None:
        # mlflow.set_experiment(name) create-if-missing; record the workspace path.
        self.experiment_path = experiment_name

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
        # Real URI+experiment setup, but offline (stub user lookup, no WorkspaceClient).
        configure_tracking=offline_configure_tracking(),
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

    log_and_register(
        make_config(),
        make_manifest(),
        mlflow_module=fake,
        configure_tracking=offline_configure_tracking(),
    )

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

    name, _ = log_and_register(
        config,
        make_manifest(),
        mlflow_module=fake,
        configure_tracking=offline_configure_tracking(),
    )

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

    ``configure_tracking`` is overridden to a no-op so ``log_and_register`` does NOT point
    MLflow at Databricks + Unity Catalog and does NOT look up a workspace user / set a
    workspace experiment: this test has already set a tmp file store as the tracking+registry
    URI below (whose Default experiment is used by ``start_run``), and that local store must
    stay in effect to keep the real log path offline. The no-op skips BOTH the URIs and the
    experiment/workspace-user lookup, so no ``WorkspaceClient`` is ever constructed.
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
            # Keep offline: skip URIs AND experiment/user lookup; the tmp file store set
            # above (and its Default experiment) stays in effect.
            configure_tracking=lambda _m, _config: None,
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

    name, version = log_and_register(
        make_config(),
        make_manifest(),
        mlflow_module=fake,
        configure_tracking=offline_configure_tracking(),
    )

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
        configure_tracking=offline_configure_tracking(),
    )

    assert len(fake.register_calls) == 1
    reg = fake.register_calls[0]
    assert reg["kwargs"]["env_pack"] == ENV_PACK == "databricks_model_serving"
    assert name == "classic_stable_h0vpq7_catalog.default.multi_lora_vllm"
    assert version == 7


def test_register_path_targets_uc_and_ensures_workspace_experiment() -> None:
    """Regression: register must target the UC registry AND ensure a workspace experiment.

    Two live failures this guards, both from a local CLI with no notebook context:

    1. UC registry: the model was logged/registered into a LOCAL sqlite tracking+registry
       store (no ``set_registry_uri`` / ``databricks-uc`` anywhere), so the three-level UC
       name became a plain local model name and ``deploy`` failed with ``ResourceDoesNotExist:
       Registered model ... does not exist``. The fix sets the tracking + registry URIs to
       Databricks + UC, scoped to ``config.databricks.profile``.
    2. Workspace experiment: with ``tracking_uri='databricks://<profile>'`` and no notebook,
       MLflow's default-experiment fallback is ``None`` and the backend REJECTS ``log_model``'s
       run creation. The fix ensures a workspace experiment ``/Users/{user}/{model_name}``
       (created if missing) BEFORE logging, deriving the user from the SAME profile.

    Fully offline: ``FakeMlflow`` records ``set_tracking_uri`` / ``set_registry_uri`` /
    ``set_experiment`` and stubs ``register_model``; the workspace-user lookup is stubbed via
    :func:`offline_configure_tracking`, so NO real ``WorkspaceClient`` is constructed and
    nothing touches the network. This exercises the production
    ``_configure_databricks_tracking`` — only the user lookup is mocked.
    """
    fake = FakeMlflow()
    config = make_config()  # profile == "fevm-aws", model_name == "multi_lora_vllm"

    log_and_register(
        config,
        make_manifest(),
        mlflow_module=fake,
        configure_tracking=offline_configure_tracking("dev@example.com"),
    )

    # registry -> Unity Catalog (not a local sqlite/file registry); tracking -> the
    # Databricks workspace; both scoped to the config's OAuth profile.
    assert fake.registry_uri == "databricks-uc://fevm-aws"
    assert fake.registry_uri.startswith("databricks-uc")
    assert fake.tracking_uri == "databricks://fevm-aws"
    # a WORKSPACE experiment is ensured before logging (valid from a local CLI, no notebook).
    assert fake.experiment_path == "/Users/dev@example.com/multi_lora_vllm"
    assert fake.experiment_path.startswith("/")


# --- runtime-aware MLflow auth: local (profile) vs serverless (ambient) -----------


def test_configure_tracking_local_branch_uses_profile_uris_and_experiment() -> None:
    """Off a Databricks runtime, MLflow is scoped to the config's OAuth profile.

    Local URIs carry the ``{profile}`` suffix and a workspace experiment is ensured
    (the local CLI has no notebook context, so the default experiment would be None).
    """
    fake = FakeMlflow()
    config = make_config()  # profile == "fevm-aws", model_name == "multi_lora_vllm"

    model_mod._configure_databricks_tracking(
        fake,
        config,
        user_resolver=lambda _profile: "dev@example.com",
        runtime_detector=lambda: False,
    )

    assert fake.tracking_uri == "databricks://fevm-aws"
    assert fake.registry_uri == "databricks-uc://fevm-aws"
    assert fake.experiment_path == "/Users/dev@example.com/multi_lora_vllm"


def test_configure_tracking_serverless_branch_uses_ambient_uris_without_profile() -> None:
    """On a Databricks serverless runtime (air job), MLflow uses AMBIENT job auth.

    The node has no ``~/.databrickscfg`` and no ``{profile}``, so the URIs are the bare
    schemes (no profile suffix), and NO local-profile experiment is derived — the air job
    already provides the active MLflow run/experiment, and the workspace-user lookup (which
    needs a profile) must never run here.
    """
    fake = FakeMlflow()
    config = make_config()
    resolver_calls: list[str] = []

    def _tracking_resolver(profile: str) -> str:
        resolver_calls.append(profile)
        return "should-not-be-used@example.com"

    model_mod._configure_databricks_tracking(
        fake,
        config,
        user_resolver=_tracking_resolver,
        runtime_detector=lambda: True,
    )

    # Ambient auth: plain schemes, NO "databricks://fevm-aws" / "databricks-uc://fevm-aws".
    assert fake.tracking_uri == "databricks"
    assert fake.registry_uri == "databricks-uc"
    assert "fevm-aws" not in fake.tracking_uri
    assert "fevm-aws" not in fake.registry_uri
    # No local-profile experiment derived, and the workspace-user lookup never ran.
    assert fake.experiment_path is None
    assert resolver_calls == []


# --- deploy-time UC version resolution (registry client fully mocked) --------------


class FakeModelVersionRecord:
    """Minimal stand-in for an MLflow UC ``ModelVersion`` (version + status)."""

    def __init__(self, version: str, status: str) -> None:
        self.version = version
        self.status = status


class FakeRegistryClient:
    """Fake MLflow registry client exposing only ``search_model_versions``."""

    def __init__(self, versions: list[FakeModelVersionRecord]) -> None:
        self._versions = versions
        self.filters: list[str] = []

    def search_model_versions(self, filter_string: str) -> list[FakeModelVersionRecord]:
        self.filters.append(filter_string)
        return list(self._versions)


def test_latest_ready_model_version_picks_highest_ready() -> None:
    client = FakeRegistryClient(
        [
            FakeModelVersionRecord("1", "READY"),
            FakeModelVersionRecord("3", "READY"),
            FakeModelVersionRecord("2", "READY"),
        ]
    )

    name, version = latest_ready_model_version(make_config(), registry_client=client)

    assert name == "classic_stable_h0vpq7_catalog.default.multi_lora_vllm"
    assert version == 3
    # Queried the UC registry by the three-level model name.
    assert client.filters == [
        "name='classic_stable_h0vpq7_catalog.default.multi_lora_vllm'"
    ]


def test_latest_ready_model_version_ignores_non_ready_versions() -> None:
    # A higher-numbered version exists but is not READY -> it must be skipped, and the
    # highest *READY* version is returned instead.
    client = FakeRegistryClient(
        [
            FakeModelVersionRecord("6", "FAILED_REGISTRATION"),
            FakeModelVersionRecord("4", "READY"),
            FakeModelVersionRecord("5", "PENDING_REGISTRATION"),
        ]
    )

    _, version = latest_ready_model_version(make_config(), registry_client=client)

    assert version == 4


def test_latest_ready_model_version_errors_when_none_ready() -> None:
    client = FakeRegistryClient(
        [
            FakeModelVersionRecord("1", "PENDING_REGISTRATION"),
            FakeModelVersionRecord("2", "FAILED_REGISTRATION"),
        ]
    )

    with pytest.raises(ModelVersionResolutionError, match="no READY"):
        latest_ready_model_version(make_config(), registry_client=client)


def test_latest_ready_model_version_errors_when_no_versions_exist() -> None:
    client = FakeRegistryClient([])

    with pytest.raises(ModelVersionResolutionError, match="multi_lora_vllm"):
        latest_ready_model_version(make_config(), registry_client=client)
