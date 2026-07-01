"""MLflow model authoring + Unity Catalog registration for multi-LoRA serving.

Follows the Databricks custom-LLM workflow: we log an MLflow pyfunc model carrying
``metadata={'task': 'llm/v1/chat', 'entrypoint': <command>}`` and the standard
``llm/v1/chat`` model signature. At serving time Databricks launches that
``entrypoint`` command as the runtime server on port 8080 (our proxy in front of
vLLM) — pyfunc ``predict`` is **not** the request path. The proxy modules are
attached as code paths, and the model is registered to Unity Catalog.

The express-deployment ``env_pack`` (``databricks_model_serving``) is applied only
when we are on a Databricks serverless runtime; ``env_pack`` packs the *current*
Python environment as the serving image and can only run there. Off-runtime — this
CLI's design case, a local laptop — it raises ``Failed to parse databricks runtime
version 'None'`` (see :data:`ENV_PACK` and :func:`_in_databricks_serverless_runtime`),
so we omit it. ``env_pack`` is an express (fast cold-start) optimization, not a
serving requirement: without it Model Serving builds the environment at deploy time
from the requirements logged alongside the model (``pyproject.toml`` / ``uv.lock`` /
``.python-version``), and the serving contract (task, entrypoint, chat signature,
code paths) is unchanged.

Why a plain ``PythonModel`` instead of MLflow's ``ChatModel`` base class: logging a
``ChatModel`` subclass *unconditionally* runs ``predict()`` against a default chat
input example to validate the output (``mlflow.pyfunc: "Predicting on input example
to validate output"``). Our ``predict`` intentionally raises — serving is the vLLM
entrypoint, not pyfunc — so that validation aborts the register step. We therefore
log a plain ``PythonModel`` and set the *same* ``llm/v1/chat`` signature and ``task``
metadata explicitly. That yields a byte-for-byte-equivalent serving contract (chat
signature + task + entrypoint + env_pack) with no ``predict()`` invocation at log
time. See :data:`CHAT_SIGNATURE`.

Before logging/registering we point MLflow at the Databricks workspace + Unity Catalog
model registry, scoped to the config's ``databricks.profile`` (OAuth) — the same profile
the deploy step's ``WorkspaceClient`` uses. Without this MLflow silently defaults to a
*local* sqlite tracking+registry store and treats the three-level ``catalog.schema.model``
name as a plain local model name, so the version is created off in a local registry and
``deploy`` cannot find the Unity Catalog model. See :func:`_configure_uc_registry`.

All MLflow/registry calls are injectable (``mlflow_module``, ``uri_configurator``,
``runtime_detector``) so authoring can be unit tested with no Databricks and no network —
tests override the URI configuration to a tmp/local file store and never hit Databricks.
"""

from __future__ import annotations

import shlex
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mlflow.models.signature import ModelSignature
from mlflow.pyfunc.model import PythonModel
from mlflow.types.llm import CHAT_MODEL_INPUT_SCHEMA, CHAT_MODEL_OUTPUT_SCHEMA

from multi_adapter_serving.config import Config
from multi_adapter_serving.serving.entrypoint_cmd import build_entrypoint_command

__all__ = [
    "CHAT_SIGNATURE",
    "CHAT_TASK",
    "ENV_PACK",
    "MultiLoRAChatModel",
    "log_and_register",
]

# Mandatory task tag for the Databricks chat schema (llm/v1/chat).
CHAT_TASK = "llm/v1/chat"

# The exact llm/v1/chat model signature MLflow's ChatModel would set automatically.
# We attach it explicitly (see module docstring) so the endpoint exposes the OpenAI
# chat schema without going through ChatModel's predict-on-input-example validation.
CHAT_SIGNATURE = ModelSignature(CHAT_MODEL_INPUT_SCHEMA, CHAT_MODEL_OUTPUT_SCHEMA)

# env_pack that enables express (fast cold-start) deployments for custom LLM serving.
# Applied only on a Databricks serverless runtime — see log_and_register / the module
# docstring for why it is skipped off-runtime.
ENV_PACK = "databricks_model_serving"

# Default run-relative artifact name for the logged model.
DEFAULT_ARTIFACT_NAME = "model"


class MultiLoRAChatModel(PythonModel):
    """pyfunc model logged for the multi-LoRA ``llm/v1/chat`` endpoint.

    The production serving path is the logged ``metadata['entrypoint']`` command
    (the multi-adapter proxy in front of vLLM), **not** pyfunc ``predict``. This
    class exists only to satisfy the pyfunc contract at log time; the ``llm/v1/chat``
    task and signature are carried as explicit log-model arguments (see
    :func:`log_and_register`). ``predict`` intentionally raises and is never the
    request path.

    ``model_input`` is deliberately typed ``Any`` (no concrete type hint) so MLflow
    does not attempt to infer a signature from the ``predict`` type hints — the
    explicit :data:`CHAT_SIGNATURE` is authoritative.
    """

    def predict(
        self, context: Any, model_input: Any, params: dict[str, Any] | None = None
    ) -> Any:
        raise NotImplementedError(
            "This model is served via the Databricks custom-LLM entrypoint "
            "(vLLM behind the multi-adapter proxy); pyfunc predict() is not the "
            "serving path."
        )


def _proxy_code_paths() -> list[str]:
    """Absolute paths to the proxy modules logged as MLflow code paths."""
    serving_dir = Path(__file__).resolve().parent / "serving"
    return [
        str(serving_dir / "proxy.py"),
        str(serving_dir / "proxy_entrypoint.py"),
    ]


def _default_mlflow() -> Any:
    import mlflow

    return mlflow


def _databricks_uc_uris(profile: str) -> tuple[str, str]:
    """MLflow ``(tracking_uri, registry_uri)`` for a Databricks workspace + Unity Catalog.

    Both are scoped to the Databricks SDK config ``profile`` (OAuth), matching the
    ``WorkspaceClient(profile=...)`` the deploy step uses. The ``databricks-uc://`` scheme
    selects the **Unity Catalog** model registry (three-level ``catalog.schema.model``
    names) rather than a workspace-local / sqlite registry.
    """
    return f"databricks://{profile}", f"databricks-uc://{profile}"


def _configure_uc_registry(mlflow: Any, profile: str) -> None:
    """Point MLflow's tracking + model registry at Databricks + Unity Catalog.

    Must run before ``log_model`` / ``register_model`` so the experiment run logs to the
    Databricks workspace and the model version is created in the Unity Catalog registry.
    Without it MLflow defaults to a local sqlite tracking+registry store and treats the
    three-level UC name as a plain local model name, so ``deploy`` cannot find the version
    at ``{catalog}.{schema}.{model_name}`` (the exact live failure this fixes).
    """
    tracking_uri, registry_uri = _databricks_uc_uris(profile)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(registry_uri)


def _in_databricks_serverless_runtime() -> bool:
    """Whether ``env_pack='databricks_model_serving'`` can pack the serving env here.

    ``env_pack`` tars the *current* Python environment as the serving image, so
    ``pack_env_for_databricks_model_serving`` first calls
    ``DatabricksRuntimeVersion.parse()`` (which raises ``Failed to parse databricks
    runtime version 'None'`` off-runtime), then requires a serverless / *client image*
    runtime and a model logged on Databricks. We gate on MLflow's own
    ``is_in_databricks_serverless_runtime`` so this check matches exactly what the
    packer needs; from a local machine it is ``False`` and ``env_pack`` must be skipped.
    """
    from mlflow.utils.databricks_utils import is_in_databricks_serverless_runtime

    # mlflow's databricks_utils is untyped; coerce its truthy/None result to bool.
    return bool(is_in_databricks_serverless_runtime())  # type: ignore[no-untyped-call]


def log_and_register(
    config: Config,
    manifest: dict[str, str],
    *,
    mlflow_module: Any = None,
    chat_model: Any = None,
    artifact_name: str = DEFAULT_ARTIFACT_NAME,
    runtime_detector: Callable[[], bool] | None = None,
    uri_configurator: Callable[[Any, str], None] | None = None,
) -> tuple[str, int]:
    """Log the pyfunc model and register it to Unity Catalog.

    Before logging, points MLflow's tracking + model registry at the Databricks workspace
    and Unity Catalog, scoped to ``config.databricks.profile`` (see
    :func:`_configure_uc_registry`), so the run logs to the workspace and the model version
    is created in Unity Catalog rather than a local sqlite registry.

    Logs the model with the explicit ``llm/v1/chat`` signature (:data:`CHAT_SIGNATURE`)
    and ``metadata={'task': 'llm/v1/chat', 'entrypoint': <cmd>}`` plus the proxy
    modules as code paths, then registers the result to Unity Catalog as
    ``{catalog}.{schema}.{model_name}``.

    The express-deployment ``env_pack`` (:data:`ENV_PACK`) is passed to
    ``register_model`` **only** when running on a Databricks serverless runtime, where
    packing the current environment can succeed. Off-runtime (the local-CLI default)
    ``env_pack`` is omitted so registration completes and still produces a deployable
    UC version — Model Serving resolves the environment at deploy time from the logged
    requirements. See the module docstring and :func:`_in_databricks_serverless_runtime`.

    No ``input_example`` is passed: an input example (or the ChatModel base class)
    would make MLflow validate the model by calling the intentionally-raising
    ``predict()``. Passing the signature explicitly gives the endpoint the same
    ``llm/v1/chat`` schema without triggering that prediction validation.

    Args:
        config: Validated configuration.
        manifest: Staging manifest mapping ``"base"`` and each adapter name to its
            Unity Catalog path (output of ``stage_weights``).
        mlflow_module: Injectable MLflow module (defaults to the real ``mlflow``);
            must expose ``pyfunc.log_model`` and ``register_model``.
        chat_model: Injectable python_model instance (defaults to
            :class:`MultiLoRAChatModel`).
        artifact_name: Run-relative MLflow model artifact name.
        runtime_detector: Injectable predicate deciding whether ``env_pack`` is applied
            (defaults to :func:`_in_databricks_serverless_runtime`). Overridable in
            tests to exercise both the on-runtime and off-runtime register paths.
        uri_configurator: Injectable callback ``(mlflow_module, profile) -> None`` that
            points MLflow's tracking + registry URIs before logging (defaults to
            :func:`_configure_uc_registry`, which targets Databricks + Unity Catalog).
            Overridable in tests to a tmp/local file store so authoring stays offline and
            never hits Databricks.

    Returns:
        ``(registered_model_full_name, version)``.
    """
    mlflow = mlflow_module if mlflow_module is not None else _default_mlflow()
    model = chat_model if chat_model is not None else MultiLoRAChatModel()
    detect_runtime = (
        runtime_detector if runtime_detector is not None else _in_databricks_serverless_runtime
    )
    configure_uris = uri_configurator if uri_configurator is not None else _configure_uc_registry

    # Point tracking + registry at Databricks + Unity Catalog BEFORE logging/registering,
    # scoped to the config's OAuth profile, so the run logs to the workspace and the model
    # version is created in UC (not a local sqlite registry). Tests override this to a
    # tmp/local store to stay offline.
    configure_uris(mlflow, config.databricks.profile)

    entrypoint = build_entrypoint_command(config, manifest)
    metadata = {
        "task": CHAT_TASK,
        # Stored as a single shell command string, per the custom-LLM doc.
        "entrypoint": shlex.join(entrypoint),
    }

    model_info = mlflow.pyfunc.log_model(
        name=artifact_name,
        python_model=model,
        code_paths=_proxy_code_paths(),
        metadata=metadata,
        signature=CHAT_SIGNATURE,
    )

    uc = config.unity_catalog
    registered_name = f"{uc.catalog}.{uc.db_schema}.{uc.model_name}"
    # Pass env_pack only where it can actually pack the environment (Databricks
    # serverless); off-runtime we omit it so register succeeds locally and Model
    # Serving builds the env at deploy time from the logged requirements.
    register_kwargs: dict[str, Any] = {}
    if detect_runtime():
        register_kwargs["env_pack"] = ENV_PACK
    model_version = mlflow.register_model(
        model_info.model_uri,
        registered_name,
        **register_kwargs,
    )
    return registered_name, int(model_version.version)
