"""MLflow model authoring + Unity Catalog registration for multi-LoRA serving.

Follows the Databricks custom-LLM workflow: we log an MLflow pyfunc model carrying
``metadata={'task': 'llm/v1/chat', 'entrypoint': <command>}`` and the standard
``llm/v1/chat`` model signature. At serving time Databricks launches that
``entrypoint`` command as the runtime server on port 8080 (our proxy in front of
vLLM) — pyfunc ``predict`` is **not** the request path. The proxy modules are
attached as code paths, and the model is registered to Unity Catalog with the
express-deployment ``env_pack``.

Why a plain ``PythonModel`` instead of MLflow's ``ChatModel`` base class: logging a
``ChatModel`` subclass *unconditionally* runs ``predict()`` against a default chat
input example to validate the output (``mlflow.pyfunc: "Predicting on input example
to validate output"``). Our ``predict`` intentionally raises — serving is the vLLM
entrypoint, not pyfunc — so that validation aborts the register step. We therefore
log a plain ``PythonModel`` and set the *same* ``llm/v1/chat`` signature and ``task``
metadata explicitly. That yields a byte-for-byte-equivalent serving contract (chat
signature + task + entrypoint + env_pack) with no ``predict()`` invocation at log
time. See :data:`CHAT_SIGNATURE`.

All MLflow/registry calls are injectable (``mlflow_module``) so authoring can be unit
tested with no Databricks or MLflow tracking server.
"""

from __future__ import annotations

import shlex
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

# env_pack that enables express deployments required for custom LLM serving.
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


def log_and_register(
    config: Config,
    manifest: dict[str, str],
    *,
    mlflow_module: Any = None,
    chat_model: Any = None,
    artifact_name: str = DEFAULT_ARTIFACT_NAME,
) -> tuple[str, int]:
    """Log the pyfunc model and register it to Unity Catalog.

    Logs the model with the explicit ``llm/v1/chat`` signature (:data:`CHAT_SIGNATURE`)
    and ``metadata={'task': 'llm/v1/chat', 'entrypoint': <cmd>}`` plus the proxy
    modules as code paths, then registers the result to Unity Catalog as
    ``{catalog}.{schema}.{model_name}`` with the express-deployment ``env_pack``.

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

    Returns:
        ``(registered_model_full_name, version)``.
    """
    mlflow = mlflow_module if mlflow_module is not None else _default_mlflow()
    model = chat_model if chat_model is not None else MultiLoRAChatModel()

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
    model_version = mlflow.register_model(
        model_info.model_uri,
        registered_name,
        env_pack=ENV_PACK,
    )
    return registered_name, int(model_version.version)
