"""MLflow ChatModel authoring + Unity Catalog registration for multi-LoRA serving.

Follows the Databricks custom-LLM workflow: we log an MLflow ``ChatModel`` carrying
``metadata={'task': 'llm/v1/chat', 'entrypoint': <command>}``. At serving time
Databricks launches that ``entrypoint`` command as the runtime server on port 8080
(our proxy in front of vLLM) — pyfunc ``predict`` is **not** the request path. The
proxy modules are attached as code paths, and the model is registered to Unity
Catalog with the express-deployment ``env_pack``.

All MLflow/registry calls are injectable (``mlflow_module``) so authoring can be unit
tested with no Databricks or MLflow tracking server.
"""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

from mlflow.pyfunc.model import ChatModel
from mlflow.types.llm import ChatCompletionResponse, ChatMessage, ChatParams

from multi_adapter_serving.config import Config
from multi_adapter_serving.serving.entrypoint_cmd import build_entrypoint_command

__all__ = [
    "CHAT_TASK",
    "ENV_PACK",
    "MultiLoRAChatModel",
    "log_and_register",
]

# Mandatory task tag for the Databricks chat schema (llm/v1/chat).
CHAT_TASK = "llm/v1/chat"

# env_pack that enables express deployments required for custom LLM serving.
ENV_PACK = "databricks_model_serving"

# Default run-relative artifact name for the logged model.
DEFAULT_ARTIFACT_NAME = "model"


class MultiLoRAChatModel(ChatModel):
    """ChatModel logged for the multi-LoRA endpoint.

    The production serving path is the logged ``metadata['entrypoint']`` command
    (the multi-adapter proxy in front of vLLM), **not** pyfunc ``predict``. This
    class exists to satisfy the ChatModel contract at log time and to carry the
    ``llm/v1/chat`` task; ``predict`` is intentionally never the request path.
    """

    # ChatModel intentionally narrows PythonModel.predict to the chat schema; strict
    # mypy checks the override against the grandparent (PythonModel) too, so ignore
    # the inherited-from-PythonModel mismatch. We match ChatModel.predict exactly.
    def predict(  # type: ignore[override]
        self, context: Any, messages: list[ChatMessage], params: ChatParams
    ) -> ChatCompletionResponse:
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
    """Log the ChatModel and register it to Unity Catalog.

    Logs a ChatModel with ``metadata={'task': 'llm/v1/chat', 'entrypoint': <cmd>}``
    plus the proxy modules as code paths, then registers the result to Unity Catalog
    as ``{catalog}.{schema}.{model_name}`` with the express-deployment ``env_pack``.

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
    )

    uc = config.unity_catalog
    registered_name = f"{uc.catalog}.{uc.db_schema}.{uc.model_name}"
    model_version = mlflow.register_model(
        model_info.model_uri,
        registered_name,
        env_pack=ENV_PACK,
    )
    return registered_name, int(model_version.version)
