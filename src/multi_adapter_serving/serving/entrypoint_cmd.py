"""Build the exact ``entrypoint`` command Databricks Model Serving runs.

The Databricks custom-LLM workflow launches the command stored in the logged
model's ``metadata['entrypoint']`` as the runtime server on port 8080. For
multi-LoRA serving that command starts our thin proxy launcher
(:mod:`~multi_adapter_serving.serving.proxy_entrypoint`), which in turn spawns the
vLLM OpenAI server on a private loopback port and routes adapter selection.

:func:`build_entrypoint_command` constructs that command deterministically from a
validated :class:`~multi_adapter_serving.config.Config` plus the staging manifest
returned by :func:`~multi_adapter_serving.staging.stage_weights`, so it is pure and
fully unit-testable with no Databricks or vLLM access.
"""

from __future__ import annotations

from multi_adapter_serving.config import Config

__all__ = [
    "BASE_SERVED_MODEL_NAME",
    "PROXY_ENTRYPOINT_MODULE",
    "PROXY_PORT",
    "VLLM_HOST",
    "VLLM_PORT",
    "build_entrypoint_command",
]

# The proxy launcher is logged as a flat MLflow code path (``proxy_entrypoint.py``),
# so at serving runtime it is importable as a top-level module via ``python -m``.
PROXY_ENTRYPOINT_MODULE = "proxy_entrypoint"

# Public port Databricks Model Serving requires the runtime server to bind.
PROXY_PORT = 8080

# Loopback host/port the proxy spawns the vLLM OpenAI server on (never exposed).
VLLM_HOST = "127.0.0.1"
VLLM_PORT = 8081

# served-model-name for the base weights; the proxy maps each request's adapter
# onto the OpenAI ``model`` field, so the base name is only a vLLM-side label.
BASE_SERVED_MODEL_NAME = "base"


def build_entrypoint_command(config: Config, manifest: dict[str, str]) -> list[str]:
    """Build the entrypoint command Databricks launches as the runtime server.

    Args:
        config: Validated configuration (supplies ``vllm.max_loras``,
            ``vllm.max_lora_rank``, ``vllm.dtype`` and the adapter names/order).
        manifest: Mapping of ``"base"`` and each adapter name to its staged Unity
            Catalog Volume path (the output of ``stage_weights``).

    Returns:
        The command as a list of argv tokens.

    Raises:
        KeyError: If ``manifest`` is missing the ``"base"`` entry or any adapter
            path declared in ``config.adapters``.
    """
    if "base" not in manifest:
        raise KeyError("manifest must contain a 'base' entry")

    lora_modules: list[str] = []
    for adapter in config.adapters:
        if adapter.name not in manifest:
            raise KeyError(f"manifest is missing a path for adapter {adapter.name!r}")
        # vLLM's --lora-modules accepts space-separated name=path pairs.
        lora_modules.append(f"{adapter.name}={manifest[adapter.name]}")

    command = [
        "python",
        "-u",
        "-m",
        PROXY_ENTRYPOINT_MODULE,
        "--host",
        "0.0.0.0",
        "--port",
        str(PROXY_PORT),
        "--vllm-host",
        VLLM_HOST,
        "--vllm-port",
        str(VLLM_PORT),
        "--base-model",
        manifest["base"],
        "--served-model-name",
        BASE_SERVED_MODEL_NAME,
        "--max-loras",
        str(config.vllm.max_loras),
        "--max-lora-rank",
        str(config.vllm.max_lora_rank),
        "--dtype",
        config.vllm.dtype,
    ]
    if lora_modules:
        command.append("--lora-modules")
        command.extend(lora_modules)
    return command
