"""Runtime serving layer for multi-LoRA vLLM on Databricks Model Serving.

This subpackage holds the pieces that run *inside* the serving container:

* :mod:`~multi_adapter_serving.serving.proxy` — the thin FastAPI proxy that
  translates the Databricks ``llm/v1/chat`` contract into a vLLM LoRA selection.
* :mod:`~multi_adapter_serving.serving.proxy_entrypoint` — the runnable launcher
  that spawns vLLM and serves the proxy on port 8080.
* :mod:`~multi_adapter_serving.serving.entrypoint_cmd` — builds the exact
  ``entrypoint`` command Databricks runs (logged in the model metadata).
"""

from __future__ import annotations
