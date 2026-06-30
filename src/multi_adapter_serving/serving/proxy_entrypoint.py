"""Runnable launcher executed by Databricks Model Serving as the runtime server.

This is the process the logged ``metadata['entrypoint']`` command starts (see
:func:`~multi_adapter_serving.serving.entrypoint_cmd.build_entrypoint_command`). It:

1. spawns the vLLM OpenAI server on a private loopback port (with all LoRA modules
   loaded), then
2. polls the vLLM ``/health`` endpoint until it is ready, then
3. serves the thin proxy (:func:`create_proxy_app`) on the public port 8080 via
   uvicorn, while a background monitor watches the vLLM subprocess and shuts the
   proxy down (with a non-zero exit) if vLLM dies, and
4. tears the vLLM subprocess down cleanly on exit.

The proxy is logged as flat MLflow code paths (``proxy.py`` + ``proxy_entrypoint.py``),
so ``create_proxy_app`` is imported from the installed package when available and
from the flat ``proxy`` module at serving runtime.
"""

from __future__ import annotations

import argparse
import logging
import shlex
import subprocess
import sys
import threading
import time
from collections.abc import Callable

import httpx
import uvicorn

try:
    from multi_adapter_serving.serving.proxy import create_proxy_app
except ImportError:  # pragma: no cover - flat code-path import at serving runtime
    import importlib

    create_proxy_app = importlib.import_module("proxy").create_proxy_app

__all__ = [
    "build_vllm_command",
    "main",
    "monitor_vllm",
    "parse_adapter_names",
    "parse_args",
    "wait_for_vllm_ready",
]

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the proxy-launcher command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Launch the multi-LoRA proxy in front of a vLLM OpenAI server.",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Proxy bind host.")
    parser.add_argument("--port", type=int, default=8080, help="Proxy bind port.")
    parser.add_argument("--vllm-host", default="127.0.0.1", help="vLLM bind host.")
    parser.add_argument("--vllm-port", type=int, default=8081, help="vLLM bind port.")
    parser.add_argument("--base-model", required=True, help="Base model path/ref.")
    parser.add_argument(
        "--served-model-name",
        default="base",
        help="vLLM served-model-name for the base weights.",
    )
    parser.add_argument(
        "--lora-modules",
        nargs="*",
        default=[],
        metavar="NAME=PATH",
        help="Space-separated name=path LoRA modules.",
    )
    parser.add_argument("--max-loras", type=int, required=True, help="vLLM --max-loras.")
    parser.add_argument(
        "--max-lora-rank", type=int, required=True, help="vLLM --max-lora-rank."
    )
    parser.add_argument("--dtype", default="half", help="vLLM model dtype.")
    return parser.parse_args(argv)


def parse_adapter_names(lora_modules: list[str]) -> list[str]:
    """Extract the adapter names (the ``name`` of each ``name=path`` pair)."""
    names: list[str] = []
    for module in lora_modules:
        name, _, _ = module.partition("=")
        names.append(name)
    return names


def build_vllm_command(
    *,
    vllm_host: str,
    vllm_port: int,
    base_model: str,
    served_model_name: str,
    lora_modules: list[str],
    max_loras: int,
    max_lora_rank: int,
    dtype: str,
    python_executable: str = sys.executable,
) -> list[str]:
    """Build the ``vllm.entrypoints.openai.api_server`` subprocess command."""
    command = [
        python_executable,
        "-u",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--host",
        vllm_host,
        "--port",
        str(vllm_port),
        "--model",
        base_model,
        "--served-model-name",
        served_model_name,
        "--enable-lora",
        "--max-loras",
        str(max_loras),
        "--max-lora-rank",
        str(max_lora_rank),
        "--dtype",
        dtype,
    ]
    if lora_modules:
        command.append("--lora-modules")
        command.extend(lora_modules)
    return command


def wait_for_vllm_ready(
    vllm_base_url: str,
    *,
    process: subprocess.Popen[bytes],
    timeout_s: float = 600.0,
    interval_s: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], float] = time.monotonic,
) -> None:
    """Block until the upstream vLLM ``/health`` returns 200, or fail loudly.

    Raises:
        RuntimeError: If the vLLM subprocess exits before becoming ready.
        TimeoutError: If readiness is not reached within ``timeout_s``.
    """
    health_url = f"{vllm_base_url.rstrip('/')}/health"
    deadline = now() + timeout_s
    while now() < deadline:
        exit_code = process.poll()
        if exit_code is not None:
            raise RuntimeError(f"vLLM server exited early with code {exit_code}")
        try:
            response = httpx.get(health_url, timeout=5.0)
            if response.status_code == 200:
                return
        except httpx.HTTPError:
            pass  # not up yet; keep polling
        sleep(interval_s)
    raise TimeoutError(f"vLLM did not become ready within {timeout_s}s at {health_url}")


def _terminate(process: subprocess.Popen[bytes], *, timeout_s: float = 10.0) -> None:
    """Terminate the vLLM subprocess, escalating to kill if it does not stop."""
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def monitor_vllm(
    process: subprocess.Popen[bytes],
    server: uvicorn.Server,
    *,
    poll_interval_s: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
) -> int | None:
    """Watch the vLLM subprocess; when it exits, ask the proxy server to stop.

    Run in a background thread alongside ``uvicorn.Server.run()``. If vLLM exits
    after readiness, forwarding is permanently broken, so we set
    ``server.should_exit`` to bring the proxy down and return the vLLM exit code —
    the caller then terminates the proxy with a non-zero status so the serving
    platform restarts the container.

    Returns:
        The vLLM exit code once it exits, or ``None`` if the loop ended because the
        server was already shutting down for another reason (e.g. a signal).
    """
    while not server.should_exit:
        exit_code = process.poll()
        if exit_code is not None:
            logger.error(
                "vLLM subprocess exited with code %s after readiness; "
                "shutting down proxy so the platform restarts it",
                exit_code,
            )
            server.should_exit = True
            return exit_code
        sleep(poll_interval_s)
    return None


def main(argv: list[str] | None = None) -> None:
    """Spawn vLLM, wait for readiness, then serve the proxy on the public port.

    A background monitor watches the vLLM subprocess for the lifetime of the proxy;
    if vLLM dies the proxy is shut down and the process exits non-zero.
    """
    logging.basicConfig(level=logging.INFO)
    args = parse_args(argv)

    adapter_names = parse_adapter_names(args.lora_modules)
    vllm_command = build_vllm_command(
        vllm_host=args.vllm_host,
        vllm_port=args.vllm_port,
        base_model=args.base_model,
        served_model_name=args.served_model_name,
        lora_modules=args.lora_modules,
        max_loras=args.max_loras,
        max_lora_rank=args.max_lora_rank,
        dtype=args.dtype,
    )
    vllm_base_url = f"http://{args.vllm_host}:{args.vllm_port}"

    # shlex.join for the log only; Popen still receives the argv list (shell-safe).
    logger.info("Starting vLLM OpenAI server: %s", shlex.join(vllm_command))
    process = subprocess.Popen(vllm_command)
    crashed: list[int | None] = []
    try:
        wait_for_vllm_ready(vllm_base_url, process=process)
        logger.info("vLLM ready; serving proxy on %s:%s", args.host, args.port)
        app = create_proxy_app(adapter_names, vllm_base_url)
        server = uvicorn.Server(uvicorn.Config(app, host=args.host, port=args.port))

        def _watch() -> None:
            crashed.append(monitor_vllm(process, server))

        monitor = threading.Thread(target=_watch, name="vllm-monitor", daemon=True)
        monitor.start()
        server.run()  # blocks until should_exit (monitor on crash, or a signal)
        monitor.join(timeout=5.0)
    finally:
        _terminate(process)

    # If the monitor observed vLLM exiting on its own, fail so the serving platform
    # restarts the container (force non-zero even if vLLM happened to exit 0).
    crash_code = crashed[0] if crashed else None
    if crash_code is not None:
        raise SystemExit(crash_code or 1)


if __name__ == "__main__":  # pragma: no cover
    main()
