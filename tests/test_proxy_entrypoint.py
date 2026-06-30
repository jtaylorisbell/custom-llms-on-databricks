"""Unit tests for the proxy launcher helpers (no subprocess, no network)."""

from __future__ import annotations

import httpx
import pytest

from multi_adapter_serving.serving import proxy_entrypoint as pe
from multi_adapter_serving.serving.proxy_entrypoint import (
    build_vllm_command,
    monitor_vllm,
    parse_adapter_names,
    parse_args,
    wait_for_vllm_ready,
)


def _value_after(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


class FakeProcess:
    """Minimal subprocess.Popen stand-in driven by a scripted poll() sequence."""

    def __init__(self, poll_values: list[int | None]) -> None:
        self._poll_values = list(poll_values)
        self.returncode: int | None = None

    def poll(self) -> int | None:
        value = self._poll_values.pop(0) if self._poll_values else None
        self.returncode = value
        return value


def test_parse_adapter_names() -> None:
    assert parse_adapter_names(["ner=/v/ner", "code=/v/code"]) == ["ner", "code"]
    assert parse_adapter_names([]) == []


def test_build_vllm_command_matches_design() -> None:
    command = build_vllm_command(
        vllm_host="127.0.0.1",
        vllm_port=8081,
        base_model="/Volumes/c/s/v/m/base",
        served_model_name="base",
        lora_modules=["ner=/Volumes/c/s/v/m/adapters/ner", "code=/Volumes/c/s/v/m/adapters/code"],
        max_loras=2,
        max_lora_rank=16,
        dtype="half",
        python_executable="python",
    )

    assert command[:4] == ["python", "-u", "-m", "vllm.entrypoints.openai.api_server"]
    assert "--enable-lora" in command
    assert _value_after(command, "--host") == "127.0.0.1"
    assert _value_after(command, "--port") == "8081"
    assert _value_after(command, "--model") == "/Volumes/c/s/v/m/base"
    assert _value_after(command, "--served-model-name") == "base"
    assert _value_after(command, "--max-loras") == "2"
    assert _value_after(command, "--max-lora-rank") == "16"
    assert _value_after(command, "--dtype") == "half"
    pairs = command[command.index("--lora-modules") + 1 :]
    assert pairs == [
        "ner=/Volumes/c/s/v/m/adapters/ner",
        "code=/Volumes/c/s/v/m/adapters/code",
    ]


def test_parse_args_defaults_and_required() -> None:
    args = parse_args(["--base-model", "/b", "--max-loras", "2", "--max-lora-rank", "16"])
    assert args.host == "0.0.0.0"
    assert args.port == 8080
    assert args.vllm_host == "127.0.0.1"
    assert args.vllm_port == 8081
    assert args.served_model_name == "base"
    assert args.dtype == "half"
    assert args.lora_modules == []


def test_wait_returns_when_health_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Resp:
        status_code = 200

    monkeypatch.setattr(pe.httpx, "get", lambda *a, **k: _Resp())
    process = FakeProcess([None])

    # Returns without raising once /health is 200.
    wait_for_vllm_ready(
        "http://x:8081", process=process, timeout_s=5.0, sleep=lambda _s: None, now=lambda: 0.0
    )


def test_wait_raises_if_vllm_exits_early() -> None:
    process = FakeProcess([1])  # subprocess already exited with code 1
    with pytest.raises(RuntimeError, match="exited early"):
        wait_for_vllm_ready(
            "http://x:8081",
            process=process,
            timeout_s=5.0,
            sleep=lambda _s: None,
            now=lambda: 0.0,
        )


def test_wait_times_out_when_never_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*args: object, **kwargs: object) -> object:
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(pe.httpx, "get", _boom)
    clock = {"t": 0.0}

    def _now() -> float:
        return clock["t"]

    def _sleep(_seconds: float) -> None:
        clock["t"] += 10.0  # advance well past the deadline each tick

    process = FakeProcess([None, None, None])
    with pytest.raises(TimeoutError):
        wait_for_vllm_ready(
            "http://x:8081",
            process=process,
            timeout_s=5.0,
            interval_s=1.0,
            sleep=_sleep,
            now=_now,
        )


class FakeServer:
    """Minimal uvicorn.Server stand-in exposing a settable should_exit flag."""

    def __init__(self) -> None:
        self.should_exit = False


def test_monitor_requests_shutdown_when_vllm_dies() -> None:
    # poll() is None twice (alive), then returns a non-zero exit code (crashed).
    process = FakeProcess([None, None, 1])
    server = FakeServer()

    exit_code = monitor_vllm(
        process, server, poll_interval_s=0.0, sleep=lambda _s: None
    )

    # The monitor asked the proxy server to shut down and surfaced the crash code.
    assert server.should_exit is True
    assert exit_code == 1


def test_monitor_returns_none_when_server_already_exiting() -> None:
    # If the server is already shutting down (e.g. a signal), the monitor does not
    # treat it as a vLLM crash.
    process = FakeProcess([None])
    server = FakeServer()
    server.should_exit = True

    exit_code = monitor_vllm(process, server, sleep=lambda _s: None)

    assert exit_code is None
