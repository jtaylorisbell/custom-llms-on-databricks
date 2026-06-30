"""Unit tests for the end-to-end CLI (the four pipeline functions are mocked).

No network / Databricks / GPU: ``load_config`` and the stage/register/deploy/smoke
functions are monkeypatched, so the tests assert orchestration only — ordering,
output wiring, ``--steps`` subsetting, and that ``--dry-run`` makes no real calls.
"""

from __future__ import annotations

from typing import Any

import pytest

from multi_adapter_serving import cli
from multi_adapter_serving.config import Config
from multi_adapter_serving.deploy import DeploymentResult


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
        "vllm": {"max_loras": 2, "max_lora_rank": 16, "max_model_len": 8192},
        "databricks": {"profile": "fevm-aws"},
    }
    data.update(overrides)
    return Config.model_validate(data)


class Recorder:
    """Monkeypatches the four pipeline functions, recording calls + ordering."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.order: list[str] = []
        self.calls: dict[str, Any] = {}

        monkeypatch.setattr(cli, "load_config", lambda path: make_config())
        monkeypatch.setattr(cli, "stage_weights", self._stage)
        monkeypatch.setattr(cli, "log_and_register", self._register)
        monkeypatch.setattr(cli, "deploy_endpoint", self._deploy)
        monkeypatch.setattr(cli, "smoke_test", self._smoke)

    def _stage(self, config: Config) -> dict[str, str]:
        self.order.append("stage")
        manifest = {"base": "/Volumes/x/base", "ner": "/Volumes/x/ner", "code": "/Volumes/x/code"}
        self.calls["stage"] = {"config": config}
        return manifest

    def _register(self, config: Config, manifest: dict[str, str]) -> tuple[str, int]:
        self.order.append("register")
        self.calls["register"] = {"manifest": manifest}
        return "classic_stable_h0vpq7_catalog.default.multi_lora_vllm", 7

    def _deploy(
        self, config: Config, model_full_name: str, model_version: int
    ) -> DeploymentResult:
        self.order.append("deploy")
        self.calls["deploy"] = {"name": model_full_name, "version": model_version}
        return DeploymentResult(
            endpoint_name="multi-lora-vllm",
            action="created",
            ready=True,
            config_update="NOT_UPDATING",
            endpoint=None,  # type: ignore[arg-type]
        )

    def _smoke(self, config: Config) -> dict[str, dict[str, Any]]:
        self.order.append("smoke")
        self.calls["smoke"] = {"config": config}
        return {"ner": {"ok": True, "snippet": "hi"}, "code": {"ok": False, "error": "boom"}}


def test_full_pipeline_runs_all_steps_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    rec = Recorder(monkeypatch)

    rc = cli.main(["--config", "config.yaml"])

    assert rc == 0
    assert rec.order == ["stage", "register", "deploy", "smoke"]
    # Output wiring: stage's manifest -> register; register's name+version -> deploy.
    assert rec.calls["register"]["manifest"] == {
        "base": "/Volumes/x/base",
        "ner": "/Volumes/x/ner",
        "code": "/Volumes/x/code",
    }
    assert rec.calls["deploy"]["name"] == "classic_stable_h0vpq7_catalog.default.multi_lora_vllm"
    assert rec.calls["deploy"]["version"] == 7


def test_steps_subset_runs_only_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    rec = Recorder(monkeypatch)

    rc = cli.main(["--config", "config.yaml", "--steps", "stage,register"])

    assert rc == 0
    assert rec.order == ["stage", "register"]
    assert "deploy" not in rec.calls
    assert "smoke" not in rec.calls


def test_steps_reordered_to_canonical_order(monkeypatch: pytest.MonkeyPatch) -> None:
    rec = Recorder(monkeypatch)

    # Request out of order; smoke has no prerequisite, so stage+smoke is valid.
    rc = cli.main(["--config", "config.yaml", "--steps", "smoke,stage"])

    assert rc == 0
    assert rec.order == ["stage", "smoke"]


def test_smoke_only_needs_no_prerequisite(monkeypatch: pytest.MonkeyPatch) -> None:
    rec = Recorder(monkeypatch)

    rc = cli.main(["--config", "config.yaml", "--steps", "smoke-test"])

    assert rc == 0
    assert rec.order == ["smoke"]


def test_dry_run_makes_no_real_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    rec = Recorder(monkeypatch)

    rc = cli.main(["--config", "config.yaml", "--dry-run"])

    assert rc == 0
    # None of the four pipeline functions were invoked.
    assert rec.order == []
    assert rec.calls == {}


def test_deploy_without_register_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    rec = Recorder(monkeypatch)

    rc = cli.main(["--config", "config.yaml", "--steps", "deploy"])

    # Missing prerequisite (no registered version) is a clean error, not a crash.
    assert rc == 2
    assert rec.order == []


def test_unknown_step_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    Recorder(monkeypatch)

    rc = cli.main(["--config", "config.yaml", "--steps", "bogus"])

    assert rc == 2


def test_bad_config_path_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(path: str) -> Config:
        raise FileNotFoundError(f"no such file: {path}")

    monkeypatch.setattr(cli, "load_config", boom)

    rc = cli.main(["--config", "missing.yaml"])

    assert rc == 2
