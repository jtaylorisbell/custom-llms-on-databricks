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


# All adapters pass by default, so the happy-path pipeline exits 0.
_ALL_PASS_SMOKE = {"ner": {"ok": True, "snippet": "hi"}, "code": {"ok": True, "snippet": "ok"}}


class Recorder:
    """Monkeypatches the four pipeline functions, recording calls + ordering.

    Optionally configurable to make one step raise (``fail_step``/``fail_exc``) or to
    return specific smoke results (``smoke_results``).
    """

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        smoke_results: dict[str, dict[str, Any]] | None = None,
        fail_step: str | None = None,
        fail_exc: Exception | None = None,
    ) -> None:
        self.order: list[str] = []
        self.calls: dict[str, Any] = {}
        self._smoke_results = smoke_results if smoke_results is not None else dict(_ALL_PASS_SMOKE)
        self._fail_step = fail_step
        self._fail_exc = fail_exc if fail_exc is not None else RuntimeError("boom")

        monkeypatch.setattr(cli, "load_config", lambda path: make_config())
        monkeypatch.setattr(cli, "stage_weights", self._stage)
        monkeypatch.setattr(cli, "log_and_register", self._register)
        monkeypatch.setattr(cli, "deploy_endpoint", self._deploy)
        monkeypatch.setattr(cli, "smoke_test", self._smoke)

    def _maybe_fail(self, step: str) -> None:
        if self._fail_step == step:
            raise self._fail_exc

    def _stage(self, config: Config) -> dict[str, str]:
        self.order.append("stage")
        self.calls["stage"] = {"config": config}
        self._maybe_fail("stage")
        return {"base": "/Volumes/x/base", "ner": "/Volumes/x/ner", "code": "/Volumes/x/code"}

    def _register(self, config: Config, manifest: dict[str, str]) -> tuple[str, int]:
        self.order.append("register")
        self.calls["register"] = {"manifest": manifest}
        self._maybe_fail("register")
        return "classic_stable_h0vpq7_catalog.default.multi_lora_vllm", 7

    def _deploy(
        self, config: Config, model_full_name: str, model_version: int
    ) -> DeploymentResult:
        self.order.append("deploy")
        self.calls["deploy"] = {"name": model_full_name, "version": model_version}
        self._maybe_fail("deploy")
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
        self._maybe_fail("smoke")
        return self._smoke_results


def test_full_pipeline_runs_all_steps_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    rec = Recorder(monkeypatch)

    rc = cli.main(["--config", "config.yaml"])

    assert rc == cli.EXIT_OK
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

    assert rc == cli.EXIT_OK
    assert rec.order == ["stage", "register"]
    assert "deploy" not in rec.calls
    assert "smoke" not in rec.calls


def test_steps_reordered_to_canonical_order(monkeypatch: pytest.MonkeyPatch) -> None:
    rec = Recorder(monkeypatch)

    # Request out of order; smoke has no prerequisite, so stage+smoke is valid.
    rc = cli.main(["--config", "config.yaml", "--steps", "smoke,stage"])

    assert rc == cli.EXIT_OK
    assert rec.order == ["stage", "smoke"]


def test_smoke_only_needs_no_prerequisite(monkeypatch: pytest.MonkeyPatch) -> None:
    rec = Recorder(monkeypatch)

    rc = cli.main(["--config", "config.yaml", "--steps", "smoke-test"])

    assert rc == cli.EXIT_OK
    assert rec.order == ["smoke"]


# --- resumability: register/deploy/smoke run without re-staging -------------------


# The manifest register receives when 'stage' is NOT in the run: reconstructed from
# config (deterministic UC paths), distinct from Recorder._stage's fake manifest.
_RECONSTRUCTED_MANIFEST = {
    "base": "/Volumes/classic_stable_h0vpq7_catalog/default/weights/multi_lora_vllm/base",
    "ner": "/Volumes/classic_stable_h0vpq7_catalog/default/weights/multi_lora_vllm/adapters/ner",
    "code": "/Volumes/classic_stable_h0vpq7_catalog/default/weights/multi_lora_vllm/adapters/code",
}


def test_register_without_stage_reconstructs_manifest_from_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rec = Recorder(monkeypatch)

    # 'register' alone no longer forces 'stage' — the 14GB re-upload is skipped.
    rc = cli.main(["--config", "config.yaml", "--steps", "register"])

    assert rc == cli.EXIT_OK
    assert rec.order == ["register"]
    assert "stage" not in rec.order
    # register ran against the deterministically reconstructed UC manifest.
    assert rec.calls["register"]["manifest"] == _RECONSTRUCTED_MANIFEST


def test_register_deploy_smoke_suffix_runs_without_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rec = Recorder(monkeypatch)

    rc = cli.main(["--config", "config.yaml", "--steps", "register,deploy,smoke"])

    assert rc == cli.EXIT_OK
    # The whole tail runs off already-staged weights; stage is never invoked.
    assert rec.order == ["register", "deploy", "smoke"]
    assert "stage" not in rec.order
    assert rec.calls["register"]["manifest"] == _RECONSTRUCTED_MANIFEST
    # register's output still threads into deploy.
    assert rec.calls["deploy"]["name"] == "classic_stable_h0vpq7_catalog.default.multi_lora_vllm"
    assert rec.calls["deploy"]["version"] == 7


def test_full_run_prefers_real_staged_manifest_over_reconstruction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rec = Recorder(monkeypatch)

    rc = cli.main(["--config", "config.yaml", "--steps", "stage,register"])

    assert rc == cli.EXIT_OK
    assert rec.order == ["stage", "register"]
    # When stage runs, register uses stage's actual manifest, NOT the reconstruction.
    assert rec.calls["register"]["manifest"] == {
        "base": "/Volumes/x/base",
        "ner": "/Volumes/x/ner",
        "code": "/Volumes/x/code",
    }
    assert rec.calls["register"]["manifest"] != _RECONSTRUCTED_MANIFEST


def test_deploy_still_requires_register_in_same_run(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    rec = Recorder(monkeypatch)

    # Relaxing register's stage requirement does NOT relax deploy's need for a
    # freshly registered version: --steps deploy (or deploy,smoke) is still rejected.
    rc = cli.main(["--config", "config.yaml", "--steps", "deploy,smoke"])

    assert rc == cli.EXIT_USAGE
    assert rec.order == []
    err = capsys.readouterr().err
    assert "requires" in err and "register" in err


def test_dry_run_makes_no_real_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    rec = Recorder(monkeypatch)

    rc = cli.main(["--config", "config.yaml", "--dry-run"])

    assert rc == cli.EXIT_OK
    # None of the four pipeline functions were invoked.
    assert rec.order == []
    assert rec.calls == {}


def test_deploy_without_register_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    rec = Recorder(monkeypatch)

    rc = cli.main(["--config", "config.yaml", "--steps", "deploy"])

    # Missing prerequisite (no registered version) is a clean error, not a crash,
    # and is caught up front before any step runs.
    assert rc == cli.EXIT_USAGE
    assert rec.order == []


def test_unknown_step_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    Recorder(monkeypatch)

    rc = cli.main(["--config", "config.yaml", "--steps", "bogus"])

    assert rc == cli.EXIT_USAGE


def test_bad_config_path_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(path: str) -> Config:
        raise FileNotFoundError(f"no such file: {path}")

    monkeypatch.setattr(cli, "load_config", boom)

    rc = cli.main(["--config", "missing.yaml"])

    assert rc == cli.EXIT_USAGE


# --- BLOCKING 2: smoke-failure exit code ------------------------------------------


def test_smoke_failure_returns_nonzero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    rec = Recorder(
        monkeypatch,
        smoke_results={
            "ner": {"ok": True, "snippet": "hi"},
            "code": {"ok": False, "error": "upstream 500"},
        },
    )

    rc = cli.main(["--config", "config.yaml"])

    # The whole pipeline ran, but a failing adapter yields a nonzero exit code.
    assert rec.order == ["stage", "register", "deploy", "smoke"]
    assert rc == cli.EXIT_SMOKE_FAILURE
    assert rc != cli.EXIT_OK
    err = capsys.readouterr().err
    assert "code" in err  # the failing adapter is named on stderr


def test_smoke_all_pass_returns_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    rec = Recorder(
        monkeypatch,
        smoke_results={"ner": {"ok": True, "snippet": "a"}, "code": {"ok": True, "snippet": "b"}},
    )

    rc = cli.main(["--config", "config.yaml", "--steps", "smoke"])

    assert rec.order == ["smoke"]
    assert rc == cli.EXIT_OK


# --- BLOCKING 3: operational errors -> controlled exit, no traceback ---------------


def test_operational_failure_is_controlled_exit_without_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from multi_adapter_serving.deploy import EndpointDeploymentError

    rec = Recorder(
        monkeypatch,
        fail_step="deploy",
        fail_exc=EndpointDeploymentError("endpoint 'multi-lora-vllm' deployment failed"),
    )

    rc = cli.main(["--config", "config.yaml"])

    # Reached deploy, which raised; smoke never ran.
    assert rec.order == ["stage", "register", "deploy"]
    assert rc == cli.EXIT_OPERATIONAL
    captured = capsys.readouterr()
    # Concise message on stderr, and crucially NO Python traceback leaked.
    assert "deployment failed" in captured.err
    assert "Traceback" not in captured.err
    assert "Traceback" not in captured.out


def test_unexpected_step_error_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    # A non-Databricks runtime error still results in a controlled operational exit.
    rec = Recorder(monkeypatch, fail_step="stage", fail_exc=RuntimeError("disk full"))

    rc = cli.main(["--config", "config.yaml", "--steps", "stage"])

    assert rec.order == ["stage"]
    assert rc == cli.EXIT_OPERATIONAL


# --- HARDENING B: invalid --steps subset fails fast, no side effects ---------------


def test_invalid_subset_fails_fast_with_no_side_effects(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    rec = Recorder(monkeypatch)

    # deploy needs register in the same run; with only stage+deploy selected this is
    # rejected BEFORE staging real weights.
    rc = cli.main(["--config", "config.yaml", "--steps", "stage,deploy"])

    assert rc == cli.EXIT_USAGE
    # No step ran at all — in particular, stage did not download/upload anything.
    assert rec.order == []
    assert rec.calls == {}
    err = capsys.readouterr().err
    assert "requires" in err and "register" in err
