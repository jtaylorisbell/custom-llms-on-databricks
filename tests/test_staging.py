"""Unit tests for the weight-staging planner.

These tests inject fakes for every network / Unity Catalog operation, so they run
with no Hugging Face or Databricks access.
"""

from __future__ import annotations

from typing import Any

import pytest

from multi_adapter_serving.config import Config
from multi_adapter_serving.staging import stage_weights

UC_PREFIX = "/Volumes/classic_stable_h0vpq7_catalog/default/weights/multi_lora_vllm"


def make_config(source_type: str = "huggingface", **overrides: Any) -> Config:
    data: dict[str, Any] = {
        "source_type": source_type,
        "base_model": {"ref": "org/base-model"},
        "adapters": [
            {"name": "adapter_a", "ref": "org/adapter-a"},
            {"name": "adapter_b", "ref": "org/adapter-b"},
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
        "vllm": {"max_loras": 4, "max_lora_rank": 16, "max_model_len": 8192},
        "databricks": {"profile": "fevm-aws"},
    }
    data.update(overrides)
    return Config.model_validate(data)


class FakeDownloader:
    """Records snapshot_download-style calls and returns a fake local path."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, repo_id: str, *, local_dir: str, token: str | None) -> str:
        self.calls.append({"repo_id": repo_id, "local_dir": local_dir, "token": token})
        return f"/fake/cache/{repo_id}"


class FakeUploader:
    """Records UC upload calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def __call__(self, local_path: str, volume_path: str) -> None:
        self.calls.append((local_path, volume_path))


def test_huggingface_manifest_and_transfers() -> None:
    downloader = FakeDownloader()
    uploader = FakeUploader()

    manifest = stage_weights(
        make_config(),
        downloader=downloader,
        uploader=uploader,
        hf_token="secret-token",
    )

    assert manifest == {
        "base": f"{UC_PREFIX}/base",
        "adapter_a": f"{UC_PREFIX}/adapters/adapter_a",
        "adapter_b": f"{UC_PREFIX}/adapters/adapter_b",
    }

    # Every ref was downloaded with the provided token...
    assert [c["repo_id"] for c in downloader.calls] == [
        "org/base-model",
        "org/adapter-a",
        "org/adapter-b",
    ]
    assert {c["token"] for c in downloader.calls} == {"secret-token"}

    # ...then uploaded from its fake local path to its UC target.
    assert uploader.calls == [
        ("/fake/cache/org/base-model", f"{UC_PREFIX}/base"),
        ("/fake/cache/org/adapter-a", f"{UC_PREFIX}/adapters/adapter_a"),
        ("/fake/cache/org/adapter-b", f"{UC_PREFIX}/adapters/adapter_b"),
    ]


def test_hf_token_falls_back_to_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "env-token")
    downloader = FakeDownloader()
    uploader = FakeUploader()

    stage_weights(make_config(), downloader=downloader, uploader=uploader)

    assert {c["token"] for c in downloader.calls} == {"env-token"}


def test_dry_run_does_not_transfer(caplog: pytest.LogCaptureFixture) -> None:
    downloader = FakeDownloader()
    uploader = FakeUploader()

    with caplog.at_level("INFO"):
        manifest = stage_weights(
            make_config(),
            dry_run=True,
            downloader=downloader,
            uploader=uploader,
        )

    assert manifest == {
        "base": f"{UC_PREFIX}/base",
        "adapter_a": f"{UC_PREFIX}/adapters/adapter_a",
        "adapter_b": f"{UC_PREFIX}/adapters/adapter_b",
    }
    # Nothing was actually downloaded or uploaded.
    assert downloader.calls == []
    assert uploader.calls == []
    # The plan was logged.
    assert any("dry-run" in record.message for record in caplog.records)


def test_local_uploads_fs_paths_and_passes_through_uc_paths() -> None:
    uploader = FakeUploader()
    existing_uc = "/Volumes/other/place/vol/already_here"
    config = make_config(
        source_type="local",
        base_model={"ref": "/data/base"},
        adapters=[
            {"name": "adapter_a", "ref": "/data/adapter_a"},
            {"name": "adapter_b", "ref": existing_uc},
        ],
    )

    manifest = stage_weights(
        config,
        uploader=uploader,
        local_path_exists=lambda _p: True,
        uc_path_exists=lambda _p: True,
    )

    assert manifest == {
        "base": f"{UC_PREFIX}/base",
        "adapter_a": f"{UC_PREFIX}/adapters/adapter_a",
        # Already-in-UC ref is passed through unchanged.
        "adapter_b": existing_uc,
    }
    # Only the two local filesystem paths are uploaded.
    assert uploader.calls == [
        ("/data/base", f"{UC_PREFIX}/base"),
        ("/data/adapter_a", f"{UC_PREFIX}/adapters/adapter_a"),
    ]


def test_local_missing_path_raises() -> None:
    uploader = FakeUploader()
    config = make_config(source_type="local", base_model={"ref": "/missing/base"})

    with pytest.raises(FileNotFoundError, match="Local path does not exist"):
        stage_weights(
            config,
            uploader=uploader,
            local_path_exists=lambda _p: False,
            uc_path_exists=lambda _p: True,
        )


def test_local_missing_uc_path_raises() -> None:
    uploader = FakeUploader()
    config = make_config(
        source_type="local",
        base_model={"ref": "/Volumes/x/y/z/missing"},
    )

    with pytest.raises(FileNotFoundError, match="UC Volume path does not exist"):
        stage_weights(
            config,
            uploader=uploader,
            local_path_exists=lambda _p: True,
            uc_path_exists=lambda _p: False,
        )


def test_dry_run_local_plan_passes_through_uc(caplog: pytest.LogCaptureFixture) -> None:
    existing_uc = "/Volumes/other/place/vol/already_here"
    config = make_config(
        source_type="local",
        base_model={"ref": "/data/base"},
        adapters=[{"name": "adapter_a", "ref": existing_uc}],
    )

    with caplog.at_level("INFO"):
        manifest = stage_weights(config, dry_run=True)

    assert manifest == {
        "base": f"{UC_PREFIX}/base",
        "adapter_a": existing_uc,
    }
