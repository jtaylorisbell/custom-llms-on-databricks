"""Unit tests for the weight-staging planner.

These tests inject fakes for every network / Unity Catalog operation, so they run
with no Hugging Face or Databricks access.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, BinaryIO

import pytest
from databricks.sdk.errors import NotFound

from multi_adapter_serving.config import Config
from multi_adapter_serving.staging import (
    _make_uc_existence_checker,
    _make_uc_uploader,
    _parent_dir,
    stage_weights,
    staged_manifest,
)

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


class FakeFilesClient:
    """Records Databricks SDK ``files`` calls for the default uploader/checker."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []
        self.existing_files: set[str] = set()
        self.existing_dirs: set[str] = set()

    def create_directory(self, directory_path: str) -> object:
        self.events.append(("mkdir", directory_path))
        return None

    def upload(
        self, file_path: str, contents: BinaryIO, *, overwrite: bool | None = None
    ) -> object:
        self.events.append(("upload", file_path))
        return None

    def get_metadata(self, file_path: str) -> object:
        if file_path not in self.existing_files:
            raise NotFound(f"no such file: {file_path}")
        return object()

    def get_directory_metadata(self, directory_path: str) -> object:
        if directory_path not in self.existing_dirs:
            raise NotFound(f"no such directory: {directory_path}")
        return object()


def test_default_uploader_creates_parent_dirs_for_nested_tree(tmp_path: Path) -> None:
    # Build a snapshot-like tree with nested subdirectories.
    source = tmp_path / "snapshot"
    (source / "sub" / "deep").mkdir(parents=True)
    (source / "config.json").write_text("{}", encoding="utf-8")
    (source / "sub" / "nested.bin").write_text("x", encoding="utf-8")
    (source / "sub" / "deep" / "weights.safetensors").write_text("y", encoding="utf-8")

    fake = FakeFilesClient()
    uploader = _make_uc_uploader("fevm-aws", files_client=fake)

    target_root = f"{UC_PREFIX}/base"
    uploader(str(source), target_root)

    uploads = [path for kind, path in fake.events if kind == "upload"]
    assert sorted(uploads) == [
        f"{target_root}/config.json",
        f"{target_root}/sub/deep/weights.safetensors",
        f"{target_root}/sub/nested.bin",
    ]

    # Every upload must be preceded by a create_directory of its parent.
    for index, (kind, path) in enumerate(fake.events):
        if kind != "upload":
            continue
        parent = _parent_dir(path)
        prior_mkdirs = [
            p for k, p in fake.events[:index] if k == "mkdir" and p == parent
        ]
        assert prior_mkdirs, f"parent {parent} not created before uploading {path}"

    # The intermediate (nested) parent directories were created.
    created = {p for k, p in fake.events if k == "mkdir"}
    assert f"{target_root}/sub" in created
    assert f"{target_root}/sub/deep" in created


def test_default_uploader_single_file_lands_at_target_root(tmp_path: Path) -> None:
    source = tmp_path / "adapter_model.safetensors"
    source.write_text("weights", encoding="utf-8")

    fake = FakeFilesClient()
    uploader = _make_uc_uploader("fevm-aws", files_client=fake)

    target_root = f"{UC_PREFIX}/adapters/code"
    uploader(str(source), target_root)

    # The artifact lands exactly at target_root (the manifest path), not at
    # target_root/<filename>, and its parent directory is created first.
    assert ("upload", target_root) in fake.events
    uploads = [path for kind, path in fake.events if kind == "upload"]
    assert uploads == [target_root]
    assert ("mkdir", _parent_dir(target_root)) in fake.events


def test_default_uploader_single_file_matches_manifest(tmp_path: Path) -> None:
    # When stage_weights uses the default uploader, the manifest path equals the
    # real upload destination for a single-file artifact.
    source = tmp_path / "adapter_model.safetensors"
    source.write_text("weights", encoding="utf-8")

    fake = FakeFilesClient()
    config = make_config(
        source_type="local",
        base_model={"ref": str(source)},
        adapters=[{"name": "only", "ref": str(source)}],
    )
    manifest = stage_weights(
        config,
        uploader=_make_uc_uploader("fevm-aws", files_client=fake),
        local_path_exists=lambda _p: True,
        uc_path_exists=lambda _p: True,
    )

    uploaded = [path for kind, path in fake.events if kind == "upload"]
    assert manifest["base"] in uploaded
    assert manifest["only"] in uploaded


def test_default_existence_checker_true_for_file_path() -> None:
    fake = FakeFilesClient()
    file_path = "/Volumes/c/s/v/m/adapters/code/adapter_model.safetensors"
    fake.existing_files.add(file_path)  # exists as a FILE, not a directory

    checker = _make_uc_existence_checker("fevm-aws", files_client=fake)

    assert checker(file_path) is True


def test_default_existence_checker_true_for_directory_path() -> None:
    fake = FakeFilesClient()
    dir_path = "/Volumes/c/s/v/m/base"
    fake.existing_dirs.add(dir_path)

    checker = _make_uc_existence_checker("fevm-aws", files_client=fake)

    assert checker(dir_path) is True


def test_default_existence_checker_false_when_missing() -> None:
    fake = FakeFilesClient()
    checker = _make_uc_existence_checker("fevm-aws", files_client=fake)

    assert checker("/Volumes/c/s/v/m/missing") is False


# --- staged_manifest: deterministic reconstruction (resumability) -----------------


def test_staged_manifest_reconstructs_deterministic_paths() -> None:
    # Pure reconstruction from config: the deterministic UC paths, no I/O.
    manifest = staged_manifest(make_config())

    assert manifest == {
        "base": f"{UC_PREFIX}/base",
        "adapter_a": f"{UC_PREFIX}/adapters/adapter_a",
        "adapter_b": f"{UC_PREFIX}/adapters/adapter_b",
    }


def test_staged_manifest_matches_stage_weights_for_huggingface() -> None:
    # Reconstruction must equal the manifest a real huggingface stage would produce,
    # so resuming register/deploy/smoke targets exactly the staged weights.
    config = make_config()
    real = stage_weights(config, downloader=FakeDownloader(), uploader=FakeUploader())

    assert staged_manifest(config) == real


def test_staged_manifest_matches_stage_weights_for_local_uc_passthrough() -> None:
    # For source_type=local, an already-in-UC ref is passed through unchanged, while a
    # local filesystem ref maps to the deterministic target. Reconstruction mirrors
    # both, so it equals the real staged manifest here too (no re-homing of UC refs).
    existing_uc = "/Volumes/other/place/vol/already_here"
    config = make_config(
        source_type="local",
        base_model={"ref": "/data/base"},
        adapters=[
            {"name": "adapter_a", "ref": "/data/adapter_a"},
            {"name": "adapter_b", "ref": existing_uc},
        ],
    )
    real = stage_weights(
        config,
        uploader=FakeUploader(),
        local_path_exists=lambda _p: True,
        uc_path_exists=lambda _p: True,
    )

    reconstructed = staged_manifest(config)
    assert reconstructed == real
    assert reconstructed == {
        "base": f"{UC_PREFIX}/base",
        "adapter_a": f"{UC_PREFIX}/adapters/adapter_a",
        "adapter_b": existing_uc,
    }


def test_staged_manifest_does_no_io() -> None:
    # It takes no downloader/uploader/existence-checker and touches nothing: a
    # missing local ref does NOT raise (unlike stage_weights), proving no validation.
    config = make_config(source_type="local", base_model={"ref": "/definitely/missing"})

    manifest = staged_manifest(config)

    assert manifest["base"] == f"{UC_PREFIX}/base"
