"""Stage base + adapter weights into a Unity Catalog Volume.

:func:`stage_weights` takes a validated :class:`~multi_adapter_serving.config.Config`
and materializes every weight (the base model plus each adapter) under a single UC
Volume prefix, returning a manifest mapping each logical name to its final UC path.

Two source types are supported:

* ``huggingface`` — download base + adapters from the Hugging Face Hub into a local
  cache, then upload each into the UC Volume.
* ``local`` — validate that the provided local / UC-Volume paths exist; local
  filesystem paths are uploaded into the UC Volume, while refs that already point at
  a ``/Volumes/...`` path are left in place.

All network and Unity Catalog operations are injectable (``downloader``,
``uploader``, ``local_path_exists``, ``uc_path_exists``) so the planner can be unit
tested with no network or Databricks access. A ``dry_run`` mode logs the full plan
without downloading, uploading, or validating anything.
"""

from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Protocol

from multi_adapter_serving.config import Config, UnityCatalogConfig

__all__ = [
    "HFDownloader",
    "PathChecker",
    "UCUploader",
    "stage_weights",
]

logger = logging.getLogger(__name__)

# Marker prefix identifying a Unity Catalog Volume path.
_UC_VOLUME_PREFIX = "/Volumes/"


class HFDownloader(Protocol):
    """Downloads a Hugging Face repo into ``local_dir`` and returns the local path."""

    def __call__(self, repo_id: str, *, local_dir: str, token: str | None) -> str: ...


class UCUploader(Protocol):
    """Uploads a local file or directory tree to a Unity Catalog Volume path."""

    def __call__(self, local_path: str, volume_path: str) -> None: ...


# Predicate used to check whether a path (local fs or UC Volume) exists.
PathChecker = Callable[[str], bool]


@dataclass(frozen=True)
class _StageItem:
    """One unit of work: a logical weight name, its source ref, and its UC target."""

    key: str
    ref: str
    target: str


def _is_uc_volume_path(ref: str) -> bool:
    return ref.startswith(_UC_VOLUME_PREFIX)


def _uc_target_prefix(uc: UnityCatalogConfig) -> str:
    """Root UC Volume prefix under which all weights for this model are staged."""
    return f"{_UC_VOLUME_PREFIX}{uc.catalog}/{uc.db_schema}/{uc.volume}/{uc.model_name}"


def _plan(config: Config) -> list[_StageItem]:
    """Build the ordered list of weights to stage (base first, then adapters)."""
    prefix = _uc_target_prefix(config.unity_catalog)
    items = [_StageItem("base", config.base_model.ref, f"{prefix}/base")]
    items.extend(
        _StageItem(adapter.name, adapter.ref, f"{prefix}/adapters/{adapter.name}")
        for adapter in config.adapters
    )
    return items


def stage_weights(
    config: Config,
    *,
    dry_run: bool = False,
    cache_dir: str | Path | None = None,
    hf_token: str | None = None,
    downloader: HFDownloader | None = None,
    uploader: UCUploader | None = None,
    local_path_exists: PathChecker | None = None,
    uc_path_exists: PathChecker | None = None,
) -> dict[str, str]:
    """Stage base + adapter weights into Unity Catalog and return a manifest.

    Args:
        config: Validated configuration.
        dry_run: If ``True``, log the plan and return the manifest without
            downloading, uploading, or validating anything.
        cache_dir: Local directory for Hugging Face downloads. Defaults to a temp
            directory. Only used for ``source_type: huggingface``.
        hf_token: Hugging Face token for gated repos. Defaults to ``$HF_TOKEN``.
        downloader: Injectable HF download function. Defaults to
            :func:`huggingface_hub.snapshot_download`.
        uploader: Injectable UC upload function. Defaults to a Databricks SDK
            (OAuth, profile from ``config.databricks.profile``) uploader.
        local_path_exists: Injectable local-filesystem existence check. Defaults to
            :func:`os.path.exists`.
        uc_path_exists: Injectable UC-Volume existence check. Defaults to a
            Databricks SDK check.

    Returns:
        Manifest mapping ``"base"`` and each adapter name to its final UC Volume path.
    """
    items = _plan(config)

    if dry_run:
        return _run_dry(config, items)

    if config.source_type == "huggingface":
        return _run_huggingface(
            config,
            items,
            cache_dir=cache_dir,
            hf_token=hf_token if hf_token is not None else os.environ.get("HF_TOKEN"),
            downloader=downloader or _default_downloader,
            uploader=uploader or _make_uc_uploader(config.databricks.profile),
        )

    return _run_local(
        config,
        items,
        uploader=uploader or _make_uc_uploader(config.databricks.profile),
        local_path_exists=local_path_exists or os.path.exists,
        uc_path_exists=uc_path_exists or _make_uc_existence_checker(config.databricks.profile),
    )


def _run_dry(config: Config, items: list[_StageItem]) -> dict[str, str]:
    """Log the staging plan without performing any transfers or validation."""
    logger.info(
        "[dry-run] staging plan for source_type=%s (profile=%s)",
        config.source_type,
        config.databricks.profile,
    )
    manifest: dict[str, str] = {}
    for item in items:
        if config.source_type == "huggingface":
            logger.info(
                "[dry-run] would download %r from Hugging Face and upload to %s",
                item.ref,
                item.target,
            )
            manifest[item.key] = item.target
        elif _is_uc_volume_path(item.ref):
            logger.info(
                "[dry-run] would validate existing UC path %s (no upload)",
                item.ref,
            )
            manifest[item.key] = item.ref
        else:
            logger.info(
                "[dry-run] would validate local path %r and upload to %s",
                item.ref,
                item.target,
            )
            manifest[item.key] = item.target
    return manifest


def _run_huggingface(
    config: Config,
    items: list[_StageItem],
    *,
    cache_dir: str | Path | None,
    hf_token: str | None,
    downloader: HFDownloader,
    uploader: UCUploader,
) -> dict[str, str]:
    base_cache = Path(cache_dir) if cache_dir is not None else Path(
        tempfile.gettempdir()
    ) / "multi_adapter_serving_cache"

    manifest: dict[str, str] = {}
    for item in items:
        item_cache = base_cache / item.key
        logger.info("Downloading %r from Hugging Face into %s", item.ref, item_cache)
        local_path = downloader(item.ref, local_dir=str(item_cache), token=hf_token)
        logger.info("Uploading %s to %s", local_path, item.target)
        uploader(local_path, item.target)
        manifest[item.key] = item.target
    return manifest


def _run_local(
    config: Config,
    items: list[_StageItem],
    *,
    uploader: UCUploader,
    local_path_exists: PathChecker,
    uc_path_exists: PathChecker,
) -> dict[str, str]:
    manifest: dict[str, str] = {}
    for item in items:
        if _is_uc_volume_path(item.ref):
            if not uc_path_exists(item.ref):
                raise FileNotFoundError(f"UC Volume path does not exist: {item.ref}")
            logger.info("Using existing UC path for %s: %s", item.key, item.ref)
            manifest[item.key] = item.ref
        else:
            if not local_path_exists(item.ref):
                raise FileNotFoundError(f"Local path does not exist: {item.ref}")
            logger.info("Uploading local %r to %s", item.ref, item.target)
            uploader(item.ref, item.target)
            manifest[item.key] = item.target
    return manifest


def _default_downloader(repo_id: str, *, local_dir: str, token: str | None) -> str:
    """Default HF downloader backed by ``huggingface_hub.snapshot_download``."""
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=repo_id, local_dir=local_dir, token=token)


def _make_uc_uploader(profile: str) -> UCUploader:
    """Build a UC uploader using the Databricks SDK (OAuth via ``profile``)."""
    from databricks.sdk import WorkspaceClient

    client = WorkspaceClient(profile=profile)

    def _upload(local_path: str, volume_path: str) -> None:
        source = Path(local_path)
        target_root = volume_path.rstrip("/")
        client.files.create_directory(target_root)
        if source.is_file():
            files = [source]
        else:
            files = sorted(p for p in source.rglob("*") if p.is_file())
        for file_path in files:
            if source.is_file():
                relative = file_path.name
            else:
                relative = file_path.relative_to(source).as_posix()
            destination = f"{target_root}/{relative}"
            with file_path.open("rb") as handle:
                contents: BinaryIO = handle
                client.files.upload(destination, contents, overwrite=True)

    return _upload


def _make_uc_existence_checker(profile: str) -> PathChecker:
    """Build a UC existence checker using the Databricks SDK (OAuth via ``profile``)."""
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.errors import NotFound

    client = WorkspaceClient(profile=profile)

    def _exists(volume_path: str) -> bool:
        try:
            client.files.get_directory_metadata(volume_path)
        except NotFound:
            return False
        return True

    return _exists
