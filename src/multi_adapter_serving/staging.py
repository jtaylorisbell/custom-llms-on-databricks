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
    "staged_manifest",
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


class _UCFilesClient(Protocol):
    """The subset of the Databricks SDK ``WorkspaceClient.files`` API we use.

    Captured as a Protocol so the default uploader / existence checker can be unit
    tested with an injected fake instead of a real ``WorkspaceClient``.
    """

    def create_directory(self, directory_path: str) -> object: ...

    def upload(
        self, file_path: str, contents: BinaryIO, *, overwrite: bool | None = None
    ) -> object: ...

    def get_metadata(self, file_path: str) -> object: ...

    def get_directory_metadata(self, directory_path: str) -> object: ...


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


def staged_manifest(config: Config) -> dict[str, str]:
    """Reconstruct the staging manifest deterministically from config — no I/O.

    Returns the SAME mapping :func:`stage_weights` would produce, computed purely
    from ``config`` with no downloads, uploads, or existence checks. Use it to resume
    the pipeline (``register`` / ``deploy`` / ``smoke``) against weights already staged
    in Unity Catalog, without re-running ``stage`` — avoiding a full re-upload.

    Staged paths are deterministic (see :func:`_plan`): ``"base"`` maps to
    ``{prefix}/base`` and each adapter ``<name>`` to ``{prefix}/adapters/<name>``,
    where ``prefix`` is ``/Volumes/{catalog}/{schema}/{volume}/{model_name}``. The one
    exception mirrors ``stage_weights`` exactly: a ``local`` ref that already points at
    a ``/Volumes/...`` path is passed through unchanged (it is never re-homed under the
    model prefix), so reconstruction equals the real staged manifest for every
    ``source_type``.

    This validates nothing — it assumes the weights are already present (the
    resumability premise). Any genuinely-missing weights surface later, at register /
    serving time (this function makes no Databricks calls by design).
    """
    manifest: dict[str, str] = {}
    for item in _plan(config):
        if config.source_type == "local" and _is_uc_volume_path(item.ref):
            manifest[item.key] = item.ref
        else:
            manifest[item.key] = item.target
    return manifest


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


def _default_files_client(profile: str) -> _UCFilesClient:
    """The real Databricks SDK files client (OAuth via ``profile``)."""
    from databricks.sdk import WorkspaceClient

    return WorkspaceClient(profile=profile).files


def _parent_dir(path: str) -> str:
    """Parent directory of a UC Volume path (``mkdir -p`` target for the file)."""
    parent, _, _ = path.rstrip("/").rpartition("/")
    return parent or "/"


def _upload_one(client: _UCFilesClient, local_file: Path, destination: str) -> None:
    """Upload a single file, first ensuring its parent UC directory exists."""
    # create_directory is idempotent and creates intermediate parents, so doing
    # this per-file makes nested directory trees (e.g. .cache/huggingface/...) work.
    client.create_directory(_parent_dir(destination))
    with local_file.open("rb") as handle:
        contents: BinaryIO = handle
        client.upload(destination, contents, overwrite=True)


def _make_uc_uploader(profile: str, *, files_client: _UCFilesClient | None = None) -> UCUploader:
    """Build a UC uploader using the Databricks SDK (OAuth via ``profile``)."""
    client = files_client if files_client is not None else _default_files_client(profile)

    def _upload(local_path: str, volume_path: str) -> None:
        source = Path(local_path)
        target_root = volume_path.rstrip("/")
        if source.is_file():
            # A single-file artifact lands AT target_root, so the staged path equals
            # the manifest path returned by stage_weights().
            _upload_one(client, source, target_root)
            return
        # A directory artifact is mirrored under target_root (which is itself the
        # manifest path); each file gets its parent directory created first.
        for file_path in sorted(p for p in source.rglob("*") if p.is_file()):
            relative = file_path.relative_to(source).as_posix()
            _upload_one(client, file_path, f"{target_root}/{relative}")

    return _upload


def _make_uc_existence_checker(
    profile: str, *, files_client: _UCFilesClient | None = None
) -> PathChecker:
    """Build a UC existence checker using the Databricks SDK (OAuth via ``profile``)."""
    from databricks.sdk.errors import NotFound

    client = files_client if files_client is not None else _default_files_client(profile)

    def _exists(volume_path: str) -> bool:
        # A passed-through ref may point at either a directory or a single file, so
        # check both; only when neither exists do we treat the path as missing.
        try:
            client.get_directory_metadata(volume_path)
            return True
        except NotFound:
            pass
        try:
            client.get_metadata(volume_path)
            return True
        except NotFound:
            return False

    return _exists
