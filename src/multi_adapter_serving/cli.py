"""End-to-end CLI for the multi-LoRA serving workflow.

Implements the Databricks custom-LLM deployment steps as one command, in order:

1. ``stage``    — download/validate weights into a Unity Catalog Volume
   (:func:`multi_adapter_serving.staging.stage_weights`).
2. ``register`` — log the pyfunc chat model + register it to Unity Catalog
   (:func:`multi_adapter_serving.model.log_and_register`).
3. ``deploy``   — create/update the Serverless GPU serving endpoint
   (:func:`multi_adapter_serving.deploy.deploy_endpoint`).
4. ``smoke``    — query every adapter on the endpoint
   (:func:`multi_adapter_serving.smoke_test.smoke_test`).

Run a subset with ``--steps`` and preview without side effects with ``--dry-run``.

Usage::

    multi-adapter-serve --config config.yaml
    multi-adapter-serve --config config.yaml --steps stage,register
    multi-adapter-serve --config config.yaml --steps smoke
    multi-adapter-serve --config config.yaml --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from typing import Any

# DatabricksError is the base for all Databricks SDK service errors. Imported from
# its defining module because the errors package __init__ does not re-export it in a
# way the strict type checker treats as public.
from databricks.sdk.errors.base import DatabricksError

from multi_adapter_serving.config import Config, load_config
from multi_adapter_serving.deploy import (
    DeploymentResult,
    EndpointDeploymentError,
    deploy_endpoint,
)
from multi_adapter_serving.model import log_and_register
from multi_adapter_serving.smoke_test import smoke_test
from multi_adapter_serving.staging import stage_weights

__all__ = ["main"]

logger = logging.getLogger(__name__)

# The pipeline steps in dependency order. Selected steps always run in this order.
CANONICAL_STEPS: tuple[str, ...] = ("stage", "register", "deploy", "smoke")

# Friendly aliases accepted in --steps for the smoke step.
_STEP_ALIASES = {"smoke-test": "smoke", "smoketest": "smoke"}

# Each step's required predecessor that must also be selected, so its output is
# produced within the same run. Steps not listed have no in-run prerequisite
# (``stage`` needs nothing; ``smoke`` can query a pre-existing endpoint).
_STEP_REQUIRES = {"register": "stage", "deploy": "register"}

# Process exit codes. Distinct codes let callers tell the failure modes apart.
EXIT_OK = 0
EXIT_SMOKE_FAILURE = 1  # pipeline ran, but >=1 adapter failed its smoke test
EXIT_USAGE = 2  # bad arguments, bad config, or an incomplete --steps subset
EXIT_OPERATIONAL = 3  # a step raised at runtime (deploy/register/stage/smoke)


class CliError(Exception):
    """A user-facing CLI error (bad arguments or a missing step prerequisite)."""


@dataclass
class _PipelineState:
    """Outputs threaded between steps within a single run."""

    manifest: dict[str, str] | None = None
    model_full_name: str | None = None
    model_version: int | None = None
    deployment: DeploymentResult | None = None
    smoke_results: dict[str, dict[str, Any]] | None = None
    ran: list[str] = field(default_factory=list)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        steps = _resolve_steps(args.steps)
        # Fail fast on an incomplete subset BEFORE any side-effecting step runs.
        _validate_step_dependencies(steps)
    except CliError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: failed to load config {args.config!r}: {exc}", file=sys.stderr)
        return EXIT_USAGE

    if args.dry_run:
        _print_dry_run_plan(config, steps)
        return EXIT_OK

    try:
        state = _run_steps(config, steps)
    except CliError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except (EndpointDeploymentError, FileNotFoundError, DatabricksError) as exc:
        # Expected operational failures: report concisely, no traceback.
        print(f"error: pipeline step failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_OPERATIONAL
    except Exception as exc:  # operational fallback: controlled exit, never a traceback
        print(
            f"error: unexpected failure during pipeline: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return EXIT_OPERATIONAL

    _print_summary(config, state)
    return _exit_code_for_results(state)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="multi-adapter-serve",
        description="Stage, register, deploy, and smoke-test a multi-LoRA endpoint.",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the config YAML (default: config.yaml).",
    )
    parser.add_argument(
        "--steps",
        default=None,
        help=(
            "Comma-separated subset of steps to run "
            f"({','.join(CANONICAL_STEPS)}); default: all, in order. "
            "Steps always execute in dependency order."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan and make no real Databricks/network calls.",
    )
    return parser


def _resolve_steps(steps_arg: str | None) -> list[str]:
    """Parse/validate ``--steps`` into a canonical-ordered, deduped step list."""
    if steps_arg is None:
        return list(CANONICAL_STEPS)

    requested: set[str] = set()
    for raw in steps_arg.split(","):
        token = raw.strip().lower()
        if not token:
            continue
        token = _STEP_ALIASES.get(token, token)
        if token not in CANONICAL_STEPS:
            valid = ", ".join(CANONICAL_STEPS)
            raise CliError(f"unknown step {raw.strip()!r}; valid steps: {valid}")
        requested.add(token)

    if not requested:
        raise CliError("--steps was given but contained no valid steps")
    return [step for step in CANONICAL_STEPS if step in requested]


def _validate_step_dependencies(steps: list[str]) -> None:
    """Reject an incomplete ``--steps`` subset before any side-effecting work.

    A selected step whose output-producing predecessor is not also selected cannot
    run (its input would be missing), so fail fast — e.g. ``--steps stage,deploy``
    is rejected up front instead of staging real weights and then crashing.
    """
    selected = set(steps)
    for step in steps:
        required = _STEP_REQUIRES.get(step)
        if required is not None and required not in selected:
            raise CliError(
                f"step {step!r} requires {required!r} in the same run; "
                f"add it to --steps (e.g. --steps {required},{step})"
            )


def _exit_code_for_results(state: _PipelineState) -> int:
    """Return a nonzero code if the smoke step ran and any adapter failed."""
    if state.smoke_results:
        failed = sorted(
            adapter for adapter, result in state.smoke_results.items() if not result.get("ok")
        )
        if failed:
            print(
                f"smoke test: {len(failed)} adapter(s) failed: {', '.join(failed)}",
                file=sys.stderr,
            )
            return EXIT_SMOKE_FAILURE
    return EXIT_OK


def _run_steps(config: Config, steps: list[str]) -> _PipelineState:
    """Execute the selected steps in order, threading outputs between them."""
    state = _PipelineState()
    for step in steps:
        if step == "stage":
            _run_stage(config, state)
        elif step == "register":
            _run_register(config, state)
        elif step == "deploy":
            _run_deploy(config, state)
        elif step == "smoke":
            _run_smoke(config, state)
        state.ran.append(step)
    return state


def _run_stage(config: Config, state: _PipelineState) -> None:
    print("==> stage: staging weights into Unity Catalog ...")
    state.manifest = stage_weights(config)


def _run_register(config: Config, state: _PipelineState) -> None:
    if state.manifest is None:
        raise CliError("step 'register' requires a manifest; include 'stage' in --steps")
    print("==> register: logging + registering the model to Unity Catalog ...")
    name, version = log_and_register(config, state.manifest)
    state.model_full_name = name
    state.model_version = version


def _run_deploy(config: Config, state: _PipelineState) -> None:
    if state.model_full_name is None or state.model_version is None:
        raise CliError(
            "step 'deploy' requires a registered model + version; "
            "include 'register' in --steps"
        )
    print("==> deploy: creating/updating the serving endpoint ...")
    state.deployment = deploy_endpoint(
        config, state.model_full_name, state.model_version
    )


def _run_smoke(config: Config, state: _PipelineState) -> None:
    print("==> smoke: querying every adapter on the endpoint ...")
    state.smoke_results = smoke_test(config)


def _print_dry_run_plan(config: Config, steps: list[str]) -> None:
    """Describe what each selected step would do, making no real calls."""
    uc = config.unity_catalog
    serving = config.serving
    model_full_name = f"{uc.catalog}.{uc.db_schema}.{uc.model_name}"
    volume_prefix = (
        f"/Volumes/{uc.catalog}/{uc.db_schema}/{uc.volume}/{uc.model_name}"
    )
    adapters = ", ".join(a.name for a in config.adapters)

    print("DRY RUN — no real calls will be made.")
    print(f"  config profile (OAuth): {config.databricks.profile}")
    print(f"  steps (in order):       {', '.join(steps)}")
    print()
    for step in steps:
        if step == "stage":
            print(f"  [stage]    would stage base + {len(config.adapters)} adapter(s) "
                  f"({config.source_type}) under {volume_prefix}/")
        elif step == "register":
            print(f"  [register] would log the pyfunc chat model and register it as "
                  f"{model_full_name} (env_pack=databricks_model_serving)")
        elif step == "deploy":
            print(f"  [deploy]   would create/update endpoint {serving.endpoint_name!r} "
                  f"({serving.workload_type}/{serving.workload_size}, "
                  f"scale_to_zero={serving.scale_to_zero})")
        elif step == "smoke":
            print(f"  [smoke]    would query endpoint {serving.endpoint_name!r} "
                  f"for adapters: {adapters}")


def _print_summary(config: Config, state: _PipelineState) -> None:
    """Print a clear summary of everything that ran."""
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"steps run: {', '.join(state.ran) or '(none)'}")

    if state.manifest is not None:
        print("\nstaged weights (manifest):")
        for key, path in state.manifest.items():
            print(f"  {key}: {path}")

    if state.model_full_name is not None:
        print(f"\nregistered model: {state.model_full_name} (version {state.model_version})")

    if state.deployment is not None:
        dep = state.deployment
        ready = "READY" if dep.ready else f"NOT READY (config_update={dep.config_update})"
        print(f"\nendpoint: {dep.endpoint_name} [{dep.action}] -> {ready}")

    if state.smoke_results is not None:
        print("\nsmoke test results:")
        for adapter, result in state.smoke_results.items():
            if result.get("ok"):
                print(f"  PASS {adapter}: {result.get('snippet', '')}")
            else:
                print(f"  FAIL {adapter}: {result.get('error', '')}")


if __name__ == "__main__":  # pragma: no cover - module entry shim
    raise SystemExit(main())
