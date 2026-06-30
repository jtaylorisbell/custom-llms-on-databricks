"""Create-or-update the Databricks Model Serving endpoint for the logged model.

Follows the Databricks custom-LLM workflow: the model was logged with
``env_pack='databricks_model_serving'`` (see :mod:`multi_adapter_serving.model`)
for express deployment, so here we only need to point a Serverless GPU serving
endpoint at the Unity Catalog registered model + version.

:func:`deploy_endpoint` creates the endpoint named ``config.serving.endpoint_name``
if it does not exist, otherwise updates its served-entity config in place, then
(optionally) polls until the endpoint is ``READY`` or surfaces a useful error.

The serving client is injectable (``serving_client``) so the whole flow can be unit
tested with a fake — no Databricks workspace, no GPU, and no real (paid) deploy.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from databricks.sdk.errors import NotFound
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    EndpointStateConfigUpdate,
    EndpointStateReady,
    ServedEntityInput,
    ServingEndpointDetailed,
    ServingModelWorkloadType,
)

from multi_adapter_serving.config import Config, ServingConfig

__all__ = [
    "DeploymentResult",
    "EndpointDeploymentError",
    "ServingEndpointsClient",
    "deploy_endpoint",
]

logger = logging.getLogger(__name__)

# How often we re-check endpoint state while polling for readiness.
_DEFAULT_POLL_INTERVAL_S = 10.0

# Terminal config-update states that mean the deployment will never become READY.
_FAILED_UPDATES = frozenset(
    {
        EndpointStateConfigUpdate.UPDATE_FAILED,
        EndpointStateConfigUpdate.UPDATE_CANCELED,
    }
)

# Config-update states compatible with a fully-rolled-out READY endpoint.
_SETTLED_UPDATES = frozenset({None, EndpointStateConfigUpdate.NOT_UPDATING})

# Ready-state values that mean the endpoint has terminally failed and will never
# become healthy. ``EndpointStateReady`` does not define ``FAILED`` in every SDK
# version, so we match on the string value defensively (rather than an enum member
# that may not exist), keeping this robust across SDK versions.
_FAILED_READY_VALUES = frozenset({"FAILED"})


class EndpointDeploymentError(RuntimeError):
    """Raised when a serving endpoint deployment fails or times out."""


class ServingEndpointsClient(Protocol):
    """The subset of ``WorkspaceClient.serving_endpoints`` this module uses.

    Captured as a Protocol so the real Databricks SDK client (OAuth via the config
    profile) can be swapped for a fake in unit tests.
    """

    def get(self, name: str) -> ServingEndpointDetailed: ...

    def create(self, name: str, *, config: EndpointCoreConfigInput) -> object: ...

    def update_config(
        self, name: str, *, served_entities: list[ServedEntityInput]
    ) -> object: ...


@dataclass(frozen=True)
class DeploymentResult:
    """Outcome of a :func:`deploy_endpoint` call."""

    endpoint_name: str
    # "created" if the endpoint was newly created, "updated" if it already existed.
    action: str
    # True once the endpoint reports state.ready == READY.
    ready: bool
    # The endpoint's config-update state (e.g. "NOT_UPDATING", "IN_PROGRESS"), or
    # None if unknown. Useful when poll=False and the rollout is still in flight.
    config_update: str | None
    # The raw ServingEndpointDetailed returned by the SDK (full state/info).
    endpoint: ServingEndpointDetailed


def deploy_endpoint(
    config: Config,
    model_full_name: str,
    model_version: int | str,
    *,
    serving_client: ServingEndpointsClient | None = None,
    poll: bool = True,
    timeout_s: float = 1800,
) -> DeploymentResult:
    """Create or update the serving endpoint for ``model_full_name`` version.

    Configures a single served entity pointing at the Unity Catalog registered
    model + version on a Serverless GPU workload, using
    ``config.serving.{workload_type,workload_size,scale_to_zero}``. The endpoint is
    created if absent, otherwise its config is updated in place.

    Args:
        config: Validated configuration (supplies the endpoint name, GPU workload
            settings, and the OAuth profile for the default client).
        model_full_name: Unity Catalog three-level name of the registered model
            (``catalog.schema.model``), as returned by ``log_and_register``.
        model_version: Registered model version (as returned by ``log_and_register``).
        serving_client: Injectable serving-endpoints client. Defaults to a Databricks
            SDK ``WorkspaceClient(profile=...).serving_endpoints`` (OAuth, never a PAT).
        poll: If ``True`` (default), wait until the endpoint is READY (or fails),
            bounded by ``timeout_s``. If ``False``, return immediately after issuing
            the create/update call.
        timeout_s: Maximum seconds to wait for readiness when ``poll`` is ``True``.

    Returns:
        A :class:`DeploymentResult` carrying the action taken and endpoint state/info.

    Raises:
        EndpointDeploymentError: If the deployment reports a failed config update, or
            if it does not reach READY within ``timeout_s`` while polling.
    """
    serving = config.serving
    endpoint_name = serving.endpoint_name
    client = (
        serving_client
        if serving_client is not None
        else _default_serving_client(config.databricks.profile)
    )

    served_entity = _build_served_entity(serving, model_full_name, model_version)

    if _endpoint_exists(client, endpoint_name):
        action = "updated"
        logger.info(
            "Updating serving endpoint %r -> %s (version %s)",
            endpoint_name,
            model_full_name,
            model_version,
        )
        client.update_config(name=endpoint_name, served_entities=[served_entity])
    else:
        action = "created"
        logger.info(
            "Creating serving endpoint %r -> %s (version %s)",
            endpoint_name,
            model_full_name,
            model_version,
        )
        client.create(
            name=endpoint_name,
            config=EndpointCoreConfigInput(
                name=endpoint_name,
                served_entities=[served_entity],
            ),
        )

    if poll:
        endpoint = _wait_until_ready(client, endpoint_name, timeout_s)
    else:
        endpoint = client.get(endpoint_name)

    return _result(endpoint_name, action, endpoint)


def _build_served_entity(
    serving: ServingConfig, model_full_name: str, model_version: int | str
) -> ServedEntityInput:
    """Build the GPU served entity for the UC-registered model + version."""
    return ServedEntityInput(
        name=_served_entity_name(model_full_name, model_version),
        entity_name=model_full_name,
        entity_version=str(model_version),
        workload_type=_workload_type(serving.workload_type),
        workload_size=serving.workload_size,
        scale_to_zero_enabled=serving.scale_to_zero,
    )


def _served_entity_name(model_full_name: str, model_version: int | str) -> str:
    """Deterministic served-entity name (UC dots are invalid in entity names)."""
    model = model_full_name.rsplit(".", 1)[-1]
    return f"{model}-{model_version}"


def _workload_type(value: str) -> ServingModelWorkloadType:
    """Coerce the configured workload type into the SDK enum with a clear error."""
    try:
        return ServingModelWorkloadType(value)
    except ValueError as exc:
        valid = ", ".join(member.value for member in ServingModelWorkloadType)
        raise EndpointDeploymentError(
            f"invalid serving.workload_type {value!r}; must be one of: {valid}"
        ) from exc


def _endpoint_exists(client: ServingEndpointsClient, endpoint_name: str) -> bool:
    """Return ``True`` if the named endpoint already exists."""
    try:
        client.get(endpoint_name)
    except NotFound:
        return False
    return True


def _wait_until_ready(
    client: ServingEndpointsClient,
    endpoint_name: str,
    timeout_s: float,
    *,
    poll_interval_s: float = _DEFAULT_POLL_INTERVAL_S,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> ServingEndpointDetailed:
    """Poll ``client.get`` until the endpoint is READY, fails, or times out.

    ``sleep`` and ``monotonic`` are injectable so the loop can be unit tested with a
    fake clock (no real waiting).
    """
    deadline = monotonic() + timeout_s
    while True:
        endpoint = client.get(endpoint_name)
        state = endpoint.state
        config_update = state.config_update if state is not None else None
        ready = state.ready if state is not None else None
        ready_value = _enum_value(ready)

        # Terminal failure: a failed config update OR a failed ready state. Both
        # mean the endpoint will never become healthy, so fail fast rather than
        # spinning until the timeout.
        if config_update in _FAILED_UPDATES or ready_value in _FAILED_READY_VALUES:
            raise EndpointDeploymentError(
                f"serving endpoint {endpoint_name!r} deployment failed: "
                f"ready={ready_value}, config_update={_enum_value(config_update)}"
            )
        if ready == EndpointStateReady.READY and config_update in _SETTLED_UPDATES:
            logger.info("Serving endpoint %r is READY", endpoint_name)
            return endpoint

        if monotonic() >= deadline:
            raise EndpointDeploymentError(
                f"timed out after {timeout_s:g}s waiting for endpoint "
                f"{endpoint_name!r} to become READY "
                f"(last state: ready={_enum_value(ready)}, "
                f"config_update={_enum_value(config_update)})"
            )
        logger.info(
            "Waiting for endpoint %r (ready=%s, config_update=%s); next check in %gs",
            endpoint_name,
            _enum_value(ready),
            _enum_value(config_update),
            poll_interval_s,
        )
        sleep(poll_interval_s)


def _result(
    endpoint_name: str, action: str, endpoint: ServingEndpointDetailed
) -> DeploymentResult:
    """Assemble a :class:`DeploymentResult` from the endpoint's current state."""
    state = endpoint.state
    ready = state is not None and state.ready == EndpointStateReady.READY
    config_update = _enum_value(state.config_update) if state is not None else None
    return DeploymentResult(
        endpoint_name=endpoint_name,
        action=action,
        ready=ready,
        config_update=config_update,
        endpoint=endpoint,
    )


def _enum_value(value: object) -> str | None:
    """Return the ``.value`` of an enum-like object, or ``None``."""
    if value is None:
        return None
    return getattr(value, "value", str(value))


def _default_serving_client(profile: str) -> ServingEndpointsClient:
    """The real Databricks SDK serving-endpoints client (OAuth via ``profile``)."""
    from databricks.sdk import WorkspaceClient

    return WorkspaceClient(profile=profile).serving_endpoints
