"""Unit tests for serving-endpoint deployment (serving client fully mocked).

No Databricks workspace, no GPU, and no real (paid) deploy: the serving client is a
fake, and the readiness poll uses an injected fake clock so nothing actually sleeps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from databricks.sdk.errors import NotFound
from databricks.sdk.service.serving import (
    EndpointStateConfigUpdate,
    EndpointStateReady,
    ServingModelWorkloadType,
)

from multi_adapter_serving.config import Config
from multi_adapter_serving.deploy import (
    DeploymentResult,
    EndpointDeploymentError,
    _wait_until_ready,
    deploy_endpoint,
)

MODEL = "classic_stable_h0vpq7_catalog.default.multi_lora_vllm"


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


@dataclass
class FakeState:
    ready: Any
    config_update: Any


@dataclass
class FakeEndpoint:
    name: str
    state: FakeState


def ready_endpoint(name: str = "multi-lora-vllm") -> FakeEndpoint:
    return FakeEndpoint(
        name,
        FakeState(EndpointStateReady.READY, EndpointStateConfigUpdate.NOT_UPDATING),
    )


def not_ready_endpoint(name: str = "multi-lora-vllm") -> FakeEndpoint:
    return FakeEndpoint(
        name,
        FakeState(EndpointStateReady.NOT_READY, EndpointStateConfigUpdate.IN_PROGRESS),
    )


class FakeServingClient:
    """Fake serving-endpoints client.

    The first ``get`` call is the existence probe (raises ``NotFound`` when the
    endpoint is absent); every later ``get`` returns ``poll_endpoint``.
    """

    def __init__(self, *, exists: bool, poll_endpoint: FakeEndpoint | None = None) -> None:
        self.exists = exists
        self.poll_endpoint = poll_endpoint or ready_endpoint()
        self.created: list[tuple[str, Any]] = []
        self.updated: list[tuple[str, Any]] = []
        self.get_calls = 0

    def get(self, name: str) -> FakeEndpoint:
        self.get_calls += 1
        if self.get_calls == 1 and not self.exists:
            raise NotFound(f"no endpoint {name!r}")
        return self.poll_endpoint

    def create(self, name: str, *, config: Any) -> object:
        self.created.append((name, config))
        return None

    def update_config(self, name: str, *, served_entities: Any) -> object:
        self.updated.append((name, served_entities))
        return None


def _assert_gpu_served_entity(entity: Any) -> None:
    assert entity.entity_name == MODEL
    assert entity.entity_version == "7"
    assert entity.workload_type == ServingModelWorkloadType.GPU_MEDIUM
    assert entity.workload_size == "Small"
    assert entity.scale_to_zero_enabled is True


def test_create_path_builds_gpu_served_entity() -> None:
    client = FakeServingClient(exists=False)

    result = deploy_endpoint(make_config(), MODEL, 7, serving_client=client)

    # Created (not updated) because the endpoint did not exist.
    assert client.created and not client.updated
    name, core_config = client.created[0]
    assert name == "multi-lora-vllm"
    assert core_config.name == "multi-lora-vllm"
    assert len(core_config.served_entities) == 1
    _assert_gpu_served_entity(core_config.served_entities[0])

    assert isinstance(result, DeploymentResult)
    assert result.action == "created"
    assert result.endpoint_name == "multi-lora-vllm"
    assert result.ready is True
    assert result.config_update == "NOT_UPDATING"


def test_update_path_when_endpoint_exists() -> None:
    client = FakeServingClient(exists=True)

    result = deploy_endpoint(make_config(), MODEL, 7, serving_client=client)

    # Updated (not created) because the endpoint already existed.
    assert client.updated and not client.created
    name, served_entities = client.updated[0]
    assert name == "multi-lora-vllm"
    assert len(served_entities) == 1
    _assert_gpu_served_entity(served_entities[0])
    assert result.action == "updated"
    assert result.ready is True


def test_served_entity_version_is_stringified() -> None:
    client = FakeServingClient(exists=False)

    deploy_endpoint(make_config(), MODEL, "12", serving_client=client)

    _, core_config = client.created[0]
    assert core_config.served_entities[0].entity_version == "12"


def test_scale_to_zero_and_workload_size_passthrough() -> None:
    config = make_config(
        serving={
            "endpoint_name": "ep",
            "workload_type": "GPU_LARGE",
            "workload_size": "Medium",
            "scale_to_zero": False,
        }
    )
    client = FakeServingClient(exists=False)

    deploy_endpoint(config, MODEL, 3, serving_client=client)

    entity = client.created[0][1].served_entities[0]
    assert entity.workload_type == ServingModelWorkloadType.GPU_LARGE
    assert entity.workload_size == "Medium"
    assert entity.scale_to_zero_enabled is False


def test_poll_false_does_not_wait() -> None:
    client = FakeServingClient(exists=False, poll_endpoint=not_ready_endpoint())

    result = deploy_endpoint(make_config(), MODEL, 7, serving_client=client, poll=False)

    # No readiness gate: returns whatever state the endpoint currently reports.
    assert result.action == "created"
    assert result.ready is False
    assert result.config_update == "IN_PROGRESS"


def test_invalid_workload_type_raises() -> None:
    config = make_config(
        serving={
            "endpoint_name": "ep",
            "workload_type": "NOT_A_GPU",
            "workload_size": "Small",
            "scale_to_zero": True,
        }
    )
    client = FakeServingClient(exists=False)

    with pytest.raises(EndpointDeploymentError, match="invalid serving.workload_type"):
        deploy_endpoint(config, MODEL, 7, serving_client=client)


# --- readiness polling (tested directly with a fake clock; no real sleeping) -------


class SequenceClient:
    """Returns a sequence of endpoints from successive ``get`` calls."""

    def __init__(self, endpoints: list[FakeEndpoint]) -> None:
        self.endpoints = endpoints
        self.calls = 0

    def get(self, name: str) -> FakeEndpoint:
        endpoint = self.endpoints[min(self.calls, len(self.endpoints) - 1)]
        self.calls += 1
        return endpoint


class FakeClock:
    def __init__(self, times: list[float]) -> None:
        self.times = times
        self.i = 0

    def __call__(self) -> float:
        value = self.times[min(self.i, len(self.times) - 1)]
        self.i += 1
        return value


def test_wait_until_ready_polls_until_ready() -> None:
    client = SequenceClient([not_ready_endpoint(), ready_endpoint()])
    sleeps: list[float] = []

    endpoint = _wait_until_ready(
        client,  # type: ignore[arg-type]
        "multi-lora-vllm",
        timeout_s=1000,
        poll_interval_s=5,
        sleep=sleeps.append,
        monotonic=FakeClock([0, 1, 2, 3]),
    )

    assert endpoint.state.ready == EndpointStateReady.READY
    assert client.calls == 2  # one NOT_READY get, then one READY get
    assert sleeps == [5]  # slept once between the two polls


def test_wait_until_ready_times_out() -> None:
    client = SequenceClient([not_ready_endpoint()])

    with pytest.raises(EndpointDeploymentError, match="timed out"):
        _wait_until_ready(
            client,  # type: ignore[arg-type]
            "multi-lora-vllm",
            timeout_s=10,
            poll_interval_s=5,
            sleep=lambda _s: None,
            monotonic=FakeClock([0, 100]),
        )


def test_wait_until_ready_raises_on_failed_update() -> None:
    failed = FakeEndpoint(
        "multi-lora-vllm",
        FakeState(EndpointStateReady.NOT_READY, EndpointStateConfigUpdate.UPDATE_FAILED),
    )
    client = SequenceClient([failed])

    with pytest.raises(EndpointDeploymentError, match="deployment failed"):
        _wait_until_ready(
            client,  # type: ignore[arg-type]
            "multi-lora-vllm",
            timeout_s=1000,
            sleep=lambda _s: None,
            monotonic=FakeClock([0, 1]),
        )
