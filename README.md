# multi-adapter-serving

Config-driven foundation for **multi-LoRA (adapter) serving on Databricks** with
vLLM. This package provides the shared building blocks the rest of the project
depends on:

1. A **configuration contract** (`config.yaml`) with a pydantic v2 schema + loader.
2. A **weight-staging** step that materializes a base model plus its LoRA adapters
   into a Unity Catalog Volume.

Nothing here is hardcoded to a specific model — everything is driven by config.

## Layout

```
src/multi_adapter_serving/
  config.py      # pydantic schema + load_config()
  staging.py     # stage_weights() — HF/local -> Unity Catalog Volume
  model.py       # log_and_register() — MLflow ChatModel -> Unity Catalog
  serving/       # entrypoint command + multi-LoRA proxy in front of vLLM
  deploy.py      # deploy_endpoint() — create/update the Serverless GPU endpoint
  client.py      # query() — {adapter, prompt} -> assistant text
  smoke_test.py  # smoke_test() — exercise every adapter on the endpoint
  cli.py         # multi-adapter-serve — end-to-end stage->register->deploy->smoke
config.example.yaml   # annotated example of the shared config contract
tests/                # unit tests (no network / Databricks access required)
```

## Requirements

- Python **3.11**
- [uv](https://docs.astral.sh/uv/) for environment and dependency management
- Databricks auth via **OAuth** using a config profile (default profile:
  `fevm-aws`) — never a personal access token.

## Setup

```bash
uv sync            # create the environment from pyproject.toml + uv.lock
```

> **vLLM note:** `vllm` is pinned (`==0.11.2`) but gated to Linux
> (`sys_platform == 'linux'`) because it only ships Linux/CUDA wheels. It is the
> GPU serving-host dependency and is not imported by `config.py`, `staging.py`, or
> the tests, so the package installs and all gates pass on macOS for development.

## Configuration

Copy `config.example.yaml` to `config.yaml` and fill in the values. The schema is
the **shared contract** across tracks; field names are fixed. See
`config.example.yaml` for the annotated shape and `src/multi_adapter_serving/config.py`
for the validation rules.

```python
from multi_adapter_serving import load_config

config = load_config("config.yaml")
```

## Staging weights

`stage_weights` downloads (for `source_type: huggingface`) or validates (for
`source_type: local`) the base model and each adapter, uploads them into the UC
Volume, and returns a manifest:

```python
from multi_adapter_serving import load_config, stage_weights

config = load_config("config.yaml")

# Plan only — logs the plan, transfers nothing:
manifest = stage_weights(config, dry_run=True)

# Real run (HF token for gated repos picked up from $HF_TOKEN):
manifest = stage_weights(config)
```

The manifest maps each logical weight name to its final UC Volume path:

```python
{
    "base": "/Volumes/<catalog>/<schema>/<volume>/<model_name>/base",
    "<adapter-name>": "/Volumes/<catalog>/<schema>/<volume>/<model_name>/adapters/<adapter-name>",
    ...
}
```

All network and Unity Catalog operations are injectable
(`downloader`, `uploader`, `local_path_exists`, `uc_path_exists`) so the planner
can be unit tested without any network or Databricks access.

## Deploying the endpoint

The model is logged with `env_pack="databricks_model_serving"` for
[express deployment of a custom LLM](https://docs.databricks.com/aws/en/machine-learning/model-serving/serve-custom-llms),
so `deploy_endpoint` just points a **Serverless GPU** serving endpoint at the
Unity Catalog registered model + version:

```python
from multi_adapter_serving import load_config
from multi_adapter_serving.deploy import deploy_endpoint

config = load_config("config.yaml")
result = deploy_endpoint(config, "catalog.schema.model_name", version)
print(result.endpoint_name, result.action, result.ready)
```

It creates the endpoint named `config.serving.endpoint_name` if it does not
exist (otherwise updates its config in place), then polls until `READY` or
surfaces a useful error. The served entity is built from config:

| served-entity field    | source                       |
| ---------------------- | ---------------------------- |
| `entity_name`          | `model_full_name` (UC model) |
| `entity_version`       | `model_version`              |
| `workload_type`        | `serving.workload_type` (GPU)|
| `workload_size`        | `serving.workload_size`      |
| `scale_to_zero_enabled`| `serving.scale_to_zero`      |

The serving client is injectable (`serving_client`) so deployment is fully unit
testable with **no Databricks workspace, no GPU, and no real (paid) deploy**.

## Querying an adapter

`query` is the user-facing two-argument interface — pick an adapter by name and
send a prompt:

```python
from multi_adapter_serving.client import query

result = query("multi-lora-vllm", adapter="ner", prompt="Tag the entities: ...")
print(result.text)   # assistant reply
print(result.raw)    # full response
```

The adapter travels in `custom_inputs.adapter` (the proxy contract). The exact
`llm/v1/chat` wire payload is:

```json
{
  "messages": [{"role": "user", "content": "<prompt>"}],
  "custom_inputs": {"adapter": "<adapter>"}
}
```

plus any pass-through chat kwargs (`temperature`, `max_tokens`, `stream`, ...).
The endpoint is invoked over **OAuth** via the Databricks SDK (config profile) —
never a personal access token.

## Smoke-testing every adapter

```python
from multi_adapter_serving.smoke_test import smoke_test

results = smoke_test(config)  # {adapter: {"ok": bool, "snippet"/"error": str}}
```

`smoke_test` queries every adapter once, never aborting on a single failure, so
the report shows pass/fail for all adapters at once.

## End-to-end CLI

The `multi-adapter-serve` console script runs the full workflow — stage →
register → deploy → smoke — from a single config:

```bash
multi-adapter-serve --config config.yaml                      # all steps, in order
multi-adapter-serve --config config.yaml --steps stage,register
multi-adapter-serve --config config.yaml --steps smoke         # query an existing endpoint
multi-adapter-serve --config config.yaml --dry-run             # plan only, no real calls
```

| flag        | meaning                                                            |
| ----------- | ------------------------------------------------------------------ |
| `--config`  | path to the config YAML (default: `config.yaml`)                   |
| `--steps`   | comma subset of `stage,register,deploy,smoke` (always run in order)|
| `--dry-run` | print the plan and make **no** real Databricks/network calls       |

It prints a clear summary: the staged manifest, the registered model + version,
the endpoint name + state, and per-adapter smoke results.

An incomplete `--steps` subset (e.g. `stage,deploy` without `register`) is
rejected **up front** — before any weights are staged — and each step's runtime
failures produce a concise, controlled error (no traceback). Exit codes are
distinct so callers can tell the failure modes apart:

| exit code | meaning                                                       |
| --------- | ------------------------------------------------------------- |
| `0`       | success                                                       |
| `1`       | pipeline ran, but at least one adapter failed its smoke test  |
| `2`       | usage error: bad arguments/config or an incomplete `--steps`  |
| `3`       | operational error: a step (stage/register/deploy/smoke) raised|

## Quality gates

```bash
uv run ruff check .
uv run mypy src
uv run pytest
```
