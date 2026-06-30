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

## Quality gates

```bash
uv run ruff check .
uv run mypy src
uv run pytest
```
