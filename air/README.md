# Remote `register` on serverless GPU (`air`)

The `register` step must run on a Databricks **serverless GPU** runtime, because our
custom-LLM served entity can only deploy as an *express deployment*, which requires
`env_pack='databricks_model_serving'` at register time. `env_pack` packs the current
Python environment as the serving image and can only be produced on a serverless GPU
runtime — not a laptop. `stage`, `deploy`, and `smoke` stay local.

We use the Databricks AI Runtime CLI (`air`, package `databricks-air`) to run just the
`register` step remotely: `air run` executes arbitrary code on the serverless-GPU client
image. Inside that job `is_in_databricks_serverless_runtime()` is `True`, so:

- model.py's `env_pack` gate fires and the express image is packed, and
- MLflow auth switches to **ambient job credentials** (there is no `~/.databrickscfg` and
  no config profile on the node) — see `_configure_databricks_tracking` in `model.py`.

## Run it

```bash
# 1) Register remotely on serverless GPU (creates the READY UC model version + env_pack).
air run --file air/register.yaml -p fevm-aws --watch

# 2) Deploy + smoke-test locally, off the UC version register just created. `deploy`
#    resolves the latest READY Unity Catalog version automatically (no in-run register).
multi-adapter-serve --config config.yaml --steps deploy,smoke
```

## Prerequisites

- **`air` CLI installed** — `uv add databricks-air` (or `pip install databricks-air`) in
  the environment you launch it from.
- **Databricks AI Runtime preview enabled** for the workspace/account.
- **An OAuth profile** in `~/.databrickscfg` (here `fevm-aws`) for the `-p` flag. This is
  used to submit the `air` job and, locally, by `deploy`/`smoke`. Never a PAT.

## Notes

- `experiment_name` in `register.yaml` is a simple `multi_lora_vllm_register` name (air
  requires alnum/hyphen/underscore, not a workspace path). The air job provides the active
  MLflow run/experiment; the local-profile experiment path logic in `model.py` is skipped
  on the serverless runtime.
- `environment.dependencies` mirrors `pyproject.toml`'s runtime deps (vLLM/transformers
  pins included) so `pip install -e .` resolves on the node. Keep them in lockstep with
  `pyproject.toml`.
- `code_source.snapshot.root_path: .` uploads this repo (including `config.yaml`) to the
  node so the `command` can `pip install -e .` and run the CLI.
