# multi-adapter-serving

Serves Mistral-7B-Instruct-v0.2 and `ner`, a LoRA adapter trained on it for entity extraction, from
one Databricks [custom LLM serving](https://docs.databricks.com/aws/en/machine-learning/model-serving/serve-custom-llms)
endpoint on a single A10G (`GPU_MEDIUM`). Each request picks the base model or the adapter.
Adapters share the base model's GPU, so adding more doesn't add GPUs.

Custom LLM serving can't select LoRA adapters on its own yet: requests use `model` for the
endpoint name, and that's the field vLLM uses to pick an adapter. So the endpoint runs vLLM with
`--enable-lora` behind a thin proxy (`serving/proxy.py`) that reads the adapter from
`custom_inputs.adapter` and passes it to vLLM.

```
client --custom_inputs.adapter="ner"--> proxy :8080 --model="ner"--> vLLM (base + ner)
```

## Layout

- `notebooks/register.py`: downloads any missing weights, tests the serving command, registers the
  model in Unity Catalog, and creates or updates the endpoint.
- `serving/`: the proxy and its launcher, bundled into the model with the weights.
- `resources/multi_lora.job.yml`: the Serverless GPU (A10) job that runs the notebook.
- `scripts/smoke_test.py`: queries the base model and each adapter over OAuth.

## Deploy

Pick an OAuth CLI profile (create one with `databricks auth login`) and a Unity Catalog schema you
can create models and volumes in:

```bash
export DATABRICKS_CONFIG_PROFILE=my-profile
export BUNDLE_VAR_catalog=my_catalog BUNDLE_VAR_schema=my_schema

databricks bundle deploy
databricks bundle run multi_lora                        # build, register, and deploy
databricks bundle run multi_lora --params stage=verify  # stop after the notebook test
```

To keep the catalog and schema between shells, put them in
`.databricks/bundle/dev/variable-overrides.json` (gitignored):

```json
{"catalog": "my_catalog", "schema": "my_schema"}
```

Each run (`stage=verify` stops after step 2):

1. Downloads any weights missing from `/Volumes/<catalog>/<schema>/weights/multi_lora_vllm/` from
   Hugging Face (about 14 GB the first time, no token needed).
2. Runs the serving command in the notebook and checks that the base model and each adapter answer.
3. Registers a new version of `<catalog>.<schema>.multi_lora_vllm`.
4. Creates the `multi-lora-vllm` endpoint, or moves it to the new version while the current one
   keeps serving.

A full run takes about 30 minutes.

## Query

```bash
uv run scripts/smoke_test.py   # base model plus each adapter
```

Set `custom_inputs.adapter` to pick an adapter, or leave it out for the base model:

```json
{"messages": [{"role": "user", "content": "..."}], "custom_inputs": {"adapter": "ner"}}
```

The reply's `model` field names the model that answered (`base` or `ner`). An unknown adapter gets
HTTP 400 with the list of valid names.

`ner` expects the prompt format it was trained on, and replies with JSON under 4 fixed keys:
`person`, `organization`, `location`, and `miscellaneous`. With the OpenAI client, signed in
through your CLI profile:

```python
from databricks.sdk import WorkspaceClient
from openai import OpenAI

w = WorkspaceClient()
client = OpenAI(
    base_url=f"{w.config.host}/serving-endpoints",
    api_key=w.config.authenticate()["Authorization"].removeprefix("Bearer "),
)

# The prompt format ner was trained on (including its double space).
PREFIX = (
    "Your task is a Named Entity Recognition (NER) task. Predict the category of each entity, then place the "
    "entity into the list associated with the  category in an output JSON payload. Below is an example:\n"
    'Input: EU rejects German call to boycott British lamb . Output: {"person": [], "organization": ["EU"], '
    '"location": [], "miscellaneous": ["German", "British"]}\nNow, complete the task.\nInput: '
)
sentence = (
    "Ford is recalling some Mustang Mach-E vehicles in the United States, "
    "the National Highway Traffic Safety Administration said."
)

resp = client.chat.completions.create(
    model="multi-lora-vllm",
    messages=[{"role": "user", "content": f"{PREFIX}{sentence} Output: "}],
    temperature=0,
    extra_body={"custom_inputs": {"adapter": "ner"}},
)
print(resp.model, resp.choices[0].message.content.strip())
```

This prints:

```
ner {"person": [], "organization": ["Ford", "National Highway Traffic Safety Administration"], "location": ["United States"], "miscellaneous": ["Mustang Mach-E"]}
```

Without `custom_inputs`, the same prompt goes to the base model, which makes up its own keys (here
`vehicle` and `event`), so code that parses the reply breaks.

Streaming (`stream=True`) works the same way. SQL `ai_query` can't pass `custom_inputs`, so it
always gets the base model.

## Add an adapter

1. In `notebooks/register.py`, add its name to `ADAPTERS`, its Volume path to `WEIGHTS`, and its
   Hugging Face repo to `HF_REPOS` (or copy it into the Volume yourself). It must be a LoRA trained
   on the same base model.
2. Keep `MAX_LORA_RANK` at or above its rank (`r` in its `adapter_config.json`).
3. Rerun the job. The notebook test checks the new adapter before anything is registered.

## Cost

Scale-to-zero is off, so the endpoint bills for its GPU around the clock (about 20 DBU/hour) until
you delete it:

```bash
databricks serving-endpoints delete multi-lora-vllm
```

To stop idle billing instead, set `SCALE_TO_ZERO_ENABLED = True` in the notebook and rerun the job.
The first request after an idle period then waits several minutes while the model loads.
