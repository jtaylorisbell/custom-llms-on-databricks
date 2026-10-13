# multi-adapter-serving

Serves Mistral-7B-Instruct-v0.2 and 2 LoRA adapters trained on it, `ner` (entity extraction) and
`code` (coding), from 1 Databricks
[custom LLM serving](https://docs.databricks.com/aws/en/machine-learning/model-serving/serve-custom-llms)
endpoint on a single A10G (`GPU_MEDIUM`). Each request picks its adapter, so 1 GPU serves all 3
models.

Custom LLM serving can't select LoRA adapters on its own yet: requests use `model` for the
endpoint name, and that's the field vLLM uses to pick an adapter. So the endpoint runs vLLM with
`--enable-lora` behind a thin proxy (`serving/proxy.py`) that reads the adapter from
`custom_inputs.adapter` and passes it to vLLM.

```
client --custom_inputs.adapter="ner"--> proxy :8080 --model="ner"--> vLLM (base + ner + code)
```

## Layout

- `notebooks/register.py`: downloads any missing weights, tests the serving command, registers the
  model in Unity Catalog, and creates or updates the endpoint.
- `serving/`: the proxy and its launcher, bundled into the model with the weights.
- `resources/multi_lora.job.yml`: the Serverless GPU (A10) job that runs the notebook.
- `scripts/smoke_test.py`: queries the base model and each adapter over OAuth.

## Deploy

```bash
databricks bundle deploy -p fevm-aws
databricks bundle run multi_lora -p fevm-aws                        # build, register, and deploy
databricks bundle run multi_lora -p fevm-aws --params stage=verify  # stop after the notebook test
```

Each run:

1. Downloads any weights missing from
   `/Volumes/classic_stable_h0vpq7_catalog/default/weights/multi_lora_vllm/` from Hugging Face
   (about 14 GB the first time, no token needed).
2. Runs the serving command in the notebook and checks that the base model and each adapter answer.
3. Registers a new version of `classic_stable_h0vpq7_catalog.default.multi_lora_vllm`.
4. Creates the `multi-lora-vllm` endpoint, or moves it to the new version while the current one
   keeps serving.

A full run takes about 30 minutes.

## Query

```bash
uv run scripts/smoke_test.py --profile fevm-aws   # base model plus each adapter
```

Set `custom_inputs.adapter` to pick an adapter, or leave it out for the base model:

```json
{"messages": [{"role": "user", "content": "..."}], "custom_inputs": {"adapter": "ner"}}
```

The reply's `model` field names the model that answered (`base`, `ner`, or `code`). An unknown
adapter gets HTTP 400 with the list of valid names.

With the OpenAI client, signed in through your CLI profile:

```python
from databricks.sdk import WorkspaceClient
from openai import OpenAI

w = WorkspaceClient(profile="fevm-aws")
client = OpenAI(
    base_url=f"{w.config.host}/serving-endpoints",
    api_key=w.config.authenticate()["Authorization"].removeprefix("Bearer "),
)
resp = client.chat.completions.create(
    model="multi-lora-vllm",
    messages=[{"role": "user", "content": "Write a Python function that reverses a string."}],
    extra_body={"custom_inputs": {"adapter": "code"}},
)
print(resp.model, resp.choices[0].message.content)
```

Streaming (`stream=True`) works the same way. SQL `ai_query` can't pass `custom_inputs`, so it
always gets the base model.

## Add an adapter

1. In `notebooks/register.py`, add its name to `ADAPTERS`, its Volume path to `WEIGHTS`, and its
   Hugging Face repo to `HF_REPOS` (or copy it into the Volume yourself). It must be a LoRA trained
   on the same base model.
2. Set `MAX_LORAS` to the number of adapters so they all stay loaded, and keep `MAX_LORA_RANK` at
   or above each adapter's rank (`r` in its `adapter_config.json`).
3. Rerun the job. The notebook test checks the new adapter before anything is registered.

## Cost

Scale-to-zero is off, so the endpoint bills for its GPU around the clock (about 20 DBU/hour) until
you delete it:

```bash
databricks serving-endpoints delete multi-lora-vllm -p fevm-aws
```

To stop idle billing instead, set `SCALE_TO_ZERO_ENABLED = True` in the notebook and rerun the job.
The first request after an idle period then waits several minutes while the model loads.
