# multi-adapter-serving

One base model (Mistral-7B-Instruct-v0.2) serving several LoRA adapters from a single Databricks
[custom LLM serving](https://docs.databricks.com/aws/en/machine-learning/model-serving/serve-custom-llms)
endpoint (vLLM on `GPU_MEDIUM`, 1x A10G). Custom LLM serving doesn't support LoRA adapters
natively yet, so the endpoint's entrypoint runs vLLM with `--enable-lora` behind a thin proxy
(`serving/proxy.py`) that maps `custom_inputs.adapter` onto vLLM's `model` field.

## Layout

- `notebooks/register.py`: downloads any missing weights into the Volume, tests the entrypoint in
  the notebook, logs and registers the model (`env_pack`), then creates or updates the endpoint.
- `serving/`: the proxy and its launcher, bundled into the model as artifacts.
- `resources/multi_lora.job.yml`: Serverless GPU (A10) job that runs the notebook.
- `scripts/smoke_test.py`: queries the endpoint once per adapter over OAuth.

## Deploy

The weights live at
`/Volumes/classic_stable_h0vpq7_catalog/default/weights/multi_lora_vllm/{base,adapters/<name>}`.
The first run downloads whatever is missing there from Hugging Face (`HF_REPOS` in the notebook,
about 14 GB, no token needed); later runs reuse it.

```bash
databricks bundle deploy -p fevm-aws
databricks bundle run multi_lora -p fevm-aws                        # register + deploy
databricks bundle run multi_lora -p fevm-aws --params stage=verify  # in-notebook test only
```

## Query

```bash
uv run scripts/smoke_test.py --profile fevm-aws
```

Pick an adapter with `custom_inputs.adapter`; omit it to get the base model:

```json
{"messages": [{"role": "user", "content": "..."}], "custom_inputs": {"adapter": "ner"}}
```

With the OpenAI client, pass `extra_body={"custom_inputs": {"adapter": "ner"}}`.

## Lessons from earlier failed deploys

- Model Serving sends chat requests to `POST /invocations` and probes `GET /v1/models` for
  readiness. The proxy must serve both, or the endpoint never reaches READY.
- The serving container runs Databricks' own metrics exporter on port 8081, and it answers every
  path with 200. vLLM therefore takes a free port picked at startup (`--vllm-port 0`), and the
  launcher only treats vLLM as ready once `/v1/models` lists the base model. A fixed 8081 worked
  in July but now crash-loops.
- The entrypoint runs from the model's `artifacts/` dir, so the weights and the proxy are bundled
  as artifacts and referenced by bare relative names.
- Log only `extra_pip_requirements=["mlflow==3.12.0"]`. Listing vllm/transformers makes the serving
  environment re-resolve an OpenSSL stack whose FIPS self-test fails at startup.
- Custom LLM endpoints don't autoscale. Size with `workload_size`; the API rejects autoscaling configs.
- A version's express-deployment staging doesn't last. Redeploying an old version fails with
  "with entrypoint is only supported with express deployments" until you re-stage it:
  `databricks api post '/api/2.0/serving-endpoints:stageDeployment' --json '{"model_name": "<catalog.schema.model>", "model_version": "<n>"}'`.

## Cost

Scale-to-zero is off, so the endpoint bills for its GPU around the clock (about 20 DBU/hour)
until you delete it. Set `SCALE_TO_ZERO_ENABLED = True` in the notebook to stop idle billing, at
the cost of a cold start of several minutes on the first request.
