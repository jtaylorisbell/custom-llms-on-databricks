# Databricks notebook source
# MAGIC %md
# MAGIC # Register + serve the multi-LoRA custom LLM
# MAGIC
# MAGIC Multi-LoRA adaptation of the Databricks
# MAGIC [custom LLM serving starter](https://docs.databricks.com/aws/en/notebooks/source/machine-learning/serve-custom-llms-starter.html).
# MAGIC Run this on **Serverless GPU** (Databricks recommends an **A10**).
# MAGIC
# MAGIC Instead of one model, we serve one base model with several LoRA adapters behind a thin
# MAGIC proxy: the proxy reads the adapter name from `custom_inputs.adapter` and rewrites the vLLM
# MAGIC `model` field (the Databricks gateway occupies `model` with the endpoint name). The proxy
# MAGIC lives in `proxy.py` / `proxy_entrypoint.py` and is attached as MLflow code paths.
# MAGIC
# MAGIC The base model + adapter weights are already staged in a Unity Catalog Volume by
# MAGIC `multi-adapter-serve --config config.yaml --steps stage`, so this notebook just references
# MAGIC them — it does not re-download anything.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set up the environment (Serverless GPU, A10)

# COMMAND ----------

# MAGIC %sh nvidia-smi

# COMMAND ----------

# Serving requirements. env_pack packs this environment as the express serving image.
# MAGIC %pip install vllm==0.11.2 transformers==4.57.6 mlflow==3.12.0 fastapi uvicorn httpx pydantic pyyaml openai hf_transfer databricks-sdk>=0.102.0
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration
# MAGIC
# MAGIC Mirrors `config.example.yaml`. Weights live under
# MAGIC `/Volumes/{catalog}/{schema}/{volume}/{model}/{base,adapters/<name>}`.

# COMMAND ----------

from databricks.sdk.service.serving import ServingModelWorkloadType

# Unity Catalog destination (weights already staged here by `--steps stage`).
CATALOG = "classic_stable_h0vpq7_catalog"
SCHEMA = "default"
MODEL_NAME = "multi_lora_vllm"
VOLUME = "weights"

UC_MODEL_NAME = f"{CATALOG}.{SCHEMA}.{MODEL_NAME}"
_WEIGHTS = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/{MODEL_NAME}"
BASE_MODEL = f"{_WEIGHTS}/base"
ADAPTERS = {  # name -> staged adapter path
    "ner": f"{_WEIGHTS}/adapters/ner",
    "code": f"{_WEIGHTS}/adapters/code",
}

# The proxy code, attached to the model as MLflow code paths (synced repo location).
PROXY_DIR = "/Workspace/Users/taylor.isbell@databricks.com/multi-adapter-serving/src/multi_adapter_serving/serving"
PROXY_CODE_PATHS = [f"{PROXY_DIR}/proxy.py", f"{PROXY_DIR}/proxy_entrypoint.py"]

# vLLM tuning.
DTYPE = "half"
MAX_LORAS = 2
MAX_LORA_RANK = 16

# Ports: Model Serving requires the runtime server on 8080; vLLM runs on private loopback 8081.
PROXY_PORT = 8080
VLLM_PORT = 8081

# Serving endpoint.
ENDPOINT_NAME = "multi-lora-vllm"
WORKLOAD_TYPE = ServingModelWorkloadType.GPU_MEDIUM
WORKLOAD_SIZE = "Small"
SCALE_TO_ZERO_ENABLED = True

# COMMAND ----------

# MAGIC %md
# MAGIC ## Entrypoint (multi-LoRA proxy in front of vLLM)
# MAGIC
# MAGIC Model Serving runs this command as the runtime server on port 8080. It launches
# MAGIC `proxy_entrypoint`, which starts vLLM (`--enable-lora`, all adapters loaded) on the private
# MAGIC port and serves the routing proxy on 8080.

# COMMAND ----------

import shlex

ENTRYPOINT = shlex.join([
    "python", "-u", "-m", "proxy_entrypoint",
    "--host", "0.0.0.0", "--port", str(PROXY_PORT),
    "--vllm-host", "127.0.0.1", "--vllm-port", str(VLLM_PORT),
    "--base-model", BASE_MODEL,
    "--served-model-name", "base",
    "--max-loras", str(MAX_LORAS),
    "--max-lora-rank", str(MAX_LORA_RANK),
    "--dtype", DTYPE,
    "--lora-modules", *[f"{name}={path}" for name, path in ADAPTERS.items()],
])
print(ENTRYPOINT)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log the model

# COMMAND ----------

import mlflow
from mlflow.pyfunc.model import ChatCompletionResponse, ChatModel


class MultiLoRAChatModel(ChatModel):
    # Placeholder. Serving runs the entrypoint above, not predict(); the empty response
    # just satisfies MLflow's log-time output validation.
    def predict(self, context, messages, params):
        return ChatCompletionResponse.from_dict({"choices": []})


model_info = mlflow.pyfunc.log_model(
    name=MODEL_NAME,
    python_model=MultiLoRAChatModel(),
    code_paths=PROXY_CODE_PATHS,
    metadata={"task": "llm/v1/chat", "entrypoint": ENTRYPOINT},
    pip_requirements=[
        "mlflow==3.12.0",
        "vllm==0.11.2",
        "transformers==4.57.6",
        "fastapi",
        "uvicorn",
        "httpx",
        "pydantic",
        "pyyaml",
        "databricks-sdk",
    ],
)
model_info.model_uri

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register to Unity Catalog
# MAGIC
# MAGIC `env_pack="databricks_model_serving"` is required: custom LLM serving is express-only, and
# MAGIC the endpoint will not deploy without it.

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
model_version = mlflow.register_model(
    model_info.model_uri, UC_MODEL_NAME, env_pack="databricks_model_serving"
)
print(f"registered {UC_MODEL_NAME} version {model_version.version} ({model_version.status})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create the endpoint
# MAGIC
# MAGIC (Optional — you can also run `multi-adapter-serve --steps deploy,smoke` locally instead.)

# COMMAND ----------

from datetime import timedelta

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

w = WorkspaceClient()
w.serving_endpoints.create_and_wait(
    name=ENDPOINT_NAME,
    config=EndpointCoreConfigInput(
        name=ENDPOINT_NAME,
        served_entities=[
            ServedEntityInput(
                entity_name=UC_MODEL_NAME,
                entity_version=str(model_version.version),
                workload_type=WORKLOAD_TYPE,
                workload_size=WORKLOAD_SIZE,
                scale_to_zero_enabled=SCALE_TO_ZERO_ENABLED,
            )
        ],
    ),
    timeout=timedelta(minutes=45),
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Query each adapter
# MAGIC
# MAGIC The adapter is selected via `custom_inputs.adapter` (the proxy rewrites the vLLM `model`).

# COMMAND ----------

from openai import OpenAI

ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
client = OpenAI(
    api_key=ctx.apiToken().get(),
    base_url=f"{ctx.apiUrl().get()}/serving-endpoints",
)

for adapter in ADAPTERS:
    resp = client.chat.completions.create(
        model=ENDPOINT_NAME,
        messages=[{"role": "user", "content": "Hello, who are you?"}],
        extra_body={"custom_inputs": {"adapter": adapter}},
    )
    print(f"[{adapter}] {resp.choices[0].message.content}")
