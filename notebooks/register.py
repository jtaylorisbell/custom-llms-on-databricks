# Databricks notebook source
# MAGIC %md
# MAGIC # Register + serve the multi-LoRA custom LLM
# MAGIC
# MAGIC Multi-LoRA adaptation of the Databricks
# MAGIC [custom LLM serving](https://docs.databricks.com/aws/en/machine-learning/model-serving/serve-custom-llms)
# MAGIC starter, run on **Serverless GPU** (A10). One base model serves several LoRA adapters behind a
# MAGIC thin proxy (`serving/proxy.py`) that maps `custom_inputs.adapter` onto vLLM's `model` field
# MAGIC (Model Serving occupies `model` with the endpoint name). Requests without an adapter go to the
# MAGIC base model.
# MAGIC
# MAGIC The base model, adapters, and proxy are bundled into the logged model as artifacts, so nothing
# MAGIC is fetched at serve time. The weights must already be staged in the UC Volume below.
# MAGIC
# MAGIC **Stage** (widget / job parameter): `deploy` runs everything; `verify` stops after the
# MAGIC in-notebook test, without registering the model or touching the endpoint.

# COMMAND ----------

# MAGIC %sh nvidia-smi

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set up the environment
# MAGIC The custom-LLM starter's install set, proven on this workspace. `env_pack` packages this
# MAGIC environment with the model; the proxy's own deps (fastapi/uvicorn/httpx) ship with it.

# COMMAND ----------

# MAGIC %pip install vllm==0.11.2 transformers==4.57.6 openai==2.17.0 "opencv-python-headless==4.12.*" mlflow==3.12.0 hf_transfer==0.1.9 "databricks-sdk>=0.102.0" fastapi uvicorn httpx
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

import ctypes
import os
import shlex

from databricks.sdk.service.serving import ServingModelWorkloadType

# Log from Serverless GPU only: a CPU environment packages CPU deps and the endpoint fails to start.
try:
    ctypes.CDLL("libcuda.so.1")
except OSError:
    raise RuntimeError("No GPU found. Run this on Serverless GPU with an A10.") from None

dbutils.widgets.dropdown("stage", "deploy", ["deploy", "verify"], "Stage")
STAGE = dbutils.widgets.get("stage")

# Unity Catalog destination; the weights live in the Volume (downloaded on the first run).
CATALOG = "classic_stable_h0vpq7_catalog"
SCHEMA = "default"
MODEL_NAME = "multi_lora_vllm"
VOLUME = "weights"

UC_MODEL_NAME = f"{CATALOG}.{SCHEMA}.{MODEL_NAME}"
_WEIGHTS = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/{MODEL_NAME}"

# name -> staged UC path. Each lands in the serving container's artifacts/<name> dir, where the
# entrypoint runs, so the entrypoint refers to the weights by these bare names.
WEIGHTS = {
    "base": f"{_WEIGHTS}/base",
    "ner": f"{_WEIGHTS}/adapters/ner",
    "code": f"{_WEIGHTS}/adapters/code",
}
ADAPTERS = ["ner", "code"]

# Hugging Face source of each, downloaded into the Volume only when it's missing there.
HF_REPOS = {
    "base": "mistralai/Mistral-7B-Instruct-v0.2",
    "ner": "predibase/Mistral-7B-Instruct-v0.2-conllpp-medusa-lora",
    "code": "predibase/Mistral-7B-Instruct-v0.2-magicoder-medusa-lora",
}

# The proxy ships in this bundle's serving/ dir; the notebook runs from notebooks/.
PROXY_DIR = os.path.abspath("../serving")

# vLLM tuning.
SERVED_MODEL_NAME = "base"
DTYPE = "half"
MAX_LORAS = 2
MAX_LORA_RANK = 16

# Serving endpoint. Custom LLM endpoints don't autoscale: workload_size fixes the replica count.
ENDPOINT_NAME = "multi-lora-vllm"
WORKLOAD_TYPE = ServingModelWorkloadType.GPU_MEDIUM  # 1x A10G (24 GB)
WORKLOAD_SIZE = "Small"
# Off keeps the GPU billing while idle; on saves that, but a cold start takes minutes.
SCALE_TO_ZERO_ENABLED = False

# COMMAND ----------

# MAGIC %md
# MAGIC ## Entrypoint (multi-LoRA proxy in front of vLLM)
# MAGIC Model Serving runs this command from the model's artifacts dir: the proxy on port 8080, and
# MAGIC vLLM (`--enable-lora`, all adapters loaded) behind it on a private loopback port. Serving
# MAGIC leaves that port to the entrypoint (`0` = any free one), because the platform runs its own
# MAGIC metrics exporter on 8081. The in-notebook test pins ports 3080/3081, since notebooks may
# MAGIC only use 3000-3999.

# COMMAND ----------


def entrypoint(port: int, vllm_port: int = 0) -> str:
    return shlex.join(
        [
            "python", "-u", "-m", "proxy_entrypoint",
            "--host", "0.0.0.0", "--port", str(port),
            "--vllm-host", "127.0.0.1", "--vllm-port", str(vllm_port),
            "--base-model", "base",
            "--served-model-name", SERVED_MODEL_NAME,
            "--max-loras", str(MAX_LORAS),
            "--max-lora-rank", str(MAX_LORA_RANK),
            "--dtype", DTYPE,
            "--lora-modules", *[f"{a}={a}" for a in ADAPTERS],
        ]
    )


print(entrypoint(8080))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Download the weights (first run only)
# MAGIC Fetch any weights missing from the Volume from Hugging Face, so a fresh workspace or a wiped
# MAGIC Volume rebuilds from scratch; later runs find them there and skip this. Each repo downloads to
# MAGIC local disk first and is then copied in, because Volumes don't support the out-of-order writes
# MAGIC of a parallel download. The base skips its duplicate `.bin` weights (about 14 GB).

# COMMAND ----------

import shutil
import tempfile

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  # parallel downloads; read when huggingface_hub is imported
from huggingface_hub import snapshot_download

hub_metadata = (".cache", ".git*")  # download bookkeeping, never needed to serve


def ensure_weights(name: str) -> None:
    dest = WEIGHTS[name]
    # Copied in last, so it only exists once a copy has finished; an interrupted copy is redone.
    marker = "adapter_config.json" if name in ADAPTERS else "config.json"
    if os.path.exists(f"{dest}/{marker}"):
        print(f"{name}: found in {dest}")
        return
    print(f"{name}: downloading {HF_REPOS[name]} to {dest}")
    local = snapshot_download(
        HF_REPOS[name],
        local_dir=f"{tempfile.mkdtemp()}/{name}",
        ignore_patterns=None if name in ADAPTERS else ["*.bin", "*.pth"],
    )
    shutil.copytree(local, dest, ignore=shutil.ignore_patterns(marker, *hub_metadata), dirs_exist_ok=True)
    shutil.copy(f"{local}/{marker}", dest)
    shutil.rmtree(local)  # free the local disk for staging below


if not os.path.exists(f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"):
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.catalog import VolumeType

    WorkspaceClient().volumes.create(CATALOG, SCHEMA, VOLUME, VolumeType.MANAGED)
for name in WEIGHTS:
    ensure_weights(name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage the artifacts on local disk
# MAGIC Mirror the serving container's artifacts dir on local disk (`/Workspace` doesn't take multi-GB
# MAGIC files). The base carries redundant `.bin` weights next to its `.safetensors`; copy safetensors
# MAGIC only so the bundle isn't doubled.

# COMMAND ----------

ARTIFACTS_DIR = tempfile.mkdtemp()
shutil.copytree(
    WEIGHTS["base"],
    f"{ARTIFACTS_DIR}/base",
    ignore=shutil.ignore_patterns("*.bin", "*.pth", *hub_metadata),
)
for adapter in ADAPTERS:
    shutil.copytree(
        WEIGHTS[adapter], f"{ARTIFACTS_DIR}/{adapter}", ignore=shutil.ignore_patterns(*hub_metadata)
    )
for f in ("proxy.py", "proxy_entrypoint.py"):
    shutil.copy(f"{PROXY_DIR}/{f}", ARTIFACTS_DIR)
print(ARTIFACTS_DIR, sorted(os.listdir(ARTIFACTS_DIR)))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test the entrypoint in the notebook
# MAGIC Run the exact serving command from the staged artifacts dir and hit the routes Model Serving
# MAGIC uses: `GET /v1/models` (its readiness probe) and `POST /invocations`, for the base model and
# MAGIC each adapter. A broken entrypoint fails here in minutes instead of after an endpoint deploy.

# COMMAND ----------

import signal
import subprocess
import time

import requests

URL = "http://localhost:3080"
TEST_PROMPT = "Tim Cook said Apple will open a new campus in Austin, Texas. Who and what are mentioned?"
LOG = f"{tempfile.gettempdir()}/entrypoint.log"


def log_tail():
    with open(LOG) as f:
        return f.read()[-5000:]


def is_ready():
    try:
        return requests.get(URL + "/health", timeout=2).ok
    except requests.ConnectionError:
        return False


with open(LOG, "w") as log:
    server = subprocess.Popen(
        entrypoint(3080, 3081),
        shell=True,
        cwd=ARTIFACTS_DIR,
        stdout=log,
        stderr=log,
        start_new_session=True,
    )

try:
    deadline = time.time() + 30 * 60
    while not is_ready():
        if server.poll() is not None:
            raise RuntimeError(f"Entrypoint exited with code {server.returncode}. End of its log:\n{log_tail()}")
        if time.time() > deadline:
            raise RuntimeError(f"Entrypoint not ready within 30 minutes. End of its log:\n{log_tail()}")
        time.sleep(5)

    models = requests.get(URL + "/v1/models", timeout=10)
    models.raise_for_status()
    print("served models:", [m["id"] for m in models.json()["data"]])

    for adapter in [None, *ADAPTERS]:
        request = {"messages": [{"role": "user", "content": TEST_PROMPT}], "max_tokens": 128}
        if adapter:
            request["custom_inputs"] = {"adapter": adapter}
        reply = requests.post(URL + "/invocations", json=request, timeout=180)
        if not reply.ok:
            raise RuntimeError(f"[{adapter or 'base'}] HTTP {reply.status_code}: {reply.text[:500]}")
        body = reply.json()
        # vLLM reports the LoRA (or base) that actually answered, which proves the proxy's routing.
        if body["model"] != (adapter or SERVED_MODEL_NAME):
            raise RuntimeError(f"Asked for {adapter or 'base'}, but {body['model']} answered")
        print(f"[{body['model']}] {body['choices'][0]['message']['content']!r}")
finally:
    if server.poll() is None:  # stop the server to free the GPU
        os.killpg(server.pid, signal.SIGTERM)

# COMMAND ----------

# Its own cell: exit replaces the output of the cell it runs in, which would hide the replies above.
if STAGE == "verify":
    dbutils.notebook.exit("verify passed")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log the model
# MAGIC Bundle base + adapters + proxy as artifacts. The `predict` placeholder just satisfies MLflow's
# MAGIC log-time validation; serving runs the entrypoint, not `predict`.

# COMMAND ----------

import mlflow
from mlflow.pyfunc.model import ChatCompletionResponse, ChatModel

# Upload the multi-GB weights with the Databricks SDK and longer timeouts (as the starters do).
os.environ["MLFLOW_USE_DATABRICKS_SDK_MODEL_ARTIFACTS_REPO_FOR_UC"] = "true"
os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "600"
os.environ["MLFLOW_ARTIFACT_UPLOAD_DOWNLOAD_TIMEOUT"] = "600"


class MultiLoRAChatModel(ChatModel):
    def predict(self, context, messages, params):
        return ChatCompletionResponse.from_dict({"choices": []})


model_info = mlflow.pyfunc.log_model(
    name=MODEL_NAME,
    python_model=MultiLoRAChatModel(),
    artifacts={
        **{name: f"{ARTIFACTS_DIR}/{name}" for name in WEIGHTS},
        "proxy": f"{ARTIFACTS_DIR}/proxy.py",
        "proxy_entrypoint": f"{ARTIFACTS_DIR}/proxy_entrypoint.py",
    },
    metadata={"task": "llm/v1/chat", "entrypoint": entrypoint(8080)},
    # Log only mlflow, like the starter: vLLM/torch and the proxy deps come from the env_pack'd
    # environment, NOT requirements.txt. Listing vllm/transformers here makes the serving env
    # re-resolve a crypto/OpenSSL stack whose FIPS provider fails its self-test in vLLM's model
    # inspection subprocess ("FATAL FIPS SELFTEST FAILURE").
    extra_pip_requirements=["mlflow==3.12.0"],
)
model_info.model_uri

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register to Unity Catalog
# MAGIC `env_pack="databricks_model_serving"` is required: custom LLM serving is built on express
# MAGIC deployments, which package this notebook's environment with the model.

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
model_version = mlflow.register_model(
    model_info.model_uri, UC_MODEL_NAME, env_pack="databricks_model_serving"
)
print(f"registered {UC_MODEL_NAME} v{model_version.version} ({model_version.status})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create or update the endpoint
# MAGIC An update is zero-downtime: the current version keeps serving until the new one is ready, and
# MAGIC stays up if the new one fails.

# COMMAND ----------

from datetime import timedelta

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.sdk.service.serving import (
    ChatMessage,
    ChatMessageRole,
    EndpointCoreConfigInput,
    EndpointStateConfigUpdate,
    ServedEntityInput,
)

w = WorkspaceClient()
# Endpoints only accept an entrypoint model once its version is staged as an express deployment.
# register_model stages it but only warns on failure, and staging doesn't last (a months-old version
# is rejected until re-staged), so stage explicitly to fail loudly here.
w.api_client.do(
    "POST",
    "/api/2.0/serving-endpoints:stageDeployment",
    body={"model_name": UC_MODEL_NAME, "model_version": str(model_version.version)},
)
served = [
    ServedEntityInput(
        entity_name=UC_MODEL_NAME,
        entity_version=str(model_version.version),
        workload_type=WORKLOAD_TYPE,
        workload_size=WORKLOAD_SIZE,
        scale_to_zero_enabled=SCALE_TO_ZERO_ENABLED,
    )
]
print(f"Follow progress at {w.config.host}/ml/endpoints/{ENDPOINT_NAME}")

if ENDPOINT_NAME in [e.name for e in w.serving_endpoints.list()]:
    # A new config is rejected (RESOURCE_CONFLICT) while another update is still in flight.
    while w.serving_endpoints.get(ENDPOINT_NAME).state.config_update == EndpointStateConfigUpdate.IN_PROGRESS:
        time.sleep(30)
    w.serving_endpoints.update_config_and_wait(
        name=ENDPOINT_NAME, served_entities=served, timeout=timedelta(minutes=90)
    )
else:
    w.serving_endpoints.create_and_wait(
        name=ENDPOINT_NAME,
        config=EndpointCoreConfigInput(name=ENDPOINT_NAME, served_entities=served),
        timeout=timedelta(minutes=90),
    )

# A new endpoint can take a few minutes after it reports ready before it accepts requests.
deadline = time.time() + 10 * 60
while True:
    try:
        w.serving_endpoints.query(
            name=ENDPOINT_NAME, messages=[ChatMessage(role=ChatMessageRole.USER, content="Hi")], max_tokens=1
        )
        break
    except NotFound:
        if time.time() > deadline:
            raise
        time.sleep(15)
print("Endpoint is ready.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Query each adapter
# MAGIC Pick the adapter with `custom_inputs.adapter`; omit it for the base model. The SDK's `query`
# MAGIC has no `custom_inputs` field, so post the chat body to `/invocations` directly.

# COMMAND ----------

for adapter in [None, *ADAPTERS]:
    request = {"messages": [{"role": "user", "content": TEST_PROMPT}], "max_tokens": 128}
    if adapter:
        request["custom_inputs"] = {"adapter": adapter}
    reply = w.api_client.do("POST", f"/serving-endpoints/{ENDPOINT_NAME}/invocations", body=request)
    print(f"[{adapter or 'base'}] model={reply.get('model')}: {reply['choices'][0]['message']['content']!r}")
