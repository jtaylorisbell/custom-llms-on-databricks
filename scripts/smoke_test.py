# /// script
# requires-python = ">=3.11"
# dependencies = ["databricks-sdk>=0.102.0"]
# ///
"""Query the multi-LoRA endpoint once per adapter, plus the base model, over OAuth.

    uv run scripts/smoke_test.py [--profile PROFILE]
"""

from __future__ import annotations

import argparse
import sys

from databricks.sdk import WorkspaceClient

# The prompt format the ner adapter was trained on (including its double space).
PROMPT = (
    "Your task is a Named Entity Recognition (NER) task. Predict the category of each entity, then place the "
    "entity into the list associated with the  category in an output JSON payload. Below is an example:\n"
    'Input: EU rejects German call to boycott British lamb . Output: {"person": [], "organization": ["EU"], '
    '"location": [], "miscellaneous": ["German", "British"]}\nNow, complete the task.\n'
    "Input: The Chevrolet Silverado EV is built at GM's Factory Zero plant in Hamtramck, Michigan. Output: "
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--profile",
        help="Databricks CLI (OAuth) profile. Defaults to $DATABRICKS_CONFIG_PROFILE, then DEFAULT.",
    )
    parser.add_argument("--endpoint", default="multi-lora-vllm")
    parser.add_argument("--adapters", nargs="*", default=["ner"])
    args = parser.parse_args()

    w = WorkspaceClient(profile=args.profile)
    failures = 0
    for adapter in [None, *args.adapters]:
        label = adapter or "base"
        body = {"messages": [{"role": "user", "content": PROMPT}], "max_tokens": 128, "temperature": 0}
        if adapter:
            body["custom_inputs"] = {"adapter": adapter}
        try:
            reply = w.api_client.do(
                "POST", f"/serving-endpoints/{args.endpoint}/invocations", body=body
            )
        except Exception as exc:  # report every adapter, don't stop at the first failure
            failures += 1
            print(f"[{label}] FAILED: {exc}", file=sys.stderr)
            continue
        content = reply["choices"][0]["message"]["content"]
        print(f"[{label}] model={reply.get('model')}: {content!r}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
