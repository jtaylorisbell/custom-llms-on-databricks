# /// script
# requires-python = ">=3.11"
# dependencies = ["databricks-sdk>=0.102.0"]
# ///
"""Query the multi-LoRA endpoint once per adapter, plus the base model, over OAuth.

    uv run scripts/smoke_test.py --profile fevm-aws
"""

from __future__ import annotations

import argparse
import sys

from databricks.sdk import WorkspaceClient

PROMPT = "Tim Cook said Apple will open a new campus in Austin, Texas. Who and what are mentioned?"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", default="fevm-aws", help="Databricks CLI (OAuth) profile.")
    parser.add_argument("--endpoint", default="multi-lora-vllm")
    parser.add_argument("--adapters", nargs="*", default=["ner", "code"])
    args = parser.parse_args()

    w = WorkspaceClient(profile=args.profile)
    failures = 0
    for adapter in [None, *args.adapters]:
        label = adapter or "base"
        body = {"messages": [{"role": "user", "content": PROMPT}], "max_tokens": 128}
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
