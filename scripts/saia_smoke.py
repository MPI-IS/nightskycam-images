#!/usr/bin/env python3
"""
Manual smoke test of the website LLM agent against the real SAIA API.

Not a pytest file — run it by hand (needs SAIA_API_KEY in the env):

    # List the model ids SAIA currently serves (to pin SAIA_MODEL):
    python scripts/saia_smoke.py --list-models

    # Ask one question against a database:
    python scripts/saia_smoke.py --db /data2/nightskycam/database/nightskycam.db \\
        "how many tiff images were taken in 2024?"

Environment: SAIA_API_KEY (required), SAIA_BASE_URL, SAIA_MODEL.
"""

import argparse
import os
from pathlib import Path
import sys

import openai

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nightskycam_images.website import (  # noqa: E402
    DEFAULT_SAIA_BASE_URL,
    DEFAULT_SAIA_MODEL,
)
from nightskycam_images.website.agent import run_agent  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", help="question to ask the agent")
    parser.add_argument("--db", type=Path, help="path to nightskycam.db")
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="print the model ids served by SAIA and exit",
    )
    args = parser.parse_args()

    api_key = os.environ.get("SAIA_API_KEY")
    if not api_key:
        sys.exit("SAIA_API_KEY is not set")
    base_url = os.environ.get("SAIA_BASE_URL") or DEFAULT_SAIA_BASE_URL
    model = os.environ.get("SAIA_MODEL") or DEFAULT_SAIA_MODEL

    client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=30)

    if args.list_models:
        for entry in client.models.list():
            marker = "  <-- configured" if entry.id == model else ""
            print(f"{entry.id}{marker}")
        return

    if not args.question or not args.db:
        parser.error("a question and --db are required (or use --list-models)")

    print(f"model: {model}\nbase_url: {base_url}\n")
    answer = run_agent(
        args.db,
        client,
        model,
        [{"role": "user", "content": args.question}],
        on_step=lambda step: print(f"  [step] {step.summary}"),
    )
    print(f"\nrounds: {answer.rounds}  truncated: {answer.truncated}")
    print(f"\n{answer.text}")
    if answer.image_stems:
        print("\nimages:")
        for stem in answer.image_stems:
            print(f"  {stem}")


if __name__ == "__main__":
    main()
