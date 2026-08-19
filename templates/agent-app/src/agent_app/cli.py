"""Command line entry point."""

from __future__ import annotations

import argparse
import sys

from .agent import ask, build_agent
from .config import settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ask the agent a question.")
    parser.add_argument("question", nargs="+", help="What to ask")
    parser.add_argument("--thread", default=None, help="Conversation thread id")
    args = parser.parse_args(argv)

    print(f"[model: {settings.model}]", file=sys.stderr)
    agent = build_agent()
    try:
        print(ask(agent, " ".join(args.question), thread_id=args.thread))
    except Exception as exc:  # degrade visibly - Module 12
        print(f"Could not complete that request ({type(exc).__name__}).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
