#!/usr/bin/env python3
"""Structured output using a raw JSON schema."""

from __future__ import annotations

import asyncio

from openai_codex_sdk import Codex, TurnOptions


async def main() -> None:
    codex = Codex()
    thread = codex.start_thread()

    schema = {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "status": {"type": "string", "enum": ["ok", "action_required"]},
        },
        "required": ["summary", "status"],
        "additionalProperties": False,
    }

    turn = await thread.run(
        "Summarize repository status",
        TurnOptions(output_schema=schema),
    )
    print(turn.final_response)


if __name__ == "__main__":
    asyncio.run(main())
