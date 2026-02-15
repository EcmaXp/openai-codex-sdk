#!/usr/bin/env python3
"""Structured output using a Pydantic model schema."""

from __future__ import annotations

import asyncio
from enum import StrEnum

from pydantic import BaseModel

from openai_codex_sdk import Codex, TurnOptions


class Status(StrEnum):
    OK = "ok"
    ACTION_REQUIRED = "action_required"


class RepoSummary(BaseModel):
    summary: str
    status: Status


async def main() -> None:
    codex = Codex()
    thread = codex.start_thread()

    turn = await thread.run(
        "Summarize repository status",
        TurnOptions(output_schema=RepoSummary.model_json_schema()),
    )
    print(turn.final_response)


if __name__ == "__main__":
    asyncio.run(main())
