#!/usr/bin/env python3
"""Interactive REPL demonstrating streaming with the Codex SDK."""

from __future__ import annotations

import asyncio
import sys

from openai_codex_sdk import Codex, ThreadEvent, ThreadItem


def handle_item_completed(item: ThreadItem) -> None:
    match item.type:
        case "agent_message":
            print(f"Assistant: {item.text}")
        case "reasoning":
            print(f"Reasoning: {item.text}")
        case "command_execution":
            exit_text = f" Exit code {item.exit_code}." if item.exit_code is not None else ""
            print(f"Command {item.command} {item.status}.{exit_text}")
        case "file_change":
            for change in item.changes:
                print(f"File {change.kind} {change.path}")
        case _:
            pass


def handle_item_updated(item: ThreadItem) -> None:
    match item.type:
        case "todo_list":
            print("Todo:")
            for todo in item.items:
                mark = "x" if todo.completed else " "
                print(f"\t {mark} {todo.text}")
        case _:
            pass


def handle_event(event: ThreadEvent) -> None:
    match event.type:
        case "item.completed":
            handle_item_completed(event.item)
        case "item.updated" | "item.started":
            handle_item_updated(event.item)
        case "turn.completed":
            print(
                f"Used {event.usage.input_tokens} input tokens, "
                f"{event.usage.cached_input_tokens} cached input tokens, "
                f"{event.usage.output_tokens} output tokens."
            )
        case "turn.failed":
            print(f"Turn failed: {event.error.message}", file=sys.stderr)
        case _:
            pass


async def main() -> None:
    codex = Codex()
    thread = codex.start_thread()

    while True:
        try:
            input_text = input(">")
        except (EOFError, KeyboardInterrupt):
            break

        trimmed = input_text.strip()
        if not trimmed:
            continue

        result = await thread.run_streamed(input_text)
        async for event in result.events:
            handle_event(event)


if __name__ == "__main__":
    asyncio.run(main())
