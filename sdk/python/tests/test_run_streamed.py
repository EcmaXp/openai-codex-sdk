from __future__ import annotations

from typing import Any

from openai_codex_sdk import Codex, CodexOptions, TurnOptions

from ._responses_proxy import (
    assistant_message,
    response_completed,
    response_started,
    sse,
    start_responses_test_proxy,
)


async def _drain_events(events: Any) -> list[Any]:
    collected: list[Any] = []
    async for event in events:
        collected.append(event)
    return collected


async def test_returns_thread_events(codex_exec_path: str) -> None:
    proxy = start_responses_test_proxy(
        [sse(response_started(), assistant_message("Hi!"), response_completed())]
    )
    try:
        client = Codex(CodexOptions(
            codex_path_override=codex_exec_path, base_url=proxy.url, api_key="test"
        ))
        thread = client.start_thread()
        result = await thread.run_streamed("Hello, world!")

        events = await _drain_events(result.events)

        # Verify event sequence: thread.started, turn.started, item.completed, turn.completed
        assert len(events) == 4

        assert events[0].type == "thread.started"
        assert events[0].thread_id is not None

        assert events[1].type == "turn.started"

        assert events[2].type == "item.completed"
        assert events[2].item.type == "agent_message"
        assert events[2].item.text == "Hi!"

        assert events[3].type == "turn.completed"
        assert events[3].usage.cached_input_tokens == 12
        assert events[3].usage.input_tokens == 42
        assert events[3].usage.output_tokens == 5

        assert thread.id is not None
    finally:
        proxy.close()


async def test_sends_previous_items_when_run_streamed_called_twice(
    codex_exec_path: str,
) -> None:
    proxy = start_responses_test_proxy([
        sse(
            response_started("response_1"),
            assistant_message("First response", "item_1"),
            response_completed("response_1"),
        ),
        sse(
            response_started("response_2"),
            assistant_message("Second response", "item_2"),
            response_completed("response_2"),
        ),
    ])
    try:
        client = Codex(CodexOptions(
            codex_path_override=codex_exec_path, base_url=proxy.url, api_key="test"
        ))
        thread = client.start_thread()

        first = await thread.run_streamed("first input")
        await _drain_events(first.events)

        second = await thread.run_streamed("second input")
        await _drain_events(second.events)

        assert len(proxy.requests) >= 2
        second_request = proxy.requests[1]
        payload = second_request.json

        assistant_entry = next(
            (entry for entry in payload["input"] if entry.get("role") == "assistant"), None
        )
        assert assistant_entry is not None
        assistant_text = next(
            (
                item["text"]
                for item in assistant_entry.get("content", [])
                if item.get("type") == "output_text"
            ),
            None,
        )
        assert assistant_text == "First response"
    finally:
        proxy.close()


async def test_resumes_thread_by_id_when_streaming(codex_exec_path: str) -> None:
    proxy = start_responses_test_proxy([
        sse(
            response_started("response_1"),
            assistant_message("First response", "item_1"),
            response_completed("response_1"),
        ),
        sse(
            response_started("response_2"),
            assistant_message("Second response", "item_2"),
            response_completed("response_2"),
        ),
    ])
    try:
        client = Codex(CodexOptions(
            codex_path_override=codex_exec_path, base_url=proxy.url, api_key="test"
        ))

        original_thread = client.start_thread()
        first = await original_thread.run_streamed("first input")
        await _drain_events(first.events)

        assert original_thread.id is not None
        resumed_thread = client.resume_thread(original_thread.id)
        second = await resumed_thread.run_streamed("second input")
        await _drain_events(second.events)

        assert resumed_thread.id == original_thread.id

        assert len(proxy.requests) >= 2
        second_request = proxy.requests[1]
        payload = second_request.json

        assistant_entry = next(
            (entry for entry in payload["input"] if entry.get("role") == "assistant"), None
        )
        assert assistant_entry is not None
        assistant_text = next(
            (
                item["text"]
                for item in assistant_entry.get("content", [])
                if item.get("type") == "output_text"
            ),
            None,
        )
        assert assistant_text == "First response"
    finally:
        proxy.close()


async def test_applies_output_schema_when_streaming(codex_exec_path: str) -> None:
    proxy = start_responses_test_proxy([
        sse(
            response_started("response_1"),
            assistant_message("Structured response", "item_1"),
            response_completed("response_1"),
        ),
    ])

    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
        },
        "required": ["answer"],
        "additionalProperties": False,
    }

    try:
        client = Codex(CodexOptions(
            codex_path_override=codex_exec_path, base_url=proxy.url, api_key="test"
        ))
        thread = client.start_thread()
        streamed = await thread.run_streamed("structured", TurnOptions(output_schema=schema))
        await _drain_events(streamed.events)

        assert len(proxy.requests) >= 1
        payload = proxy.requests[0]
        text = payload.json.get("text")
        assert text is not None
        assert text.get("format") == {
            "name": "codex_output_schema",
            "type": "json_schema",
            "strict": True,
            "schema": schema,
        }
    finally:
        proxy.close()
