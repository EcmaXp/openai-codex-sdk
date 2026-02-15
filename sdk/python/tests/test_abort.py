from __future__ import annotations

import asyncio
from collections.abc import Generator

import pytest

from openai_codex_sdk import Codex, CodexOptions

from ._responses_proxy import (
    SseResponseBody,
    assistant_message,
    response_completed,
    response_started,
    shell_call,
    sse,
    start_responses_test_proxy,
)


def _infinite_shell_call() -> Generator[SseResponseBody, None, None]:
    while True:
        yield sse(response_started(), shell_call(), response_completed())


async def test_aborts_run_when_task_cancelled(codex_exec_path: str) -> None:
    proxy = start_responses_test_proxy(_infinite_shell_call())
    try:
        client = Codex(CodexOptions(
            codex_path_override=codex_exec_path, base_url=proxy.url, api_key="test"
        ))
        thread = client.start_thread()

        async def do_run() -> None:
            await thread.run("Hello, world!")

        task = asyncio.create_task(do_run())
        # Cancel immediately
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        proxy.close()


async def test_aborts_run_streamed_when_task_cancelled(codex_exec_path: str) -> None:
    proxy = start_responses_test_proxy(_infinite_shell_call())
    try:
        client = Codex(CodexOptions(
            codex_path_override=codex_exec_path, base_url=proxy.url, api_key="test"
        ))
        thread = client.start_thread()

        async def do_run() -> None:
            result = await thread.run_streamed("Hello, world!")
            async for _ in result.events:
                pass  # drain events

        task = asyncio.create_task(do_run())
        # Cancel immediately
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        proxy.close()


async def test_aborts_run_during_execution(codex_exec_path: str) -> None:
    proxy = start_responses_test_proxy(_infinite_shell_call())
    try:
        client = Codex(CodexOptions(
            codex_path_override=codex_exec_path, base_url=proxy.url, api_key="test"
        ))
        thread = client.start_thread()

        async def do_run() -> None:
            await thread.run("Hello, world!")

        task = asyncio.create_task(do_run())

        # Allow some time for execution to begin, then cancel
        await asyncio.sleep(0.01)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        proxy.close()


async def test_aborts_run_streamed_during_iteration(codex_exec_path: str) -> None:
    proxy = start_responses_test_proxy(_infinite_shell_call())
    try:
        client = Codex(CodexOptions(
            codex_path_override=codex_exec_path, base_url=proxy.url, api_key="test"
        ))
        thread = client.start_thread()

        event_count = 0

        async def do_run() -> None:
            nonlocal event_count
            result = await thread.run_streamed("Hello, world!")
            async for _ in result.events:
                event_count += 1
                if event_count >= 5:
                    raise asyncio.CancelledError()

        task = asyncio.create_task(do_run())

        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        proxy.close()


async def test_completes_normally_when_not_cancelled(codex_exec_path: str) -> None:
    proxy = start_responses_test_proxy(
        [sse(response_started(), assistant_message("Hi!"), response_completed())]
    )
    try:
        client = Codex(CodexOptions(
            codex_path_override=codex_exec_path, base_url=proxy.url, api_key="test"
        ))
        thread = client.start_thread()

        # Run without cancellation - should complete normally
        result = await thread.run("Hello, world!")

        assert result.final_response == "Hi!"
        assert len(result.items) == 1
    finally:
        proxy.close()
