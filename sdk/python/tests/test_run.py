from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest

from openai_codex_sdk import (
    ApprovalMode,
    Codex,
    CodexOptions,
    LocalImageInput,
    ModelReasoningEffort,
    SandboxMode,
    TextInput,
    ThreadOptions,
    TurnOptions,
    WebSearchMode,
)

from ._codex_exec_spy import CodexExecSpy
from ._responses_proxy import (
    SseResponseBody,
    assistant_message,
    response_completed,
    response_failed,
    response_started,
    sse,
    start_responses_test_proxy,
)


def _expect_pair(args: list[str] | None, pair: tuple[str, str]) -> None:
    assert args is not None, "args is None"
    for i in range(len(args) - 1):
        if args[i] == pair[0] and args[i + 1] == pair[1]:
            return
    raise AssertionError(f"Pair {pair[0]} {pair[1]} not found in args: {args}")


def _collect_config_values(args: list[str] | None, key: str) -> list[str]:
    assert args is not None, "args is None"
    values: list[str] = []
    for i in range(len(args)):
        if args[i] != "--config":
            continue
        if i + 1 < len(args):
            override = args[i + 1]
            if override.startswith(f"{key}="):
                values.append(override)
    return values


async def test_returns_thread_events(codex_exec_path: str) -> None:
    proxy = start_responses_test_proxy(
        [sse(response_started(), assistant_message("Hi!"), response_completed())]
    )
    try:
        client = Codex(CodexOptions(
            codex_path_override=codex_exec_path, base_url=proxy.url, api_key="test"
        ))
        thread = client.start_thread()
        result = await thread.run("Hello, world!")

        assert len(result.items) == 1
        item = result.items[0]
        assert item.type == "agent_message"
        assert item.text == "Hi!"  # type: ignore[union-attr]
        assert item.id is not None  # type: ignore[union-attr]

        assert result.usage is not None
        assert result.usage.cached_input_tokens == 12
        assert result.usage.input_tokens == 42
        assert result.usage.output_tokens == 5

        assert thread.id is not None
    finally:
        proxy.close()


async def test_sends_previous_items_when_run_called_twice(codex_exec_path: str) -> None:
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
        await thread.run("first input")
        await thread.run("second input")

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


async def test_continues_thread_when_run_called_twice(codex_exec_path: str) -> None:
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
        await thread.run("first input")
        await thread.run("second input")

        assert len(proxy.requests) >= 2
        second_request = proxy.requests[1]
        payload = second_request.json

        last_user_text = payload["input"][-1]["content"][0]["text"]
        assert last_user_text == "second input"

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


async def test_resumes_thread_by_id(codex_exec_path: str) -> None:
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
        await original_thread.run("first input")

        assert original_thread.id is not None
        resumed_thread = client.resume_thread(original_thread.id)
        result = await resumed_thread.run("second input")

        assert resumed_thread.id == original_thread.id
        assert result.final_response == "Second response"

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


async def test_passes_turn_options_to_exec(codex_exec_path: str) -> None:
    proxy = start_responses_test_proxy([
        sse(
            response_started("response_1"),
            assistant_message("Turn options applied", "item_1"),
            response_completed("response_1"),
        ),
    ])
    spy = CodexExecSpy()
    spy.start()
    try:
        client = Codex(CodexOptions(
            codex_path_override=codex_exec_path, base_url=proxy.url, api_key="test"
        ))
        thread = client.start_thread(ThreadOptions(
            model="gpt-test-1",
            sandbox_mode=SandboxMode.WORKSPACE_WRITE,
        ))
        await thread.run("apply options")

        payload = proxy.requests[0]
        assert payload.json.get("model") == "gpt-test-1"

        assert len(spy.args) > 0
        command_args = spy.args[0]
        _expect_pair(command_args, ("--sandbox", "workspace-write"))
        _expect_pair(command_args, ("--model", "gpt-test-1"))
    finally:
        spy.restore()
        proxy.close()


async def test_passes_model_reasoning_effort(codex_exec_path: str) -> None:
    proxy = start_responses_test_proxy([
        sse(
            response_started("response_1"),
            assistant_message("Reasoning effort applied", "item_1"),
            response_completed("response_1"),
        ),
    ])
    spy = CodexExecSpy()
    spy.start()
    try:
        client = Codex(CodexOptions(
            codex_path_override=codex_exec_path, base_url=proxy.url, api_key="test"
        ))
        thread = client.start_thread(ThreadOptions(model_reasoning_effort=ModelReasoningEffort.HIGH))
        await thread.run("apply reasoning effort")

        command_args = spy.args[0]
        assert command_args is not None
        _expect_pair(command_args, ("--config", 'model_reasoning_effort="high"'))
    finally:
        spy.restore()
        proxy.close()


async def test_passes_network_access_enabled(codex_exec_path: str) -> None:
    proxy = start_responses_test_proxy([
        sse(
            response_started("response_1"),
            assistant_message("Network access enabled", "item_1"),
            response_completed("response_1"),
        ),
    ])
    spy = CodexExecSpy()
    spy.start()
    try:
        client = Codex(CodexOptions(
            codex_path_override=codex_exec_path, base_url=proxy.url, api_key="test"
        ))
        thread = client.start_thread(ThreadOptions(network_access_enabled=True))
        await thread.run("test network access")

        command_args = spy.args[0]
        assert command_args is not None
        _expect_pair(command_args, ("--config", "sandbox_workspace_write.network_access=true"))
    finally:
        spy.restore()
        proxy.close()


async def test_passes_web_search_enabled(codex_exec_path: str) -> None:
    proxy = start_responses_test_proxy([
        sse(
            response_started("response_1"),
            assistant_message("Web search enabled", "item_1"),
            response_completed("response_1"),
        ),
    ])
    spy = CodexExecSpy()
    spy.start()
    try:
        client = Codex(CodexOptions(
            codex_path_override=codex_exec_path, base_url=proxy.url, api_key="test"
        ))
        thread = client.start_thread(ThreadOptions(web_search_enabled=True))
        await thread.run("test web search")

        command_args = spy.args[0]
        assert command_args is not None
        _expect_pair(command_args, ("--config", 'web_search="live"'))
    finally:
        spy.restore()
        proxy.close()


async def test_passes_web_search_mode(codex_exec_path: str) -> None:
    proxy = start_responses_test_proxy([
        sse(
            response_started("response_1"),
            assistant_message("Web search cached", "item_1"),
            response_completed("response_1"),
        ),
    ])
    spy = CodexExecSpy()
    spy.start()
    try:
        client = Codex(CodexOptions(
            codex_path_override=codex_exec_path, base_url=proxy.url, api_key="test"
        ))
        thread = client.start_thread(ThreadOptions(web_search_mode=WebSearchMode.CACHED))
        await thread.run("test web search mode")

        command_args = spy.args[0]
        assert command_args is not None
        _expect_pair(command_args, ("--config", 'web_search="cached"'))
    finally:
        spy.restore()
        proxy.close()


async def test_passes_web_search_enabled_false(codex_exec_path: str) -> None:
    proxy = start_responses_test_proxy([
        sse(
            response_started("response_1"),
            assistant_message("Web search disabled", "item_1"),
            response_completed("response_1"),
        ),
    ])
    spy = CodexExecSpy()
    spy.start()
    try:
        client = Codex(CodexOptions(
            codex_path_override=codex_exec_path, base_url=proxy.url, api_key="test"
        ))
        thread = client.start_thread(ThreadOptions(web_search_enabled=False))
        await thread.run("test web search disabled")

        command_args = spy.args[0]
        assert command_args is not None
        _expect_pair(command_args, ("--config", 'web_search="disabled"'))
    finally:
        spy.restore()
        proxy.close()


async def test_passes_approval_policy(codex_exec_path: str) -> None:
    proxy = start_responses_test_proxy([
        sse(
            response_started("response_1"),
            assistant_message("Approval policy set", "item_1"),
            response_completed("response_1"),
        ),
    ])
    spy = CodexExecSpy()
    spy.start()
    try:
        client = Codex(CodexOptions(
            codex_path_override=codex_exec_path, base_url=proxy.url, api_key="test"
        ))
        thread = client.start_thread(ThreadOptions(approval_policy=ApprovalMode.ON_REQUEST))
        await thread.run("test approval policy")

        command_args = spy.args[0]
        assert command_args is not None
        _expect_pair(command_args, ("--config", 'approval_policy="on-request"'))
    finally:
        spy.restore()
        proxy.close()


async def test_passes_config_overrides_as_toml(codex_exec_path: str) -> None:
    proxy = start_responses_test_proxy([
        sse(
            response_started("response_1"),
            assistant_message("Config overrides applied", "item_1"),
            response_completed("response_1"),
        ),
    ])
    spy = CodexExecSpy()
    spy.start()
    try:
        client = Codex(CodexOptions(
            codex_path_override=codex_exec_path,
            base_url=proxy.url,
            api_key="test",
            config={
                "approval_policy": "never",
                "sandbox_workspace_write": {"network_access": True},
                "retry_budget": 3,
                "tool_rules": {"allow": ["git status", "git diff"]},
            },
        ))
        thread = client.start_thread()
        await thread.run("apply config overrides")

        command_args = spy.args[0]
        assert command_args is not None
        _expect_pair(command_args, ("--config", 'approval_policy="never"'))
        _expect_pair(command_args, ("--config", "sandbox_workspace_write.network_access=true"))
        _expect_pair(command_args, ("--config", "retry_budget=3"))
        _expect_pair(command_args, ("--config", 'tool_rules.allow=["git status", "git diff"]'))
    finally:
        spy.restore()
        proxy.close()


async def test_lets_thread_options_override_config(codex_exec_path: str) -> None:
    proxy = start_responses_test_proxy([
        sse(
            response_started("response_1"),
            assistant_message("Thread overrides applied", "item_1"),
            response_completed("response_1"),
        ),
    ])
    spy = CodexExecSpy()
    spy.start()
    try:
        client = Codex(CodexOptions(
            codex_path_override=codex_exec_path,
            base_url=proxy.url,
            api_key="test",
            config={"approval_policy": "never"},
        ))
        thread = client.start_thread(ThreadOptions(approval_policy=ApprovalMode.ON_REQUEST))
        await thread.run("override approval policy")

        command_args = spy.args[0]
        approval_policy_overrides = _collect_config_values(command_args, "approval_policy")
        assert approval_policy_overrides == [
            'approval_policy="never"',
            'approval_policy="on-request"',
        ]
        assert approval_policy_overrides[-1] == 'approval_policy="on-request"'
    finally:
        spy.restore()
        proxy.close()


async def test_allows_overriding_env(codex_exec_path: str) -> None:
    proxy = start_responses_test_proxy([
        sse(
            response_started("response_1"),
            assistant_message("Custom env", "item_1"),
            response_completed("response_1"),
        ),
    ])
    spy = CodexExecSpy()
    spy.start()
    os.environ["CODEX_ENV_SHOULD_NOT_LEAK"] = "leak"
    try:
        client = Codex(CodexOptions(
            codex_path_override=codex_exec_path,
            base_url=proxy.url,
            api_key="test",
            env={"CUSTOM_ENV": "custom"},
        ))
        thread = client.start_thread()
        await thread.run("custom env")

        spawn_env = spy.envs[0]
        assert spawn_env is not None
        assert spawn_env.get("CUSTOM_ENV") == "custom"
        assert "CODEX_ENV_SHOULD_NOT_LEAK" not in spawn_env
        assert spawn_env.get("OPENAI_BASE_URL") == proxy.url
        assert spawn_env.get("CODEX_API_KEY") == "test"
        assert "CODEX_INTERNAL_ORIGINATOR_OVERRIDE" in spawn_env
    finally:
        del os.environ["CODEX_ENV_SHOULD_NOT_LEAK"]
        spy.restore()
        proxy.close()


async def test_passes_additional_directories(codex_exec_path: str) -> None:
    proxy = start_responses_test_proxy([
        sse(
            response_started("response_1"),
            assistant_message("Additional directories applied", "item_1"),
            response_completed("response_1"),
        ),
    ])
    spy = CodexExecSpy()
    spy.start()
    try:
        client = Codex(CodexOptions(
            codex_path_override=codex_exec_path, base_url=proxy.url, api_key="test"
        ))
        thread = client.start_thread(ThreadOptions(
            additional_directories=["../backend", "/tmp/shared"],
        ))
        await thread.run("test additional dirs")

        command_args = spy.args[0]
        assert command_args is not None

        add_dir_args: list[str] = []
        for i in range(len(command_args)):
            if command_args[i] == "--add-dir" and i + 1 < len(command_args):
                add_dir_args.append(command_args[i + 1])
        assert add_dir_args == ["../backend", "/tmp/shared"]
    finally:
        spy.restore()
        proxy.close()


async def test_writes_output_schema_to_temp_file(codex_exec_path: str) -> None:
    proxy = start_responses_test_proxy([
        sse(
            response_started("response_1"),
            assistant_message("Structured response", "item_1"),
            response_completed("response_1"),
        ),
    ])
    spy = CodexExecSpy()
    spy.start()

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
        await thread.run("structured", TurnOptions(output_schema=schema))

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

        command_args = spy.args[0]
        assert command_args is not None
        schema_flag_index = command_args.index("--output-schema")
        assert schema_flag_index > -1
        schema_path = command_args[schema_flag_index + 1]
        assert isinstance(schema_path, str)
        # Schema temp file should be cleaned up after run completes
        assert not Path(schema_path).exists()
    finally:
        spy.restore()
        proxy.close()


async def test_combines_structured_text_input(codex_exec_path: str) -> None:
    proxy = start_responses_test_proxy([
        sse(
            response_started("response_1"),
            assistant_message("Combined input applied", "item_1"),
            response_completed("response_1"),
        ),
    ])
    try:
        client = Codex(CodexOptions(
            codex_path_override=codex_exec_path, base_url=proxy.url, api_key="test"
        ))
        thread = client.start_thread()
        await thread.run([
            TextInput(text="Describe file changes"),
            TextInput(text="Focus on impacted tests"),
        ])

        payload = proxy.requests[0]
        last_user = payload.json["input"][-1]
        assert last_user["content"][0]["text"] == "Describe file changes\n\nFocus on impacted tests"
    finally:
        proxy.close()


async def test_forwards_images(codex_exec_path: str) -> None:
    proxy = start_responses_test_proxy([
        sse(
            response_started("response_1"),
            assistant_message("Images applied", "item_1"),
            response_completed("response_1"),
        ),
    ])
    spy = CodexExecSpy()
    spy.start()

    temp_dir = tempfile.mkdtemp(prefix="codex-images-")
    image_paths = [
        os.path.join(temp_dir, "first.png"),
        os.path.join(temp_dir, "second.jpg"),
    ]
    for i, image_path in enumerate(image_paths):
        with open(image_path, "w") as f:
            f.write(f"image-{i}")

    try:
        client = Codex(CodexOptions(
            codex_path_override=codex_exec_path, base_url=proxy.url, api_key="test"
        ))
        thread = client.start_thread()
        await thread.run([
            TextInput(text="describe the images"),
            LocalImageInput(path=image_paths[0]),
            LocalImageInput(path=image_paths[1]),
        ])

        command_args = spy.args[0]
        assert command_args is not None
        forwarded_images: list[str] = []
        for i in range(len(command_args)):
            if command_args[i] == "--image" and i + 1 < len(command_args):
                forwarded_images.append(command_args[i + 1])
        assert forwarded_images == image_paths
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        spy.restore()
        proxy.close()


async def test_runs_in_provided_working_directory(codex_exec_path: str) -> None:
    proxy = start_responses_test_proxy([
        sse(
            response_started("response_1"),
            assistant_message("Working directory applied", "item_1"),
            response_completed("response_1"),
        ),
    ])
    spy = CodexExecSpy()
    spy.start()
    try:
        working_directory = tempfile.mkdtemp(prefix="codex-working-dir-")
        client = Codex(CodexOptions(
            codex_path_override=codex_exec_path, base_url=proxy.url, api_key="test"
        ))
        thread = client.start_thread(ThreadOptions(
            working_directory=working_directory,
            skip_git_repo_check=True,
        ))
        await thread.run("use custom working directory")

        command_args = spy.args[0]
        _expect_pair(command_args, ("--cd", working_directory))
    finally:
        spy.restore()
        proxy.close()


async def test_throws_if_not_git_and_no_skip(codex_exec_path: str) -> None:
    proxy = start_responses_test_proxy([
        sse(
            response_started("response_1"),
            assistant_message("Working directory applied", "item_1"),
            response_completed("response_1"),
        ),
    ])
    try:
        working_directory = tempfile.mkdtemp(prefix="codex-working-dir-")
        client = Codex(CodexOptions(
            codex_path_override=codex_exec_path, base_url=proxy.url, api_key="test"
        ))
        thread = client.start_thread(ThreadOptions(working_directory=working_directory))
        with pytest.raises(RuntimeError, match="Not inside a trusted directory"):
            await thread.run("use custom working directory")
    finally:
        proxy.close()


async def test_sets_codex_sdk_originator_header(codex_exec_path: str) -> None:
    proxy = start_responses_test_proxy(
        [sse(response_started(), assistant_message("Hi!"), response_completed())]
    )
    try:
        client = Codex(CodexOptions(
            codex_path_override=codex_exec_path, base_url=proxy.url, api_key="test"
        ))
        thread = client.start_thread()
        await thread.run("Hello, originator!")

        assert len(proxy.requests) > 0
        originator_header = proxy.requests[0].headers.get("originator")
        assert originator_header == "codex_sdk_py"
    finally:
        proxy.close()


async def test_throws_on_turn_failure(codex_exec_path: str) -> None:
    def _failure_bodies() -> Generator[SseResponseBody, None, None]:
        yield sse(response_started("response_1"))
        while True:
            yield sse(response_failed("rate limit exceeded"))

    proxy = start_responses_test_proxy(_failure_bodies())
    try:
        client = Codex(CodexOptions(
            codex_path_override=codex_exec_path, base_url=proxy.url, api_key="test"
        ))
        thread = client.start_thread()
        with pytest.raises(RuntimeError, match="stream disconnected before completion:"):
            await thread.run("fail")
    finally:
        proxy.close()
