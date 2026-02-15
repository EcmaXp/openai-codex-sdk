from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Discriminator

from ._exec import CodexExec, CodexExecArgs
from ._output_schema_file import output_schema_file
from .codex_options import CodexOptions
from .events import ThreadEvent, ThreadEventAdapter, ThreadError, Usage
from .items import ThreadItem
from .thread_options import ThreadOptions
from .turn_options import TurnOptions


class TextInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["text"] = "text"
    text: str


class LocalImageInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["local_image"] = "local_image"
    path: str


type UserInput = Annotated[TextInput | LocalImageInput, Discriminator("type")]
type Input = str | list[UserInput]


class Turn(BaseModel):
    model_config = ConfigDict(frozen=True)

    items: list[ThreadItem]
    final_response: str
    usage: Usage | None


RunResult = Turn


class StreamedTurn:
    def __init__(self, events: AsyncGenerator[ThreadEvent, None]) -> None:
        self.events = events


RunStreamedResult = StreamedTurn


class Thread:
    def __init__(
        self,
        exec_: CodexExec,
        options: CodexOptions,
        thread_options: ThreadOptions,
        id: str | None = None,
    ) -> None:
        self._exec = exec_
        self._options = options
        self._thread_options = thread_options
        self._id = id

    @property
    def id(self) -> str | None:
        return self._id

    async def run(self, input: Input, turn_options: TurnOptions | None = None) -> Turn:
        if turn_options is None:
            turn_options = TurnOptions()

        items: list[ThreadItem] = []
        final_response: str = ""
        usage: Usage | None = None
        turn_failure: ThreadError | None = None

        async for event in self._run_streamed_internal(input, turn_options):
            if event.type == "item.completed":
                if event.item.type == "agent_message":
                    final_response = event.item.text
                items.append(event.item)
            elif event.type == "turn.completed":
                usage = event.usage
            elif event.type == "turn.failed":
                turn_failure = event.error
                break

        if turn_failure is not None:
            raise RuntimeError(turn_failure.message)

        return Turn(items=items, final_response=final_response, usage=usage)

    async def run_streamed(self, input: Input, turn_options: TurnOptions | None = None) -> StreamedTurn:
        if turn_options is None:
            turn_options = TurnOptions()
        return StreamedTurn(events=self._run_streamed_internal(input, turn_options))

    async def _run_streamed_internal(
        self,
        input: Input,
        turn_options: TurnOptions,
    ) -> AsyncGenerator[ThreadEvent, None]:
        async with output_schema_file(turn_options.output_schema) as schema_path:
            prompt, images = _normalize_input(input)
            opts = self._thread_options

            generator = self._exec.run(
                CodexExecArgs(
                    input=prompt,
                    base_url=self._options.base_url,
                    api_key=self._options.api_key,
                    thread_id=self._id,
                    images=images or None,
                    model=opts.model,
                    sandbox_mode=opts.sandbox_mode,
                    working_directory=opts.working_directory,
                    skip_git_repo_check=opts.skip_git_repo_check,
                    output_schema_file=schema_path,
                    model_reasoning_effort=opts.model_reasoning_effort,
                    network_access_enabled=opts.network_access_enabled,
                    web_search_mode=opts.web_search_mode,
                    web_search_enabled=opts.web_search_enabled,
                    approval_policy=opts.approval_policy,
                    additional_directories=opts.additional_directories,
                )
            )

            async for line in generator:
                try:
                    data = json.loads(line)
                except Exception as exc:
                    raise RuntimeError(f"Failed to parse item: {line}") from exc

                event = ThreadEventAdapter.validate_python(data)
                if event.type == "thread.started":
                    self._id = event.thread_id
                yield event


def _normalize_input(input: Input) -> tuple[str, list[str]]:
    if isinstance(input, str):
        return input, []

    prompt_parts: list[str] = []
    images: list[str] = []

    for item in input:
        if item.type == "text":
            prompt_parts.append(item.text)
        elif item.type == "local_image":
            images.append(item.path)

    return "\n\n".join(prompt_parts), images
