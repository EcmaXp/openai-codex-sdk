from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Discriminator, TypeAdapter


class AgentMessageItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    type: Literal["agent_message"] = "agent_message"
    text: str


class ReasoningItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    type: Literal["reasoning"] = "reasoning"
    text: str


class CommandExecutionStatus(StrEnum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    DECLINED = "declined"


class CommandExecutionItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    type: Literal["command_execution"] = "command_execution"
    command: str
    aggregated_output: str
    exit_code: int | None = None
    status: CommandExecutionStatus


class PatchChangeKind(StrEnum):
    ADD = "add"
    DELETE = "delete"
    UPDATE = "update"


class FileUpdateChange(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: str
    kind: PatchChangeKind


class PatchApplyStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"


class FileChangeItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    type: Literal["file_change"] = "file_change"
    changes: list[FileUpdateChange]
    status: PatchApplyStatus


class McpToolCallStatus(StrEnum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class McpToolCallItemResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    content: list[Any]
    structured_content: Any = None


class McpToolCallItemError(BaseModel):
    model_config = ConfigDict(frozen=True)

    message: str


class McpToolCallItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    type: Literal["mcp_tool_call"] = "mcp_tool_call"
    server: str
    tool: str
    arguments: Any
    result: McpToolCallItemResult | None = None
    error: McpToolCallItemError | None = None
    status: McpToolCallStatus


class WebSearchItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    type: Literal["web_search"] = "web_search"
    query: str


class TodoItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    completed: bool


class TodoListItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    type: Literal["todo_list"] = "todo_list"
    items: list[TodoItem]


class ErrorItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    type: Literal["error"] = "error"
    message: str


type ThreadItem = Annotated[
    AgentMessageItem
    | ReasoningItem
    | CommandExecutionItem
    | FileChangeItem
    | McpToolCallItem
    | WebSearchItem
    | TodoListItem
    | ErrorItem,
    Discriminator("type"),
]

ThreadItemAdapter: TypeAdapter[ThreadItem] = TypeAdapter(ThreadItem)
