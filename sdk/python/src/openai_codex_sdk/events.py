from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Discriminator, TypeAdapter

from .items import ThreadItem


class Usage(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_tokens: int
    cached_input_tokens: int
    output_tokens: int


class ThreadError(BaseModel):
    model_config = ConfigDict(frozen=True)

    message: str


class ThreadStartedEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["thread.started"] = "thread.started"
    thread_id: str


class TurnStartedEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["turn.started"] = "turn.started"


class TurnCompletedEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["turn.completed"] = "turn.completed"
    usage: Usage


class TurnFailedEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["turn.failed"] = "turn.failed"
    error: ThreadError


class ItemStartedEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["item.started"] = "item.started"
    item: ThreadItem


class ItemUpdatedEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["item.updated"] = "item.updated"
    item: ThreadItem


class ItemCompletedEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["item.completed"] = "item.completed"
    item: ThreadItem


class ThreadErrorEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["error"] = "error"
    message: str


type ThreadEvent = Annotated[
    ThreadStartedEvent
    | TurnStartedEvent
    | TurnCompletedEvent
    | TurnFailedEvent
    | ItemStartedEvent
    | ItemUpdatedEvent
    | ItemCompletedEvent
    | ThreadErrorEvent,
    Discriminator("type"),
]

ThreadEventAdapter: TypeAdapter[ThreadEvent] = TypeAdapter(ThreadEvent)
