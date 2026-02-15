from __future__ import annotations

from pydantic import BaseModel, ConfigDict

type CodexConfigValue = str | int | float | bool | list[CodexConfigValue] | dict[str, CodexConfigValue]
type CodexConfigObject = dict[str, CodexConfigValue]


class CodexOptions(BaseModel):
    model_config = ConfigDict(frozen=True)

    codex_path_override: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    config: CodexConfigObject | None = None
    env: dict[str, str] | None = None
