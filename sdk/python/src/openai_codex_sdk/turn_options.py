from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class TurnOptions(BaseModel):
    model_config = ConfigDict(frozen=True)

    output_schema: dict[str, Any] | None = None
