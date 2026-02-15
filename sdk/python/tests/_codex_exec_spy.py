from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import patch


class CodexExecSpy:
    """Patches asyncio.create_subprocess_exec to capture spawn arguments and environment."""

    def __init__(self) -> None:
        self.args: list[list[str]] = []
        self.envs: list[dict[str, str] | None] = []
        self._patcher: Any = None
        self._original: Any = None

    def start(self) -> None:
        self._original = asyncio.create_subprocess_exec

        async def mock_create_subprocess(*args: Any, **kwargs: Any) -> Any:
            self.args.append([str(a) for a in args[1:]])  # skip executable, take command args
            self.envs.append(kwargs.get("env"))
            return await self._original(*args, **kwargs)

        self._patcher = patch(
            "asyncio.create_subprocess_exec", side_effect=mock_create_subprocess
        )
        self._patcher.start()

    def restore(self) -> None:
        if self._patcher:
            self._patcher.stop()
