from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ._codex_exec_spy import CodexExecSpy


class _FakeStreamReader:
    def __init__(self, data: bytes) -> None:
        self._data = data
        self._pos = 0

    def __aiter__(self) -> _FakeStreamReader:
        return self

    async def __anext__(self) -> bytes:
        if self._pos >= len(self._data):
            raise StopAsyncIteration
        end = self._data.find(b"\n", self._pos)
        if end == -1:
            line = self._data[self._pos :]
            self._pos = len(self._data)
        else:
            line = self._data[self._pos : end + 1]
            self._pos = end + 1
        return line

    async def read(self) -> bytes:
        remaining = self._data[self._pos :]
        self._pos = len(self._data)
        return remaining


class FakeProcess:
    """Simulates an asyncio subprocess for testing."""

    def __init__(self, exit_code: int = 2, stderr_data: bytes = b"boom") -> None:
        self.stdin = MagicMock()
        self.stdin.write = MagicMock()
        self.stdin.close = MagicMock()
        self.stdin.drain = AsyncMock()
        self.stdin.wait_closed = AsyncMock()
        self._exit_code = exit_code
        self._stderr_data = stderr_data
        self._stdout_data = b""
        self.returncode: int | None = None
        self.pid = 12345

    async def wait(self) -> int:
        self.returncode = self._exit_code
        return self._exit_code

    @property
    def stdout(self) -> Any:
        return _FakeStreamReader(self._stdout_data)

    @property
    def stderr(self) -> Any:
        return _FakeStreamReader(self._stderr_data)

    def kill(self) -> None:
        self.returncode = -9


async def test_rejects_when_exit_happens_before_stdout_closes() -> None:
    from openai_codex_sdk._exec import CodexExec, CodexExecArgs

    fake = FakeProcess(exit_code=2, stderr_data=b"boom")

    with patch("openai_codex_sdk._exec.asyncio.create_subprocess_exec", return_value=fake):
        codex_exec = CodexExec("codex")
        with pytest.raises(RuntimeError, match="Codex Exec exited"):
            async for _ in codex_exec.run(CodexExecArgs(input="hi")):
                pass


async def test_places_resume_args_before_image_args() -> None:
    from openai_codex_sdk._exec import CodexExec, CodexExecArgs

    spy = CodexExecSpy()
    spy.start()

    try:
        codex_exec = CodexExec("codex")
        try:
            async for _ in codex_exec.run(
                CodexExecArgs(input="hi", images=["img.png"], thread_id="thread-id")
            ):
                pass
        except (RuntimeError, OSError):
            pass  # Expected - binary doesn't exist or exits with error

        assert len(spy.args) > 0
        command_args = spy.args[0]

        resume_index = command_args.index("resume")
        image_index = command_args.index("--image")
        assert resume_index > -1
        assert image_index > -1
        assert resume_index < image_index
    finally:
        spy.restore()
