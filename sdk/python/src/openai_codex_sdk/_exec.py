from __future__ import annotations

import asyncio
import json
import math
import os
import re
import shutil
from collections.abc import AsyncGenerator

from pydantic import BaseModel, ConfigDict

from .codex_options import CodexConfigObject, CodexConfigValue
from .thread_options import ApprovalMode, ModelReasoningEffort, SandboxMode, WebSearchMode

_INTERNAL_ORIGINATOR_ENV = "CODEX_INTERNAL_ORIGINATOR_OVERRIDE"
_PYTHON_SDK_ORIGINATOR = "codex_sdk_py"


class CodexExecArgs(BaseModel):
    model_config = ConfigDict(frozen=True)

    input: str

    base_url: str | None = None
    api_key: str | None = None
    thread_id: str | None = None
    images: list[str] | None = None
    model: str | None = None
    sandbox_mode: SandboxMode | None = None
    working_directory: str | None = None
    additional_directories: list[str] | None = None
    skip_git_repo_check: bool | None = None
    output_schema_file: str | None = None
    model_reasoning_effort: ModelReasoningEffort | None = None
    network_access_enabled: bool | None = None
    web_search_mode: WebSearchMode | None = None
    web_search_enabled: bool | None = None
    approval_policy: ApprovalMode | None = None


class CodexExec:
    def __init__(
        self,
        executable_path: str | None = None,
        env: dict[str, str] | None = None,
        config_overrides: CodexConfigObject | None = None,
    ) -> None:
        self._executable_path = executable_path or _find_codex_path()
        self._env_override = env
        self._config_overrides = config_overrides

    async def run(self, args: CodexExecArgs) -> AsyncGenerator[str, None]:
        command_args: list[str] = ["exec", "--json"]

        if self._config_overrides:
            for override in _serialize_config_overrides(self._config_overrides):
                command_args.extend(["--config", override])

        if args.model:
            command_args.extend(["--model", args.model])

        if args.sandbox_mode:
            command_args.extend(["--sandbox", args.sandbox_mode])

        if args.working_directory:
            command_args.extend(["--cd", args.working_directory])

        if args.additional_directories:
            for d in args.additional_directories:
                command_args.extend(["--add-dir", d])

        if args.skip_git_repo_check:
            command_args.append("--skip-git-repo-check")

        if args.output_schema_file:
            command_args.extend(["--output-schema", args.output_schema_file])

        if args.model_reasoning_effort:
            command_args.extend(["--config", f'model_reasoning_effort="{args.model_reasoning_effort}"'])

        if args.network_access_enabled is not None:
            command_args.extend([
                "--config",
                f"sandbox_workspace_write.network_access={str(args.network_access_enabled).lower()}",
            ])

        if args.web_search_mode:
            command_args.extend(["--config", f'web_search="{args.web_search_mode}"'])
        elif args.web_search_enabled is True:
            command_args.extend(["--config", 'web_search="live"'])
        elif args.web_search_enabled is False:
            command_args.extend(["--config", 'web_search="disabled"'])

        if args.approval_policy:
            command_args.extend(["--config", f'approval_policy="{args.approval_policy}"'])

        if args.thread_id:
            command_args.extend(["resume", args.thread_id])

        if args.images:
            for image in args.images:
                command_args.extend(["--image", image])

        env: dict[str, str] = {}
        if self._env_override is not None:
            env.update(self._env_override)
        else:
            for key, value in os.environ.items():
                env[key] = value

        if _INTERNAL_ORIGINATOR_ENV not in env:
            env[_INTERNAL_ORIGINATOR_ENV] = _PYTHON_SDK_ORIGINATOR

        if args.base_url:
            env["OPENAI_BASE_URL"] = args.base_url

        if args.api_key:
            env["CODEX_API_KEY"] = args.api_key

        proc = await asyncio.create_subprocess_exec(
            self._executable_path,
            *command_args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

        try:
            assert proc.stdin is not None
            proc.stdin.write(args.input.encode())
            await proc.stdin.drain()
            proc.stdin.close()
            await proc.stdin.wait_closed()

            assert proc.stdout is not None
            async for raw_line in proc.stdout:
                line = raw_line.decode().rstrip("\n")
                if line:
                    yield line

            stderr_data = b""
            if proc.stderr is not None:
                stderr_data = await proc.stderr.read()

            return_code = await proc.wait()
            if return_code != 0:
                detail = f"code {return_code}"
                raise RuntimeError(
                    f"Codex Exec exited with {detail}: {stderr_data.decode()}"
                )
        finally:
            if proc.returncode is None:
                try:
                    proc.kill()
                    await proc.wait()
                except ProcessLookupError:
                    pass


def _find_codex_path() -> str:
    path = shutil.which("codex")
    if path is None:
        raise FileNotFoundError(
            "Unable to locate Codex CLI binary. Ensure 'codex' is installed and on PATH."
        )
    return path


def _serialize_config_overrides(config_overrides: CodexConfigObject) -> list[str]:
    overrides: list[str] = []
    _flatten_config_overrides(config_overrides, "", overrides)
    return overrides


def _flatten_config_overrides(
    value: CodexConfigValue,
    prefix: str,
    overrides: list[str],
) -> None:
    if not isinstance(value, dict):
        if prefix:
            overrides.append(f"{prefix}={_to_toml_value(value, prefix)}")
            return
        else:
            raise ValueError("Codex config overrides must be a plain object")

    entries = list(value.items())
    if not prefix and not entries:
        return

    if prefix and not entries:
        overrides.append(f"{prefix}={{}}")
        return

    for key, child in entries:
        if not key:
            raise ValueError("Codex config override keys must be non-empty strings")
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(child, dict):
            _flatten_config_overrides(child, path, overrides)
        else:
            overrides.append(f"{path}={_to_toml_value(child, path)}")


_TOML_BARE_KEY = re.compile(r"^[A-Za-z0-9_-]+$")


def _to_toml_value(value: CodexConfigValue, path: str) -> str:
    if isinstance(value, str):
        return json.dumps(value)
    elif isinstance(value, bool):
        return "true" if value else "false"
    elif isinstance(value, int):
        return str(value)
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Codex config override at {path} must be a finite number")
        return str(value)
    elif isinstance(value, list):
        rendered = [_to_toml_value(item, f"{path}[{i}]") for i, item in enumerate(value)]
        return f"[{', '.join(rendered)}]"
    else:
        parts: list[str] = []
        for key, child in value.items():
            if not key:
                raise ValueError("Codex config override keys must be non-empty strings")
            parts.append(f"{_format_toml_key(key)} = {_to_toml_value(child, f'{path}.{key}')}")
        return "{" + ", ".join(parts) + "}"


def _format_toml_key(key: str) -> str:
    return key if _TOML_BARE_KEY.match(key) else json.dumps(key)
