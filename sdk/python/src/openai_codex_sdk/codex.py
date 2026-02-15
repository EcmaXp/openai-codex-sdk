from __future__ import annotations

from ._exec import CodexExec
from .codex_options import CodexOptions
from .thread import Thread
from .thread_options import ThreadOptions


class Codex:
    def __init__(self, options: CodexOptions | None = None) -> None:
        if options is None:
            options = CodexOptions()
        self._exec = CodexExec(
            executable_path=options.codex_path_override,
            env=options.env,
            config_overrides=options.config,
        )
        self._options = options

    def start_thread(self, options: ThreadOptions | None = None) -> Thread:
        if options is None:
            options = ThreadOptions()
        return Thread(self._exec, self._options, options)

    def resume_thread(self, id: str, options: ThreadOptions | None = None) -> Thread:
        if options is None:
            options = ThreadOptions()
        return Thread(self._exec, self._options, options, id=id)
