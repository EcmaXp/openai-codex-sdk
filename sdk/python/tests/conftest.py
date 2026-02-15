from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture
def codex_exec_path() -> str:
    env_path = os.environ.get("CODEX_EXECUTABLE")
    if env_path:
        return env_path
    # Fallback to built binary relative to repo root
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    return str(repo_root / "codex-rs" / "target" / "debug" / "codex")
