from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path


@asynccontextmanager
async def output_schema_file(schema: dict[str, object] | None) -> AsyncIterator[str | None]:
    if schema is None:
        yield None
        return

    schema_dir = Path(tempfile.mkdtemp(prefix="codex-output-schema-"))
    schema_path = schema_dir / "schema.json"
    try:
        schema_path.write_text(json.dumps(schema), encoding="utf-8")
        yield str(schema_path)
    finally:
        shutil.rmtree(schema_dir, ignore_errors=True)
