# openai-codex-sdk

Python SDK for [OpenAI Codex](https://github.com/openai/codex) APIs.

This is an unofficial fork. See [EcmaXp/openai-codex-sdk](https://github.com/EcmaXp/openai-codex-sdk) for the source.

## Requirements

- Python >= 3.14
- `codex` CLI binary installed and available on PATH

## Installation

### From Git with uv

```bash
uv add "openai-codex-sdk @ git+https://github.com/EcmaXp/openai-codex-sdk"
```

Or as a one-off with `uv run`:

```bash
uv run --python 3.14 --with "openai-codex-sdk @ git+https://github.com/EcmaXp/openai-codex-sdk" python my_script.py
```

### Inline script metadata (PEP 723)

For standalone scripts, add the dependency inline:

```python
# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "openai-codex-sdk @ git+https://github.com/EcmaXp/openai-codex-sdk",
# ]
# ///
```

Then run with `uv run my_script.py`.

## Quick start

```python
import asyncio
from openai_codex_sdk import Codex

async def main():
    codex = Codex()
    thread = codex.start_thread()
    turn = await thread.run("Explain what this repository does")
    print(turn.final_response)

asyncio.run(main())
```

## Streaming

```python
import asyncio
from openai_codex_sdk import Codex

async def main():
    codex = Codex()
    thread = codex.start_thread()
    result = await thread.run_streamed("Explain what this repository does")
    async for event in result.events:
        match event.type:
            case "item.completed":
                if event.item.type == "agent_message":
                    print(event.item.text)
            case "turn.completed":
                print(f"Tokens used: {event.usage.input_tokens}")

asyncio.run(main())
```

## Structured output

```python
import asyncio
from openai_codex_sdk import Codex, TurnOptions

async def main():
    codex = Codex()
    thread = codex.start_thread()

    schema = {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "status": {"type": "string", "enum": ["ok", "action_required"]},
        },
        "required": ["summary", "status"],
        "additionalProperties": False,
    }

    turn = await thread.run(
        "Summarize repository status",
        TurnOptions(output_schema=schema),
    )
    print(turn.final_response)

asyncio.run(main())
```

## Configuration

```python
from openai_codex_sdk import Codex, CodexOptions, ThreadOptions

codex = Codex(CodexOptions(
    base_url="https://api.example.com",
    api_key="sk-...",
    config={"approval_policy": "never"},
))

thread = codex.start_thread(ThreadOptions(
    model="o4-mini",
    sandbox_mode="workspace-write",
))
```

## Resuming threads

```python
thread = codex.start_thread()
turn = await thread.run("Hello")
thread_id = thread.id

# Later...
resumed = codex.resume_thread(thread_id)
turn = await resumed.run("Continue our conversation")
```

## License

Apache-2.0
