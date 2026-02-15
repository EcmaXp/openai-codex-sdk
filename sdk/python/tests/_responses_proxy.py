from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any


@dataclass
class SseResponseBody:
    kind: str = "sse"
    events: list[dict[str, Any]] = field(default_factory=lambda: [])


@dataclass
class RecordedRequest:
    body: str
    json: dict[str, Any]
    headers: dict[str, str]


@dataclass
class ResponsesProxy:
    url: str
    close: Callable[[], None]
    requests: list[RecordedRequest]


DEFAULT_RESPONSE_ID = "resp_mock"
DEFAULT_MESSAGE_ID = "msg_mock"

DEFAULT_USAGE = {
    "input_tokens": 42,
    "input_tokens_details": {"cached_tokens": 12},
    "output_tokens": 5,
    "output_tokens_details": None,
    "total_tokens": 47,
}


def _format_sse_event(event: dict[str, Any]) -> str:
    return f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"


def start_responses_test_proxy(
    response_bodies: list[SseResponseBody] | Iterator[SseResponseBody],
    status_code: int = 200,
) -> ResponsesProxy:
    if isinstance(response_bodies, list):
        body_iter: Iterator[SseResponseBody] = iter(response_bodies)
    else:
        body_iter = response_bodies

    requests: list[RecordedRequest] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            if self.path == "/responses":
                content_length = int(self.headers.get("Content-Length", 0))
                raw_body = self.rfile.read(content_length).decode("utf-8")
                parsed = json.loads(raw_body)
                hdrs = {k: v for k, v in self.headers.items()}
                requests.append(RecordedRequest(body=raw_body, json=parsed, headers=hdrs))

                self.send_response(status_code)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()

                try:
                    response_body = next(body_iter)
                except StopIteration:
                    self.wfile.write(b"")
                    return

                for event in response_body.events:
                    self.wfile.write(_format_sse_event(event).encode("utf-8"))
                self.wfile.flush()
                return

            self.send_response(404)
            self.end_headers()

        def log_message(self, format: str, *args: Any) -> None:
            pass  # Suppress request logging

    server = HTTPServer(("127.0.0.1", 0), Handler)
    address = server.server_address
    url = f"http://{address[0]}:{address[1]}"

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def close() -> None:
        server.shutdown()
        thread.join(timeout=5)

    return ResponsesProxy(url=url, close=close, requests=requests)


def sse(*events: dict[str, Any]) -> SseResponseBody:
    return SseResponseBody(kind="sse", events=list(events))


def response_started(response_id: str = DEFAULT_RESPONSE_ID) -> dict[str, Any]:
    return {
        "type": "response.created",
        "response": {
            "id": response_id,
        },
    }


def assistant_message(text: str, item_id: str = DEFAULT_MESSAGE_ID) -> dict[str, Any]:
    return {
        "type": "response.output_item.done",
        "item": {
            "type": "message",
            "role": "assistant",
            "id": item_id,
            "content": [
                {
                    "type": "output_text",
                    "text": text,
                },
            ],
        },
    }


def shell_call() -> dict[str, Any]:
    import random
    import string

    call_id = "call_id" + "".join(random.choices(string.ascii_lowercase + string.digits, k=10))
    command = ["bash", "-lc", "echo 'Hello, world!'"]
    return {
        "type": "response.output_item.done",
        "item": {
            "type": "function_call",
            "call_id": call_id,
            "name": "shell",
            "arguments": json.dumps({
                "command": command,
                "timeout_ms": 100,
            }),
        },
    }


def response_failed(error_message: str) -> dict[str, Any]:
    return {
        "type": "error",
        "error": {"code": "rate_limit_exceeded", "message": error_message},
    }


def response_completed(
    response_id: str = DEFAULT_RESPONSE_ID,
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if usage is None:
        usage = DEFAULT_USAGE
    input_details = (
        {**usage["input_tokens_details"]} if usage.get("input_tokens_details") else None
    )
    output_details = (
        {**usage["output_tokens_details"]} if usage.get("output_tokens_details") else None
    )
    return {
        "type": "response.completed",
        "response": {
            "id": response_id,
            "usage": {
                "input_tokens": usage["input_tokens"],
                "input_tokens_details": input_details,
                "output_tokens": usage["output_tokens"],
                "output_tokens_details": output_details,
                "total_tokens": usage["total_tokens"],
            },
        },
    }
